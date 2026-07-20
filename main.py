import os

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora


CROSS_DATASET_TARGETS = [
    'caltech101',
    'dtd',
    'eurosat',
    'fgvc',
    'food101',
    'oxford_flowers',
    'oxford_pets',
    'stanford_cars',
    'sun397',
    'ucf101',
]
DOMAIN_GENERALIZATION_TARGETS = [
    'imagenetv2',
    'imagenet_sketch',
    'imagenet_a',
    'imagenet_r',
]
TARGET_DATASET_GROUPS = {
    'cross_dataset': CROSS_DATASET_TARGETS,
    'domain_generalization': DOMAIN_GENERALIZATION_TARGETS,
    'all_transfer': CROSS_DATASET_TARGETS + DOMAIN_GENERALIZATION_TARGETS,
    'all': CROSS_DATASET_TARGETS + DOMAIN_GENERALIZATION_TARGETS,
}


def setup_distributed(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.distributed = args.world_size > 1
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        torch.cuda.set_device(0)


def main():
    args = get_arguments()
    setup_distributed(args)
    set_random_seed(args.seed)

    try:
        if args.batch_size < 1:
            raise ValueError('batch_size must be at least 1')
        if args.batch_size % args.world_size != 0:
            raise ValueError(
                f'Effective global batch size {args.batch_size} must be '
                f'divisible by world size ({args.world_size})')
        if getattr(args, 'mrsa', False):
            if not 0. <= args.dropout_rate < 1.:
                raise ValueError(
                    'MRSA requires --dropout_rate in [0, 1)')
        if getattr(args, 'v_rpr', False):
            if args.encoder not in {'vision', 'both'}:
                raise ValueError('V-RPR requires --encoder vision or both')
            if args.r < 2:
                raise ValueError('V-RPR requires --r >= 2')
        if getattr(args, 'dp_vrpr', False):
            if args.encoder not in {'vision', 'both'}:
                raise ValueError(
                    'DP-VRPR requires --encoder vision or both')
            if args.r < 2:
                raise ValueError('DP-VRPR requires --r >= 2')
        enabled_visual_partitions = sum((
            bool(getattr(args, 'v_rpr', False)),
            bool(getattr(args, 'dp_vrpr', False)),
        ))
        if enabled_visual_partitions > 1:
            raise ValueError(
                '--v_rpr and --dp_vrpr are mutually exclusive')
        per_gpu_batch_size = args.batch_size // args.world_size

        clip_model, preprocess = clip.load(args.backbone)
        clip_model.eval()
        logit_scale = 100

        if args.rank == 0:
            print(
                f"Distributed training: {args.world_size} GPU(s); "
                f"effective global batch size: {args.batch_size}; "
                f"per-GPU batch size: {per_gpu_batch_size}; "
                f"batch formula: {per_gpu_batch_size} × "
                f"{args.world_size} = "
                f"{args.batch_size}")
            print("Preparing dataset.")

        dataset = build_dataset(
            args.dataset, args.root_path, args.shots, preprocess,
            setting=args.setting)

        if args.rank == 0 and args.setting == 'base2new':
            print(
                f'Base-to-novel split: {len(dataset.classnames)} base classes, '
                f'{len(dataset.test_new_classnames)} novel classes.')

        if uses_torchvision_loader(args.dataset):
            val_loader = torch.utils.data.DataLoader(
                dataset.val, batch_size=args.eval_batch_size, num_workers=8,
                shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                dataset.test, batch_size=args.eval_batch_size, num_workers=8,
                shuffle=False, pin_memory=True)
            test_new_loader = (
                torch.utils.data.DataLoader(
                    dataset.test_new, batch_size=args.eval_batch_size,
                    num_workers=8, shuffle=False, pin_memory=True)
                if dataset.test_new is not None else None)
        else:
            val_loader = build_data_loader(
                data_source=dataset.val, batch_size=args.eval_batch_size,
                is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
            test_loader = build_data_loader(
                data_source=dataset.test, batch_size=args.eval_batch_size,
                is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
            test_new_loader = (
                build_data_loader(
                    data_source=dataset.test_new,
                    batch_size=args.eval_batch_size, is_train=False,
                    tfm=preprocess, shuffle=False, num_workers=8)
                if dataset.test_new is not None else None)

        if test_new_loader is not None:
            test_loader = (test_loader, test_new_loader)

        train_loader = None
        if not args.eval_only:
            if args.log_interval < 1:
                raise ValueError('log_interval must be at least 1')
            if args.val_interval < 0:
                raise ValueError('val_interval cannot be negative')
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=224, scale=(0.08, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711))
            ])
            train_sampler = None
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset.train_x, num_replicas=args.world_size,
                    rank=args.rank, shuffle=True, seed=args.seed)

            if uses_torchvision_loader(args.dataset):
                train_loader = torch.utils.data.DataLoader(
                    dataset.train_x, batch_size=per_gpu_batch_size,
                    num_workers=8, shuffle=train_sampler is None,
                    sampler=train_sampler, pin_memory=True)
            else:
                train_loader = build_data_loader(
                    data_source=dataset.train_x,
                    batch_size=per_gpu_batch_size,
                    tfm=train_transform, is_train=True,
                    shuffle=train_sampler is None, sampler=train_sampler,
                    num_workers=8)

        transfer_eval_data = None
        target_datasets = expand_target_datasets(args.target_datasets)
        if target_datasets:
            if args.rank == 0:
                print(
                    "Preparing transfer evaluation target(s): "
                    + ", ".join(target_datasets))
            transfer_eval_data = []
            for target_dataset in target_datasets:
                try:
                    transfer_eval_data.append(
                        build_transfer_eval_data(
                            target_dataset, args, preprocess))
                except FileNotFoundError as error:
                    if args.rank == 0:
                        print(
                            f"WARNING: skipping transfer target "
                            f"{target_dataset!r} because data is missing: "
                            f"{error}")
            if not transfer_eval_data:
                raise RuntimeError(
                    "No transfer evaluation targets could be prepared. "
                    "Check --target_datasets and --root_path.")

        run_lora(
            args, clip_model, logit_scale, dataset,
            train_loader, val_loader, test_loader,
            transfer_eval_data=transfer_eval_data)
    finally:
        if args.distributed:
            dist.destroy_process_group()


def uses_torchvision_loader(dataset_name):
    return dataset_name == 'imagenet'


def build_eval_loader(args, dataset_name, dataset, preprocess):
    if uses_torchvision_loader(dataset_name):
        return torch.utils.data.DataLoader(
            dataset.test, batch_size=args.eval_batch_size, num_workers=8,
            shuffle=False, pin_memory=True)

    return build_data_loader(
        data_source=dataset.test,
        batch_size=args.eval_batch_size,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
        num_workers=8)


def expand_target_datasets(target_datasets):
    expanded = []
    for dataset in target_datasets:
        expanded.extend(TARGET_DATASET_GROUPS.get(dataset, [dataset]))

    deduplicated = []
    seen = set()
    for dataset in expanded:
        dataset = 'fgvc' if dataset == 'fgvc_aircraft' else dataset
        if dataset in seen:
            continue
        deduplicated.append(dataset)
        seen.add(dataset)
    return deduplicated


def build_transfer_eval_data(dataset_name, args, preprocess):
    dataset = build_dataset(
        dataset_name, args.root_path, args.shots, preprocess,
        setting='standard')
    loader = build_eval_loader(args, dataset_name, dataset, preprocess)
    return {
        'name': dataset_name,
        'dataset': dataset,
        'loader': loader,
    }


if __name__ == '__main__':
    main()
