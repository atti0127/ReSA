import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from utils import *

from loralib.utils import (
    disabled_adapters,
    mark_only_lora_as_trainable,
    apply_lora,
    get_adapter_metadata,
    get_adapter_save_dir,
    get_lora_parameters,
    load_lora,
    save_lora,
)


class TrainingLogger:
    def __init__(self, args):
        self.enabled = (
            is_main_process(args)
            and args.save_path is not None
        )
        self.file = None
        self.start_time = time.time()
        if self.enabled:
            save_dir = get_adapter_save_dir(args)
            os.makedirs(save_dir, exist_ok=True)
            log_filename = (
                'eval_log.jsonl'
                if getattr(args, 'eval_only', False)
                else 'training_log.jsonl')
            log_mode = 'a' if getattr(args, 'eval_only', False) else 'w'
            self.file = open(
                os.path.join(save_dir, log_filename),
                log_mode, encoding='utf-8')

    def log(self, event, **metrics):
        if not self.enabled:
            return
        record = {
            'event': event,
            'elapsed_seconds': time.time() - self.start_time,
            **metrics,
        }
        record = self._json_safe(record)
        self.file.write(json.dumps(record, sort_keys=True) + '\n')
        self.file.flush()

    def _json_safe(self, value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class CLIPFeatureEncoder(nn.Module):
    """Routes trainable CLIP feature extraction through DDP.forward."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images=None, texts=None):
        outputs = {}
        if images is not None:
            image_features = self.clip_model.encode_image(images)
            outputs['raw_image_features'] = image_features
            outputs['image_features'] = image_features
        if texts is not None:
            outputs['text_features'] = self.clip_model.encode_text(texts)
        return outputs


def is_main_process(args):
    return getattr(args, 'rank', 0) == 0


def shard_texts(args, texts):
    if not getattr(args, 'distributed', False):
        return texts, len(texts)

    num_texts = len(texts)
    chunk_size = (num_texts + args.world_size - 1) // args.world_size
    padded_size = chunk_size * args.world_size
    if padded_size > num_texts:
        texts = torch.cat([
            texts,
            texts[-1:].expand(padded_size - num_texts, -1),
        ])
    start = args.rank * chunk_size
    return texts[start:start + chunk_size], num_texts


def gather_text_features(args, local_features, num_texts):
    if not getattr(args, 'distributed', False):
        return local_features
    gathered = dist_nn.all_gather(local_features)
    return torch.cat(gathered, dim=0)[:num_texts]


def distributed_mean(args, value):
    value = value.detach().float()
    if getattr(args, 'distributed', False):
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= args.world_size
    return value.item()


def tensor_collection_norm(tensors):
    tensors = list(tensors)
    if not tensors:
        return torch.zeros((), device='cuda')
    squared_norm = sum(
        tensor.detach().float().square().sum() for tensor in tensors)
    return squared_norm.sqrt()


def feature_anchor_loss(adapted_features, frozen_features):
    cosine = F.cosine_similarity(
        adapted_features.float(),
        frozen_features.detach().float(),
        dim=-1,
        eps=1e-6)
    return 1. - cosine.clamp(-1., 1.).mean()


def _next_power_of_two(value):
    if value < 1:
        raise ValueError('feature dimension must be positive')
    return 1 << (value - 1).bit_length()


def fast_walsh_hadamard_transform(features):
    """Apply an orthonormal Walsh-Hadamard transform on the last axis."""
    dimension = features.shape[-1]
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError(
            'Walsh-Hadamard dimension must be a positive power of two')

    transformed = features
    block_size = 1
    while block_size < dimension:
        original_shape = transformed.shape
        paired = transformed.reshape(
            *original_shape[:-1], -1, 2, block_size)
        first = paired[..., 0, :]
        second = paired[..., 1, :]
        transformed = torch.stack(
            (first + second, first - second), dim=-2).reshape(original_shape)
        block_size *= 2
    return transformed / math.sqrt(dimension)


def matched_random_subspace_features(
        image_features, text_features, drop_rate, seed):
    """Project both modalities through one deterministic random subspace.

    A randomized sign transform followed by a Walsh-Hadamard transform and
    shared coordinate selection implements the same structured projection for
    every image and class. Randomness is isolated in a local generator so all
    distributed ranks obtain the same projection from the same seed.
    """
    if image_features.shape[-1] != text_features.shape[-1]:
        raise ValueError('MRSA requires matching image/text feature dimensions')
    if not 0. <= drop_rate < 1.:
        raise ValueError('MRSA drop_rate must be in [0, 1)')

    dimension = image_features.shape[-1]
    if drop_rate == 0.:
        return (
            F.normalize(image_features.float(), dim=-1, eps=1e-6),
            F.normalize(text_features.float(), dim=-1, eps=1e-6),
        )

    padded_dimension = _next_power_of_two(dimension)
    retained_dimension = max(1, round((1. - drop_rate) * dimension))
    generator = torch.Generator(device=image_features.device)
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (padded_dimension,), generator=generator,
        device=image_features.device, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(dtype=torch.float32)
    coordinates = torch.randperm(
        padded_dimension, generator=generator,
        device=image_features.device)[:retained_dimension]

    def project(features):
        features = features.float()
        if padded_dimension != dimension:
            features = F.pad(features, (0, padded_dimension - dimension))
        transformed = fast_walsh_hadamard_transform(features * signs)
        projected = transformed.index_select(-1, coordinates)
        return F.normalize(projected, dim=-1, eps=1e-6)

    return project(image_features), project(text_features)


def build_class_prompts(classnames, template):
    prompt = template[0]
    return [
        prompt.format(classname.replace('_', ' '))
        for classname in classnames
    ]


def get_prototype_classnames(dataset, include_novel):
    classnames = list(dataset.classnames)
    novel_classnames = getattr(dataset, 'test_new_classnames', None)
    if include_novel and novel_classnames is not None:
        classnames.extend(novel_classnames)
    return classnames


def encode_frozen_text_features(clip_model, tokenized_texts):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            text_features = clip_model.encode_text(tokenized_texts.cuda())
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)
    return text_features.detach()


def normalized_relation_profile_row_losses(
        adapted_scores, frozen_scores, include_mask=None):
    adapted_scores = adapted_scores.float()
    frozen_scores = frozen_scores.detach().float()
    if include_mask is None:
        include_mask = torch.ones_like(adapted_scores, dtype=torch.bool)

    include_mask = include_mask.to(device=adapted_scores.device)
    mask = include_mask.to(dtype=adapted_scores.dtype)
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.)

    adapted_mean = (adapted_scores * mask).sum(dim=-1, keepdim=True) / counts
    frozen_mean = (frozen_scores * mask).sum(dim=-1, keepdim=True) / counts
    adapted_profile = (adapted_scores - adapted_mean) * mask
    frozen_profile = (frozen_scores - frozen_mean) * mask

    adapted_profile = F.normalize(adapted_profile, dim=-1, eps=1e-6)
    frozen_profile = F.normalize(frozen_profile, dim=-1, eps=1e-6)
    return 1. - (adapted_profile * frozen_profile).sum(dim=-1)


def normalized_relation_profile_loss(
        adapted_scores, frozen_scores, include_mask=None, row_weights=None):
    row_losses = normalized_relation_profile_row_losses(
        adapted_scores, frozen_scores, include_mask=include_mask)

    if row_weights is None:
        return row_losses.mean()
    row_weights = row_weights.to(
        device=row_losses.device, dtype=row_losses.dtype)
    return (row_losses * row_weights).sum() / row_weights.sum().clamp_min(1.)


def balanced_novel_row_weights(num_rows, num_train_classes, device, dtype):
    weights = torch.ones(num_rows, device=device, dtype=dtype)
    num_novel_classes = num_rows - num_train_classes
    if num_train_classes > 0 and num_novel_classes > 0:
        weights[num_train_classes:] = num_train_classes / num_novel_classes
    return weights


def class_prototype_memory_loss(
        image_features, frozen_image_features,
        adapted_text_features, frozen_text_features,
        num_train_classes):
    losses = []
    frozen_text_features = frozen_text_features.detach().float()

    if image_features is not None and frozen_image_features is not None:
        adapted_image_scores = (
            image_features.float() @ frozen_text_features.t())
        frozen_image_scores = (
            frozen_image_features.detach().float()
            @ frozen_text_features.t())
        losses.append(normalized_relation_profile_loss(
            adapted_image_scores, frozen_image_scores))

    if adapted_text_features is not None:
        adapted_text_scores = (
            adapted_text_features.float() @ frozen_text_features.t())
        frozen_text_scores = frozen_text_features @ frozen_text_features.t()
        include_mask = None
        if frozen_text_scores.shape[0] == frozen_text_scores.shape[1]:
            include_mask = ~torch.eye(
                frozen_text_scores.shape[0],
                dtype=torch.bool,
                device=frozen_text_scores.device)
        row_weights = balanced_novel_row_weights(
            adapted_text_scores.shape[0],
            num_train_classes,
            adapted_text_scores.device,
            adapted_text_scores.dtype)
        losses.append(normalized_relation_profile_loss(
            adapted_text_scores, frozen_text_scores,
            include_mask=include_mask,
            row_weights=row_weights))

    if not losses:
        device = frozen_text_features.device
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def adapter_diagnostics(list_lora_layers, adapter_parameters):
    with torch.no_grad():
        return {
            'adapter_norm': tensor_collection_norm(adapter_parameters),
        }


def evaluate_zero_shot(clip_model, loader, textual_features, logit_scale):
    clip_model.eval()
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True)
            logits = logit_scale * image_features @ textual_features
            acc += cls_acc(logits, target) * len(logits)
            tot_samples += len(logits)
    return acc / tot_samples


def harmonic_mean(first, second):
    denominator = first + second
    return 0. if denominator == 0 else 2. * first * second / denominator


def resolve_test_loaders(args, test_loader):
    if getattr(args, 'setting', 'standard') != 'base2new':
        return test_loader, None
    if not isinstance(test_loader, (tuple, list)) or len(test_loader) != 2:
        raise ValueError(
            'split evaluation requires (seen_loader, heldout_loader)')
    return test_loader[0], test_loader[1]


def evaluate_lora(
        args, clip_model, loader, dataset, classnames=None):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        if classnames is None:
            classnames = dataset.classnames
        texts = [
            template.format(classname.replace('_', ' '))
            for classname in classnames
        ]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def evaluate_test_splits(args, clip_model, test_loader, dataset):
    base_loader, novel_loader = resolve_test_loaders(args, test_loader)
    if novel_loader is None:
        return {
            'test_accuracy': evaluate_lora(
                args, clip_model, base_loader, dataset,
                classnames=dataset.test_classnames),
        }

    base_accuracy = evaluate_lora(
        args, clip_model, base_loader, dataset,
        classnames=dataset.test_classnames)
    novel_accuracy = evaluate_lora(
        args, clip_model, novel_loader, dataset,
        classnames=dataset.test_new_classnames)
    return {
        'base_accuracy': base_accuracy,
        'novel_accuracy': novel_accuracy,
        'harmonic_mean': harmonic_mean(base_accuracy, novel_accuracy),
    }


def print_test_metrics(metrics, prefix='Final'):
    if 'novel_accuracy' in metrics:
        print(
            f"**** {prefix} base accuracy: {metrics['base_accuracy']:.2f}; "
            f"novel accuracy: {metrics['novel_accuracy']:.2f}; "
            f"harmonic mean: {metrics['harmonic_mean']:.2f}. ****\n")
    else:
        print(
            f"**** {prefix} test accuracy: "
            f"{metrics['test_accuracy']:.2f}. ****\n")


def evaluate_transfer_targets(
        args, clip_model, transfer_eval_data, logger=None):
    if not transfer_eval_data:
        return {}

    metrics_by_dataset = {}
    for item in transfer_eval_data:
        dataset_name = item['name']
        target_dataset = item['dataset']
        target_loader = item['loader']
        accuracy = evaluate_lora(
            args,
            clip_model,
            target_loader,
            target_dataset,
            classnames=target_dataset.test_classnames)
        metrics = {'test_accuracy': accuracy}
        metrics_by_dataset[dataset_name] = metrics
        print_test_metrics(metrics, prefix=f'Transfer {dataset_name}')
        if logger is not None:
            logger.log(
                'transfer_eval',
                dataset=dataset_name,
                **metrics)

    if len(metrics_by_dataset) > 1:
        mean_accuracy = sum(
            metrics['test_accuracy']
            for metrics in metrics_by_dataset.values()) / len(metrics_by_dataset)
        print(
            f"**** Transfer average over {len(metrics_by_dataset)} "
            f"dataset(s): {mean_accuracy:.2f}. ****\n")
        if logger is not None:
            logger.log(
                'transfer_eval_average',
                num_datasets=len(metrics_by_dataset),
                test_accuracy=mean_accuracy)

    return metrics_by_dataset


def run_lora(
        args, clip_model, logit_scale, dataset, train_loader, val_loader,
        test_loader, transfer_eval_data=None):
    logger = TrainingLogger(args)
    test_base_loader, test_new_loader = resolve_test_loaders(args, test_loader)
    prototype_anchor_weight = getattr(args, 'prototype_anchor_weight', 0.)
    # Textual features
    if is_main_process(args):
        print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    prototype_classnames = get_prototype_classnames(
        dataset,
        include_novel=(prototype_anchor_weight > 0))
    prototype_texts = build_class_prompts(
        prototype_classnames, dataset.template)
    prototype_tokenized_texts = clip.tokenize(prototype_texts)
    frozen_prototype_text_features = encode_frozen_text_features(
        clip_model, prototype_tokenized_texts)
    num_train_classes = len(dataset.classnames)
    num_prototype_classes = len(prototype_classnames)
    joint_embedding_dimension = textual_features.shape[0]
    mrsa_retained_dimension = (
        max(
            1,
            round(
                (1. - args.dropout_rate)
                * joint_embedding_dimension))
        if getattr(args, 'mrsa', False) else None)

    if is_main_process(args):
        print("\nEvaluating zero-shot CLIP on the test set.")
        zero_shot_base = evaluate_zero_shot(
            clip_model, test_base_loader, textual_features, logit_scale)
        if test_new_loader is None:
            zero_shot_metrics = {'test_accuracy': zero_shot_base}
        else:
            novel_textual_features = clip_classifier(
                dataset.test_new_classnames, dataset.template, clip_model)
            zero_shot_novel = evaluate_zero_shot(
                clip_model, test_new_loader, novel_textual_features,
                logit_scale)
            zero_shot_metrics = {
                'base_accuracy': zero_shot_base,
                'novel_accuracy': zero_shot_novel,
                'harmonic_mean': harmonic_mean(
                    zero_shot_base, zero_shot_novel),
            }
        print_test_metrics(zero_shot_metrics, prefix='Zero-shot CLIP')
        logger.log('zero_shot', **zero_shot_metrics)

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        if is_main_process(args):
            load_lora(args, list_lora_layers)
            test_metrics = evaluate_test_splits(
                args, clip_model, test_loader, dataset)
            print_test_metrics(test_metrics, prefix='Loaded adapter')
            logger.log(
                'loaded_adapter',
                dataset=args.dataset,
                **test_metrics)
            evaluate_transfer_targets(
                args, clip_model, transfer_eval_data, logger)
        if getattr(args, 'distributed', False):
            dist.barrier()
        logger.close()
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    feature_encoder = CLIPFeatureEncoder(clip_model)
    if getattr(args, 'distributed', False):
        feature_encoder = DistributedDataParallel(
            feature_encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank)
    adapter_parameters = get_lora_parameters(feature_encoder)
    trainable_parameters = adapter_parameters
    trainable_count = sum(
        parameter.numel() for parameter in adapter_parameters)
    if is_main_process(args):
        config_metadata = get_adapter_metadata(args)
        print(
            f"LoRA trainable adapter parameters: {trainable_count:,}")
        logger.log(
            'config',
            metadata=config_metadata,
            trainable_parameters=trainable_count,
            lora_trainable_parameters=trainable_count,
            total_steps=total_iters,
            global_batch_size=args.batch_size,
            per_gpu_batch_size=(args.batch_size // args.world_size),
            world_size=args.world_size,
            learning_rate=args.lr,
            mrsa=getattr(args, 'mrsa', False),
            mrsa_projection=(
                'signed_hadamard_subspace'
                if getattr(args, 'mrsa', False) else None),
            mrsa_drop_rate=(
                args.dropout_rate
                if getattr(args, 'mrsa', False) else None),
            mrsa_full_dimension=(
                joint_embedding_dimension
                if getattr(args, 'mrsa', False) else None),
            mrsa_retained_dimension=mrsa_retained_dimension,
            image_anchor_weight=getattr(args, 'image_anchor_weight', 0.),
            text_anchor_weight=getattr(args, 'text_anchor_weight', 0.),
            prototype_anchor_weight=prototype_anchor_weight,
            v_rpr=getattr(args, 'v_rpr', False),
            v_rpr_global_rank=(
                (args.r + 1) // 2
                if getattr(args, 'v_rpr', False) else None),
            v_rpr_readout_rank=(
                args.r - (args.r + 1) // 2
                if getattr(args, 'v_rpr', False) else None),
            dp_vrpr=getattr(args, 'dp_vrpr', False),
            dp_vrpr_schedule=(
                'linear_absolute_depth'
                if getattr(args, 'dp_vrpr', False) else None),
            dp_vrpr_global_rank=(
                (args.r + 1) // 2
                if getattr(args, 'dp_vrpr', False) else None),
            dp_vrpr_transition_rank=(
                args.r - (args.r + 1) // 2
                if getattr(args, 'dp_vrpr', False) else None),
            dp_vrpr_first_patch_scale=(
                1. if getattr(args, 'dp_vrpr', False) else None),
            dp_vrpr_last_patch_scale=(
                0. if getattr(args, 'dp_vrpr', False) else None),
            prototype_classes=num_prototype_classes,
            train_classes=num_train_classes,
            log_interval=args.log_interval,
            val_interval=args.val_interval)
    optimizer = torch.optim.AdamW(trainable_parameters, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    optimizer_attempts = 0
    epoch = 0
    optimizer.zero_grad()
    while count_iters < total_iters:
        feature_encoder.train()
        if getattr(args, 'distributed', False):
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        image_anchor_loss_epoch = 0.
        text_anchor_loss_epoch = 0.
        prototype_anchor_loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
            classifier_text_features = text_features
        for i, batch in enumerate(
                tqdm(train_loader, disable=not is_main_process(args))):
            images, target = batch
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            tokenized_texts = None
            num_texts = None
            all_text_features = None
            if args.encoder == 'text' or args.encoder == 'both':
                tokenized_texts = prototype_tokenized_texts.cuda()
                tokenized_texts, num_texts = shard_texts(args, tokenized_texts)

            needs_image_anchor = (
                getattr(args, 'image_anchor_weight', 0.) > 0
                and args.encoder in {'vision', 'both'})
            needs_text_anchor = (
                getattr(args, 'text_anchor_weight', 0.) > 0
                and args.encoder in {'text', 'both'})
            needs_prototype_anchor = prototype_anchor_weight > 0
            needs_frozen_image = (
                needs_image_anchor
                or (needs_prototype_anchor
                    and args.encoder in {'vision', 'both'}))
            frozen_image_features = None
            if needs_frozen_image:
                with torch.no_grad(), disabled_adapters(list_lora_layers):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        frozen_image_features = clip_model.encode_image(images)
                    frozen_image_features = (
                        frozen_image_features
                        / frozen_image_features.norm(dim=-1, keepdim=True))
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                encoded = feature_encoder(
                    images=images if args.encoder in {'vision', 'both'} else None,
                    texts=tokenized_texts)

            if args.encoder == 'text' or args.encoder == 'both':
                class_embeddings = gather_text_features(
                    args, encoded['text_features'], num_texts)
                all_text_features = (
                    class_embeddings
                    / class_embeddings.norm(dim=-1, keepdim=True))
                text_features = all_text_features[:num_train_classes]
                classifier_text_features = text_features
                
            if args.encoder == 'vision' or args.encoder == 'both':
                image_features = encoded['image_features']
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            image_anchor_loss_value = image_features.new_zeros(())
            text_anchor_loss_value = image_features.new_zeros(())
            prototype_anchor_loss_value = image_features.new_zeros(())
            frozen_image_features = (
                frozen_image_features
                if frozen_image_features is not None
                else image_features.detach())
            frozen_text_features = (
                frozen_prototype_text_features
                .to(device=image_features.device, dtype=text_features.dtype))

            full_similarity = (
                image_features @ classifier_text_features.t())
            full_cosine_similarity = logit_scale * full_similarity
            if getattr(args, 'mrsa', False):
                mrsa_image_features, mrsa_text_features = (
                    matched_random_subspace_features(
                        image_features,
                        classifier_text_features,
                        drop_rate=args.dropout_rate,
                        seed=(
                            int(args.seed) * 1_000_003
                            + count_iters + 1)))
                cosine_similarity = (
                    logit_scale
                    * mrsa_image_features @ mrsa_text_features.t())
            else:
                cosine_similarity = full_cosine_similarity
            classification_loss = F.cross_entropy(cosine_similarity, target)

            if needs_image_anchor:
                image_anchor_loss_value = feature_anchor_loss(
                    image_features, frozen_image_features)
            if needs_text_anchor:
                text_anchor_loss_value = feature_anchor_loss(
                    text_features,
                    frozen_text_features[:num_train_classes])
            if needs_prototype_anchor:
                adapted_image_memory = (
                    image_features
                    if args.encoder in {'vision', 'both'}
                    else None)
                frozen_image_memory = (
                    frozen_image_features
                    if args.encoder in {'vision', 'both'}
                    else None)
                adapted_text_memory = (
                    all_text_features
                    if args.encoder in {'text', 'both'}
                    else None)
                prototype_anchor_loss_value = class_prototype_memory_loss(
                    image_features=adapted_image_memory,
                    frozen_image_features=frozen_image_memory,
                    adapted_text_features=adapted_text_memory,
                    frozen_text_features=frozen_text_features,
                    num_train_classes=num_train_classes)
            regularization_loss = (
                getattr(args, 'image_anchor_weight', 0.) * image_anchor_loss_value
                + getattr(args, 'text_anchor_weight', 0.) * text_anchor_loss_value
                + prototype_anchor_weight * prototype_anchor_loss_value
            )
            loss = classification_loss + regularization_loss
            batch_accuracy = (
                cosine_similarity.argmax(dim=1).eq(target).float().mean()
                * 100.)
            full_batch_accuracy = (
                full_cosine_similarity.argmax(dim=1).eq(target).float().mean()
                * 100.)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            image_anchor_loss_epoch += (
                image_anchor_loss_value.item() * target.shape[0])
            text_anchor_loss_epoch += (
                text_anchor_loss_value.item() * target.shape[0])
            prototype_anchor_loss_epoch += (
                prototype_anchor_loss_value.item() * target.shape[0])
            tot_samples += target.shape[0]

            step = count_iters + 1
            scheduled_log = (
                step == 1 or step == total_iters
                or step % args.log_interval == 0)
            applied_learning_rate = optimizer.param_groups[0]['lr']
            optimizer_attempts += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = tensor_collection_norm([
                parameter.grad for parameter in trainable_parameters
                if parameter.grad is not None
            ])
            grad_scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            grad_scale_after = scaler.get_scale()
            optimizer_step_skipped = grad_scale_after < grad_scale_before
            if not optimizer_step_skipped:
                scheduler.step()
                count_iters = step
            optimizer.zero_grad()

            should_log = scheduled_log or optimizer_step_skipped
            if should_log:
                diagnostics = adapter_diagnostics(
                    list_lora_layers, trainable_parameters)
                step_metrics = {
                    'step': count_iters,
                    'optimizer_attempt': optimizer_attempts,
                    'epoch': epoch,
                    'learning_rate': applied_learning_rate,
                    'classification_loss': distributed_mean(
                        args, classification_loss.detach()),
                    'total_loss': distributed_mean(
                        args, loss.detach()),
                    'image_anchor_loss': distributed_mean(
                        args, image_anchor_loss_value.detach()),
                    'text_anchor_loss': distributed_mean(
                        args, text_anchor_loss_value.detach()),
                    'prototype_anchor_loss': distributed_mean(
                        args, prototype_anchor_loss_value.detach()),
                    'batch_accuracy': distributed_mean(
                        args, batch_accuracy.detach()),
                    'full_batch_accuracy': distributed_mean(
                        args, full_batch_accuracy.detach()),
                    'gradient_norm': distributed_mean(args, gradient_norm),
                    'gradient_finite': distributed_mean(
                        args, torch.isfinite(gradient_norm).float()),
                    'grad_scale_before': grad_scale_before,
                    'grad_scale_after': grad_scale_after,
                    'optimizer_step_skipped': optimizer_step_skipped,
                }
                step_metrics.update({
                    name: distributed_mean(args, value)
                    for name, value in diagnostics.items()
                })
                logger.log('train_step', **step_metrics)

            should_validate = (
                not optimizer_step_skipped
                and args.val_interval > 0
                and count_iters < total_iters
                and count_iters % args.val_interval == 0)
            if should_validate:
                if getattr(args, 'distributed', False):
                    dist.barrier()
                if is_main_process(args):
                    validation_accuracy = evaluate_lora(
                        args, clip_model, val_loader, dataset)
                    print(
                        f"**** Step {count_iters} validation accuracy: "
                        f"{validation_accuracy:.2f}. ****")
                    logger.log(
                        'periodic_validation',
                        step=count_iters,
                        validation_accuracy=validation_accuracy)
                if getattr(args, 'distributed', False):
                    dist.barrier()
                feature_encoder.train()
            
            if count_iters == total_iters:
                break

        if count_iters < total_iters and is_main_process(args):
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            image_anchor_loss_epoch /= tot_samples
            text_anchor_loss_epoch /= tot_samples
            prototype_anchor_loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print(
                'LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, '
                'ImageAnchor: {:.6f}, TextAnchor: {:.6f}, '
                'ProtoAnchor: {:.6f}'
                .format(
                    current_lr, acc_train, loss_epoch,
                    image_anchor_loss_epoch, text_anchor_loss_epoch,
                    prototype_anchor_loss_epoch))

    if is_main_process(args):
        if args.save_path != None:
            save_lora(args, list_lora_layers)
            logger.log(
                'checkpoint_saved',
                path=(
                    os.path.join(
                        get_adapter_save_dir(args),
                        f'{args.filename}.pt')))

        test_metrics = evaluate_test_splits(
            args, clip_model, test_loader, dataset)
        print_test_metrics(test_metrics)
        logger.log('final', **test_metrics)

        evaluate_transfer_targets(
            args, clip_model, transfer_eval_data, logger)
    if getattr(args, 'distributed', False):
        dist.barrier()
    logger.close()
    return
            
    
            
