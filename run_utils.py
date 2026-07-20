
import random
import argparse  
import numpy as np 
import torch

from lora import run_lora


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--setting', default='standard',
                        choices=[
                            'standard',
                            'base2new',
                            'cross_dataset',
                            'domain_generalization',
                        ],
                        help='standard uses the same classes for training and testing; '
                             'base2new trains on the first class half and evaluates '
                             'base and held-out novel classes separately; '
                             'cross_dataset/domain_generalization train on the '
                             'source dataset and load that source checkpoint for '
                             'target-dataset evaluation')
    parser.add_argument('--checkpoint_dataset', default=None, type=str,
                        help='dataset name used to locate the saved adapter; '
                             'defaults to the current dataset, except eval-only '
                             'cross-dataset/domain-generalization target runs '
                             'default to imagenet')
    parser.add_argument('--target_datasets', default=[], nargs='+',
                        help='optional target datasets to evaluate after loading '
                             'or training an adapter. Supports dataset names and '
                             'groups: cross_dataset, domain_generalization, '
                             'all_transfer')
    # Model arguments
    parser.add_argument('--backbone', default='ViT-L/14', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='effective global training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int,
                        help='per-GPU validation and test batch size')
    parser.add_argument('--log_interval', default=10, type=int,
                        help='optimizer steps between training_log.jsonl records')
    parser.add_argument('--val_interval', default=0, type=int,
                        help='steps between validation evaluations; 0 disables them')
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--image_anchor_weight', default=0.0, type=float,
                        help='weight for distilling adapted image features toward frozen CLIP')
    parser.add_argument('--text_anchor_weight', default=0.0, type=float,
                        help='weight for distilling adapted text features toward frozen CLIP')
    parser.add_argument('--prototype_anchor_weight', default=0.0, type=float,
                        help='weight for prototype profile anchoring')
    parser.add_argument('--mrsa', default=False, action='store_true',
                        help='Matched Random-Subspace Adaptation: compute '
                             'training CE after applying the same structured '
                             'random projection to image and text features; '
                             'anchors and inference remain full-dimensional')
    parser.add_argument('--v_rpr', default=False, action='store_true',
                        help='partition visual LoRA rank channels into global '
                             'and CLS-only channels')
    parser.add_argument('--dp_vrpr', default=False, action='store_true',
                        help='progressively suppress the visual readout rank '
                             'channel on patch tokens from early to late '
                             'blocks while preserving it on CLS')
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()
    return args
    

        
