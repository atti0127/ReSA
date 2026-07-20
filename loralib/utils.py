import os
import tempfile
from contextlib import contextmanager

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA

ADAPTER_PARAMETER_MARKERS = ('lora_',)


def is_adapter_parameter(name):
    return any(marker in name for marker in ADAPTER_PARAMETER_MARKERS)


def get_adapter_metadata(args):
    return {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position,
        'setting': getattr(args, 'setting', 'standard'),
        'image_anchor_weight': getattr(args, 'image_anchor_weight', 0.),
        'text_anchor_weight': getattr(args, 'text_anchor_weight', 0.),
        'prototype_anchor_weight': getattr(args, 'prototype_anchor_weight', 0.),
        'mrsa': getattr(args, 'mrsa', False),
        'mrsa_projection': (
            'signed_hadamard_subspace'
            if getattr(args, 'mrsa', False) else None),
        'mrsa_drop_rate': (
            getattr(args, 'dropout_rate', None)
            if getattr(args, 'mrsa', False) else None),
        'v_rpr': getattr(args, 'v_rpr', False),
        'dp_vrpr': getattr(args, 'dp_vrpr', False),
        'dp_vrpr_schedule': (
            'linear_absolute_depth'
            if getattr(args, 'dp_vrpr', False) else None),
    }


@contextmanager
def disabled_adapters(list_lora_layers):
    modules = []
    previous_states = []
    previous_merged = []
    for layer in list_lora_layers:
        for module in layer.modules():
            if hasattr(module, 'adapters_disabled'):
                modules.append(module)
                previous_states.append(module.adapters_disabled)
                previous_merged.append(getattr(module, 'merged', False))
                if (
                        getattr(module, 'merged', False)
                        and hasattr(module, 'sub_lora_data')):
                    module.sub_lora_data()
                    module.merged = False
                module.adapters_disabled = True
    try:
        yield
    finally:
        for module, previous_state, was_merged in zip(
                modules, previous_states, previous_merged):
            module.adapters_disabled = previous_state
            if (
                    was_merged
                    and not getattr(module, 'merged', False)
                    and hasattr(module, 'add_lora_data')):
                module.add_lora_data()
                module.merged = True


def get_checkpoint_dataset(args):
    if getattr(args, 'checkpoint_dataset', None):
        return args.checkpoint_dataset
    if (
            getattr(args, 'eval_only', False)
            and getattr(args, 'setting', 'standard') in {
                'cross_dataset',
                'domain_generalization',
            }
            and args.dataset != 'imagenet'):
        return 'imagenet'
    return args.dataset


def get_adapter_save_dir(args):
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    checkpoint_dataset = get_checkpoint_dataset(args)
    save_dir = (
        f'{args.save_path}/{backbone}/{checkpoint_dataset}/'
        f'{args.shots}shots/seed{args.seed}')
    setting = getattr(args, 'setting', 'standard')
    return f'{save_dir}/{setting}'


INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        p.requires_grad = is_adapter_parameter(n)
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if is_adapter_parameter(k)
        }
    elif bias == 'all':
        return {
            k: my_state_dict[k]
            for k in my_state_dict
            if is_adapter_parameter(k) or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if is_adapter_parameter(k):
                to_return[k] = my_state_dict[k]
            if 'lora_' in k:
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if is_adapter_parameter(name) and param.requires_grad:
                params.append(param)
        elif bias == 'all':
            if is_adapter_parameter(name) or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if is_adapter_parameter(name):
                params.append(param)
            if 'lora_' in name:
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if getattr(args, 'rank', 0) == 0:
                print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,
                            enable_lora=(args.params if i in indices else []),
                            r=args.r,
                            lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        num_visual_blocks = len(vision_encoder.resblocks)
        for i, block in enumerate(vision_encoder.resblocks):
            if getattr(args, 'rank', 0) == 0:
                print(f"Residual Attention Block {i}: {block}")
            install_lora = i in indices
            if install_lora:
                if getattr(args, 'dp_vrpr', False):
                    readout_kind = 'visual_cls_progressive'
                    readout_depth = (
                        i / (num_visual_blocks - 1)
                        if num_visual_blocks > 1 else 1.)
                elif getattr(args, 'v_rpr', False):
                    readout_kind = 'visual_cls'
                    readout_depth = None
                else:
                    readout_kind = None
                    readout_depth = None
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,
                            enable_lora=args.params,
                            r=args.r,
                            lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate,
                            readout_kind=readout_kind,
                            readout_depth=readout_depth)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if isinstance(layer, PlainMultiheadAttentionLoRA):
            if hasattr(layer.q_proj, 'w_lora_A'):
                layer_weights['q_proj'] = {
                    'w_lora_A': layer.q_proj.w_lora_A.detach().cpu(),
                    'w_lora_B': layer.q_proj.w_lora_B.detach().cpu()
                }
            if hasattr(layer.k_proj, 'w_lora_A'):
                layer_weights['k_proj'] = {
                    'w_lora_A': layer.k_proj.w_lora_A.detach().cpu(),
                    'w_lora_B': layer.k_proj.w_lora_B.detach().cpu()
                }
            if hasattr(layer.v_proj, 'w_lora_A'):
                layer_weights['v_proj'] = {
                    'w_lora_A': layer.v_proj.w_lora_A.detach().cpu(),
                    'w_lora_B': layer.v_proj.w_lora_B.detach().cpu()
                }
            if hasattr(layer.proj, 'w_lora_A'):
                layer_weights['proj'] = {
                    'w_lora_A': layer.proj.w_lora_A.detach().cpu(),
                    'w_lora_B': layer.proj.w_lora_B.detach().cpu()
                }
        else:
            raise TypeError(
                f'Unsupported adapter layer type: {type(layer).__name__}')

        weights[f'layer_{i}'] = layer_weights

    return save_adapter_data(args, weights, get_adapter_metadata(args))


def save_adapter_data(args, weights, metadata=None):
    if metadata is None:
        metadata = get_adapter_metadata(args)
    save_data = {'weights': weights, 'metadata': metadata}
    save_dir = get_adapter_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    fd, tmp_path = tempfile.mkstemp(
        prefix=f'.{args.filename}.', suffix='.tmp', dir=save_dir)
    try:
        with os.fdopen(fd, 'wb') as file:
            torch.save(save_data, file)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, save_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    print(f'Adapter weights saved to {save_path}')


def load_lora(args, list_lora_layers):
    load_dir = get_adapter_save_dir(args)
    load_path = f'{load_dir}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path, map_location='cpu')

    metadata = loaded_data['metadata']
    stored_setting = metadata.get('setting', 'standard')
    expected_setting = getattr(args, 'setting', 'standard')
    if stored_setting != expected_setting:
        raise ValueError(
            f"Setting mismatch: expected {expected_setting}, found {stored_setting}")
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")
    stored_image_anchor = metadata.get('image_anchor_weight', 0.)
    expected_image_anchor = getattr(args, 'image_anchor_weight', 0.)
    if stored_image_anchor != expected_image_anchor:
        raise ValueError(
            f"Image anchor weight mismatch: expected {expected_image_anchor}, "
            f"found {stored_image_anchor}")
    stored_text_anchor = metadata.get('text_anchor_weight', 0.)
    expected_text_anchor = getattr(args, 'text_anchor_weight', 0.)
    if stored_text_anchor != expected_text_anchor:
        raise ValueError(
            f"Text anchor weight mismatch: expected {expected_text_anchor}, "
            f"found {stored_text_anchor}")
    stored_prototype_anchor = metadata.get('prototype_anchor_weight', 0.)
    expected_prototype_anchor = getattr(args, 'prototype_anchor_weight', 0.)
    if stored_prototype_anchor != expected_prototype_anchor:
        raise ValueError(
            f"Prototype anchor weight mismatch: expected {expected_prototype_anchor}, "
            f"found {stored_prototype_anchor}")
    stored_mrsa = metadata.get('mrsa', False)
    expected_mrsa = getattr(args, 'mrsa', False)
    if stored_mrsa != expected_mrsa:
        raise ValueError(
            f"MRSA mismatch: expected {expected_mrsa}, found {stored_mrsa}")
    if expected_mrsa:
        stored_mrsa_projection = metadata.get('mrsa_projection')
        if stored_mrsa_projection != 'signed_hadamard_subspace':
            raise ValueError(
                'MRSA projection mismatch: expected '
                'signed_hadamard_subspace, found '
                f'{stored_mrsa_projection}')
    stored_v_rpr = metadata.get('v_rpr', False)
    expected_v_rpr = getattr(args, 'v_rpr', False)
    if stored_v_rpr != expected_v_rpr:
        raise ValueError(
            f"V-RPR mismatch: expected {expected_v_rpr}, found {stored_v_rpr}")
    stored_dp_vrpr = metadata.get('dp_vrpr', False)
    expected_dp_vrpr = getattr(args, 'dp_vrpr', False)
    if stored_dp_vrpr != expected_dp_vrpr:
        raise ValueError(
            "DP-VRPR mismatch: expected "
            f"{expected_dp_vrpr}, found {stored_dp_vrpr}")
    stored_dp_vrpr_schedule = metadata.get('dp_vrpr_schedule')
    expected_dp_vrpr_schedule = (
        'linear_absolute_depth' if expected_dp_vrpr else None)
    if stored_dp_vrpr_schedule != expected_dp_vrpr_schedule:
        raise ValueError(
            "DP-VRPR schedule mismatch: expected "
            f"{expected_dp_vrpr_schedule}, "
            f"found {stored_dp_vrpr_schedule}")
    weights = loaded_data['weights']
    expected_layer_keys = {
        f'layer_{i}' for i in range(len(list_lora_layers))}
    if set(weights) != expected_layer_keys:
        raise ValueError(
            'Adapter layer layout mismatch: expected '
            f'{len(expected_layer_keys)} layers, found {len(weights)}')
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if isinstance(layer, PlainMultiheadAttentionLoRA):
            if hasattr(layer.q_proj, 'w_lora_A') and 'q_proj' in layer_weights:
                layer.q_proj.w_lora_A.data.copy_(
                    layer_weights['q_proj']['w_lora_A'])
                layer.q_proj.w_lora_B.data.copy_(
                    layer_weights['q_proj']['w_lora_B'])
            if hasattr(layer.k_proj, 'w_lora_A') and 'k_proj' in layer_weights:
                layer.k_proj.w_lora_A.data.copy_(
                    layer_weights['k_proj']['w_lora_A'])
                layer.k_proj.w_lora_B.data.copy_(
                    layer_weights['k_proj']['w_lora_B'])
            if hasattr(layer.v_proj, 'w_lora_A') and 'v_proj' in layer_weights:
                layer.v_proj.w_lora_A.data.copy_(
                    layer_weights['v_proj']['w_lora_A'])
                layer.v_proj.w_lora_B.data.copy_(
                    layer_weights['v_proj']['w_lora_B'])
            if hasattr(layer.proj, 'w_lora_A') and 'proj' in layer_weights:
                layer.proj.w_lora_A.data.copy_(
                    layer_weights['proj']['w_lora_A'])
                layer.proj.w_lora_B.data.copy_(
                    layer_weights['proj']['w_lora_B'])
        else:
            raise TypeError(
                f'Unsupported adapter layer type: {type(layer).__name__}')
    print(f'Adapter weights loaded from {load_path}')
