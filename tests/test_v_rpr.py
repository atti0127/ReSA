import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from clip.model import CLIP
from loralib.layers import LinearLoRA, PlainMultiheadAttentionLoRA
from loralib.utils import apply_lora, get_adapter_metadata, load_lora, save_lora


def make_args(save_path, v_rpr=True, dp_vrpr=False):
    return SimpleNamespace(
        r=2,
        alpha=1,
        encoder='both',
        params=['q', 'k', 'v'],
        position='all',
        setting='standard',
        image_anchor_weight=1.,
        text_anchor_weight=1.,
        prototype_anchor_weight=1.,
        v_rpr=v_rpr,
        dp_vrpr=dp_vrpr,
        backbone='ViT-B/16',
        dataset='imagenet',
        shots=16,
        seed=1,
        save_path=save_path,
        filename='lora_weights',
        checkpoint_dataset=None,
        eval_only=False,
    )


def make_attention(
        partitioned=True, readout_kind='visual_cls', readout_depth=None):
    torch.manual_seed(7)
    base = nn.MultiheadAttention(8, 2, dropout=0.)
    return PlainMultiheadAttentionLoRA(
        base,
        enable_lora=['q', 'k', 'v'],
        r=2,
        lora_alpha=1,
        dropout_rate=0.,
        readout_kind=readout_kind if partitioned else None,
        readout_depth=readout_depth if partitioned else None,
    )


class VisualReadoutPartitionedRankTest(unittest.TestCase):
    def test_zero_initialized_attention_matches_ordinary_lora(self):
        partitioned = make_attention(partitioned=True)
        ordinary = make_attention(partitioned=False)
        inputs = torch.randn(5, 3, 8)
        partitioned_output = partitioned(
            inputs, inputs, inputs, need_weights=False)[0]
        ordinary_output = ordinary(
            inputs, inputs, inputs, need_weights=False)[0]
        torch.testing.assert_close(partitioned_output, ordinary_output)

    def test_partition_does_not_change_parameter_count(self):
        partitioned = make_attention(partitioned=True)
        ordinary = make_attention(partitioned=False)
        partitioned_count = sum(
            parameter.numel() for parameter in partitioned.parameters())
        ordinary_count = sum(
            parameter.numel() for parameter in ordinary.parameters())
        self.assertEqual(partitioned_count, ordinary_count)

    def test_progressive_partition_does_not_change_parameter_count(self):
        progressive = make_attention(
            partitioned=True,
            readout_kind='visual_cls_progressive',
            readout_depth=0.5)
        ordinary = make_attention(partitioned=False)
        progressive_count = sum(
            parameter.numel() for parameter in progressive.parameters())
        ordinary_count = sum(
            parameter.numel() for parameter in ordinary.parameters())
        self.assertEqual(progressive_count, ordinary_count)

    def test_rank_channels_are_partitioned_by_cls_role(self):
        base = nn.Linear(2, 2)
        base.weight.data.zero_()
        base.bias.data.zero_()
        layer = LinearLoRA(
            base,
            r=2,
            lora_alpha=1,
            dropout_rate=0.,
            readout_partition=True,
        )
        with torch.no_grad():
            layer.w_lora_A.copy_(torch.eye(2))
            layer.w_lora_B.copy_(torch.eye(2))

        inputs = torch.tensor([
            [[2., 3.]],
            [[5., 7.]],
            [[11., 13.]],
        ])
        cls_mask = torch.tensor([[[1.]], [[0.]], [[0.]]])
        actual = layer(inputs, readout_mask=cls_mask)
        scale = 1. / (2. ** 0.5)
        expected = torch.tensor([
            [[2., 3.]],
            [[5., 0.]],
            [[11., 0.]],
        ]) * scale
        torch.testing.assert_close(actual, expected)

    def test_partition_is_preserved_in_evaluation_mode(self):
        base = nn.Linear(2, 2)
        base.weight.data.zero_()
        base.bias.data.zero_()
        layer = LinearLoRA(
            base,
            r=2,
            lora_alpha=1,
            dropout_rate=0.,
            readout_partition=True,
        )
        with torch.no_grad():
            layer.w_lora_A.copy_(torch.eye(2))
            layer.w_lora_B.copy_(torch.eye(2))
        inputs = torch.ones(2, 1, 2)
        mask = torch.tensor([[[1.]], [[0.]]])
        training_output = layer(inputs, readout_mask=mask)
        layer.eval()
        evaluation_output = layer(inputs, readout_mask=mask)
        self.assertFalse(layer.merged)
        torch.testing.assert_close(evaluation_output, training_output)

    def test_global_and_cls_channels_both_receive_gradients(self):
        base = nn.Linear(2, 2)
        base.weight.data.zero_()
        base.bias.data.zero_()
        layer = LinearLoRA(
            base,
            r=2,
            lora_alpha=1,
            dropout_rate=0.,
            readout_partition=True,
        )
        with torch.no_grad():
            layer.w_lora_A.copy_(torch.eye(2))
        inputs = torch.tensor([[[2., 3.]], [[5., 7.]]])
        mask = torch.tensor([[[1.]], [[0.]]])
        layer(inputs, readout_mask=mask).sum().backward()
        self.assertGreater(
            layer.w_lora_B.grad[:, :layer.global_rank].abs().sum().item(),
            0.)
        self.assertGreater(
            layer.w_lora_B.grad[:, layer.global_rank:].abs().sum().item(),
            0.)

    def test_disabled_partitioned_adapter_is_exact_frozen_layer(self):
        torch.manual_seed(9)
        base = nn.Linear(3, 4)
        layer = LinearLoRA(
            base,
            r=2,
            lora_alpha=1,
            dropout_rate=0.,
            readout_partition=True,
        )
        layer.w_lora_B.data.normal_()
        inputs = torch.randn(3, 2, 3)
        mask = torch.tensor([[[1.]], [[0.]], [[0.]]])
        layer.adapters_disabled = True
        actual = layer(inputs, readout_mask=mask)
        expected = nn.functional.linear(inputs, base.weight, base.bias)
        torch.testing.assert_close(actual, expected)

    def test_attention_marks_only_visual_cls_as_readout(self):
        layer = make_attention(partitioned=True)
        captured_masks = []

        def capture_mask(module, args, kwargs):
            captured_masks.append(kwargs['readout_mask'].detach().clone())

        handles = [
            projection.register_forward_pre_hook(
                capture_mask, with_kwargs=True)
            for projection in (layer.q_proj, layer.k_proj, layer.v_proj)
        ]
        try:
            inputs = torch.randn(5, 3, 8)
            layer(inputs, inputs, inputs, need_weights=False)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(captured_masks), 3)
        for mask in captured_masks:
            self.assertEqual(tuple(mask.shape), (5, 3, 1))
            self.assertTrue(torch.equal(mask[0], torch.ones_like(mask[0])))
            self.assertEqual(mask[1:].count_nonzero().item(), 0)

    def test_progressive_mask_follows_absolute_depth(self):
        inputs = torch.randn(5, 3, 8)
        for depth, expected_patch_scale in ((0., 1.), (0.5, 0.5), (1., 0.)):
            layer = make_attention(
                partitioned=True,
                readout_kind='visual_cls_progressive',
                readout_depth=depth)
            captured_masks = []

            def capture_mask(module, args, kwargs):
                captured_masks.append(
                    kwargs['readout_mask'].detach().clone())

            handles = [
                projection.register_forward_pre_hook(
                    capture_mask, with_kwargs=True)
                for projection in (layer.q_proj, layer.k_proj, layer.v_proj)
            ]
            try:
                layer(inputs, inputs, inputs, need_weights=False)
            finally:
                for handle in handles:
                    handle.remove()

            expected = torch.full((5, 3, 1), expected_patch_scale)
            expected[0] = 1.
            self.assertEqual(len(captured_masks), 3)
            for mask in captured_masks:
                torch.testing.assert_close(mask, expected)

    def test_progressive_endpoints_match_lora_and_v_rpr(self):
        inputs = torch.randn(5, 3, 8)
        early = make_attention(
            partitioned=True,
            readout_kind='visual_cls_progressive',
            readout_depth=0.)
        ordinary = make_attention(partitioned=False)
        late = make_attention(
            partitioned=True,
            readout_kind='visual_cls_progressive',
            readout_depth=1.)
        static = make_attention(partitioned=True)

        with torch.no_grad():
            for projection_name in ('q_proj', 'k_proj', 'v_proj'):
                early_projection = getattr(early, projection_name)
                ordinary_projection = getattr(ordinary, projection_name)
                late_projection = getattr(late, projection_name)
                static_projection = getattr(static, projection_name)
                update = torch.randn_like(early_projection.w_lora_B)
                early_projection.w_lora_B.copy_(update)
                ordinary_projection.w_lora_B.copy_(update)
                late_projection.w_lora_B.copy_(update)
                static_projection.w_lora_B.copy_(update)

        early_output = early(
            inputs, inputs, inputs, need_weights=False)[0]
        ordinary_output = ordinary(
            inputs, inputs, inputs, need_weights=False)[0]
        late_output = late(
            inputs, inputs, inputs, need_weights=False)[0]
        static_output = static(
            inputs, inputs, inputs, need_weights=False)[0]
        torch.testing.assert_close(early_output, ordinary_output)
        torch.testing.assert_close(late_output, static_output)

    def test_progressive_readout_configuration_is_strict(self):
        with self.assertRaisesRegex(ValueError, 'depth in'):
            make_attention(
                partitioned=True,
                readout_kind='visual_cls_progressive')
        with self.assertRaisesRegex(ValueError, 'depth in'):
            make_attention(
                partitioned=True,
                readout_kind='visual_cls_progressive',
                readout_depth=1.1)
        with self.assertRaisesRegex(ValueError, 'only valid'):
            make_attention(
                partitioned=True,
                readout_kind='visual_cls',
                readout_depth=0.5)
    def test_apply_lora_assigns_absolute_visual_depths(self):
        model = CLIP(
            embed_dim=8,
            image_resolution=32,
            vision_layers=3,
            vision_width=64,
            vision_patch_size=16,
            context_length=6,
            vocab_size=20,
            transformer_width=8,
            transformer_heads=1,
            transformer_layers=1,
        )
        args = make_args(None, v_rpr=False, dp_vrpr=True)
        args.encoder = 'vision'
        args.dropout_rate = 0.
        args.rank = 1
        layers = apply_lora(args, model)
        self.assertEqual(len(layers), 3)
        self.assertEqual(
            [layer.readout_kind for layer in layers],
            ['visual_cls_progressive'] * 3)
        self.assertEqual(
            [layer.readout_depth for layer in layers],
            [0., 0.5, 1.])

    def test_text_style_attention_remains_ordinary_lora(self):
        layer = make_attention(partitioned=False)
        self.assertFalse(layer.q_proj.readout_partition)
        inputs = torch.randn(5, 3, 8)
        layer(inputs, inputs, inputs, need_weights=False)

    def test_checkpoint_roundtrip_records_v_rpr(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_args(directory)
            source = make_attention(partitioned=True)
            with torch.no_grad():
                source.q_proj.w_lora_B.normal_()
            save_lora(args, [source])
            restored = make_attention(partitioned=True)
            load_lora(args, [restored])
            torch.testing.assert_close(
                restored.q_proj.w_lora_B,
                source.q_proj.w_lora_B)
            self.assertTrue(get_adapter_metadata(args)['v_rpr'])

    def test_checkpoint_roundtrip_records_dp_vrpr(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_args(directory, v_rpr=False, dp_vrpr=True)
            source = make_attention(
                partitioned=True,
                readout_kind='visual_cls_progressive',
                readout_depth=0.5)
            with torch.no_grad():
                source.q_proj.w_lora_B.normal_()
            save_lora(args, [source])
            restored = make_attention(
                partitioned=True,
                readout_kind='visual_cls_progressive',
                readout_depth=0.5)
            load_lora(args, [restored])
            torch.testing.assert_close(
                restored.q_proj.w_lora_B,
                source.q_proj.w_lora_B)
            metadata = get_adapter_metadata(args)
            self.assertTrue(metadata['dp_vrpr'])
            self.assertEqual(
                metadata['dp_vrpr_schedule'],
                'linear_absolute_depth')

if __name__ == '__main__':
    unittest.main()
