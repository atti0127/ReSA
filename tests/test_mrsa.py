import tempfile
from types import SimpleNamespace
import unittest

import torch

from lora import (
    fast_walsh_hadamard_transform,
    matched_random_subspace_features,
)
from loralib.layers import PlainMultiheadAttentionLoRA
from loralib.utils import get_adapter_metadata, load_lora, save_lora


def make_args(save_path=None, mrsa=True):
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
        mrsa=mrsa,
        dropout_rate=0.25,
        v_rpr=False,
        dp_vrpr=True,
        backbone='ViT-B/16',
        dataset='imagenet',
        shots=16,
        seed=1,
        save_path=save_path,
        filename='lora_weights',
        checkpoint_dataset=None,
        eval_only=False,
    )


def make_attention():
    base = torch.nn.MultiheadAttention(8, 2, dropout=0.)
    return PlainMultiheadAttentionLoRA(
        base,
        enable_lora=['q', 'k', 'v'],
        r=2,
        lora_alpha=1,
        dropout_rate=0.,
        readout_kind='visual_cls_progressive',
        readout_depth=0.5,
    )


class MatchedRandomSubspaceAdaptationTest(unittest.TestCase):
    def test_hadamard_transform_is_orthonormal(self):
        torch.manual_seed(3)
        features = torch.randn(7, 8, dtype=torch.float64)
        transformed = fast_walsh_hadamard_transform(features)
        torch.testing.assert_close(
            transformed.square().sum(dim=-1),
            features.square().sum(dim=-1))

    def test_projection_is_matched_across_modalities(self):
        torch.manual_seed(5)
        features = torch.randn(6, 16)
        image_features, text_features = matched_random_subspace_features(
            features, features.clone(), drop_rate=0.25, seed=19)
        torch.testing.assert_close(image_features, text_features)
        torch.testing.assert_close(
            (image_features * text_features).sum(dim=-1),
            torch.ones(6))
        self.assertEqual(image_features.shape[-1], 12)

    def test_projection_is_deterministic_per_step_seed(self):
        torch.manual_seed(7)
        images = torch.randn(4, 12)
        texts = torch.randn(9, 12)
        first = matched_random_subspace_features(
            images, texts, drop_rate=0.25, seed=101)
        repeated = matched_random_subspace_features(
            images, texts, drop_rate=0.25, seed=101)
        changed = matched_random_subspace_features(
            images, texts, drop_rate=0.25, seed=102)
        torch.testing.assert_close(first[0], repeated[0])
        torch.testing.assert_close(first[1], repeated[1])
        self.assertFalse(torch.equal(first[0], changed[0]))

    def test_projection_supports_non_power_of_two_embeddings(self):
        images = torch.randn(3, 10)
        texts = torch.randn(5, 10)
        projected_images, projected_texts = (
            matched_random_subspace_features(
                images, texts, drop_rate=0.2, seed=23))
        self.assertEqual(tuple(projected_images.shape), (3, 8))
        self.assertEqual(tuple(projected_texts.shape), (5, 8))
        torch.testing.assert_close(
            projected_images.norm(dim=-1), torch.ones(3))
        torch.testing.assert_close(
            projected_texts.norm(dim=-1), torch.ones(5))

    def test_projection_preserves_gradients_to_both_modalities(self):
        images = torch.randn(3, 8, requires_grad=True)
        texts = torch.randn(5, 8, requires_grad=True)
        projected_images, projected_texts = (
            matched_random_subspace_features(
                images, texts, drop_rate=0.25, seed=29))
        (projected_images @ projected_texts.t()).sum().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0.)
        self.assertGreater(texts.grad.abs().sum().item(), 0.)

    def test_zero_drop_rate_is_identity_after_normalization(self):
        images = torch.randn(3, 10)
        texts = torch.randn(5, 10)
        projected_images, projected_texts = (
            matched_random_subspace_features(
                images, texts, drop_rate=0., seed=31))
        torch.testing.assert_close(
            projected_images,
            torch.nn.functional.normalize(images, dim=-1))
        torch.testing.assert_close(
            projected_texts,
            torch.nn.functional.normalize(texts, dim=-1))

    def test_metadata_records_mrsa(self):
        metadata = get_adapter_metadata(make_args())
        self.assertTrue(metadata['mrsa'])
        self.assertEqual(
            metadata['mrsa_projection'], 'signed_hadamard_subspace')
        self.assertEqual(metadata['mrsa_drop_rate'], 0.25)

    def test_checkpoint_rejects_mrsa_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source_args = make_args(directory, mrsa=True)
            save_lora(source_args, [make_attention()])

            target_args = make_args(directory, mrsa=False)
            with self.assertRaisesRegex(ValueError, 'MRSA mismatch'):
                load_lora(target_args, [make_attention()])


if __name__ == '__main__':
    unittest.main()
