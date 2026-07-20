import unittest

import torch
import torch.nn as nn

from loralib.layers import LinearLoRA, PlainMultiheadAttentionLoRA
from loralib.utils import disabled_adapters


class AdapterDisableTest(unittest.TestCase):
    def test_eval_does_not_remerge_disabled_linear_adapter(self):
        torch.manual_seed(3)
        base = nn.Linear(4, 4)
        layer = LinearLoRA(base, r=2, lora_alpha=1, dropout_rate=0.)
        layer.w_lora_B.data.normal_()
        inputs = torch.randn(3, 4)
        frozen = nn.functional.linear(inputs, base.weight, base.bias)

        layer.eval()
        adapted = layer(inputs)
        self.assertFalse(torch.allclose(adapted, frozen))

        with disabled_adapters([layer]):
            layer.eval()
            ablated = layer(inputs)
            self.assertTrue(torch.allclose(ablated, frozen, atol=1e-6))
            self.assertFalse(layer.merged)

        restored = layer(inputs)
        self.assertTrue(torch.allclose(restored, adapted, atol=1e-6))

    def test_eval_does_not_remerge_disabled_attention_block(self):
        torch.manual_seed(7)
        base = nn.MultiheadAttention(8, 2)
        layer = PlainMultiheadAttentionLoRA(
            base, enable_lora=['q', 'k', 'v'], r=2,
            lora_alpha=1, dropout_rate=0.)
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj):
            projection.w_lora_B.data.normal_()
        inputs = torch.randn(5, 2, 8)

        layer.eval()
        adapted, _ = layer(inputs, inputs, inputs, need_weights=False)
        with disabled_adapters([layer]):
            # This mirrors evaluate_lora(), which always calls model.eval().
            layer.eval()
            ablated, _ = layer(inputs, inputs, inputs, need_weights=False)
        restored, _ = layer(inputs, inputs, inputs, need_weights=False)

        self.assertFalse(torch.allclose(adapted, ablated))
        self.assertTrue(torch.allclose(restored, adapted, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
