import unittest

import torch

from src.error_budget_residual import (
    dequantize_binary_residual,
    dequantize_error_budget_residual,
    estimate_binary_residual_bpw,
    estimate_error_budget_residual_bpw,
    pack_binary_sign,
    pack_signed_int2,
    quantize_binary_residual,
    quantize_error_budget_residual,
    unpack_binary_sign,
    unpack_signed_int2,
)
from src.groupwise_int4 import dequantize_groupwise_int4, quantize_groupwise_int4


class BinaryResidualQuantizationTests(unittest.TestCase):
    def test_pack_signed_int2_round_trips_values(self):
        values = torch.tensor([-1, 0, 1, -1, 1, 0, -1], dtype=torch.int8)

        packed = pack_signed_int2(values)
        unpacked = unpack_signed_int2(packed, values.numel())

        self.assertTrue(torch.equal(unpacked, values))

    def test_pack_binary_sign_round_trips_signs(self):
        values = torch.tensor([-1, 1, 1, -1, -1, 1, -1, 1, 1], dtype=torch.int8)

        packed = pack_binary_sign(values)
        unpacked = unpack_binary_sign(packed, values.numel())

        self.assertTrue(torch.equal(unpacked, values))

    def test_binary_residual_dequantizes_to_original_shape(self):
        weight = torch.tensor(
            [
                [-1.0, -0.75, -0.2, 0.0, 0.15],
                [0.0, 0.3, 0.55, 0.8, 1.0],
            ],
            dtype=torch.float32,
        )

        entry = quantize_binary_residual(weight, group_size=4)
        restored = dequantize_binary_residual(entry)

        self.assertEqual(tuple(restored.shape), tuple(weight.shape))
        self.assertEqual(entry["format"], "int2_binary_residual")
        self.assertEqual(entry["base_bits"], 2)
        self.assertEqual(entry["residual_bits"], 1)
        self.assertIn("base_packed", entry)
        self.assertIn("residual_packed", entry)

    def test_binary_residual_improves_over_int2_base(self):
        weight = torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).reshape(4, 8)

        entry = quantize_binary_residual(weight, group_size=8)
        restored = dequantize_binary_residual(entry)
        base_only = dequantize_binary_residual(entry, include_residual=False)

        residual_mse = (restored - weight).pow(2).mean().item()
        base_mse = (base_only - weight).pow(2).mean().item()

        self.assertLess(residual_mse, base_mse)

    def test_binary_residual_uses_less_than_groupwise_int4_bpw(self):
        shape = (16, 128)

        residual_bpw = estimate_binary_residual_bpw(shape, group_size=128)
        int4_bpw = quantize_groupwise_int4(torch.zeros(shape), group_size=128)["bpw"]

        self.assertLess(residual_bpw, int4_bpw)
        self.assertAlmostEqual(residual_bpw, 3.25)

    def test_binary_residual_is_reasonable_but_not_forced_to_beat_int4(self):
        weight = torch.randn(8, 64, generator=torch.Generator().manual_seed(7))

        residual = dequantize_binary_residual(quantize_binary_residual(weight, group_size=16))
        int4 = dequantize_groupwise_int4(quantize_groupwise_int4(weight, group_size=16))

        residual_mse = (residual - weight).pow(2).mean().item()
        int4_mse = (int4 - weight).pow(2).mean().item()

        self.assertGreater(residual_mse, 0.0)
        self.assertGreater(int4_mse, 0.0)
        self.assertLess(residual_mse, 10 * int4_mse)

    def test_error_budget_residual_stays_below_int4_bpw(self):
        shape = (16, 128)

        residual_bpw = estimate_error_budget_residual_bpw(
            shape,
            group_size=128,
            outliers_per_group=8,
        )
        int4_bpw = quantize_groupwise_int4(torch.zeros(shape), group_size=128)["bpw"]

        self.assertLess(residual_bpw, int4_bpw)
        self.assertAlmostEqual(residual_bpw, 4.0625)

    def test_error_budget_side_channel_targets_largest_residuals(self):
        weight = torch.tensor(
            [[-1.0, -0.75, -0.4, -0.1, 0.05, 0.35, 0.7, 1.0]],
            dtype=torch.float32,
        )

        binary = dequantize_binary_residual(quantize_binary_residual(weight, group_size=8))
        error_budget = dequantize_error_budget_residual(
            quantize_error_budget_residual(weight, group_size=8, outliers_per_group=2)
        )

        binary_mse = (binary - weight).pow(2).mean().item()
        error_budget_mse = (error_budget - weight).pow(2).mean().item()

        self.assertLess(error_budget_mse, binary_mse)

    def test_error_budget_residual_supports_256_wide_groups(self):
        weight = torch.linspace(-1.0, 1.0, steps=256, dtype=torch.float32).reshape(1, 256)

        entry = quantize_error_budget_residual(weight, group_size=256, outliers_per_group=16)
        restored = dequantize_error_budget_residual(entry)

        self.assertEqual(tuple(restored.shape), tuple(weight.shape))


if __name__ == "__main__":
    unittest.main()
