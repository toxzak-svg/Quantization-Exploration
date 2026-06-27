import unittest

import torch

from src.groupwise_int4 import (
    dequantize_groupwise_int4,
    estimate_groupwise_int4_bpw,
    pack_signed_int4,
    quantize_groupwise_int4,
    unpack_signed_int4,
)


class GroupwiseInt4Tests(unittest.TestCase):
    def test_pack_signed_int4_round_trips_odd_length_values(self):
        values = torch.tensor([-8, -7, -1, 0, 1, 6, 7], dtype=torch.int8)

        packed = pack_signed_int4(values)
        unpacked = unpack_signed_int4(packed, values.numel())

        self.assertTrue(torch.equal(unpacked, values))
        self.assertEqual(packed.dtype, torch.uint8)

    def test_groupwise_int4_dequantizes_to_original_shape_with_bounded_error(self):
        weight = torch.tensor(
            [
                [-1.0, -0.5, 0.0, 0.5, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ],
            dtype=torch.float32,
        )

        entry = quantize_groupwise_int4(weight, group_size=4)
        restored = dequantize_groupwise_int4(entry)

        self.assertEqual(tuple(restored.shape), tuple(weight.shape))
        self.assertEqual(entry["format"], "groupwise_int4")
        self.assertEqual(entry["bits"], 4)

        scales = entry["scales"].to(torch.float32)
        worst_scale = scales.max().item()
        self.assertLessEqual((restored - weight).abs().max().item(), worst_scale / 2 + 1e-6)

    def test_groupwise_int4_handles_zero_groups(self):
        weight = torch.zeros((2, 7), dtype=torch.float32)

        entry = quantize_groupwise_int4(weight, group_size=4)
        restored = dequantize_groupwise_int4(entry)

        self.assertTrue(torch.equal(restored, weight))
        self.assertTrue(torch.all(entry["scales"] > 0))

    def test_bpw_estimate_includes_group_scale_overhead(self):
        # Shape has 10 weights and four per-row scale groups:
        # data = 10 * 4 bits, scales = 4 * 16 bits.
        bpw = estimate_groupwise_int4_bpw((2, 5), group_size=4, scale_bits=16)

        self.assertAlmostEqual(bpw, 10.4)


if __name__ == "__main__":
    unittest.main()
