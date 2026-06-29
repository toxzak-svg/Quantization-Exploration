import unittest

import torch

from scripts.quantize_mixed_budget import build_selection_map, quantize_weight_for_method


class MixedCheckpointBuilderTests(unittest.TestCase):
    def test_quantize_weight_for_method_marks_int2_base_format(self):
        weight = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32).reshape(1, 128)

        entry = quantize_weight_for_method(weight, "int2_base", group_size=128)

        self.assertEqual(entry["format"], "int2_base")
        self.assertEqual(entry["orig_shape"], [1, 128])
        self.assertIn("base_packed", entry)
        self.assertAlmostEqual(entry["bpw"], 2.125)

    def test_quantize_weight_for_method_parses_error_budget_k(self):
        weight = torch.linspace(-1.0, 1.0, steps=8, dtype=torch.float32).reshape(1, 8)

        entry = quantize_weight_for_method(weight, "int2_error_budget_k4", group_size=8)

        self.assertEqual(entry["format"], "int2_error_budget_residual")
        self.assertEqual(entry["outliers_per_group"], 4)

    def test_build_selection_map_uses_selected_layers_by_key(self):
        scan = {
            "mixed_allocation": {
                "selected_layers": [
                    {"idx": 3, "key": "b.weight", "method": "groupwise_int4"},
                    {"idx": 1, "key": "a.weight", "method": "int2_base"},
                ]
            }
        }

        selection = build_selection_map(scan)

        self.assertEqual(selection["a.weight"], "int2_base")
        self.assertEqual(selection["b.weight"], "groupwise_int4")


if __name__ == "__main__":
    unittest.main()
