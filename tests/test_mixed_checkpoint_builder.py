import unittest
import tempfile
from pathlib import Path

import torch

from scripts.quantize_mixed_budget import (
    build_selection_map,
    load_layer_shard,
    quantize_weight_for_method,
    save_layer_shard,
)


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

    def test_layer_shards_round_trip_with_validation_metadata(self):
        entry = {
            "format": "groupwise_int4",
            "orig_shape": [1, 4],
            "packed_int4": torch.tensor([1, 2], dtype=torch.uint8),
        }
        layer_stat = {
            "idx": 4,
            "key": "model.language_model.layers.0.mlp.up_proj.weight",
            "method": "groupwise_int4",
            "shape": [1, 4],
            "params": 4,
            "bpw": 4.125,
            "mse": 0.01,
            "rmse": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmp:
            shard_path = save_layer_shard(Path(tmp), layer_stat, entry, group_size=128)
            loaded = load_layer_shard(
                shard_path,
                key=layer_stat["key"],
                method=layer_stat["method"],
                group_size=128,
            )

        self.assertEqual(loaded["layer_stat"], layer_stat)
        self.assertTrue(torch.equal(loaded["entry"]["packed_int4"], entry["packed_int4"]))

    def test_layer_shard_validation_rejects_wrong_method(self):
        entry = {"format": "groupwise_int4", "orig_shape": [1, 4]}
        layer_stat = {
            "idx": 0,
            "key": "w",
            "method": "groupwise_int4",
            "shape": [1, 4],
            "params": 4,
            "bpw": 4.125,
            "mse": None,
            "rmse": None,
        }

        with tempfile.TemporaryDirectory() as tmp:
            shard_path = save_layer_shard(Path(tmp), layer_stat, entry, group_size=128)

            with self.assertRaises(ValueError):
                load_layer_shard(shard_path, key="w", method="int2_base", group_size=128)


if __name__ == "__main__":
    unittest.main()
