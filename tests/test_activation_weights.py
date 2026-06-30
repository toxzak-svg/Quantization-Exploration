import tempfile
import unittest
from pathlib import Path

import torch

from scripts.collect_activation_weights import (
    ActivationAccumulator,
    module_weight_key,
    normalize_activation_weights,
)


class ActivationWeightTests(unittest.TestCase):
    def test_module_weight_key_maps_module_name_to_checkpoint_key(self):
        self.assertEqual(
            module_weight_key("model.language_model.layers.0.mlp.up_proj"),
            "model.language_model.layers.0.mlp.up_proj.weight",
        )

    def test_accumulator_records_mean_square_activation_by_weight_key(self):
        acc = ActivationAccumulator()
        activation = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        acc.record("layer.weight", activation)
        stats = acc.to_stats()

        self.assertEqual(stats["layer.weight"]["samples"], 4)
        self.assertAlmostEqual(stats["layer.weight"]["mean_square"], 7.5)

    def test_normalize_activation_weights_centers_positive_weights_at_one(self):
        stats = {
            "quiet.weight": {"mean_square": 2.0, "samples": 10},
            "hot.weight": {"mean_square": 8.0, "samples": 10},
        }

        weights = normalize_activation_weights(stats)

        self.assertAlmostEqual(weights["quiet.weight"], 0.4)
        self.assertAlmostEqual(weights["hot.weight"], 1.6)

    def test_accumulator_saves_stats_and_weights_atomically(self):
        acc = ActivationAccumulator()
        acc.record("layer.weight", torch.ones((2, 2)))

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "stats.json"
            weights_path = Path(tmp) / "weights.json"
            acc.save(stats_path, weights_path)

            self.assertTrue(stats_path.exists())
            self.assertTrue(weights_path.exists())
            self.assertFalse(stats_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
