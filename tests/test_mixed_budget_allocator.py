import unittest

from src.mixed_budget import allocate_mixed_budget, summarize_allocation


class MixedBudgetAllocatorTests(unittest.TestCase):
    def test_allocator_spends_budget_on_best_error_reduction_per_bit(self):
        layers = [
            {
                "idx": 0,
                "key": "layer0",
                "params": 100,
                "candidates": [
                    {"method": "cheap", "bpw": 3.0, "mse": 0.10},
                    {"method": "upgrade", "bpw": 4.0, "mse": 0.01},
                ],
            },
            {
                "idx": 1,
                "key": "layer1",
                "params": 100,
                "candidates": [
                    {"method": "cheap", "bpw": 3.0, "mse": 0.10},
                    {"method": "upgrade", "bpw": 4.0, "mse": 0.08},
                ],
            },
        ]

        allocation = allocate_mixed_budget(layers, target_avg_bpw=3.5)

        chosen = {item["key"]: item["method"] for item in allocation["selected_layers"]}
        self.assertEqual(chosen, {"layer0": "upgrade", "layer1": "cheap"})
        self.assertLessEqual(allocation["avg_bpw"], 3.5)

    def test_allocator_uses_activation_weight_in_score(self):
        layers = [
            {
                "idx": 0,
                "key": "quiet",
                "params": 100,
                "activation_weight": 1.0,
                "candidates": [
                    {"method": "cheap", "bpw": 3.0, "mse": 0.10},
                    {"method": "upgrade", "bpw": 4.0, "mse": 0.02},
                ],
            },
            {
                "idx": 1,
                "key": "hot",
                "params": 100,
                "activation_weight": 10.0,
                "candidates": [
                    {"method": "cheap", "bpw": 3.0, "mse": 0.10},
                    {"method": "upgrade", "bpw": 4.0, "mse": 0.08},
                ],
            },
        ]

        allocation = allocate_mixed_budget(layers, target_avg_bpw=3.5)

        chosen = {item["key"]: item["method"] for item in allocation["selected_layers"]}
        self.assertEqual(chosen, {"quiet": "cheap", "hot": "upgrade"})

    def test_summarize_allocation_reports_weighted_metrics(self):
        selected = [
            {"params": 100, "bpw": 3.0, "mse": 0.09, "weighted_mse": 0.09},
            {"params": 300, "bpw": 4.0, "mse": 0.01, "weighted_mse": 0.01},
        ]

        summary = summarize_allocation(selected)

        self.assertAlmostEqual(summary["avg_bpw"], 3.75)
        self.assertAlmostEqual(summary["weighted_mse"], 0.03)
        self.assertAlmostEqual(summary["weighted_rmse"], 0.03**0.5)


if __name__ == "__main__":
    unittest.main()
