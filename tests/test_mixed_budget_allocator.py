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

    def test_mi_prior_biases_high_mi_layer_to_get_upgrade(self):
        # Two layers with identical candidates. Without MI the tie-break
        # is arbitrary (defaults to lower index). With a strong MI prior,
        # the high-MI layer should win the upgrade budget.
        layers = [
            {
                "idx": 0,
                "key": "low_mi",
                "params": 1000,
                "candidates": [
                    {"method": "ternary", "bpw": 1.5, "mse": 1.0},
                    {"method": "int4", "bpw": 3.0, "mse": 0.1},
                ],
            },
            {
                "idx": 1,
                "key": "high_mi",
                "params": 1000,
                "candidates": [
                    {"method": "ternary", "bpw": 1.5, "mse": 1.0},
                    {"method": "int4", "bpw": 3.0, "mse": 0.1},
                ],
            },
        ]
        result = allocate_mixed_budget(
            layers,
            target_avg_bpw=2.5,
            mi_scores=[0.1, 10.0],
            mi_prior=10.0,
        )
        chosen = {item["key"]: item["method"] for item in result["selected_layers"]}
        self.assertEqual(chosen["high_mi"], "int4")
        self.assertEqual(chosen["low_mi"], "ternary")
        self.assertTrue(result["mi_used"])

    def test_mi_prior_zero_matches_default_behaviour(self):
        layers = [
            {
                "idx": 0,
                "key": "a",
                "params": 100,
                "candidates": [
                    {"method": "cheap", "bpw": 3.0, "mse": 0.10},
                    {"method": "upgrade", "bpw": 4.0, "mse": 0.01},
                ],
            },
            {
                "idx": 1,
                "key": "b",
                "params": 100,
                "candidates": [
                    {"method": "cheap", "bpw": 3.0, "mse": 0.10},
                    {"method": "upgrade", "bpw": 4.0, "mse": 0.08},
                ],
            },
        ]
        # mi_prior=0 with mi_scores supplied should give the same result
        # as no mi_scores.
        baseline = allocate_mixed_budget(layers, target_avg_bpw=3.5)
        with_prior = allocate_mixed_budget(
            layers,
            target_avg_bpw=3.5,
            mi_scores=[5.0, 0.1],
            mi_prior=0.0,
        )
        baseline_chosen = {item["key"]: item["method"] for item in baseline["selected_layers"]}
        prior_chosen = {item["key"]: item["method"] for item in with_prior["selected_layers"]}
        self.assertEqual(baseline_chosen, prior_chosen)


if __name__ == "__main__":
    unittest.main()
