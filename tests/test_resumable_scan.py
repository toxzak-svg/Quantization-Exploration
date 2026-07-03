import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import torch

from scripts.scan_mixed_budget import (
    append_resume_record,
    layer_mi_score,
    load_mi_report,
    load_resume_records,
    scan,
)


class ResumableScanTests(unittest.TestCase):
    def test_resume_records_round_trip_by_weight_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.layers.jsonl"
            layer = {
                "idx": 7,
                "key": "model.language_model.layers.0.mlp.up_proj.weight",
                "shape": [2, 4],
                "params": 8,
                "activation_weight": 1.0,
                "candidates": [{"method": "groupwise_int4", "bpw": 4.125, "mse": 0.01}],
            }

            append_resume_record(path, layer)

            loaded = load_resume_records(path)

        self.assertEqual(loaded[layer["key"]], layer)

    def test_resume_records_keep_latest_duplicate_for_crash_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.layers.jsonl"
            append_resume_record(path, {"idx": 1, "key": "w", "candidates": [{"mse": 0.2}]})
            append_resume_record(path, {"idx": 1, "key": "w", "candidates": [{"mse": 0.1}]})

            loaded = load_resume_records(path)

        self.assertEqual(loaded["w"]["candidates"], [{"mse": 0.1}])

    def test_load_mi_report_uses_explicit_layer_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mi.json"
            path.write_text(
                json.dumps({"layer_indices": [1, 3], "mi_scores": [0.25, 0.75]}),
                encoding="utf-8",
            )

            scores = load_mi_report(path)

        self.assertEqual(scores, {1: 0.25, 3: 0.75})

    def test_layer_mi_score_extracts_transformer_layer_from_weight_key(self):
        key = "model.language_model.layers.17.mlp.down_proj.weight"

        self.assertEqual(layer_mi_score(key, {17: 2.5}), 2.5)
        self.assertEqual(layer_mi_score(key, {16: 1.0}), 0.0)

    def test_scan_passes_mapped_mi_scores_to_allocator(self):
        resume_layers = [
            {
                "idx": 0,
                "key": "model.language_model.layers.0.mlp.down_proj.weight",
                "shape": [1, 128],
                "params": 1000,
                "activation_weight": 1.0,
                "candidates": [
                    {"method": "cheap", "bpw": 1.5, "mse": 1.0},
                    {"method": "groupwise_int4", "bpw": 3.0, "mse": 0.1},
                ],
            },
            {
                "idx": 1,
                "key": "model.language_model.layers.1.mlp.down_proj.weight",
                "shape": [1, 128],
                "params": 1000,
                "activation_weight": 1.0,
                "candidates": [
                    {"method": "cheap", "bpw": 1.5, "mse": 1.0},
                    {"method": "groupwise_int4", "bpw": 3.0, "mse": 0.1},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            resume_path = Path(tmp) / "scan.layers.jsonl"
            for layer in resume_layers:
                append_resume_record(resume_path, layer)

            with mock.patch(
                "scripts.scan_mixed_budget.iter_model_weights",
                return_value=[
                    (0, resume_layers[0]["key"], torch.zeros((1, 128))),
                    (1, resume_layers[1]["key"], torch.zeros((1, 128))),
                ],
            ):
                result = scan(
                    model_dir=Path("unused"),
                    group_size=128,
                    outlier_options=[],
                    max_layers=None,
                    target_bpw=2.5,
                    target_margin_bpw=0.0,
                    activation_weights={},
                    resume_jsonl=resume_path,
                    mi_scores_by_layer={0: 0.1, 1: 10.0},
                    mi_prior=10.0,
                )

        chosen = {
            item["key"]: item["method"]
            for item in result["mixed_allocation"]["selected_layers"]
        }
        self.assertEqual(
            chosen["model.language_model.layers.1.mlp.down_proj.weight"],
            "groupwise_int4",
        )
        self.assertEqual(
            chosen["model.language_model.layers.0.mlp.down_proj.weight"],
            "cheap",
        )
        self.assertTrue(result["mixed_allocation"]["mi_used"])


if __name__ == "__main__":
    unittest.main()
