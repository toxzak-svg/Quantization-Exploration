import tempfile
import unittest
from pathlib import Path

from scripts.scan_mixed_budget import append_resume_record, load_resume_records


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


if __name__ == "__main__":
    unittest.main()
