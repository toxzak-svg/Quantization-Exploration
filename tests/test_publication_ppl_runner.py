import unittest
import subprocess
import sys

from scripts import run_publication_ppl


class PublicationPplRunnerTests(unittest.TestCase):
    def test_parse_token_limit_accepts_all_for_full_dataset(self):
        self.assertIsNone(run_publication_ppl.parse_token_limit("all"))
        self.assertIsNone(run_publication_ppl.parse_token_limit("0"))
        self.assertEqual(run_publication_ppl.parse_token_limit("4096"), 4096)

    def test_compact_apply_stats_counts_lists_without_dropping_samples(self):
        stats = run_publication_ppl.compact_apply_stats(
            {
                "replaced": 276,
                "skipped": ["k0", "k1", "k2"],
                "missing": ["m0"],
                "shape_mismatches": [],
            },
            sample_size=2,
        )

        self.assertEqual(
            stats,
            {
                "replaced": 276,
                "skipped_count": 3,
                "missing_count": 1,
                "shape_mismatch_count": 0,
                "skipped_sample": ["k0", "k1"],
                "missing_sample": ["m0"],
                "shape_mismatch_sample": [],
            },
        )

    def test_build_publication_payload_records_exact_comparison(self):
        payload = run_publication_ppl.build_publication_payload(
            label="groupwise_int4_g128_full",
            base_result={
                "ppl": 108.4542007446289,
                "seq_len": 292282,
                "chunks": 571,
                "elapsed_s": 73.1,
                "device": "cuda",
                "runtime_dtype": "torch.bfloat16",
            },
            quantized_result={
                "ppl": 107.5655746459961,
                "seq_len": 292282,
                "chunks": 571,
                "elapsed_s": 80.3,
                "device": "cuda",
                "runtime_dtype": "torch.bfloat16",
                "apply_stats": {
                    "replaced": 276,
                    "skipped": ["shared.k"],
                    "missing": [],
                    "shape_mismatches": [],
                },
                "checkpoint_stats": {
                    "method": "groupwise_int4",
                    "avg_bpw": 4.125,
                    "weighted_rmse": 0.0028486063019064127,
                },
            },
            quantized_pt="/content/sub1quant/quantized/gemma_groupwise_int4_g128.pt",
            model_dir="/content/sub1quant/models/gemma-4-E2B",
            wikitext="/content/sub1quant/data/wiki.test.txt",
            max_length=512,
            stride=512,
            token_limit=None,
        )

        self.assertEqual(payload["label"], "groupwise_int4_g128_full")
        self.assertEqual(payload["token_limit"], "all")
        self.assertEqual(payload["quantized"]["apply_stats"]["skipped_count"], 1)
        self.assertAlmostEqual(
            payload["comparison"]["quantized_minus_base_ppl"],
            -0.8886260986328125,
        )
        self.assertAlmostEqual(
            payload["comparison"]["quantized_ppl_ratio_vs_base"],
            0.9918064391002686,
        )

    def test_script_help_works_when_executed_by_path(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_publication_ppl.py", "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tokens", result.stdout)


if __name__ == "__main__":
    unittest.main()
