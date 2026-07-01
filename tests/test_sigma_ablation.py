import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from src.sigma_ablation import (
    SIGMAConfig,
    make_blocks,
    sigma_bpw,
    sketch_learned_continuous,
    sketch_random_hadamard,
    train_generator,
    quantize_layer,
    within_bucket_variance_ratio,
)


class SigmaAblationTests(unittest.TestCase):
    def test_blocks_preserve_signs_and_normalize_magnitudes_per_block(self):
        weight = torch.tensor(
            [[-2.0, -1.0, 1.0, 2.0], [0.0, 0.5, -0.5, 1.0]],
            dtype=torch.float32,
        )

        signs, mags, blocks, stats = make_blocks(weight, block_size=4)

        self.assertEqual(tuple(signs.shape), (2, 4))
        self.assertEqual(tuple(mags.shape), (2, 4))
        self.assertEqual(tuple(blocks.shape), (2, 4))
        self.assertEqual(stats.n_blocks, 2)
        self.assertEqual(stats.block_size, 4)
        self.assertTrue(torch.equal(signs[0], torch.tensor([0, 0, 1, 1], dtype=torch.int8)))
        self.assertTrue(torch.allclose(mags.mean(dim=1), torch.ones(2), atol=1e-6))

    def test_learned_continuous_sketch_runs_without_autograd_error(self):
        generator = torch.Generator().manual_seed(11)
        signs = torch.randint(0, 2, (24, 8), generator=generator, dtype=torch.int8)
        mags = torch.rand((24, 8), generator=generator)

        buckets = sketch_learned_continuous(
            signs,
            mags,
            n_bits=3,
            steps=3,
            seed=7,
            progress=False,
        )

        self.assertEqual(tuple(buckets.shape), (24,))
        self.assertGreaterEqual(int(buckets.min().item()), 0)
        self.assertLess(int(buckets.max().item()), 8)

    def test_generator_quantizes_tiny_layer_with_expected_shapes(self):
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn((4, 8), generator=generator)
        signs, mags, blocks, stats = make_blocks(weight, block_size=4)
        buckets = sketch_random_hadamard(signs, n_bits=2, seed=3)

        model = train_generator(
            signs,
            blocks,
            buckets,
            config=SIGMAConfig(block_size=4, n_bits=2, n_taus=3, rank=4),
            steps=2,
            batch_size=4,
            device="cpu",
            seed=5,
            progress=False,
        )
        recon, tau_used, alpha_used = quantize_layer(
            signs,
            blocks,
            buckets,
            model,
            n_taus=3,
            device="cpu",
            chunk=4,
        )

        self.assertEqual(tuple(recon.shape), tuple(blocks.shape))
        self.assertEqual(tuple(tau_used.shape), (stats.n_blocks,))
        self.assertEqual(tuple(alpha_used.shape), (stats.n_blocks,))
        self.assertTrue(torch.isfinite(recon).all())

    def test_variance_ratio_and_bit_budget_are_finite(self):
        mags = torch.tensor(
            [[1.0, 2.0], [1.1, 1.9], [3.0, 4.0], [3.1, 3.9]],
            dtype=torch.float32,
        )
        buckets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        ratio = within_bucket_variance_ratio(mags, buckets, n_buckets=2)
        bpw = sigma_bpw(n_blocks=128, n_buckets=4, n_taus=3, rank=4, block_size=8)

        self.assertGreaterEqual(ratio, 0.0)
        self.assertLess(ratio, 1.0)
        self.assertGreater(bpw, 1.0)

    def test_builder_writes_notebook_with_runnable_sigma_cells(self):
        notebook_path = Path("notebook/sigma_sketch_ablation.ipynb")
        if notebook_path.exists():
            notebook_path.unlink()

        subprocess.check_call([sys.executable, "scripts/build_sigma_notebook.py"])

        data = json.loads(notebook_path.read_text(encoding="utf-8"))
        all_source = "\n".join("".join(cell["source"]) for cell in data["cells"])
        self.assertIn("from safetensors.torch import load_file, save_file", all_source)
        self.assertIn("class SIGMAGenerator", all_source)
        self.assertNotIn("rank={rank}", all_source)
        self.assertNotIn("{n_buckets} buckets", all_source)


if __name__ == "__main__":
    unittest.main()
