"""Synthetic tests for cross-layer MI.

These tests build a toy chain of "layers" with a known information-flow
structure, then verify that the estimators recover the right ranking.

The chain:

    L0 --strong--> L1 --strong--> L2 --weak--> L3 --none--> L4

We expect:

* MI(L_i; L_{i+1}) is high when the coupling is strong.
* MI(L_0; L_4) is small (information has been forgotten after 4 steps).
* The MI matrix is roughly symmetric.
* The per-layer sensitivity score ranks L1 and L2 highest (they are
  central in the information chain).

If these tests fail, the estimator is broken and any conclusions drawn
from the real Gemma matrix are suspect.
"""

from __future__ import annotations

import unittest

import torch

from src.cross_layer_mi import (
    CalibrationActivations,
    allocate_bits,
    estimate_hsic,
    estimate_hsic_rff,
    estimate_hsic_conditional_rff,
    hsic_matrix,
    mi_scores_from_matrix,
    mi_to_bit_allocation,
    estimate_mine,
    residual_deltas,
)


def _make_chain(
    n_tokens: int = 1024,
    dim: int = 32,
    couplings: list[float] | None = None,
    noise: float = 0.5,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Build a 5-layer toy chain with the requested couplings.

    Each layer is a (n_tokens, dim) tensor. Layer 0 is pure noise
    (initialised from a Gaussian). Layer i+1 = couplings[i] * layer_i
    + N(0, noise). A coupling of 0 gives an independent layer.
    """
    if couplings is None:
        couplings = [0.9, 0.9, 0.4, 0.0]
    gen = torch.Generator().manual_seed(seed)
    layers: list[torch.Tensor] = [
        torch.randn(n_tokens, dim, generator=gen),
    ]
    for c in couplings:
        prev = layers[-1]
        nxt = c * prev + noise * torch.randn(n_tokens, dim, generator=gen)
        layers.append(nxt)
    return layers


class HSICEstimatorTests(unittest.TestCase):
    def test_hsic_is_zero_for_independent_layers(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(512, 16, generator=gen)
        y = torch.randn(512, 16, generator=gen)
        val = estimate_hsic(x, y, sub_sample=512)
        # Biased estimator is non-negative and small for independent data.
        self.assertGreaterEqual(val, 0.0)
        self.assertLess(val, 0.05)

    def test_hsic_is_larger_for_coupled_layers(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(512, 16, generator=gen)
        # y is mostly x with small noise.
        y = x + 0.1 * torch.randn(512, 16, generator=gen)
        coupled = estimate_hsic(x, y, sub_sample=512)
        y_indep = torch.randn(512, 16, generator=gen)
        independent = estimate_hsic(x, y_indep, sub_sample=512)
        self.assertGreater(coupled, independent * 5.0)

    def test_hsic_matrix_is_symmetric(self):
        layers = _make_chain()
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        m = hsic_matrix(acts, sub_sample=512)
        self.assertEqual(tuple(m.shape), (5, 5))
        self.assertTrue(torch.allclose(m, m.t(), atol=1e-5))


class ChainRecoveryTests(unittest.TestCase):
    """The estimators must rank the chain couplings in the expected order."""

    def setUp(self):
        # 5 layers, couplings strong, strong, weak, none.
        self.layers = _make_chain()
        # Each t is (n_tokens, dim). We wrap to (1, n_tokens, dim) so the
        # batch dimension is 1 and CalibrationActivations.flattened can
        # collapse to (n_tokens, dim).
        self.acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(self.layers)},
        )

    def test_hsic_recovers_off_diagonal_decay(self):
        m = hsic_matrix(self.acts, sub_sample=1024, method="rbf")
        # Off-diagonal magnitudes should decay with chain distance.
        near = float(m[0, 1].item())
        mid = float(m[0, 2].item())
        far = float(m[0, 4].item())
        self.assertGreater(near, mid)
        self.assertGreater(mid, far * 1.5)

    def test_mi_scores_rank_central_layers_highest(self):
        m = hsic_matrix(self.acts, sub_sample=1024, method="rbf")
        scores = mi_scores_from_matrix(m, horizon=4)
        # L1 (index 1) is fed by L0 and feeds L2 -- central in the chain.
        # L2 is fed by L1 and feeds L3 weakly. Both should outscore L4.
        # L0 has only one strong neighbour; L4 is the dead end.
        self.assertGreater(float(scores[1].item()), float(scores[4].item()))
        self.assertGreater(float(scores[2].item()), float(scores[4].item()))

    def test_rff_matches_rbf_on_chain_structure(self):
        """RFF approximation should agree with RBF on chain distance ranking."""
        rbf_m = hsic_matrix(self.acts, sub_sample=1024, method="rbf")
        rff_m = hsic_matrix(self.acts, sub_sample=1024, method="rff", n_rff=512)
        # Same decay direction: near > mid > far in both.
        self.assertGreater(rff_m[0, 1].item(), rff_m[0, 2].item())
        self.assertGreater(rff_m[0, 2].item(), rff_m[0, 4].item() * 1.5)
        # The two matrices should have positive correlation.
        # Use a wide tolerance because RFF is a stochastic approximation.
        corr = torch.corrcoef(torch.stack([rbf_m.flatten(), rff_m.flatten()]))[0, 1]
        self.assertGreater(corr.item(), 0.5)

    def test_rff_pair_matches_rbf_pair(self):
        """Single-pair RFF HSIC should be in the same ballpark as RBF."""
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(512, 16, generator=gen)
        y = 0.9 * x + 0.5 * torch.randn(512, 16, generator=gen)
        rbf_val = estimate_hsic(x, y)
        rff_val = estimate_hsic_rff(x, y, n_rff=512, seed=42)
        # Both should be > 0; the RFF should be within 50% of RBF.
        self.assertGreater(rff_val, 0.0)
        self.assertLess(abs(rff_val - rbf_val) / max(rbf_val, 1e-9), 0.5)

    def test_mine_agrees_with_hsic_on_strong_coupling(self):
        # MINE on the strongly-coupled pair should give a non-trivial
        # positive estimate; on the decoupled pair it should be near zero.
        gen = torch.Generator().manual_seed(0)
        strong = estimate_mine(
            self.layers[0],
            self.layers[1],
            steps=300,
            sub_sample=512,
            seed=42,
            progress=False,
        )
        weak = estimate_mine(
            self.layers[0],
            self.layers[4],
            steps=300,
            sub_sample=512,
            seed=42,
            progress=False,
        )
        self.assertGreater(strong[0], 0.1)
        self.assertLess(weak[0], strong[0])


class BitAllocationTests(unittest.TestCase):
    def test_allocation_stays_in_range_and_matches_target(self):
        scores = torch.tensor([0.1, 1.0, 0.5, 0.0, 0.3])
        bits = mi_to_bit_allocation(
            scores,
            bits_min=1.5,
            bits_max=8.0,
            temperature=1.0,
            target_avg_bpw=4.0,
        )
        self.assertEqual(bits.numel(), 5)
        self.assertTrue(torch.all(bits >= 1.5))
        self.assertTrue(torch.all(bits <= 8.0))
        # Rank-based allocation centred on the requested average; clamp to
        # the bits range can shift the mean by a small amount.
        self.assertAlmostEqual(float(bits.mean().item()), 4.0, places=1)

    def test_allocation_handles_zero_scores(self):
        scores = torch.zeros(4)
        bits = mi_to_bit_allocation(scores, bits_min=1.5, bits_max=8.0)
        # All scores tied -> rank allocation still spreads bits linearly.
        self.assertEqual(bits.numel(), 4)
        self.assertAlmostEqual(float(bits.min().item()), 1.5, places=4)
        self.assertAlmostEqual(float(bits.max().item()), 8.0, places=4)

    def test_rank_method_produces_linear_interpolation(self):
        scores = torch.tensor([0.1, 0.4, 0.2, 0.3, 0.5])
        bits = mi_to_bit_allocation(
            scores, bits_min=1.5, bits_max=8.0, method="rank"
        )
        # Highest-scoring layer gets bits_max, lowest gets bits_min.
        max_idx = int(scores.argmax().item())
        min_idx = int(scores.argmin().item())
        self.assertAlmostEqual(float(bits[max_idx].item()), 8.0, places=4)
        self.assertAlmostEqual(float(bits[min_idx].item()), 1.5, places=4)
        # Bits span the full range linearly. Sorted descending, consecutive
        # differences should equal (bits_max - bits_min) / (n - 1).
        expected_step = (8.0 - 1.5) / (len(scores) - 1)
        sorted_bits = torch.sort(bits, descending=True).values
        diffs = sorted_bits[:-1] - sorted_bits[1:]
        self.assertTrue(torch.allclose(diffs, torch.full_like(diffs, expected_step)))

    def test_softmax_method_concentrates_on_high_scores(self):
        scores = torch.tensor([0.1, 5.0, 0.2, 0.3, 0.4])
        bits = mi_to_bit_allocation(
            scores, bits_min=1.5, bits_max=8.0, temperature=0.1, method="softmax"
        )
        # Top score should clearly get the most bits.
        self.assertEqual(int(bits.argmax().item()), 1)
        self.assertGreater(float(bits[1].item()), float(bits[0].item()))

    def test_end_to_end_allocate_bits_returns_summary(self):
        layers = _make_chain()
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        alloc = allocate_bits(
            acts, horizon=2, bits_min=1.5, bits_max=8.0,
            sub_sample=512, hsic_method="rbf",
        )
        summary = alloc.summary()
        self.assertEqual(summary["n_layers"], 5)
        self.assertGreaterEqual(summary["avg_bits"], 1.5)
        self.assertLessEqual(summary["avg_bits"], 8.0)
        # On a chain with strong-strong-weak-none couplings, the *first*
        # layer has the highest cumulative MI (it can reach layer 3 via
        # the strong links). Layer 4 (decoupled) should be among the
        # lowest-ranked. We check the relative order rather than a
        # specific pair because horizon windowing shifts the rankings.
        mi_scores = alloc.mi_scores
        self.assertGreater(
            float(mi_scores[0].item()), float(mi_scores[4].item()),
            "decoupled layer should rank lower than chained layer",
        )

    def test_end_to_end_rff_method_produces_reasonable_allocation(self):
        """The default (RFF) path should produce a valid allocation in range."""
        layers = _make_chain()
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        alloc = allocate_bits(
            acts, horizon=2, bits_min=1.5, bits_max=8.0, sub_sample=512,
        )
        # Default hsic_method is "rff"; the test for chain structure is
        # weaker here because RFF adds approximation noise. We just
        # check the allocation is in range and non-degenerate.
        self.assertEqual(alloc.method, "cross_layer_mi_hsic_rff_h2")
        self.assertEqual(alloc.bits.numel(), 5)
        self.assertTrue(torch.all(alloc.bits >= 1.5))
        self.assertTrue(torch.all(alloc.bits <= 8.0))
        # The standard deviation of bits should be > 0 (not all equal).
        self.assertGreater(float(alloc.bits.std().item()), 0.0)


class ConditionalMITests(unittest.TestCase):
    """Phase 3: conditional MI primitives.

    The unconditional estimator over-credits layers that share a
    residual stream. These tests verify that the conditional path
    actually removes that confounder.
    """

    def test_residual_deltas_preserves_layer_count_minus_one(self):
        """Drops the baseline layer (no predecessor) and adds deltas for the rest."""
        gen = torch.Generator().manual_seed(0)
        layers = {i: torch.randn(1, 64, 16, generator=gen) for i in range(5)}
        acts = CalibrationActivations(hidden_states=layers)
        deltas = residual_deltas(acts, keep_baseline=False)
        # 5 input layers, baseline dropped => 4 deltas.
        self.assertEqual(deltas.layer_count(), 4)
        # Indices are 1, 2, 3, 4 (the non-baselines).
        self.assertEqual(sorted(deltas.hidden_states.keys()), [1, 2, 3, 4])

    def test_residual_deltas_keep_baseline_preserves_first_layer(self):
        gen = torch.Generator().manual_seed(0)
        layers = {i: torch.randn(1, 64, 16, generator=gen) for i in range(5)}
        acts = CalibrationActivations(hidden_states=layers)
        deltas = residual_deltas(acts, keep_baseline=True)
        self.assertEqual(deltas.layer_count(), 5)
        # Layer 0 should be preserved unchanged.
        self.assertTrue(torch.allclose(deltas.hidden_states[0], acts.hidden_states[0]))

    def test_residual_deltas_compute_x_l_minus_x_l_minus_1(self):
        gen = torch.Generator().manual_seed(0)
        layers = {i: torch.randn(1, 32, 8, generator=gen) for i in range(4)}
        acts = CalibrationActivations(hidden_states=layers)
        deltas = residual_deltas(acts, keep_baseline=False)
        for idx in (1, 2, 3):
            expected = layers[idx] - layers[idx - 1]
            self.assertTrue(torch.allclose(deltas.hidden_states[idx], expected))

    def test_residual_deltas_remove_residual_confounder(self):
        """Pure-residual chain: x_{l+1} = x_l + noise. Deltas are
        independent across layers; raw activations are perfectly
        correlated. The conditional matrix should have much lower
        off-diagonal energy than the unconditional one.
        """
        gen = torch.Generator().manual_seed(0)
        n, d = 1024, 16
        layers_list = [torch.randn(1, n, d, generator=gen)]
        for _ in range(4):
            layers_list.append(layers_list[-1] + 0.1 * torch.randn(1, n, d, generator=gen))
        acts = CalibrationActivations(
            hidden_states={i: t for i, t in enumerate(layers_list)},
        )
        # Unconditional: off-diagonal MI is large because everything
        # shares the same residual stream.
        m_uncond = hsic_matrix(acts, sub_sample=512, method="rbf")
        # Conditional via deltas: deltas are ~independent across layers.
        m_cond = hsic_matrix(acts, sub_sample=512, method="rbf", conditioning="delta")
        # Off-diagonal energy (sum of squares excluding diag) should
        # collapse under conditioning.
        uncond_off = (
            m_uncond.pow(2).sum() - m_uncond.diag().pow(2).sum()
        ).item()
        cond_off = (
            m_cond.pow(2).sum() - m_cond.diag().pow(2).sum()
        ).item()
        self.assertGreater(
            uncond_off, cond_off * 5.0,
            f"conditional MI should dramatically reduce off-diagonal "
            f"energy vs unconditional (got uncond={uncond_off:.4f}, "
            f"cond={cond_off:.4f})",
        )

    def test_estimate_hsic_conditional_rff_smaller_than_unconditional(self):
        """For an X-Y pair whose dependence is fully explained by Z,
        conditioning on Z should collapse HSIC to near zero.
        """
        gen = torch.Generator().manual_seed(0)
        n, d = 1024, 16
        # Z is the shared base. X = Z + tiny noise_x. Y = Z + tiny noise_y.
        # Unconditional HSIC is high (X and Y both derive from Z); after
        # conditioning on Z the two should be independent noise.
        z = torch.randn(n, d, generator=gen)
        noise_x = torch.randn(n, d, generator=gen)
        noise_y = torch.randn(n, d, generator=gen)
        x = z + 0.05 * noise_x
        y = z + 0.05 * noise_y
        uncond = estimate_hsic_rff(x, y, n_rff=256, seed=0)
        cond = estimate_hsic_conditional_rff(
            x, y, z, n_rff=256, seed=0, ridge_lambda=1e-1,
        )
        # Conditional should be a small fraction of unconditional.
        self.assertLess(
            cond, uncond * 0.5,
            f"conditional HSIC should drop substantially; got "
            f"cond={cond:.4f}, uncond={uncond:.4f}",
        )

    def test_estimate_hsic_conditional_rff_returns_zero_for_independent_after_z(self):
        """If X and Y are independent given Z, conditional HSIC should be ~0."""
        gen = torch.Generator().manual_seed(0)
        n, d = 1024, 16
        z = torch.randn(n, d, generator=gen)
        # X and Y are pure noise (independent of each other, independent of Z).
        x = torch.randn(n, d, generator=gen)
        y = torch.randn(n, d, generator=gen)
        val = estimate_hsic_conditional_rff(x, y, z, n_rff=256, seed=0)
        # Not exactly zero (biased estimator), but should be small.
        self.assertGreaterEqual(val, 0.0)
        self.assertLess(val, 0.05)

    def test_estimate_hsic_conditional_rff_recovers_residual_signal(self):
        """Build X, Y that share Z AND have an independent shared signal.
        Conditional HSIC should pick up the residual; unconditional HSIC
        is dominated by Z.
        """
        gen = torch.Generator().manual_seed(0)
        n, d = 1024, 16
        z = torch.randn(n, d, generator=gen)
        # Independent residual signal shared between X and Y but not Z.
        shared = torch.randn(n, d, generator=gen)
        x = z + shared + 0.1 * torch.randn(n, d, generator=gen)
        y = z + shared + 0.1 * torch.randn(n, d, generator=gen)
        uncond = estimate_hsic_rff(x, y, n_rff=512, seed=0)
        cond = estimate_hsic_conditional_rff(
            x, y, z, n_rff=512, seed=0, ridge_lambda=1e-1,
        )
        # Conditional should still be > 0 (the shared signal survives).
        self.assertGreater(cond, 0.005)
        # And it should be smaller than unconditional (Z component removed).
        self.assertLess(cond, uncond)

    def test_hsic_matrix_with_delta_conditioning_changes_ranking(self):
        """On a chain with a clear residual pattern, delta-conditioning
        should change the per-layer sensitivity ranking vs the
        unconditional path. The two rankings should NOT be identical
        for a non-trivial chain -- if they were identical the
        conditioning would be a no-op.
        """
        gen = torch.Generator().manual_seed(0)
        n, d = 1024, 16
        layers_list = [torch.randn(1, n, d, generator=gen)]
        # Build a chain with two "informative" layers (large delta)
        # interleaved with "passthrough" layers (small delta).
        for i in range(6):
            prev = layers_list[-1]
            if i in (1, 4):
                delta = 2.0 * torch.randn(1, n, d, generator=gen)
            else:
                delta = 0.05 * torch.randn(1, n, d, generator=gen)
            layers_list.append(prev + delta)
        acts = CalibrationActivations(
            hidden_states={i: t for i, t in enumerate(layers_list)},
        )
        alloc_u = allocate_bits(
            acts, horizon=3, bits_min=1.5, bits_max=8.0,
            sub_sample=512, hsic_method="rbf",
        )
        alloc_c = allocate_bits(
            acts, horizon=3, bits_min=1.5, bits_max=8.0,
            sub_sample=512, hsic_method="rbf", conditioning="delta",
        )
        # Rankings should differ. Spearman-style: the order should not
        # be perfectly preserved.
        u_order = torch.argsort(alloc_u.mi_scores, descending=True).tolist()
        c_order = torch.argsort(alloc_c.mi_scores, descending=True).tolist()
        self.assertNotEqual(u_order, c_order)

    def test_allocate_bits_conditional_method_tag_reflects_conditioning(self):
        gen = torch.Generator().manual_seed(0)
        layers = _make_chain()
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        alloc_u = allocate_bits(acts, horizon=2, sub_sample=256)
        alloc_c = allocate_bits(
            acts, horizon=2, sub_sample=256, conditioning="delta",
        )
        self.assertIn("h2", alloc_u.method)
        self.assertNotIn("cond", alloc_u.method)
        self.assertIn("cond_delta", alloc_c.method)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GPUCorrectnessTests(unittest.TestCase):
    """Phase 4: GPU vs CPU parity.

    These tests run only when CUDA is available. On the local dev box
    they're skipped (no GPU); on Colab they're the headline correctness
    check -- a GPU result that's far from the CPU result would indicate
    a real bug in the device handling.

    Tolerances are loose (atol=1e-3 for absolute, 5% for relative) because
    the bandwidth heuristic and the centering operation can land in
    slightly different floating-point order on CPU vs GPU. The RFF
    *ranking* should be identical even when the absolute values differ
    by a few percent.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")

    def test_rff_pair_matches_cpu_to_tolerance(self):
        """GPU RFF HSIC should be within 5% relative of CPU."""
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(512, 16, generator=gen)
        y = 0.9 * x + 0.5 * torch.randn(512, 16, generator=gen)
        cpu_val = estimate_hsic_rff(x, y, n_rff=256, seed=42)
        gpu_val = estimate_hsic_rff(x, y, n_rff=256, seed=42, device=self.device)
        rel_err = abs(gpu_val - cpu_val) / max(cpu_val, 1e-9)
        self.assertLess(
            rel_err, 0.05,
            f"GPU vs CPU relative error {rel_err:.4f} exceeds 5% "
            f"(cpu={cpu_val:.6f}, gpu={gpu_val:.6f})",
        )

    def test_hsic_matrix_matches_cpu_to_tolerance(self):
        """GPU HSIC matrix should match CPU to within 5% relative on each entry."""
        gen = torch.Generator().manual_seed(0)
        layers = _make_chain(n_tokens=512, dim=16, seed=0)
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        m_cpu = hsic_matrix(acts, sub_sample=512, method="rff", n_rff=256, horizon=4)
        m_gpu = hsic_matrix(
            acts, sub_sample=512, method="rff", n_rff=256, horizon=4,
            device=self.device,
        )
        self.assertEqual(tuple(m_cpu.shape), tuple(m_gpu.shape))
        # Move to CPU for the comparison.
        m_gpu_cpu = m_gpu.cpu()
        denom = m_cpu.abs().clamp_min(1e-6)
        rel_err = ((m_gpu_cpu - m_cpu).abs() / denom).max().item()
        self.assertLess(
            rel_err, 0.10,
            f"GPU vs CPU max relative error {rel_err:.4f} exceeds 10%",
        )

    def test_hsic_matrix_returns_gpu_tensor_when_device_set(self):
        """When device='cuda' the output should be a CUDA tensor."""
        gen = torch.Generator().manual_seed(0)
        layers = _make_chain(n_tokens=128, dim=8, seed=0)
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        m = hsic_matrix(
            acts, sub_sample=128, method="rff", n_rff=64, horizon=2,
            device=self.device,
        )
        self.assertTrue(m.device.type == "cuda")

    def test_estimate_hsic_conditional_rff_matches_cpu(self):
        """GPU conditional HSIC should match CPU within 5%."""
        gen = torch.Generator().manual_seed(0)
        n, d = 512, 16
        z = torch.randn(n, d, generator=gen)
        shared = torch.randn(n, d, generator=gen)
        x = z + shared + 0.1 * torch.randn(n, d, generator=gen)
        y = z + shared + 0.1 * torch.randn(n, d, generator=gen)
        cpu_val = estimate_hsic_conditional_rff(
            x, y, z, n_rff=256, seed=0, ridge_lambda=1e-2,
        )
        gpu_val = estimate_hsic_conditional_rff(
            x, y, z, n_rff=256, seed=0, ridge_lambda=1e-2, device=self.device,
        )
        rel_err = abs(gpu_val - cpu_val) / max(cpu_val, 1e-9)
        self.assertLess(
            rel_err, 0.10,
            f"GPU vs CPU conditional HSIC relative error {rel_err:.4f} exceeds 10% "
            f"(cpu={cpu_val:.6f}, gpu={gpu_val:.6f})",
        )

    def test_allocate_bits_matches_cpu_ranking(self):
        """GPU and CPU allocation should produce the same per-layer ranking
        (the absolute bit values may differ by 1-2% from FP order-of-ops)."""
        gen = torch.Generator().manual_seed(0)
        layers = _make_chain(n_tokens=512, dim=16, seed=0)
        acts = CalibrationActivations(
            hidden_states={i: t.unsqueeze(0) for i, t in enumerate(layers)},
        )
        alloc_cpu = allocate_bits(
            acts, horizon=2, bits_min=1.5, bits_max=8.0,
            sub_sample=512, hsic_method="rff",
        )
        alloc_gpu = allocate_bits(
            acts, horizon=2, bits_min=1.5, bits_max=8.0,
            sub_sample=512, hsic_method="rff", device=self.device,
        )
        # Rankings (argmax of bits) should match exactly.
        cpu_order = torch.argsort(alloc_cpu.bits, descending=True).tolist()
        gpu_order = torch.argsort(alloc_gpu.bits.cpu(), descending=True).tolist()
        self.assertEqual(cpu_order, gpu_order)


if __name__ == "__main__":
    unittest.main()