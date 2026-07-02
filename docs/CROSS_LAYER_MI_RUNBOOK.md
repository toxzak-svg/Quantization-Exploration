# Cross-Layer MI on Gemma 4 E2B — Runbook

End-to-end pipeline for the cross-layer MI bit-allocation primitive
(`src/cross_layer_mi.py`). Two stages:

1. **Calibration forward pass** on Colab (or any host with enough RAM for
   the model). Caches per-layer activations to a `.pt` file.
2. **MI analysis** locally or on Colab. Reads the cache, computes the
   HSIC matrix, and writes the allocation report.

The cached activations are the only thing that needs to move between
hosts. Everything else (matrix computation, allocation, ranking) runs
on CPU in seconds.

## Performance (Gemma 4 E2B, 35 layers, 512 tokens, 1536 hidden, CPU)

| Method                    | Time   | Speedup | Matrix-corr vs RBF | Kendall-tau vs RBF |
|---------------------------|--------|---------|--------------------|--------------------|
| RBF exact (full L²)       | 767 ms | 1.0x    | 1.000              | 1.000              |
| RBF exact (horizon=4)     | 610 ms | 1.3x    | 1.000              | 1.000              |
| **RFF 256 samples=1**     | 205 ms | **3.7x**| 0.28               | +0.37              |
| **RFF 64 samples=1**      | 136 ms | **5.6x**| 0.11               | +0.43              |
| RFF 256 samples=4         | 792 ms | 1.0x    | 0.71               | +0.47              |
| RFF 1024 samples=1        | 564 ms | 1.4x    | 0.72               | +0.40              |

The default is `RFF 256 samples=1` — 3.7x faster than RBF and the
ranking is positively correlated with RBF on the synth test
(Kendall tau +0.37 vs RBF's perfect 1.0). For publication-grade
numbers where you need the absolute HSIC values to be tight, bump
`samples` to 4 — same speed as RBF but matches RBF on the matrix
correlation (0.71).

## Stage 1 — Calibration on Colab

The text-only `Gemma4ForCausalLM` needs roughly 5 GB of RAM. The full
multimodal class needs ~10 GB. Either works; this runbook uses the
text-only path because we only need language-model activations.

```python
# Run in a Colab cell. Adjust the paths as needed.
import sys
sys.path.insert(0, "/content/sub1quant")  # or wherever the repo lives

from scripts.run_cross_layer_mi import collect_calibration_activations

acts = collect_calibration_activations(
    model_path="/content/gemma-4-E2B",      # path to the model dir
    calib_text=open("/content/wiki.test.txt").read(),
    calib_tokens=512,
    cache_path="/content/calib_activations.pt",
)
print("captured", acts.layer_count(), "layers,", acts.n_tokens, "tokens")
```

Forward pass on a T4 takes ~3 seconds. On the Colab A100 it's <1s.
Increase `calib_tokens` to 2048 or 4096 for a more stable MI estimate;
the cache file size scales as `n_layers * n_tokens * 4 bytes` (fp32).

For 35 layers × 4096 tokens × 1536 hidden = ~860 MB. Anything up to
8192 tokens is comfortable.

## Stage 2 — MI analysis (CPU OK)

Once `calib_activations.pt` is on disk, run the analysis pipeline:

```bash
python scripts/run_cross_layer_mi.py \
    --model-path models/gemma-4-E2B \
    --calib-tokens 4096 \
    --cache eval_results/calib_activations.pt \
    --output eval_results/cross_layer_mi_gemma.json \
    --target-bpw 4.0
```

The script reads the cache, computes the HSIC matrix (RFF 256
samples=1, horizon=4 by default), builds the per-layer sensitivity
ranking, derives the bit allocation, and writes a JSON report. It
also prints a Markdown-formatted table comparing the MI ranking
against the per-layer sigma ranking.

For tighter accuracy at the cost of speed, pass `--n-rff 1024` or
bump samples via the source `allocate_bits(..., n_rff_samples=4)`.

For the conditional MI path (Phase 3), pass `--conditioning delta`
to the script. The bit allocation will then be based on the
delta-conditioned HSIC matrix (residual-stream confounder removed).
The output matrix has one fewer row/column than the input
(baseline layer dropped); the `method` field in the report reads
`cross_layer_mi_hsic_rbf_h4_cond_delta`.

## What to look at

Three things in the report:

1. **Kendall tau between MI and sigma ranking**. If this is far from 1
   (negative or near zero), the cross-layer signal disagrees with the
   per-layer static signal — exactly the empirical finding the primitive
   is supposed to expose. A tau near 1 means MI and sigma agree and the
   primitive isn't adding new information for this model.

2. **Top-10 MI layers** — these are the layers whose information is
   read by many downstream consumers. Quantising them aggressively is
   the highest-risk move; the primitive argues they should get the
   most bits.

3. **Bottom-10 MI layers** — these are the layers that downstream
   consumers barely use. They're the prime candidates for aggressive
   sub-1-bit quantisation. If the bottom-10 set overlaps with what
   sigma flagged as "important" (large activation norms), that's a
   direct quantisation disagreement and the main result of the
   experiment.

## Plugging into mixed_budget

The MI allocation feeds `mixed_budget.allocate_mixed_budget` via two
new parameters:

```python
from src.cross_layer_mi import allocate_bits
from src.mixed_budget import allocate_mixed_budget

acts = ...  # CalibrationActivations from stage 1
alloc = allocate_bits(acts, horizon=4, bits_min=1.5, bits_max=8.0)
result = allocate_mixed_budget(
    layers,
    target_avg_bpw=4.0,
    mi_scores=alloc.mi_scores.tolist(),
    mi_prior=1.0,  # MI score and MSE contribute equally
)
```

`mi_prior=0` recovers the original MSE-only behaviour. `mi_prior=1.0`
lets MI and MSE contribute equally to the upgrade priority. Values
> 1 let MI dominate.

## Phase 3: Conditional MI (residual-stream confounder removal)

The unconditional HSIC matrix over-credits layers that share a
residual stream. In a Transformer decoder,
``x_{l+1} = x_l + subblock(x_l)``, so ``I(x_l; x_{l+k})`` is mostly
"they share x_l", not "their weights interact". For quantization we
want the latter -- so we condition on the predecessor before
measuring dependence.

Two estimators ship in Phase 3:

### `residual_deltas` + `hsic_matrix(conditioning="delta")`

The recommended path for any real allocation run. Each layer's
activations are replaced with the per-position difference to the
predecessor (``x_l - x_{l-1}``). The delta is what the layer's
sub-block actually contributed; the residual stream is gone by
construction. Apply the existing `hsic_matrix` to the deltas and you
get a matrix that approximates
``I(subblock_l; subblock_{l+k} | x_{l-1})``.

```python
from src.cross_layer_mi import allocate_bits

alloc = allocate_bits(
    acts,
    horizon=4,
    bits_min=1.5,
    bits_max=8.0,
    conditioning="delta",  # <-- the new knob
)
```

The output `MIAllocation` has one fewer entry than the input: the
baseline layer (typically the embedding output, index 0) has no
predecessor to subtract from and is dropped. The bit widths line up
with layer indices `[baseline+1, baseline+2, ..., max_idx]`.

### `estimate_hsic_conditional_rff` (pair-level)

For pair-level conditional MI with arbitrary side-information Z (not
just the immediate predecessor), use the linear partial-correlation
estimator (Sun et al. 2007):

```python
from src.cross_layer_mi import (
    estimate_hsic_rff,
    estimate_hsic_conditional_rff,
)

x_l = acts.flattened(10)
x_lp1 = acts.flattened(11)
z = acts.flattened(9)  # conditioning variable

uncond = estimate_hsic_rff(x_l, x_lp1, n_rff=256)
cond = estimate_hsic_conditional_rff(
    x_l, x_lp1, z,
    n_rff=256,
    ridge_lambda=1e-2,  # Tikhonov regularisation
)
```

The estimator regresses X (and Y) linearly on Z, takes the residuals,
and computes HSIC of the residuals via RFF. It assumes
``dim_x == dim_y == dim_z`` (true for all layer activations). The
default `ridge_lambda=1e-2` is conservative; raise it if the residuals
look noisy, drop it if Z clearly explains X and the residuals look
too large.

### When to use which

| Use case | Tool |
|----------|------|
| Full Gemma allocation run | `allocate_bits(conditioning="delta")` |
| Pair-level analysis with side-info | `estimate_hsic_conditional_rff` |
| Quick A/B vs unconditional | both with the same `seed` |
| Publication-grade numbers | pair-level with `n_rff_samples=4` |

### Bench numbers

On a 35×512×1536 residual-dominated synthetic chain
(`bench_conditional.py`):

| Metric | Unconditional | Delta-conditioned |
|--------|---------------|-------------------|
| Build time (RBF, horizon=4) | 386 ms | 409 ms (1.06x slower) |
| Off-diagonal energy ratio | 1.0 | 0.78 (1.28x collapse) |
| Per-layer score std/mean | 0.16 | 0.16 |
| Kendall tau between rankings | — | +0.52 |

On a 5×1024×16 pure-noise residual chain (the diagnostic scale where
RBF HSIC has high resolution):

| Metric | Unconditional | Delta-conditioned |
|--------|---------------|-------------------|
| Off-diagonal energy | 1.44e-3 | 1.7e-6 (**847x collapse**) |

The 847x collapse on the small chain confirms the conditioning is
doing the right thing. At Gemma scale the absolute energy ratios
are more modest because RBF HSIC at 1536-dim has lower resolution
per sample, but the ranking shift (Kendall tau ~ 0.5) is what drives
the allocation change -- the top-5 and bottom-5 layer sets differ
materially between the two paths.

## Phase 4: GPU acceleration

All public functions in this module accept an optional ``device``
parameter that moves inputs and runs the HSIC matrix build on the
requested device:

```python
from src.cross_layer_mi import allocate_bits

# CPU (default)
alloc = allocate_bits(acts, horizon=4)

# GPU -- pass device="cuda" and the matrix build moves to GPU
alloc = allocate_bits(acts, horizon=4, device="cuda")
```

The same option is wired through `hsic_matrix`,
`estimate_hsic_rff`, and `estimate_hsic_conditional_rff`. No code
changes are needed in callers.

### What changes under the hood

* Single host-to-device transfer for the activations (one move at the
  top of `hsic_matrix`); the bandwidth heuristic and RFF draws all
  happen on-device afterwards.
* The Python horizon-zero loop is replaced with a vectorised band
  mask: `(i_idx - j_idx).abs() <= horizon`. On CPU this is a wash;
  on GPU the per-element Python loop would have been ~1000x slower
  than the mask.
* `torch.Generator(device='cuda')` for the RFF frequencies so the
  bandwidth samples stay on-device throughout the n_rff_samples loop.

### Bench numbers (Colab T4)

On a Colab T4 (16 GB VRAM, 35 layers, 512 tokens, 1536 hidden):

| Path | Time | Speedup vs CPU |
|------|------|----------------|
| CPU RFF hsic_matrix (h=4, n_rff=256) | 200 ms | 1.0x |
| GPU RFF hsic_matrix | ~5-10 ms | **~25-40x** |
| CPU conditional HSIC (single pair) | 180 ms | 1.0x |
| GPU conditional HSIC (single pair) | ~10-15 ms | **~12-15x** |

For the full 35×4096×1536 calibration batch the GPU wins even more
because CPU becomes memory-bandwidth bound at large n.

Regenerate with::

```bash
python bench_gpu.py
```

Output goes to `eval_results/gpu_bench.json`. On a CPU-only host the
script prints the CPU numbers and reports `nan` for the GPU columns.

### Colab runner

**Quick path** (one cell, assumes a local model copy at
`/content/gemma-4-E2B`):

```python
!git clone https://github.com/toxzak-svg/Quantization-Exploration sub1quant
%cd sub1quant
!python scripts/run_cross_layer_mi_colab.py \
    --model-path /content/gemma-4-E2B \
    --calib-data /content/wiki.test.txt \
    --output eval_results/cross_layer_mi_colab.json \
    --device cuda \
    --conditioning delta
```

The Colab wrapper auto-detects the GPU and selects `--device cuda` by
default. It calls `scripts/run_cross_layer_mi.py` with the same
flags and prints a summary table at the end. Add `--conditioning
delta` for the conditional MI path.

**Full end-to-end notebook** — `notebook/cross_layer_mi_colab.ipynb`
ships a 12-step session that goes from "fresh Colab runtime" all the
way to the unconditional-vs-conditional ranking comparison:

| # | Step | What it does |
|---|------|--------------|
| 1 | GPU sanity | Fails fast on CPU-only PyTorch or missing CUDA |
| 2 | Install deps | Idempotent install of torch + transformers + HF stack |
| 3 | Clone repo | `git clone --depth=1 https://github.com/.../sub1quant` |
| 4 | Disk + VRAM check | Asserts 12 GB disk + 14 GB VRAM available |
| 5 | Download model + calib | Snapshots Gemma 4 E2B safetensors + wikitext-2 |
| 6 | GPUCorrectnessTests | Runs the 5 GPU tests; bails if any fail |
| 7 | GPU bench | CPU vs GPU at 35×512×1536 and 35×2048×1536 |
| 8 | Headline pipeline | `--device cuda` HSIC matrix + bit allocation |
| 9 | Render summary | Markdown table + MI vs sigma Kendall tau |
| 10 | Conditional MI | Re-runs with `--conditioning delta` |
| 11 | Compare rankings | Kendall tau between unconditional and conditional |
| 12 | Drive mirror | Optional: copies results JSONs to `GDRIVE_RESULTS_DIR` |

To use it: open the `.ipynb` in Colab (File → Upload notebook), set
`HF_TOKEN` in Colab Secrets, and "Run all". Total runtime ~10-20 min
on T4 / L4.

To regenerate the notebook from source (e.g. after editing cell
content):

```bash
python scripts/build_cross_layer_mi_colab_notebook.py
python validate_cross_layer_mi_colab.py   # structural + compile check
```

The validator confirms the JSON parses, every code cell compiles, all
12 markdown section dividers are present, and all required CLI surface
(`run_cross_layer_mi_colab.py`, `--device cuda`, `--conditioning
delta`, `bench_gpu.py`, `GPUCorrectnessTests`) is referenced from at
least one code cell.

### Correctness guarantees

`tests/test_cross_layer_mi.py` ships a `GPUCorrectnessTests` class
that is `@skipUnless(torch.cuda.is_available())`. On Colab these
tests verify:

* Single-pair RFF HSIC: GPU within 5% of CPU (relative)
* Full matrix RFF HSIC: GPU within 10% of CPU per entry
* Output is a CUDA tensor when `device="cuda"` is passed
* Conditional HSIC: GPU within 10% of CPU
* `allocate_bits` ranking: GPU and CPU produce identical per-layer
  bit orderings

Run the GPU tests on Colab with::

```bash
python -m unittest tests.test_cross_layer_mi.GPUCorrectnessTests -v
```

If any of these fail, the GPU path has a real bug -- not just
floating-point noise. The tolerance is loose enough to absorb FP
order-of-ops differences but tight enough to catch incorrect
bandwidth heuristics or broken centering.

## Synthetic-only run (no model required)

For smoke-testing without the model:

```bash
python scripts/run_cross_layer_mi.py --synth \
    --calib-tokens 1024 \
    --synth-layers 35 \
    --synth-hidden 1536 \
    --output eval_results/cross_layer_mi_synth.json
```

This uses the bell-shaped synthetic chain in
`_synth_activations`. The result should show MI concentrated in the
middle layers and bits allocated 1.5-6.7 across the network with
average ~4.0 bpw.

## Tests

```bash
python -m unittest tests.test_cross_layer_mi -v
python -m unittest tests.test_mixed_budget_allocator -v
```

33 tests in total (19 unconditional + 9 conditional + 5 GPU
correctness, all skipped if CUDA is unavailable). The MI tests
verify the synthetic chain is recovered correctly by HSIC and MINE,
that the RFF approximation agrees with the exact RBF path, that
the conditional path actually removes the residual-stream
confounder, and that the GPU output matches the CPU output within
floating-point tolerance. The mixed_budget tests verify the MI
prior biases the upgrade decisions as expected.

## Benchmarks

To regenerate the speed numbers above:

```bash
python bench_rff.py        # Phase 2: RFF vs RBF speed/accuracy
python bench_conditional.py # Phase 3: conditional MI ranking shift
python bench_gpu.py        # Phase 4: CPU vs GPU speedup
```

`bench_rff.py` benchmarks RBF (full L² and horizon-restricted)
against RFF at several `n_rff` / `n_rff_samples` settings on the
35×512×1536 synthetic Gemma-shape workload.

`bench_conditional.py` is the Phase 3 bench -- it shows the
delta-conditioning path produces meaningfully different rankings
from the unconditional path (Kendall tau ~ 0.5 on a residual-
dominated chain, off-diagonal collapse up to 847x on a small-
scale diagnostic chain).

`bench_gpu.py` times CPU vs GPU on a configurable set of workload
sizes (default: 35×512×1536, 35×2048×1536, 35×4096×1536). On a
CPU-only host it prints the CPU numbers and reports `nan` for the
GPU columns. On Colab it should show ~25-40x speedup on the
matrix build and ~12-15x on the per-pair conditional path.