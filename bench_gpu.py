"""Benchmark CPU vs GPU HSIC matrix computation.

Phase 4: shows the GPU speedup on Gemma-shape workloads. Designed to
run on Colab (which has a T4 by default) but skips the GPU section
gracefully on CPU-only hosts.

Headline expectation on a Colab T4:

* 35-layer, 512-token, 1536-hidden RFF HSIC matrix build: ~5-10 ms
  on GPU vs ~200-400 ms on CPU. ~30-50x speedup.
* For the full 35x4096x1536 calibration batch the GPU wins even
  more (memory bandwidth-bound on CPU).

Usage::

    python bench_gpu.py [--sizes "35,512,1536;35,4096,1536"]

Each size is ``L,n,hidden`` (semicolon-separated). The script times
CPU and GPU paths, reports speedup, and saves results to
``eval_results/gpu_bench.json`` for inclusion in the runbook.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from src.cross_layer_mi import (
    CalibrationActivations,
    hsic_matrix,
    allocate_bits,
    estimate_hsic_rff,
    estimate_hsic_conditional_rff,
)


def _make_acts(n_layers: int, n_tokens: int, hidden: int, seed: int = 0) -> CalibrationActivations:
    """Build a Gemma-shape CalibrationActivations."""
    gen = torch.Generator().manual_seed(seed)
    acts: dict[int, torch.Tensor] = {}
    x = torch.randn(1, n_tokens, hidden, generator=gen)
    acts[0] = x.clone()
    for l in range(1, n_layers):
        x = x + 0.05 * torch.randn(1, n_tokens, hidden, generator=gen)
        acts[l] = x.clone()
    return CalibrationActivations(
        hidden_states=acts,
        layer_keys=[f"l{i}" for i in range(n_layers)],
        n_tokens=n_tokens,
    )


def _time_cpu(fn) -> tuple[float, object]:
    out = None
    t0 = time.time()
    out = fn()
    return time.time() - t0, out


def _time_gpu(fn) -> tuple[float, object]:
    """Time a GPU function. Sync before and after for accurate timing."""
    if not torch.cuda.is_available():
        return float("nan"), None
    torch.cuda.synchronize()
    out = None
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return time.time() - t0, out


def _bench_size(n_layers: int, n_tokens: int, hidden: int, sub_sample: int | None) -> dict:
    """Benchmark one workload size on CPU and (if available) GPU."""
    print(f"\n=== L={n_layers}, n={n_tokens}, hidden={hidden}, sub_sample={sub_sample} ===")
    acts = _make_acts(n_layers, n_tokens, hidden)

    # CPU baseline.
    t_cpu, m_cpu = _time_cpu(
        lambda: hsic_matrix(
            acts, sub_sample=sub_sample, method="rff", n_rff=256, horizon=4,
        )
    )
    print(f"  CPU RFF hsic_matrix (h=4, n_rff=256): {t_cpu * 1000:7.1f} ms")

    # GPU path (skipped if no CUDA).
    t_gpu = float("nan")
    m_gpu = None
    rel_err = float("nan")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        t_gpu, m_gpu = _time_gpu(
            lambda: hsic_matrix(
                acts, sub_sample=sub_sample, method="rff", n_rff=256, horizon=4,
                device="cuda",
            )
        )
        # Compare to CPU. Move to CPU first.
        m_gpu_cpu = m_gpu.cpu()
        denom = m_cpu.abs().clamp_min(1e-6)
        rel_err = float(((m_gpu_cpu - m_cpu).abs() / denom).max().item())
        speedup = t_cpu / t_gpu
        print(
            f"  GPU RFF hsic_matrix:                  {t_gpu * 1000:7.1f} ms"
            f"   ({speedup:.1f}x speedup, max rel err {rel_err:.4f})"
        )
    else:
        print("  GPU RFF hsic_matrix:                  (skipped, no CUDA)")

    # Pair-level conditional HSIC for a representative pair.
    if sub_sample is not None:
        x = acts.flattened(10)[:sub_sample]
        y = acts.flattened(11)[:sub_sample]
        z = acts.flattened(9)[:sub_sample]
    else:
        x, y, z = acts.flattened(10), acts.flattened(11), acts.flattened(9)
    t_cpu_pair, _ = _time_cpu(
        lambda: estimate_hsic_conditional_rff(
            x, y, z, n_rff=256, seed=0, ridge_lambda=1e-2,
        )
    )
    print(f"  CPU conditional HSIC (pair 10,11 | 9): {t_cpu_pair * 1000:7.1f} ms")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        t_gpu_pair, _ = _time_gpu(
            lambda: estimate_hsic_conditional_rff(
                x, y, z, n_rff=256, seed=0, ridge_lambda=1e-2, device="cuda",
            )
        )
        speedup_pair = t_cpu_pair / t_gpu_pair
        print(
            f"  GPU conditional HSIC:                  {t_gpu_pair * 1000:7.1f} ms"
            f"   ({speedup_pair:.1f}x speedup)"
        )
        return {
            "n_layers": n_layers,
            "n_tokens": n_tokens,
            "hidden": hidden,
            "sub_sample": sub_sample,
            "cpu_ms": t_cpu * 1000,
            "gpu_ms": t_gpu * 1000,
            "speedup": t_cpu / t_gpu,
            "max_rel_err_vs_cpu": rel_err,
            "cpu_pair_ms": t_cpu_pair * 1000,
            "gpu_pair_ms": t_gpu_pair * 1000,
            "speedup_pair": t_cpu_pair / t_gpu_pair,
        }
    return {
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "hidden": hidden,
        "sub_sample": sub_sample,
        "cpu_ms": t_cpu * 1000,
        "gpu_ms": float("nan"),
        "speedup": float("nan"),
        "max_rel_err_vs_cpu": float("nan"),
        "cpu_pair_ms": t_cpu_pair * 1000,
        "gpu_pair_ms": float("nan"),
        "speedup_pair": float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="35,512,1536;35,2048,1536;35,4096,1536",
        help="semicolon-separated L,n,hidden triples",
    )
    parser.add_argument("--sub-sample", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/gpu_bench.json"),
    )
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results: list[dict] = []
    for size_str in args.sizes.split(";"):
        parts = [int(x.strip()) for x in size_str.split(",")]
        if len(parts) != 3:
            print(f"skipping malformed size: {size_str}")
            continue
        n_layers, n_tokens, hidden = parts
        results.append(_bench_size(n_layers, n_tokens, hidden, args.sub_sample))

    print("\n=== Summary ===")
    print(f"{'size':>22s}  {'CPU ms':>10s}  {'GPU ms':>10s}  {'speedup':>10s}")
    for r in results:
        size = f"{r['n_layers']}x{r['n_tokens']}x{r['hidden']}"
        print(
            f"  {size:>20s}  {r['cpu_ms']:10.1f}  "
            f"{r['gpu_ms']:10.1f}  {r['speedup']:10.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": results,
    }, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()