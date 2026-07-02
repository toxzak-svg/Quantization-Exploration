"""Cross-layer MI on Colab (GPU).

Single-file pipeline for running the cross-layer MI allocation on a
Colab GPU. Designed to be pasted into a single Colab cell or run as
``python scripts/run_cross_layer_mi_colab.py`` from a Colab session
that has the repo cloned.

Steps:

1. Detect environment (GPU type, RAM).
2. Load Gemma 4 E2B (text-only) -- ~5 GB RAM, ~1-2 GB VRAM.
3. Capture per-layer activations on a single calibration forward pass.
4. Build the HSIC matrix on GPU (RFF 256 samples=1, horizon=4, optional
   ``conditioning='delta'``).
5. Allocate bits at the configured target BPW.
6. Save the report (JSON + Markdown table) to ``eval_results/``.

Usage from a Colab cell::

    !git clone <repo_url> sub1quant
    %cd sub1quant
    !python scripts/run_cross_layer_mi_colab.py \\
        --model-path /content/gemma-4-E2B \\
        --calib-data /content/wiki.test.txt \\
        --output eval_results/cross_layer_mi_colab.json \\
        --device cuda \\
        [--conditioning delta]

The script is a thin wrapper around ``run_cross_layer_mi.py`` that adds
the GPU device selection and a few Colab-specific paths. All the real
work happens in the underlying module.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _detect_gpu() -> str:
    """Return a human-readable GPU description. Empty string if no CUDA."""
    try:
        import torch
    except ImportError:
        return "(torch not installed)"
    if not torch.cuda.is_available():
        return ""
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} ({mem_gb:.0f} GB)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="path or HF id for Gemma 4 E2B")
    parser.add_argument("--calib-data", default="/content/wiki.test.txt")
    parser.add_argument("--calib-tokens", type=int, default=2048)
    parser.add_argument("--cache", type=Path, default=Path("/content/calib_activations.pt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/cross_layer_mi_colab.json"),
    )
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--target-bpw", type=float, default=4.0)
    parser.add_argument("--bits-min", type=float, default=1.5)
    parser.add_argument("--bits-max", type=float, default=8.0)
    parser.add_argument("--sub-sample", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditioning", choices=[None, "delta"], default=None)
    parser.add_argument("--device", default="cuda" if _detect_gpu() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-layer MI on Colab (Phase 4 GPU)")
    print("=" * 60)
    gpu_desc = _detect_gpu()
    if gpu_desc:
        print(f"GPU: {gpu_desc}")
    else:
        print("GPU: (none -- running on CPU)")
    print(f"Device arg: {args.device}")
    print(f"Model: {args.model_path}")
    print(f"Calibration: {args.calib_data} ({args.calib_tokens} tokens)")
    print(f"Conditioning: {args.conditioning}")
    print(f"Output: {args.output}")

    # Delegate to the main script. We invoke it as a subprocess so the
    # Colab environment setup (model load, caching) is identical to the
    # CPU path.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_cross_layer_mi.py"),
        "--model-path", args.model_path,
        "--calib-data", args.calib_data,
        "--calib-tokens", str(args.calib_tokens),
        "--cache", str(args.cache),
        "--output", str(args.output),
        "--horizon", str(args.horizon),
        "--target-bpw", str(args.target_bpw),
        "--bits-min", str(args.bits_min),
        "--bits-max", str(args.bits_max),
        "--sub-sample", str(args.sub_sample),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.conditioning:
        cmd.extend(["--conditioning", args.conditioning])

    print()
    print(" ".join(cmd))
    print()
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        print(f"ERROR: underlying script exited with code {rc}")
        sys.exit(rc)

    # Read the report and print a one-line summary.
    if args.output.exists():
        report = json.loads(args.output.read_text())
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  layers: {report['n_layers']}")
        print(f"  avg bits: {report['avg_bits']:.2f} (target {args.target_bpw})")
        print(f"  MI vs sigma Kendall tau: {report['kendall_tau_vs_sigma']:+.4f}")
        print(f"  method: {report['method']}")


if __name__ == "__main__":
    main()