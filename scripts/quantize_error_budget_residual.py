"""Build an INT2 + sparse error-budget residual checkpoint from local Gemma safetensors."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantize_groupwise_int4 import iter_model_weights
from src.error_budget_residual import (
    dequantize_error_budget_residual,
    quantize_error_budget_residual,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Gemma weights with INT2 base plus sparse residual corrections")
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--output", default="quantized/gemma_int2_error_budget_g128_k16.pt")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--outliers-per-group", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--skip-reconstruction-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)

    quantized: dict[int, dict] = {}
    layer_stats: list[dict] = []
    total_params = 0
    total_bits = 0.0
    total_weighted_mse = 0.0
    total_weighted_abs_error = 0.0

    print("=" * 72)
    print("INT2 + ERROR-BUDGET RESIDUAL QUANTIZATION")
    print("=" * 72)
    print("Model dir:", model_dir)
    print("Output:", output)
    print("Group size:", args.group_size)
    print("Outliers per group:", args.outliers_per_group)
    if args.max_layers is not None:
        print("Max layers:", args.max_layers)
    print()

    for idx, key, weight in iter_model_weights(model_dir, max_layers=args.max_layers):
        entry = quantize_error_budget_residual(
            weight,
            group_size=args.group_size,
            outliers_per_group=args.outliers_per_group,
        )
        entry["key"] = key
        quantized[idx] = entry

        params = weight.numel()
        total_params += params
        total_bits += entry["bpw"] * params

        mse = None
        rmse = None
        mean_abs_error = None
        if not args.skip_reconstruction_metrics:
            restored = dequantize_error_budget_residual(entry)
            error = restored - weight
            mse = error.pow(2).mean().item()
            rmse = mse**0.5
            mean_abs_error = error.abs().mean().item()
            total_weighted_mse += mse * params
            total_weighted_abs_error += mean_abs_error * params
            del restored, error

        stat = {
            "idx": idx,
            "key": key,
            "shape": list(weight.shape),
            "params": params,
            "bpw": entry["bpw"],
            "mse": mse,
            "rmse": rmse,
            "mean_abs_error": mean_abs_error,
        }
        layer_stats.append(stat)

        if idx < 5 or idx % 50 == 0:
            metric = "" if mse is None else f", mse={mse:.6f}, rmse={rmse:.6f}"
            print(f"Layer {idx:3d}: bpw={entry['bpw']:.4f}, shape={list(weight.shape)}{metric}", flush=True)

        del weight
        gc.collect()

    if not quantized:
        raise RuntimeError(f"No quantizable language_model 2D weights found in {model_dir}")

    avg_bpw = total_bits / total_params
    stats = {
        "method": "int2_error_budget_residual",
        "format_version": 1,
        "group_size": args.group_size,
        "outliers_per_group": args.outliers_per_group,
        "layers": len(quantized),
        "total_params": total_params,
        "avg_bpw": avg_bpw,
        "compression_vs_bf16": 16.0 / avg_bpw,
        "weighted_mse": None if args.skip_reconstruction_metrics else total_weighted_mse / total_params,
        "weighted_rmse": None if args.skip_reconstruction_metrics else (total_weighted_mse / total_params) ** 0.5,
        "weighted_mean_abs_error": None if args.skip_reconstruction_metrics else total_weighted_abs_error / total_params,
        "layer_stats": layer_stats,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "quantized": quantized,
            "stats": stats,
            "method": "int2_error_budget_residual",
            "config": {
                "model_dir": str(model_dir),
                "group_size": args.group_size,
                "outliers_per_group": args.outliers_per_group,
                "max_layers": args.max_layers,
                "source_dtype": "BF16/F16/F32 safetensors",
            },
            "weight_keys": [
                {"key": item["key"], "shape": item["shape"]}
                for item in layer_stats
            ],
        },
        output,
    )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"Layers: {stats['layers']}")
    print(f"Average BPW: {stats['avg_bpw']:.4f}")
    print(f"Compression vs BF16: {stats['compression_vs_bf16']:.2f}x")
    if stats["weighted_mse"] is not None:
        print(f"Weighted MSE: {stats['weighted_mse']:.6f}")
        print(f"Weighted RMSE: {stats['weighted_rmse']:.6f}")
        print(f"Weighted mean abs error: {stats['weighted_mean_abs_error']:.6f}")
    print("Saved:", output)


if __name__ == "__main__":
    main()
