"""Compare sub-4-BPW residual candidates against groupwise INT4 reconstruction."""

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
    dequantize_binary_residual,
    dequantize_error_budget_residual,
    quantize_binary_residual,
    quantize_error_budget_residual,
)
from src.groupwise_int4 import dequantize_groupwise_int4, quantize_groupwise_int4


def mse(original: torch.Tensor, restored: torch.Tensor) -> float:
    return (original - restored).pow(2).mean().item()


def weighted_result(name: str, total_sse: float, total_params: int, total_bits: float) -> dict:
    weighted_mse = total_sse / total_params
    return {
        "method": name,
        "avg_bpw": total_bits / total_params,
        "weighted_mse": weighted_mse,
        "weighted_rmse": weighted_mse**0.5,
        "compression_vs_bf16": 16.0 / (total_bits / total_params),
    }


def scan(model_dir: Path, group_size: int, outliers_per_group: int, max_layers: int | None) -> dict:
    totals = {
        "groupwise_int4": {"sse": 0.0, "bits": 0.0},
        "int2_error_budget_residual": {"sse": 0.0, "bits": 0.0},
        "int2_binary_residual": {"sse": 0.0, "bits": 0.0},
        "int2_base_from_binary_residual": {"sse": 0.0, "bits": 0.0},
    }
    total_params = 0
    layers = []

    for idx, key, weight in iter_model_weights(model_dir, max_layers=max_layers):
        params = weight.numel()
        total_params += params

        int4_entry = quantize_groupwise_int4(weight, group_size=group_size)
        int4 = dequantize_groupwise_int4(int4_entry)
        int4_sse = mse(weight, int4) * params
        totals["groupwise_int4"]["sse"] += int4_sse
        totals["groupwise_int4"]["bits"] += int4_entry["bpw"] * params

        residual_entry = quantize_binary_residual(weight, group_size=group_size)
        residual = dequantize_binary_residual(residual_entry)
        base_only = dequantize_binary_residual(residual_entry, include_residual=False)
        residual_sse = mse(weight, residual) * params
        base_sse = mse(weight, base_only) * params
        totals["int2_binary_residual"]["sse"] += residual_sse
        totals["int2_binary_residual"]["bits"] += residual_entry["bpw"] * params

        error_budget_entry = quantize_error_budget_residual(
            weight,
            group_size=group_size,
            outliers_per_group=outliers_per_group,
        )
        error_budget = dequantize_error_budget_residual(error_budget_entry)
        error_budget_sse = mse(weight, error_budget) * params
        totals["int2_error_budget_residual"]["sse"] += error_budget_sse
        totals["int2_error_budget_residual"]["bits"] += error_budget_entry["bpw"] * params

        # Base-only accounting removes the residual sign bit and residual scale.
        base_bpw = residual_entry["bpw"] - 1.0 - (16.0 / group_size)
        totals["int2_base_from_binary_residual"]["sse"] += base_sse
        totals["int2_base_from_binary_residual"]["bits"] += base_bpw * params

        if idx < 5 or idx % 25 == 0:
            layer = {
                "idx": idx,
                "key": key,
                "shape": list(weight.shape),
                "int4_mse": int4_sse / params,
                "int2_base_mse": base_sse / params,
                "binary_residual_mse": residual_sse / params,
                "error_budget_residual_mse": error_budget_sse / params,
                "error_budget_residual_bpw": error_budget_entry["bpw"],
                "binary_residual_bpw": residual_entry["bpw"],
            }
            layers.append(layer)
            print(json.dumps(layer), flush=True)

        del weight, int4, residual, base_only, error_budget, int4_entry, residual_entry, error_budget_entry
        gc.collect()

    results = [
        weighted_result(name, item["sse"], total_params, item["bits"])
        for name, item in totals.items()
    ]
    return {
        "group_size": group_size,
        "outliers_per_group": outliers_per_group,
        "max_layers": max_layers,
        "total_params": total_params,
        "results": sorted(results, key=lambda item: item["avg_bpw"]),
        "sample_layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--outliers-per-group", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--output", default="eval_results/error_budget_residual_scan.json")
    args = parser.parse_args()

    result = scan(Path(args.model_dir), args.group_size, args.outliers_per_group, args.max_layers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(result["results"], indent=2), flush=True)
    print("WROTE", output, flush=True)


if __name__ == "__main__":
    main()
