"""Search mixed per-layer quantization budgets against a uniform INT4 baseline."""

from __future__ import annotations

import argparse
import gc
import json
import os
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
from src.mixed_budget import allocate_mixed_budget, summarize_allocation


def mse(original: torch.Tensor, restored: torch.Tensor) -> float:
    return (original - restored).pow(2).mean().item()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


ACT_WEIGHT_FLOOR = 0.05
ACT_WEIGHT_CEIL = 20.0


def load_activation_weights(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(key): max(ACT_WEIGHT_FLOOR, min(ACT_WEIGHT_CEIL, float(value)))
        for key, value in data.items()
    }


def candidate(method: str, bpw: float, value_mse: float) -> dict:
    return {"method": method, "bpw": float(bpw), "mse": float(value_mse)}


def load_resume_records(path: Path | str | None) -> dict[str, dict]:
    if not path:
        return {}
    resume_path = Path(path)
    if not resume_path.exists():
        return {}

    records: dict[str, dict] = {}
    with resume_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("key")
            if not key:
                raise ValueError(f"{resume_path}:{line_no} missing layer key")
            records[str(key)] = record
    return records


def append_resume_record(path: Path | str, record: dict) -> None:
    resume_path = Path(path)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    with resume_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def uniform_int4_candidate_from_layer(layer: dict) -> dict:
    for item in layer.get("candidates", []):
        if item.get("method") == "groupwise_int4":
            return {"params": int(layer["params"]), **item}
    raise ValueError(f"layer {layer.get('key', layer.get('idx'))} has no groupwise_int4 candidate")


def scan(
    model_dir: Path,
    group_size: int,
    outlier_options: list[int],
    max_layers: int | None,
    target_bpw: float | None,
    target_margin_bpw: float,
    activation_weights: dict[str, float],
    resume_jsonl: Path | None = None,
) -> dict:
    layers = []
    uniform_int4 = []
    resume_records = load_resume_records(resume_jsonl)

    for idx, key, weight in iter_model_weights(model_dir, max_layers=max_layers):
        if key in resume_records:
            layer = resume_records[key]
            layers.append(layer)
            uniform_int4.append(uniform_int4_candidate_from_layer(layer))
            print(json.dumps({"idx": idx, "key": key, "resumed": True}), flush=True)
            del weight
            gc.collect()
            continue

        params = weight.numel()
        layer_candidates = []

        binary_entry = quantize_binary_residual(weight, group_size=group_size)
        binary = dequantize_binary_residual(binary_entry)
        base_only = dequantize_binary_residual(binary_entry, include_residual=False)
        base_bpw = binary_entry["bpw"] - 1.0 - (16.0 / group_size)
        layer_candidates.append(candidate("int2_base", base_bpw, mse(weight, base_only)))
        layer_candidates.append(candidate("int2_binary_residual", binary_entry["bpw"], mse(weight, binary)))

        for k in outlier_options:
            if k <= 0:
                continue
            error_entry = quantize_error_budget_residual(
                weight,
                group_size=group_size,
                outliers_per_group=k,
            )
            error_budget = dequantize_error_budget_residual(error_entry)
            layer_candidates.append(
                candidate(
                    f"int2_error_budget_k{k}",
                    error_entry["bpw"],
                    mse(weight, error_budget),
                )
            )
            del error_budget, error_entry

        int4_entry = quantize_groupwise_int4(weight, group_size=group_size)
        int4 = dequantize_groupwise_int4(int4_entry)
        int4_candidate = candidate("groupwise_int4", int4_entry["bpw"], mse(weight, int4))
        layer_candidates.append(int4_candidate)
        uniform_int4.append({"params": params, **int4_candidate})

        layer = {
            "idx": idx,
            "key": key,
            "shape": list(weight.shape),
            "params": params,
            "activation_weight": activation_weights.get(key, activation_weights.get(str(idx), 1.0)),
            "candidates": sorted(layer_candidates, key=lambda item: (item["bpw"], item["mse"], item["method"])),
        }
        layers.append(layer)
        print(json.dumps({"idx": idx, "key": key, "candidates": layer["candidates"]}), flush=True)
        if resume_jsonl is not None:
            append_resume_record(resume_jsonl, layer)

        del weight, binary, base_only, int4, binary_entry, int4_entry
        gc.collect()

    if not layers:
        raise RuntimeError(f"No quantizable language_model 2D weights found in {model_dir}")

    int4_summary = summarize_allocation(uniform_int4)
    budget = target_bpw if target_bpw is not None else int4_summary["avg_bpw"] - target_margin_bpw
    allocation = allocate_mixed_budget(layers, target_avg_bpw=budget)
    return {
        "group_size": group_size,
        "outlier_options": outlier_options,
        "max_layers": max_layers,
        "target_bpw": budget,
        "uniform_int4": int4_summary,
        "mixed_allocation": allocation,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--outlier-options", default="4,8")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--target-bpw", type=float, default=None)
    parser.add_argument("--target-margin-bpw", type=float, default=0.01)
    parser.add_argument("--activation-weights", default=None)
    parser.add_argument(
        "--resume-jsonl",
        default=None,
        help="Append one completed layer record per line and reuse it on restart.",
    )
    parser.add_argument("--output", default="eval_results/mixed_budget_scan.json")
    args = parser.parse_args()

    result = scan(
        model_dir=Path(args.model_dir),
        group_size=args.group_size,
        outlier_options=parse_int_list(args.outlier_options),
        max_layers=args.max_layers,
        target_bpw=args.target_bpw,
        target_margin_bpw=args.target_margin_bpw,
        activation_weights=load_activation_weights(args.activation_weights),
        resume_jsonl=Path(args.resume_jsonl) if args.resume_jsonl else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("UNIFORM_INT4=" + json.dumps(result["uniform_int4"], indent=2), flush=True)
    print("MIXED_ALLOCATION=" + json.dumps(result["mixed_allocation"], indent=2), flush=True)
    print("WROTE", output, flush=True)


if __name__ == "__main__":
    main()
