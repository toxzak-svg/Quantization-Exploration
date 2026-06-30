"""Build a mixed-budget checkpoint from a scan_mixed_budget allocation JSON."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_quantized import reconstruct_quantized_entry
from scripts.quantize_groupwise_int4 import iter_model_weights
from src.error_budget_residual import estimate_binary_residual_bpw, quantize_binary_residual, quantize_error_budget_residual
from src.groupwise_int4 import quantize_groupwise_int4


ERROR_BUDGET_RE = re.compile(r"^int2_error_budget_k(\d+)$")
SHARD_SCHEMA_VERSION = 1


def build_selection_map(scan: dict) -> dict[str, str]:
    selected = scan.get("mixed_allocation", {}).get("selected_layers", [])
    if not selected:
        raise ValueError("scan JSON does not contain mixed_allocation.selected_layers")
    selection = {}
    for item in selected:
        key = item.get("key")
        method = item.get("method")
        if not key or not method:
            raise ValueError(f"invalid selected layer entry: {item!r}")
        selection[str(key)] = str(method)
    return selection


def quantize_weight_for_method(weight: torch.Tensor, method: str, group_size: int) -> dict:
    if method == "groupwise_int4":
        return quantize_groupwise_int4(weight, group_size=group_size)
    if method == "int2_binary_residual":
        return quantize_binary_residual(weight, group_size=group_size)
    if method == "int2_base":
        entry = quantize_binary_residual(weight, group_size=group_size)
        entry["format"] = "int2_base"
        entry["bpw"] = estimate_binary_residual_bpw(weight.shape, group_size=group_size) - 1.0 - (16.0 / group_size)
        return entry

    match = ERROR_BUDGET_RE.match(method)
    if match:
        return quantize_error_budget_residual(
            weight,
            group_size=group_size,
            outliers_per_group=int(match.group(1)),
        )

    raise ValueError(f"Unsupported mixed-budget method {method!r}")


def layer_shard_path(checkpoint_dir: Path, idx: int) -> Path:
    return checkpoint_dir / f"layer_{idx:05d}.pt"


def save_layer_shard(checkpoint_dir: Path, layer_stat: dict, entry: dict, group_size: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_path = layer_shard_path(checkpoint_dir, int(layer_stat["idx"]))
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "group_size": int(group_size),
        "key": str(layer_stat["key"]),
        "method": str(layer_stat["method"]),
        "layer_stat": layer_stat,
        "entry": entry,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(shard_path)
    return shard_path


def load_layer_shard(shard_path: Path, key: str, method: str, group_size: int) -> dict:
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError(f"{shard_path} has unsupported shard schema")
    if payload.get("group_size") != int(group_size):
        raise ValueError(f"{shard_path} group size does not match requested build")
    if payload.get("key") != key:
        raise ValueError(f"{shard_path} key does not match requested build")
    if payload.get("method") != method:
        raise ValueError(f"{shard_path} method does not match requested build")
    return payload


def build_checkpoint(
    scan: dict,
    model_dir: Path,
    group_size: int,
    max_layers: int | None,
    skip_reconstruction_metrics: bool,
    checkpoint_dir: Path | None = None,
) -> dict:
    selection = build_selection_map(scan)
    quantized: dict[int, dict] = {}
    layer_stats: list[dict] = []
    total_params = 0
    total_bits = 0.0
    total_weighted_mse = 0.0

    for idx, key, weight in iter_model_weights(model_dir, max_layers=max_layers):
        method = selection.get(key)
        if method is None:
            raise KeyError(f"No mixed-budget selection for {key}")

        shard_path = layer_shard_path(checkpoint_dir, idx) if checkpoint_dir else None
        if shard_path is not None and shard_path.exists():
            shard = load_layer_shard(shard_path, key=key, method=method, group_size=group_size)
            entry = shard["entry"]
            layer_stat = shard["layer_stat"]
            if not skip_reconstruction_metrics and layer_stat.get("mse") is None:
                raise ValueError(f"{shard_path} lacks reconstruction metrics required by this build")
            print(f"Layer {idx:3d}: resumed {method} from {shard_path}", flush=True)
        else:
            entry = quantize_weight_for_method(weight, method, group_size=group_size)
            entry["key"] = key

            params = weight.numel()
            mse = None
            rmse = None
            if not skip_reconstruction_metrics:
                restored = reconstruct_quantized_entry(entry)
                error = restored - weight
                mse = error.pow(2).mean().item()
                rmse = mse**0.5
                del restored, error

            layer_stat = {
                "idx": idx,
                "key": key,
                "method": method,
                "shape": list(weight.shape),
                "params": params,
                "bpw": entry["bpw"],
                "mse": mse,
                "rmse": rmse,
            }
            print(
                f"Layer {idx:3d}: {method}, bpw={entry['bpw']:.4f}, shape={list(weight.shape)}"
                + ("" if mse is None else f", mse={mse:.6f}, rmse={rmse:.6f}"),
                flush=True,
            )
            if checkpoint_dir is not None:
                save_layer_shard(checkpoint_dir, layer_stat, entry, group_size=group_size)

        quantized[idx] = entry
        params = int(layer_stat["params"])
        total_params += params
        total_bits += float(layer_stat["bpw"]) * params
        if not skip_reconstruction_metrics:
            total_weighted_mse += float(layer_stat["mse"]) * params
        layer_stats.append(layer_stat)

        del weight
        gc.collect()

    if not quantized:
        raise RuntimeError(f"No quantizable language_model 2D weights found in {model_dir}")

    selected_keys = {item["key"] for item in layer_stats}
    extra_keys = sorted(set(selection) - selected_keys)
    if extra_keys:
        raise KeyError(f"Scan allocation had selections not emitted by this build: {extra_keys[:5]}")

    avg_bpw = total_bits / total_params
    stats = {
        "method": "mixed_budget",
        "format_version": 1,
        "group_size": group_size,
        "layers": len(quantized),
        "total_params": total_params,
        "avg_bpw": avg_bpw,
        "compression_vs_bf16": 16.0 / avg_bpw,
        "weighted_mse": None if skip_reconstruction_metrics else total_weighted_mse / total_params,
        "weighted_rmse": None if skip_reconstruction_metrics else (total_weighted_mse / total_params) ** 0.5,
        "method_counts": _method_counts(layer_stats),
        "layer_stats": layer_stats,
        "source_allocation": scan.get("mixed_allocation"),
        "uniform_int4": scan.get("uniform_int4"),
    }
    return {
        "quantized": quantized,
        "stats": stats,
        "method": "mixed_budget",
        "config": {
            "model_dir": str(model_dir),
            "group_size": group_size,
            "max_layers": max_layers,
            "source_scan": scan.get("source_scan"),
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        },
        "weight_keys": [
            {"key": item["key"], "shape": item["shape"], "method": item["method"]}
            for item in layer_stats
        ],
    }


def _method_counts(layer_stats: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in layer_stats:
        method = str(item["method"])
        counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-json", required=True)
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--output", default="quantized/gemma_mixed_budget.pt")
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--skip-reconstruction-metrics", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Save one layer shard after each layer and reuse matching shards on restart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_path = Path(args.scan_json)
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan["source_scan"] = str(scan_path)
    group_size = args.group_size if args.group_size is not None else int(scan.get("group_size", 128))
    max_layers = args.max_layers if args.max_layers is not None else scan.get("max_layers")

    checkpoint = build_checkpoint(
        scan=scan,
        model_dir=Path(args.model_dir),
        group_size=group_size,
        max_layers=max_layers,
        skip_reconstruction_metrics=args.skip_reconstruction_metrics,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    stats = checkpoint["stats"]
    print()
    print("=" * 72)
    print("MIXED BUDGET RESULTS")
    print("=" * 72)
    print(f"Layers: {stats['layers']}")
    print(f"Average BPW: {stats['avg_bpw']:.4f}")
    print(f"Compression vs BF16: {stats['compression_vs_bf16']:.2f}x")
    if stats["weighted_mse"] is not None:
        print(f"Weighted MSE: {stats['weighted_mse']:.8f}")
        print(f"Weighted RMSE: {stats['weighted_rmse']:.6f}")
    print("Method counts:", json.dumps(stats["method_counts"], sort_keys=True))
    print("Saved:", output)


if __name__ == "__main__":
    main()
