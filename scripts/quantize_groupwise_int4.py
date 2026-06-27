"""Build a groupwise INT4 checkpoint from local Gemma safetensors.

This is the practical pivot path for this repo: preserve enough precision to
chase FP8-like perplexity, while storing weights in a packed INT4 format that
can be consumed by faster kernels later.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.groupwise_int4 import dequantize_groupwise_int4, quantize_groupwise_int4


SKIP_WEIGHT_SUBSTRINGS = (
    "lm_head",
    "embed_tokens",
    "norm",
    "audio_tower",
    "vision_tower",
    "embed_vision",
)


def iter_safetensor_headers(model_dir: Path) -> Iterator[tuple[Path, int, dict]]:
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for safetensor_path in safetensor_files:
        with safetensor_path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        yield safetensor_path, header_size, header


def should_quantize_weight(key: str, info: dict) -> bool:
    shape = info.get("shape")
    if "weight" not in key or shape is None or len(shape) != 2:
        return False
    if any(part in key for part in SKIP_WEIGHT_SUBSTRINGS):
        return False
    return "language_model" in key


def load_safetensor_slice(safetensor_path: Path, header_size: int, info: dict) -> torch.Tensor:
    begin, end = info["data_offsets"]
    dtype = info["dtype"]
    shape = tuple(info["shape"])

    with safetensor_path.open("rb") as handle:
        handle.seek(8 + header_size + begin)
        data = handle.read(end - begin)

    if dtype == "BF16":
        raw = torch.from_numpy(np.frombuffer(data, dtype=np.uint16).copy())
        return raw.view(torch.bfloat16).reshape(shape).float()
    if dtype == "F16":
        raw = torch.from_numpy(np.frombuffer(data, dtype=np.float16).copy())
        return raw.reshape(shape).float()
    if dtype == "F32":
        raw = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
        return raw.reshape(shape).float()

    raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {safetensor_path}")


def iter_model_weights(model_dir: Path, max_layers: int | None = None) -> Iterator[tuple[int, str, torch.Tensor]]:
    emitted = 0
    for safetensor_path, header_size, header in iter_safetensor_headers(model_dir):
        for key, info in header.items():
            if key == "__metadata__" or not isinstance(info, dict):
                continue
            if not should_quantize_weight(key, info):
                continue

            yield emitted, key, load_safetensor_slice(safetensor_path, header_size, info)
            emitted += 1
            if max_layers is not None and emitted >= max_layers:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Gemma weights with packed groupwise INT4")
    parser.add_argument("--model-dir", default="models/gemma-4-E2B", help="Directory containing model.safetensors")
    parser.add_argument("--output", default="quantized/gemma_groupwise_int4_g128.pt")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=None, help="Smoke-test on the first N matrices")
    parser.add_argument(
        "--skip-reconstruction-metrics",
        action="store_true",
        help="Skip per-layer MSE/RMSE measurement during quantization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")

    quantized: dict[int, dict] = {}
    layer_stats: list[dict] = []
    total_params = 0
    total_bits = 0.0
    total_weighted_mse = 0.0
    total_weighted_abs_error = 0.0

    print("=" * 72)
    print("GROUPWISE INT4 QUANTIZATION")
    print("=" * 72)
    print("Model dir:", model_dir)
    print("Output:", output)
    print("Group size:", args.group_size)
    if args.max_layers is not None:
        print("Max layers:", args.max_layers)
    print()

    for idx, key, weight in iter_model_weights(model_dir, max_layers=args.max_layers):
        entry = quantize_groupwise_int4(weight, group_size=args.group_size)
        entry["key"] = key
        quantized[idx] = entry

        params = weight.numel()
        total_params += params
        total_bits += entry["bpw"] * params

        mse = None
        rmse = None
        mean_abs_error = None
        if not args.skip_reconstruction_metrics:
            restored = dequantize_groupwise_int4(entry)
            error = restored - weight
            mse = error.pow(2).mean().item()
            rmse = mse ** 0.5
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
            print(f"Layer {idx:3d}: bpw={entry['bpw']:.4f}, shape={list(weight.shape)}{metric}")

        del weight
        gc.collect()

    if not quantized:
        raise RuntimeError(f"No quantizable language_model 2D weights found in {model_dir}")

    avg_bpw = total_bits / total_params
    stats = {
        "method": "groupwise_int4",
        "format_version": 1,
        "group_size": args.group_size,
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
            "method": "groupwise_int4",
            "config": {
                "model_dir": str(model_dir),
                "group_size": args.group_size,
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
