from __future__ import annotations

from math import ceil
from typing import Sequence

import torch


INT4_QMAX = 7


def _validate_signed_int4(values: torch.Tensor) -> torch.Tensor:
    flat = values.to(torch.int8).flatten()
    if flat.numel() == 0:
        return flat
    if flat.min().item() < -8 or flat.max().item() > 7:
        raise ValueError("signed int4 values must be in [-8, 7]")
    return flat


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values into uint8 bytes, two values per byte."""
    flat = _validate_signed_int4(values)
    encoded = (flat + 8).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = torch.cat([encoded, torch.full((1,), 8, dtype=torch.uint8, device=encoded.device)])

    low = encoded[0::2]
    high = torch.bitwise_left_shift(encoded[1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def unpack_signed_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack uint8 bytes produced by pack_signed_int4."""
    if count < 0:
        raise ValueError("count must be non-negative")
    packed = packed.to(torch.uint8).flatten()
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_right_shift(packed, 4)
    encoded = torch.stack((low, high), dim=1).flatten()[:count]
    return encoded.to(torch.int16).sub(8).to(torch.int8)


def estimate_groupwise_int4_bpw(
    shape: Sequence[int],
    group_size: int = 128,
    scale_bits: int = 16,
) -> float:
    """Estimate bits per weight for row-wise group INT4 plus per-group scales."""
    if len(shape) != 2:
        raise ValueError("shape must be a 2D weight matrix shape")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")

    groups_per_row = ceil(cols / group_size)
    weight_bits = rows * cols * 4
    scale_overhead_bits = rows * groups_per_row * scale_bits
    return (weight_bits + scale_overhead_bits) / (rows * cols)


def quantize_groupwise_int4(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float16,
) -> dict:
    """Symmetric row-wise group INT4 quantization for 2D weight matrices.

    This is an inference-oriented storage format: the packed nibbles and group
    scales can be consumed by an INT4 kernel later, while still being easy to
    dequantize for correctness/perplexity evaluation today.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    source = weight.detach().to(torch.float32)
    rows, cols = source.shape
    groups_per_row = ceil(cols / group_size)
    padded_cols = groups_per_row * group_size
    if padded_cols != cols:
        padded = torch.zeros((rows, padded_cols), dtype=source.dtype, device=source.device)
        padded[:, :cols] = source
        source = padded

    blocks = source.reshape(rows, groups_per_row, group_size)
    max_abs = blocks.abs().amax(dim=2)
    scales = torch.where(
        max_abs > 0,
        max_abs / INT4_QMAX,
        torch.ones_like(max_abs),
    )
    stored_scales = scales.to(scale_dtype).to(torch.float32)
    q = torch.round(blocks / stored_scales.unsqueeze(-1)).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)

    return {
        "format": "groupwise_int4",
        "bits": 4,
        "group_size": int(group_size),
        "orig_shape": [int(rows), int(cols)],
        "padded_shape": [int(rows), int(padded_cols)],
        "scales": stored_scales.to(scale_dtype).cpu(),
        "packed_int4": pack_signed_int4(q.cpu()),
        "bpw": estimate_groupwise_int4_bpw((rows, cols), group_size=group_size, scale_bits=16),
    }


def dequantize_groupwise_int4(entry: dict, device: str | torch.device = "cpu") -> torch.Tensor:
    """Dequantize an entry produced by quantize_groupwise_int4."""
    rows, cols = [int(dim) for dim in entry["orig_shape"]]
    padded_rows, padded_cols = [int(dim) for dim in entry.get("padded_shape", entry["orig_shape"])]
    group_size = int(entry["group_size"])
    if padded_rows != rows:
        raise ValueError("padded row count must match original row count")
    if padded_cols % group_size != 0:
        raise ValueError("padded columns must be divisible by group_size")

    groups_per_row = padded_cols // group_size
    count = rows * groups_per_row * group_size
    q = unpack_signed_int4(entry["packed_int4"].to(device), count).to(torch.float32)
    q = q.reshape(rows, groups_per_row, group_size)
    scales = entry["scales"].to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, 1)
    restored = (q * scales).reshape(rows, padded_cols)
    return restored[:, :cols].contiguous()
