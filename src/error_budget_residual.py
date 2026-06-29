from __future__ import annotations

from math import ceil, log2
from typing import Sequence

import torch

from .groupwise_int4 import pack_signed_int4, unpack_signed_int4


def pack_signed_int2(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT2 values in {-1, 0, 1}, four values per byte."""
    flat = values.to(torch.int8).flatten()
    if flat.numel() and (flat.min().item() < -1 or flat.max().item() > 1):
        raise ValueError("signed int2 values must be in {-1, 0, 1}")
    encoded = (flat + 1).to(torch.uint8)
    pad = (-encoded.numel()) % 4
    if pad:
        encoded = torch.cat([encoded, torch.ones(pad, dtype=torch.uint8, device=encoded.device)])
    encoded = encoded.reshape(-1, 4)
    return (
        encoded[:, 0]
        | torch.bitwise_left_shift(encoded[:, 1], 2)
        | torch.bitwise_left_shift(encoded[:, 2], 4)
        | torch.bitwise_left_shift(encoded[:, 3], 6)
    ).contiguous()


def unpack_signed_int2(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes produced by pack_signed_int2."""
    if count < 0:
        raise ValueError("count must be non-negative")
    packed = packed.to(torch.uint8).flatten()
    values = torch.stack(
        (
            torch.bitwise_and(packed, 0x03),
            torch.bitwise_and(torch.bitwise_right_shift(packed, 2), 0x03),
            torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x03),
            torch.bitwise_and(torch.bitwise_right_shift(packed, 6), 0x03),
        ),
        dim=1,
    ).flatten()[:count]
    return values.to(torch.int16).sub(1).to(torch.int8)


def pack_binary_sign(values: torch.Tensor) -> torch.Tensor:
    """Pack residual signs {-1, +1}, eight signs per byte."""
    flat = values.to(torch.int8).flatten()
    if flat.numel() and not torch.all((flat == -1) | (flat == 1) | (flat == 0)):
        raise ValueError("binary residual signs must be -1, 0, or 1")
    encoded = torch.where(flat > 0, torch.ones_like(flat, dtype=torch.uint8), torch.zeros_like(flat, dtype=torch.uint8))
    pad = (-encoded.numel()) % 8
    if pad:
        encoded = torch.cat([encoded, torch.zeros(pad, dtype=torch.uint8, device=encoded.device)])
    encoded = encoded.reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=encoded.device)
    return torch.sum(torch.bitwise_left_shift(encoded, shifts), dim=1).to(torch.uint8).contiguous()


def unpack_binary_sign(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes produced by pack_binary_sign into {-1, +1} signs."""
    if count < 0:
        raise ValueError("count must be non-negative")
    packed = packed.to(torch.uint8).flatten()
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_and(torch.bitwise_right_shift(packed.unsqueeze(1), shifts), 1).flatten()[:count]
    return torch.where(bits > 0, torch.ones_like(bits, dtype=torch.int8), -torch.ones_like(bits, dtype=torch.int8))


def _validate_2d_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must be a 2D weight matrix shape")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")
    return rows, cols


def estimate_binary_residual_bpw(
    shape: Sequence[int],
    group_size: int = 128,
    scale_bits: int = 16,
) -> float:
    """Estimate BPW for INT2 base + 1-bit residual signs + two group scales."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, cols = _validate_2d_shape(shape)
    groups_per_row = ceil(cols / group_size)
    weight_bits = rows * cols * 3
    scale_bits_total = rows * groups_per_row * scale_bits * 2
    return (weight_bits + scale_bits_total) / (rows * cols)


def estimate_error_budget_residual_bpw(
    shape: Sequence[int],
    group_size: int = 128,
    outliers_per_group: int = 8,
    scale_bits: int = 16,
    correction_bits: int = 4,
) -> float:
    """Estimate BPW for INT2 base plus sparse residual corrections.

    Each group stores a base scale, a residual correction scale, and up to
    outliers_per_group index+signed-correction pairs.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if outliers_per_group < 0:
        raise ValueError("outliers_per_group must be non-negative")
    rows, cols = _validate_2d_shape(shape)
    groups_per_row = ceil(cols / group_size)
    groups = rows * groups_per_row
    index_bits = max(1, ceil(log2(group_size)))
    base_and_binary_bits = rows * cols * 3
    base_and_binary_scale_bits = groups * scale_bits * 2
    outlier_scale_bits = groups * scale_bits if outliers_per_group else 0
    outlier_bits = groups * outliers_per_group * (index_bits + correction_bits)
    return (base_and_binary_bits + base_and_binary_scale_bits + outlier_scale_bits + outlier_bits) / (rows * cols)


def _reshape_blocks(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
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
    return source.reshape(rows, groups_per_row, group_size), padded_cols


def quantize_binary_residual(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float16,
) -> dict:
    """Quantize with an INT2 base and per-group 1-bit residual correction.

    The residual code stores only the sign of the base reconstruction error plus
    one learned magnitude per group. This is a compact candidate below 4 BPW.
    """
    blocks, padded_cols = _reshape_blocks(weight, group_size)
    rows, cols = weight.shape

    base_max = blocks.abs().amax(dim=2)
    safe_max = torch.where(base_max > 0, base_max, torch.ones_like(base_max))
    multipliers = torch.tensor([0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=torch.float32, device=blocks.device)

    best_sse = torch.full_like(base_max, float("inf"))
    best_q = torch.zeros_like(blocks, dtype=torch.int8)
    best_base_scales = torch.ones_like(base_max)
    best_residual_sign = torch.zeros_like(blocks, dtype=torch.int8)
    best_residual_scales = torch.zeros_like(base_max)

    for multiplier in multipliers:
        candidate_scales = (safe_max * multiplier).to(scale_dtype).to(torch.float32)
        q = torch.round(blocks / candidate_scales.unsqueeze(-1)).clamp(-1, 1)
        base_rec = q * candidate_scales.unsqueeze(-1)
        residual = blocks - base_rec
        residual_scales = residual.abs().mean(dim=2).to(scale_dtype).to(torch.float32)
        residual_sign = torch.where(residual >= 0, 1.0, -1.0)
        residual_rec = residual_sign * residual_scales.unsqueeze(-1)
        sse = (residual - residual_rec).pow(2).sum(dim=2)

        better = sse < best_sse
        best_sse = torch.where(better, sse, best_sse)
        best_base_scales = torch.where(better, candidate_scales, best_base_scales)
        best_residual_scales = torch.where(better, residual_scales, best_residual_scales)
        best_q = torch.where(better.unsqueeze(-1), q.to(torch.int8), best_q)
        best_residual_sign = torch.where(better.unsqueeze(-1), residual_sign.to(torch.int8), best_residual_sign)

    zero_groups = base_max <= 0
    best_residual_sign = torch.where(
        zero_groups.unsqueeze(-1),
        torch.zeros_like(best_residual_sign),
        best_residual_sign,
    )
    best_residual_scales = torch.where(zero_groups, torch.zeros_like(best_residual_scales), best_residual_scales)

    return {
        "format": "int2_binary_residual",
        "base_bits": 2,
        "residual_bits": 1,
        "group_size": int(group_size),
        "orig_shape": [int(rows), int(cols)],
        "padded_shape": [int(rows), int(padded_cols)],
        "base_packed": pack_signed_int2(best_q.cpu()),
        "base_count": int(best_q.numel()),
        "base_scales": best_base_scales.to(scale_dtype).cpu(),
        "residual_packed": pack_binary_sign(best_residual_sign.cpu()),
        "residual_count": int(best_residual_sign.numel()),
        "residual_scales": best_residual_scales.to(scale_dtype).cpu(),
        "bpw": estimate_binary_residual_bpw((rows, cols), group_size=group_size, scale_bits=16),
    }


def _valid_block_mask(rows: int, cols: int, padded_cols: int, group_size: int, device: torch.device) -> torch.Tensor:
    valid = torch.zeros((rows, padded_cols), dtype=torch.bool, device=device)
    valid[:, :cols] = True
    return valid.reshape(rows, padded_cols // group_size, group_size)


def quantize_error_budget_residual(
    weight: torch.Tensor,
    group_size: int = 128,
    outliers_per_group: int = 8,
    scale_dtype: torch.dtype = torch.float16,
) -> dict:
    """Quantize with an INT2 base plus sparse top-error residual corrections."""
    if outliers_per_group < 0:
        raise ValueError("outliers_per_group must be non-negative")
    if group_size > 256:
        raise ValueError("group_size must be <= 256 so residual indices fit in uint8")

    blocks, padded_cols = _reshape_blocks(weight, group_size)
    rows, cols = weight.shape
    groups_per_row = padded_cols // group_size
    valid_blocks = _valid_block_mask(rows, cols, padded_cols, group_size, blocks.device)
    k = min(int(outliers_per_group), group_size)

    base_max = blocks.abs().amax(dim=2)
    safe_max = torch.where(base_max > 0, base_max, torch.ones_like(base_max))
    multipliers = torch.tensor([0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=torch.float32, device=blocks.device)

    best_sse = torch.full_like(base_max, float("inf"))
    best_q = torch.zeros_like(blocks, dtype=torch.int8)
    best_base_scales = torch.ones_like(base_max)
    best_binary_sign = torch.zeros_like(blocks, dtype=torch.int8)
    best_binary_scales = torch.zeros_like(base_max)
    best_indices = torch.zeros((rows, groups_per_row, k), dtype=torch.long, device=blocks.device)
    best_counts = torch.zeros((rows, groups_per_row), dtype=torch.long, device=blocks.device)
    best_outlier_q = torch.zeros((rows, groups_per_row, k), dtype=torch.int8, device=blocks.device)
    best_outlier_scales = torch.zeros_like(base_max)

    for multiplier in multipliers:
        candidate_scales = (safe_max * multiplier).to(scale_dtype).to(torch.float32)
        q = torch.round(blocks / candidate_scales.unsqueeze(-1)).clamp(-1, 1)
        base_rec = q * candidate_scales.unsqueeze(-1)
        residual = blocks - base_rec
        binary_scales = residual.abs().mean(dim=2).to(scale_dtype).to(torch.float32)
        binary_sign = torch.where(residual >= 0, 1.0, -1.0)
        binary_rec = binary_sign * binary_scales.unsqueeze(-1)
        remaining = residual - binary_rec

        outlier_rec = torch.zeros_like(blocks)
        top_idx = torch.zeros((rows, groups_per_row, k), dtype=torch.long, device=blocks.device)
        selected_counts = torch.zeros((rows, groups_per_row), dtype=torch.long, device=blocks.device)
        outlier_q = torch.zeros((rows, groups_per_row, k), dtype=torch.int8, device=blocks.device)
        outlier_scales = torch.zeros_like(base_max)

        if k:
            masked_abs = torch.where(valid_blocks, remaining.abs(), torch.full_like(remaining, -1.0))
            top_abs, top_idx = torch.topk(masked_abs, k=k, dim=2)
            selected = top_abs >= 0
            selected_counts = selected.sum(dim=2)
            gathered = torch.gather(remaining, 2, top_idx)
            max_abs = torch.where(top_abs[:, :, 0] > 0, top_abs[:, :, 0], torch.zeros_like(top_abs[:, :, 0]))
            outlier_scales = (max_abs / 7.0).to(scale_dtype).to(torch.float32)
            denom = torch.where(outlier_scales > 0, outlier_scales, torch.ones_like(outlier_scales))
            outlier_q = torch.round(gathered / denom.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
            outlier_q = torch.where(selected, outlier_q, torch.zeros_like(outlier_q))
            correction = outlier_q.to(torch.float32) * outlier_scales.unsqueeze(-1)
            correction = torch.where(selected, correction, torch.zeros_like(correction))
            outlier_rec.scatter_add_(2, top_idx, correction)

        error = torch.where(valid_blocks, base_rec + binary_rec + outlier_rec - blocks, torch.zeros_like(blocks))
        sse = error.pow(2).sum(dim=2)
        better = sse < best_sse
        best_sse = torch.where(better, sse, best_sse)
        best_base_scales = torch.where(better, candidate_scales, best_base_scales)
        best_q = torch.where(better.unsqueeze(-1), q.to(torch.int8), best_q)
        best_binary_sign = torch.where(better.unsqueeze(-1), binary_sign.to(torch.int8), best_binary_sign)
        best_binary_scales = torch.where(better, binary_scales, best_binary_scales)
        best_indices = torch.where(better.unsqueeze(-1), top_idx, best_indices)
        best_counts = torch.where(better, selected_counts, best_counts)
        best_outlier_q = torch.where(better.unsqueeze(-1), outlier_q, best_outlier_q)
        best_outlier_scales = torch.where(better, outlier_scales, best_outlier_scales)

    return {
        "format": "int2_error_budget_residual",
        "base_bits": 2,
        "correction_bits": 4,
        "group_size": int(group_size),
        "outliers_per_group": int(outliers_per_group),
        "orig_shape": [int(rows), int(cols)],
        "padded_shape": [int(rows), int(padded_cols)],
        "base_packed": pack_signed_int2(best_q.cpu()),
        "base_count": int(best_q.numel()),
        "base_scales": best_base_scales.to(scale_dtype).cpu(),
        "binary_residual_packed": pack_binary_sign(best_binary_sign.cpu()),
        "binary_residual_count": int(best_binary_sign.numel()),
        "binary_residual_scales": best_binary_scales.to(scale_dtype).cpu(),
        "outlier_indices": best_indices.to(torch.uint8).cpu(),
        "outlier_counts": best_counts.to(torch.uint8).cpu(),
        "outlier_packed": pack_signed_int4(best_outlier_q.cpu()),
        "outlier_count": int(best_outlier_q.numel()),
        "outlier_scales": best_outlier_scales.to(scale_dtype).cpu(),
        "bpw": estimate_error_budget_residual_bpw(
            (rows, cols),
            group_size=group_size,
            outliers_per_group=outliers_per_group,
            scale_bits=16,
            correction_bits=4,
        ),
    }


def dequantize_binary_residual(
    entry: dict,
    device: str | torch.device = "cpu",
    include_residual: bool = True,
) -> torch.Tensor:
    """Dequantize an INT2 base plus optional 1-bit residual correction."""
    rows, cols = [int(dim) for dim in entry["orig_shape"]]
    padded_rows, padded_cols = [int(dim) for dim in entry.get("padded_shape", entry["orig_shape"])]
    group_size = int(entry["group_size"])
    if padded_rows != rows:
        raise ValueError("padded row count must match original row count")
    if padded_cols % group_size != 0:
        raise ValueError("padded columns must be divisible by group_size")

    groups_per_row = padded_cols // group_size
    if "base_packed" in entry:
        base_q = unpack_signed_int2(entry["base_packed"].to(device), int(entry["base_count"]))
    else:
        base_q = entry["base_q"].to(device=device, dtype=torch.int8).flatten()
    base_q = base_q.to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, group_size)
    base_scales = entry["base_scales"].to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, 1)
    restored = base_q * base_scales
    if include_residual:
        if "residual_packed" in entry:
            residual_sign = unpack_binary_sign(entry["residual_packed"].to(device), int(entry["residual_count"]))
        else:
            residual_sign = entry["residual_sign"].to(device=device, dtype=torch.int8).flatten()
        residual_sign = residual_sign.to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, group_size)
        residual_scales = entry["residual_scales"].to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, 1)
        restored = restored + residual_sign * residual_scales

    return restored.reshape(rows, padded_cols)[:, :cols].contiguous()


def dequantize_error_budget_residual(
    entry: dict,
    device: str | torch.device = "cpu",
    include_residual: bool = True,
) -> torch.Tensor:
    """Dequantize an INT2 base plus sparse residual correction side channel."""
    rows, cols = [int(dim) for dim in entry["orig_shape"]]
    padded_rows, padded_cols = [int(dim) for dim in entry.get("padded_shape", entry["orig_shape"])]
    group_size = int(entry["group_size"])
    if padded_rows != rows:
        raise ValueError("padded row count must match original row count")
    if padded_cols % group_size != 0:
        raise ValueError("padded columns must be divisible by group_size")

    groups_per_row = padded_cols // group_size
    base_q = unpack_signed_int2(entry["base_packed"].to(device), int(entry["base_count"]))
    base_q = base_q.to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, group_size)
    base_scales = entry["base_scales"].to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, 1)
    restored = base_q * base_scales

    if include_residual:
        binary_sign = unpack_binary_sign(
            entry["binary_residual_packed"].to(device),
            int(entry["binary_residual_count"]),
        )
        binary_sign = binary_sign.to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, group_size)
        binary_scales = entry["binary_residual_scales"].to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, 1)
        restored = restored + binary_sign * binary_scales

        k = int(entry["outliers_per_group"])
        if k <= 0:
            return restored.reshape(rows, padded_cols)[:, :cols].contiguous()

        outlier_q = unpack_signed_int4(entry["outlier_packed"].to(device), int(entry["outlier_count"]))
        outlier_q = outlier_q.to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, k)
        indices = entry["outlier_indices"].to(device=device, dtype=torch.long).reshape(rows, groups_per_row, k)
        counts = entry["outlier_counts"].to(device=device, dtype=torch.long).reshape(rows, groups_per_row)
        outlier_scales = entry["outlier_scales"].to(device=device, dtype=torch.float32).reshape(rows, groups_per_row, 1)

        valid = torch.arange(k, device=device).reshape(1, 1, k) < counts.unsqueeze(-1)
        correction = torch.where(valid, outlier_q * outlier_scales, torch.zeros_like(outlier_q))
        restored_flat = restored.reshape(rows * groups_per_row, group_size)
        group_ids = torch.arange(rows * groups_per_row, device=device).reshape(rows, groups_per_row, 1).expand_as(indices)
        restored_flat.index_put_(
            (group_ids.reshape(-1)[valid.reshape(-1)], indices.reshape(-1)[valid.reshape(-1)]),
            correction.reshape(-1)[valid.reshape(-1)],
            accumulate=True,
        )

    return restored.reshape(rows, padded_cols)[:, :cols].contiguous()
