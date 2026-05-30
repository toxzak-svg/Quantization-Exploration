"""
Ternary GEMV kernel for sub1bit inference.
Fused operations: unpack + matmul for W = U @ diag(S) @ Vt
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def ternary_gemv(
    U_packed: torch.Tensor,      # [U_packed_bytes] packed ternary U
    Vt_packed: torch.Tensor,     # [Vt_packed_bytes] packed ternary Vt
    S: torch.Tensor,             # [rank] singular values (quantized)
    U_scale: float,             # scale for U
    Vt_scale: float,            # scale for Vt
    S_scale: float,             # scale for S
    x: torch.Tensor,            # [out_features] input vector
    U_shape: Tuple[int, int],   # (in_features, rank)
    Vt_shape: Tuple[int, int],  # (rank, out_features)
) -> torch.Tensor:
    """
    Compute y = (U * diag(S) * Vt) @ x using fused ternary operations.

    Args:
        U_packed: packed ternary U matrix [packed_bytes]
        Vt_packed: packed ternary Vt matrix [packed_bytes]
        S: quantized singular values [rank]
        U_scale: dequantization scale for U
        Vt_scale: dequantization scale for Vt
        S_scale: dequantization scale for S
        x: input vector [out_features]
        U_shape: (in_features, rank)
        Vt_shape: (rank, out_features)

    Returns:
        y: output vector [in_features]
    """
    in_features, rank = U_shape
    rank_out, out_features = Vt_shape

    U = _unpack_ternary(U_packed, U_shape).to(x.device) * U_scale
    Vt = _unpack_ternary(Vt_packed, Vt_shape).to(x.device) * Vt_scale
    S_float = S.float() * S_scale

    temp1 = torch.matmul(Vt, x)
    temp2 = temp1 * S_float
    y = torch.matmul(U, temp2)
    return y


def _unpack_ternary(packed: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Unpack 5 ternary values per byte into int8 tensor."""
    out_features, in_features = shape
    total = in_features * out_features

    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=packed.device)
    packed_int = packed.to(torch.int32).unsqueeze(-1)
    expanded = packed_int // weights % 3
    flat = (expanded - 1).flatten()[:total]
    return flat.reshape(shape).to(torch.int8)


def ternary_gemv_high_bw(
    U_packed: torch.Tensor,
    Vt_packed: torch.Tensor,
    S: torch.Tensor,
    U_scale: float,
    Vt_scale: float,
    S_scale: float,
    x: torch.Tensor,
    U_shape: Tuple[int, int],
    Vt_shape: Tuple[int, int],
    block_size: int = 256,
) -> torch.Tensor:
    """
    High-bandwidth ternary GEMV with memory coalescing.

    Processes in blocks to improve cache utilization and memory access patterns.
    """
    in_features, rank = U_shape
    rank_out, out_features = Vt_shape

    U = _unpack_ternary(U_packed, U_shape).to(x.device) * U_scale
    Vt = _unpack_ternary(Vt_packed, Vt_shape).to(x.device) * Vt_scale
    S_float = S.float() * S_scale

    if out_features <= 4096:
        return ternary_gemv(U_packed, Vt_packed, S, U_scale, Vt_scale, S_scale, x, U_shape, Vt_shape)

    y = torch.zeros(in_features, dtype=torch.float32, device=x.device)

    for start in range(0, out_features, block_size):
        end = min(start + block_size, out_features)
        Vt_block = Vt[:, start:end]
        x_block = x[start:end]

        temp1 = torch.matmul(Vt_block, x_block)
        temp2 = temp1 * S_float
        y += torch.matmul(U, temp2)

    return y