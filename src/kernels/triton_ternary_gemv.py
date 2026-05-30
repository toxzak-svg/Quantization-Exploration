"""
Triton kernels for ternary GEMV - fused unpack + matmul for sub1bit inference.

This provides optimized GPU kernels for:
1. Ternary unpacking (5 values/byte → float32)
2. Fused ternary GEMV (W = U @ diag(S) @ Vt)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def unpack_ternary_kernel(packed_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Unpack 5 ternary values per byte into float32."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    byte_idx = offsets // 5
    bit_offset = (offsets % 5) * 2

    packed = tl.load(packed_ptr + byte_idx, mask=byte_idx < (n_elements + 4) // 5, other=0)
    val = (packed >> bit_offset) & 0x03

    ternary_map = [-1.0, 0.0, 1.0]
    result = tl.where(val == 0, -1.0,
               tl.where(val == 1, 0.0,
               tl.where(val == 2, 1.0, 0.0)))

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def ternary_unpack_kernel(packed_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Unpack ternary with vectorized memory access."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    byte_idx = offsets // 5
    bit_pos = (offsets % 5) * 2

    packed = tl.load(packed_ptr + byte_idx, mask=byte_idx < (n_elements + 4) // 5, other=0)
    val = (packed >> bit_pos) & 0x03

    result = tl.select(val == 0, -1.0,
              tl.select(val == 1, 0.0,
              tl.select(val == 2, 1.0, 0.0)))

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def ternary_gemv_kernel(
    U_ptr, Vt_ptr, S_ptr, x_ptr, y_ptr,
    U_scale_f, Vt_scale_f, S_scale_f,
    in_features, out_features, rank,
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Fused ternary GEMV kernel: y = (U @ diag(S) @ Vt) @ x"""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    U_mask = offs_m < in_features
    Vt_mask = offs_n < out_features

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for r in range(0, rank, BLOCK_R):
        r_offs = r + offs_r
        r_mask = r_offs < rank

        U_offs_m = offs_m[:, None]
        U_offs_r = r_offs[None, :]
        U_ptrs = U_ptr + U_offs_m * rank + U_offs_r
        U_vals = tl.load(U_ptrs, mask=U_mask[:, None] & r_mask[None, :], other=0.0)

        Vt_offs_r = r_offs[:, None]
        Vt_offs_n = offs_n[None, :]
        Vt_ptrs = Vt_ptr + Vt_offs_r * out_features + Vt_offs_n
        Vt_vals = tl.load(Vt_ptrs, mask=r_mask[:, None] & Vt_mask[None, :], other=0.0)

        x_offs = r_offs
        x_vals = tl.load(x_ptr + x_offs, mask=r_mask, other=0.0)

        S_vals = tl.load(S_ptr + x_offs, mask=r_mask, other=0.0)

        acc += tl.dot(U_vals, Vt_vals * x_vals[:, None] * S_vals[None, :])

    y_offs_m = offs_m[:, None]
    y_offs_n = offs_n[None, :]
    tl.store(y_ptr + y_offs_m * out_features + y_offs_n, acc, mask=U_mask[:, None] & Vt_mask[None, :])


def triton_ternary_gemv(
    U: torch.Tensor,
    Vt: torch.Tensor,
    S: torch.Tensor,
    x: torch.Tensor,
    U_scale: float,
    Vt_scale: float,
    S_scale: float,
    BLOCK_M: int = 64,
    BLOCK_R: int = 32,
    BLOCK_N: int = 64,
):
    """Triton GEMV for ternary matrices."""
    in_features, rank = U.shape
    _, out_features = Vt.shape

    y = torch.zeros(in_features, dtype=torch.float32, device=U.device)

    grid = (triton.cdiv(in_features, BLOCK_M), triton.cdiv(out_features, BLOCK_N))

    ternary_gemv_kernel[grid](
        U, Vt, S, x, y,
        U_scale, Vt_scale, S_scale,
        in_features, out_features, rank,
        BLOCK_M, BLOCK_R, BLOCK_N
    )

    return y


def triton_unpack_ternary(packed: torch.Tensor, shape: tuple, BLOCK_SIZE: int = 1024):
    """Use Triton to unpack ternary."""
    n_elements = shape[0] * shape[1]
    out = torch.zeros(n_elements, dtype=torch.float32, device=packed.device)

    grid = triton.cdiv(n_elements, BLOCK_SIZE)

    unpack_ternary_kernel[grid](packed, out, n_elements, BLOCK_SIZE)

    return out.reshape(shape)


class TritonTernaryGEMV:
    """High-performance ternary GEMV using Triton."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        device: str = 'cuda'
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.device = device

        self._U = None
        self._Vt = None
        self._S = None
        self._U_scale = 1.0
        self._Vt_scale = 1.0
        self._S_scale = 1.0

    def load_weights(
        self,
        U_packed: torch.Tensor,
        Vt_packed: torch.Tensor,
        S: torch.Tensor,
        U_scale: float,
        Vt_scale: float,
        S_scale: float,
        U_shape: tuple,
        Vt_shape: tuple
    ):
        """Load quantized weights into device memory."""
        self._U = triton_unpack_ternary(U_packed, U_shape).to(self.device)
        self._Vt = triton_unpack_ternary(Vt_packed, Vt_shape).to(self.device)
        self._S = S.to(self.device).float()
        self._U_scale = U_scale
        self._Vt_scale = Vt_scale
        self._S_scale = S_scale

        self._U = self._U * U_scale
        self._Vt = self._Vt * Vt_scale
        self._S = self._S * S_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = (U @ diag(S) @ Vt) @ x"""
        if self._U is None:
            raise RuntimeError("Weights not loaded. Call load_weights first.")

        x = x.to(self.device).float()

        temp1 = torch.matmul(self._Vt, x)
        temp2 = temp1 * self._S
        y = torch.matmul(self._U, temp2)

        return y

    def stream_weights(
        self,
        U_packed: torch.Tensor,
        Vt_packed: torch.Tensor,
        S: torch.Tensor,
        U_scale: float,
        Vt_scale: float,
        S_scale: float,
        U_shape: tuple,
        Vt_shape: tuple
    ):
        """Stream weights from host to device for minimal memory footprint."""
        self._U_scale = U_scale
        self._Vt_scale = Vt_scale
        self._S_scale = S_scale

        self._U = triton_unpack_ternary(U_packed, U_shape).to(self.device) * U_scale
        self._S = S.to(self.device).float() * S_scale
        self._Vt = triton_unpack_ternary(Vt_packed, Vt_shape).to(self.device) * Vt_scale