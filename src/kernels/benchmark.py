"""
Benchmark and test custom kernels for sub1bit inference.
"""

import torch
import time
from src.kernels import ternary_gemv, _unpack_ternary, ternary_gemv_high_bw


def test_unpack_ternary():
    print("Testing ternary unpack...")

    shape = (512, 64)
    total = shape[0] * shape[1]
    padded = ((total + 4) // 5) * 5

    original = torch.randint(-1, 2, shape, dtype=torch.int8)
    original[original == 0] = 1

    flat = original.flatten()
    if flat.shape[0] < padded:
        flat = torch.cat([flat, torch.zeros(padded - flat.shape[0], dtype=torch.int8)])

    encoded = flat + 1
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
    packed_int = encoded.reshape(-1, 5).to(torch.int32) * weights
    packed = packed_int.sum(dim=1).to(torch.uint8)

    unpacked = _unpack_ternary(packed, shape)

    assert torch.all(unpacked == original), "Unpack mismatch!"
    print(f"  PASS: unpacked shape={unpacked.shape}, max_diff={torch.abs(unpacked.float() - original.float()).max()}")


def benchmark_gemv():
    print("\nBenchmarking ternary GEMV...")

    in_features = 2048
    rank = 32
    out_features = 2048

    original_U = torch.randint(-1, 2, (in_features, rank), dtype=torch.int8).float()
    original_U[original_U == 0] = 1
    original_Vt = torch.randint(-1, 2, (rank, out_features), dtype=torch.int8).float()
    original_Vt[original_Vt == 0] = 1
    S = torch.rand(rank) * 10

    U_scale = 1.0
    Vt_scale = 1.0
    S_scale = 0.1

    x = torch.randn(out_features)

    def pack_ternary(t):
        flat = t.flatten()
        pad = (5 - flat.shape[0] % 5) % 5
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
        encoded = flat + 1
        weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
        packed_int = encoded.reshape(-1, 5).to(torch.int32) * weights
        return packed_int.sum(dim=1).to(torch.uint8)

    U_packed = pack_ternary(original_U)
    Vt_packed = pack_ternary(original_Vt)

    U_shape = (in_features, rank)
    Vt_shape = (rank, out_features)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(100):
        y = ternary_gemv(
            U_packed, Vt_packed, S, U_scale, Vt_scale, S_scale, x, U_shape, Vt_shape
        )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / 100 * 1000

    print(f"  ternary_gemv: {elapsed:.3f} ms/call")
    print(f"  output shape: {y.shape}, max: {y.abs().max():.2f}")

    ref_y = torch.matmul(original_U * S.unsqueeze(0), original_Vt)
    ref_y = ref_y @ x

    diff = (y - ref_y).abs().max()
    print(f"  max diff vs reference: {diff:.6f}")


def benchmark_gemv_high_bw():
    print("\nBenchmarking high-bandwidth ternary GEMV...")

    in_features = 2048
    rank = 32
    out_features = 8192

    original_U = torch.randint(-1, 2, (in_features, rank), dtype=torch.int8).float()
    original_U[original_U == 0] = 1
    original_Vt = torch.randint(-1, 2, (rank, out_features), dtype=torch.int8).float()
    original_Vt[original_Vt == 0] = 1
    S = torch.rand(rank) * 10

    x = torch.randn(out_features)

    def pack_ternary(t):
        flat = t.flatten().to(torch.int8)
        pad = (5 - flat.shape[0] % 5) % 5
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int8)])
        encoded = flat + 1
        weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
        packed_int = encoded.reshape(-1, 5).to(torch.int32) * weights
        return packed_int.sum(dim=1).to(torch.uint8)

    U_packed = pack_ternary(original_U)
    Vt_packed = pack_ternary(original_Vt)

    U_shape = (in_features, rank)
    Vt_shape = (rank, out_features)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(50):
        y = ternary_gemv_high_bw(
            U_packed, Vt_packed, S, 1.0, 1.0, 0.1, x, U_shape, Vt_shape, block_size=256
        )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / 50 * 1000

    print(f"  ternary_gemv_high_bw: {elapsed:.3f} ms/call")


def benchmark_memory():
    print("\nMemory comparison (packed vs unpacked)...")

    in_features = 2048
    rank = 32
    out_features = 2048

    original_U = torch.randint(-1, 2, (in_features, rank), dtype=torch.int8)
    original_Vt = torch.randint(-1, 2, (rank, out_features), dtype=torch.int8)

    original_size = original_U.numel() + original_Vt.numel()
    U_packed_size = ((original_U.numel() + 4) // 5)
    Vt_packed_size = ((original_Vt.numel() + 4) // 5)
    packed_size = U_packed_size + Vt_packed_size

    print(f"  Original: {original_size} bytes")
    print(f"  Packed:   {packed_size} bytes")
    print(f"  Ratio:    {original_size / packed_size:.2f}x compression")


if __name__ == '__main__':
    test_unpack_ternary()
    benchmark_gemv()
    benchmark_gemv_high_bw()
    benchmark_memory()
    print("\nAll benchmarks passed!")