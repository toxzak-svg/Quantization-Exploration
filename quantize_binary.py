import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import struct
import json
import os
import gc

os.chdir(r'C:\Users\Zwmar\projects\sub1quant')
import sys
sys.path.insert(0, r'.\src')

from quantize import SubOneBitQuantizer

def pack_binary_ternary(tensor: torch.Tensor) -> tuple:
    """Pack ternary {-1, 0, +1} values into bits.
    For binary {-1, +1}: 1 bit per value
    For ternary {-1, 0, +1}: need 2 bits per value
    """
    t = tensor.clone()
    # Normalize to -1, 0, +1 range
    t = torch.sign(t)
    t[t == 0] = 1  # Treat 0 as +1 for binary case

    # For binary {-1, +1}: pack 8 values into 1 byte
    # Sign bit: -1 -> 0, +1 -> 1
    binary = (t > 0).to(torch.uint8)
    packed = torch.zeros(binary.numel() // 8, dtype=torch.uint8)
    for i in range(8):
        packed |= binary[i::8] << i
    return packed, t.shape


def unpack_binary(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Unpack binary {-1, +1} values."""
    total = shape[0] * shape[1]
    result = torch.zeros(total, dtype=torch.uint8, device=packed.device)
    for i in range(8):
        result[i::8] = (packed >> i) & 1
    # 0 -> -1, 1 -> +1
    result = result * 2 - 1
    return result.reshape(shape)


def quantize_binary(x: torch.Tensor):
    """Quantize to binary {-1, +1}."""
    scale = x.abs().max()
    if scale == 0:
        scale = 1.0
    normalized = x / scale
    binary = torch.sign(normalized)
    binary[binary == 0] = 1
    return binary.to(torch.int8), scale


def quantize_sigma(sigma: torch.Tensor, num_bits: int = 2):
    """Quantize singular values to num_bits."""
    max_val = sigma.abs().max()
    if max_val == 0:
        max_val = 1.0
    scale = max_val / (2 ** (num_bits - 1) - 1)
    normalized = (sigma / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return normalized.to(torch.int8), scale


def main():
    print("=" * 60)
    print("Sub-1-Bit Quantization with Proper Binary Encoding")
    print("=" * 60)

    device = 'cpu'
    max_rank = 16

    quantizer = SubOneBitQuantizer(r'.\models\llama-2-7b-hf', device=device)
    quantizer.load_model_weights()

    output_dir = r'.\quantized\rank16_binary'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}\checkpoints', exist_ok=True)

    print(f"\nProcessing {quantizer.total_weights} weights with rank cap={max_rank}...")

    quantized = {}
    total_bits = 0
    total_factor_params = 0

    for idx in range(quantizer.total_weights):
        W = quantizer.load_weight(idx)

        # SVD
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)

        # Cap rank
        r = min(max_rank, len(S))
        U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]

        # Quantize U and Vt to binary {-1, +1}
        q_U, scale_U = quantize_binary(U_r)
        q_Vt, scale_Vt = quantize_binary(Vt_r)

        # Quantize S to 2-bit
        q_S, scale_S = quantize_sigma(S_r, num_bits=2)

        # Calculate bits
        n_U = U_r.shape[0] * U_r.shape[1]
        n_S = S_r.shape[0]
        n_Vt = Vt_r.shape[0] * Vt_r.shape[1]
        bits_U = n_U * 0.5  # 0.5 bits per value for binary
        bits_S = n_S * 2  # 2 bits per value for sigma
        bits_Vt = n_Vt * 0.5
        total_bits += bits_U + bits_S + bits_Vt
        total_factor_params += n_U + n_S + n_Vt

        quantized[idx] = {
            'U': q_U.numpy(),
            'S': q_S.numpy(),
            'Vt': q_Vt.numpy(),
            'U_shape': list(U_r.shape),
            'S_shape': list(S_r.shape),
            'Vt_shape': list(Vt_r.shape),
            'rank': r,
            'original_shape': list(W.shape),
            'scale_U': scale_U.item(),
            'scale_S': scale_S.item(),
            'scale_Vt': scale_Vt.item(),
        }

        if idx % 50 == 0 or idx < 5:
            energy = (S[:r]**2).sum() / (S**2).sum()
            bits_per_param = (bits_U + bits_S + bits_Vt) / (n_U + n_S + n_Vt)
            print(f"  Layer {idx}: shape={W.shape}, rank={r}, energy@rank={energy:.4f}, bits/param={bits_per_param:.3f}")

        del U, S, Vt, W, U_r, S_r, Vt_r
        gc.collect()

    print(f"\n=== Quantization Summary ===")
    avg_bits = total_bits / total_factor_params
    print(f"Total factor params: {total_factor_params:,}")
    print(f"Total bits: {total_bits:,}")
    print(f"Average bits/param: {avg_bits:.4f}")
    print(f"Estimated size: {total_bits/8/1024/1024:.2f} MB")

    # Save checkpoint
    torch.save({'quantized': quantized, 'config': {'max_rank': max_rank, 'encoding': 'binary'}},
               f'{output_dir}\\model.pt')
    print(f"\nSaved to {output_dir}\\model.pt")


if __name__ == '__main__':
    main()