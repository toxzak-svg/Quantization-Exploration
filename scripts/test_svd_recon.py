"""
Proper Reconstruction Evaluation for SVD Sub1Bit
"""

import torch
import numpy as np
import json
import struct
from pathlib import Path
import os

def load_single_weight(model_dir: str, key: str) -> torch.Tensor:
    model_dir = Path(model_dir)
    safetensor_path = model_dir / 'model.safetensors'

    with open(safetensor_path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hs))

    info = header[key]
    begin, end = info['data_offsets']
    dtype_map = {'F16': np.float16, 'BF16': np.float16, 'F32': np.float32}
    numpy_dtype = dtype_map.get(info['dtype'], np.float32)

    with open(safetensor_path, 'rb') as f:
        f.seek(8 + hs + begin)
        data = f.read(end - begin)

    return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape']).float()

def unpack_ternary(packed, shape, device='cpu'):
    """Unpack ternary: 8 bits -> 5 values"""
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=device)
    expanded = packed.to(device).to(torch.int32).unsqueeze(-1) // weights % 3
    flat = (expanded - 1).flatten()[:shape[0] * shape[1]]
    return flat.view(shape[0], shape[1])

def reconstruct_svd_sub1bit(q_entry, device='cpu'):
    """Properly reconstruct from SVD Sub1Bit format"""
    U_packed = q_entry['U_packed'].to(device)
    Vt_packed = q_entry['Vt_packed'].to(device)
    U_scale = q_entry['U_scale']
    Vt_scale = q_entry['Vt_scale']
    S = q_entry['S'].to(device).float()
    S_scale = q_entry['S_scale']

    U_shape = tuple(q_entry['U_shape'])
    Vt_shape = tuple(q_entry['Vt_shape'])

    U = unpack_ternary(U_packed, U_shape, device).float() * U_scale
    Vt = unpack_ternary(Vt_packed, Vt_shape, device).float() * Vt_scale
    S = S * S_scale

    W_rec = torch.matmul(U * S.unsqueeze(0), Vt)
    return W_rec

def build_key_to_weight_map(model_dir: str):
    """Build mapping from shape to weight key for matching"""
    model_dir = Path(model_dir)
    safetensor_path = model_dir / 'model.safetensors'

    with open(safetensor_path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hs))

    shape_to_key = {}
    key_to_shape = {}

    for key, info in header.items():
        if key == '__metadata__' or 'weight' not in key or len(info['shape']) != 2:
            continue
        if any(x in key for x in ['lm_head', 'embed_tokens', 'norm', 'audio_tower', 'vision_tower', 'embed_vision']):
            continue
        if 'language_model' not in key:
            continue

        shape = tuple(info['shape'])
        key_to_shape[key] = shape
        if shape not in shape_to_key:
            shape_to_key[shape] = []
        shape_to_key[shape].append(key)

    return shape_to_key, key_to_shape

def test_svd_sub1bit():
    print("Testing SVD Sub1Bit reconstruction...")
    print("=" * 60)

    q_data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
    quantized = q_data['quantized']

    shape_to_key, key_to_shape = build_key_to_weight_map('models/gemma-4-E2B')

    print(f"Loaded {len(quantized)} quantized entries")
    print(f"Found {len(key_to_shape)} LM weight keys")
    print()

    total_mse = 0
    total_params = 0

    for layer_idx in range(min(10, len(quantized))):
        q_entry = quantized[layer_idx]
        orig_shape = tuple(q_entry['original_shape'])

        # Find matching key
        key = None
        for k, s in key_to_shape.items():
            if s == orig_shape:
                key = k
                break

        if key is None:
            print(f"Layer {layer_idx}: No matching key for shape {orig_shape}")
            continue

        W_orig = load_single_weight('models/gemma-4-E2B', key)
        W_rec = reconstruct_svd_sub1bit(q_entry, 'cpu')

        print(f"Layer {layer_idx}: shape={orig_shape}, key={key[:40]}...")
        print(f"  Reconstructed shape: {W_rec.shape}")

        # Ensure same size
        if W_rec.shape != W_orig.shape:
            print(f"  Shape mismatch, adjusting...")
            W_rec = W_rec[:W_orig.shape[0], :W_orig.shape[1]]

        mse = ((W_orig - W_rec) ** 2).mean().item()
        max_err = (W_orig - W_rec).abs().max().item()
        params = W_orig.numel()

        total_mse += mse * params
        total_params += params

        # Compare with truncated SVD
        rank = q_entry['rank']
        U, S, Vt = torch.linalg.svd(W_orig, full_matrices=False)
        W_trunc = torch.matmul(U[:, :rank] * S[:rank], Vt[:rank, :])
        trunc_mse = ((W_orig - W_trunc) ** 2).mean().item()

        print(f"  MSE: {mse:.6f}, MaxErr: {max_err:.4f}")
        print(f"  Truncated SVD (rank={rank}) MSE: {trunc_mse:.6f}")
        print()

    if total_params > 0:
        avg_mse = total_mse / total_params
        print(f"\nOverall Avg MSE: {avg_mse:.6f}")
        print(f"Avg RMSE: {avg_mse**0.5:.6f}")

if __name__ == "__main__":
    test_svd_sub1bit()