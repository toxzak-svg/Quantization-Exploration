"""
Final Summary: Reconstruction Quality Results (Memory-Safe Version)
"""

import torch
import numpy as np
import json
import struct
from pathlib import Path
import os
import gc

def load_weight_by_idx(model_dir, idx):
    model_dir = Path(model_dir)
    safetensor_path = model_dir / 'model.safetensors'
    with open(safetensor_path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hs))

    weight_keys = []
    for key, info in header.items():
        if key == '__metadata__' or 'weight' not in key or len(info['shape']) != 2:
            continue
        if any(x in key for x in ['lm_head', 'embed_tokens', 'norm', 'audio_tower', 'vision_tower', 'embed_vision']):
            continue
        if 'language_model' not in key:
            continue
        weight_keys.append((key, info))

    key, info = weight_keys[idx]
    begin, end = info['data_offsets']
    dtype_map = {'F16': np.float16, 'BF16': np.float16, 'F32': np.float32}
    numpy_dtype = dtype_map.get(info['dtype'], np.float32)

    with open(safetensor_path, 'rb') as f:
        f.seek(8 + hs + begin)
        data = f.read(end - begin)

    return torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape']).float(), key

def unpack_ternary(packed, shape, device='cpu'):
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=device)
    expanded = packed.to(device).to(torch.int32).unsqueeze(-1) // weights % 3
    flat = (expanded - 1).flatten()[:shape[0] * shape[1]]
    return flat.view(shape[0], shape[1])

def reconstruct_svd_sub1bit(q_entry, device='cpu'):
    U = unpack_ternary(q_entry['U_packed'], q_entry['U_shape'], device).float() * q_entry['U_scale']
    Vt = unpack_ternary(q_entry['Vt_packed'], q_entry['Vt_shape'], device).float() * q_entry['Vt_scale']
    S = q_entry['S'].to(device).float() * q_entry['S_scale']
    return torch.matmul(U * S.unsqueeze(0), Vt)

def reconstruct_ternary_aggressive(q_entry, device='cpu'):
    packed = q_entry['packed']
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=device)
    expanded = packed.to(device).to(torch.int32).unsqueeze(-1) // weights % 3
    flat = (expanded - 1).flatten()[:q_entry['orig_shape'][0] * q_entry['orig_shape'][1]]
    return flat.view(q_entry['orig_shape'][0], q_entry['orig_shape'][1]).float() * q_entry['scale']

def reconstruct_magnitude_small(q_entry, device='cpu'):
    """Memory-safe version for magnitude quantization"""
    num_bits = q_entry['num_bits']
    q = q_entry['q'].to(device).float()
    scale = q_entry['scale']
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    qmax = 2 ** (num_bits - 1) - 1
    return q * scale / qmax

def main():
    print("=" * 70)
    print("GEMMA 4 E2B - RECONSTRUCTION QUALITY (Memory-Safe)")
    print("=" * 70)

    # Test on layers with SMALLEST shapes first to avoid memory issues
    test_indices = [3, 4, 8, 50, 300]  # Smaller layers

    results = []

    methods = [
        ('quantized/gemma-4-E2B-sub1bit.pt', 'SVD Sub1Bit', reconstruct_svd_sub1bit),
        ('quantized/gemma_ternary_aggressive.pt', 'Ternary Aggressive', reconstruct_ternary_aggressive),
        ('quantized/gemma_magq.pt', 'Magnitude Quant', reconstruct_magnitude_small),
    ]

    for path, name, fn in methods:
        if not os.path.exists(path):
            print(f"\nSkipping {name} - file not found")
            continue

        print(f"\n{name}:")
        print("-" * 50)

        q_data = torch.load(path, map_location='cpu', weights_only=True)
        quantized = q_data['quantized']

        total_mse = 0
        total_params = 0

        for idx in test_indices:
            if idx >= len(quantized):
                continue

            q_entry = quantized[idx]
            W_orig, key = load_weight_by_idx('models/gemma-4-E2B', idx)

            # Get reconstruction
            try:
                W_rec = fn(q_entry, 'cpu')
            except Exception as e:
                print(f"  Layer {idx}: Error during reconstruction - {e}")
                continue

            # Ensure same shape
            if W_rec.shape != W_orig.shape:
                W_rec = W_rec[:W_orig.shape[0], :W_orig.shape[1]]

            mse = ((W_orig - W_rec) ** 2).mean().item()
            params = W_orig.numel()

            total_mse += mse * params
            total_params += params

            print(f"  Layer {idx:3d} ({W_orig.shape[0]:5d}x{W_orig.shape[1]:5d}): MSE={mse:.6f}")

            del W_orig, W_rec
            gc.collect()

        if total_params > 0:
            avg_mse = total_mse / total_params
            base_ppl = 5.0
            est_ppl = base_ppl * (1 + avg_mse * 50)
            print(f"\n  Overall Avg MSE: {avg_mse:.6f}")
            print(f"  Estimated PPL: {est_ppl:.1f}")
            results.append((name, avg_mse, est_ppl))

    print("\n" + "=" * 70)
    print("SUMMARY - Ordered by Quality")
    print("=" * 70)
    print(f"{'Method':<25} {'Avg MSE':<12} {'Est PPL':<12}")
    print("-" * 50)

    results.sort(key=lambda x: x[1])
    for name, mse, ppl in results:
        print(f"{name:<25} {mse:<12.6f} {ppl:<12.1f}")

if __name__ == "__main__":
    main()