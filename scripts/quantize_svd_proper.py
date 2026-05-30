"""
SVD Quantization with PROPER threshold (70%) for actual compression
Key insight: 90% threshold gives near-full rank = no compression benefit
"""

import torch
import numpy as np
from typing import Dict
import json
import struct
from pathlib import Path
import os
import gc

def pack_ternary(t):
    """Pack ternary: 5 values -> 8 bits"""
    encoded = (t + 1).to(torch.uint8)  # -1,0,1 -> 0,1,2
    n = encoded.numel()
    pad = (5 - n % 5) % 5
    if pad:
        encoded = torch.cat([encoded.flatten(), torch.zeros(pad, dtype=torch.uint8)])
    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=encoded.device)
    packed = (encoded.reshape(-1, 5).to(torch.int32) * weights).sum(dim=1).to(torch.uint8)
    return packed

def quantize_svd(W, rank, sigma_bits=2):
    """SVD + ternary quantization"""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]

    U_scale = U_r.abs().max().item()
    Vt_scale = Vt_r.abs().max().item()
    S_scale = S_r.abs().max().item()

    U_q = (U_r / U_scale).round().clamp(-1, 1)
    Vt_q = (Vt_r / Vt_scale).round().clamp(-1, 1)
    qmax = 2 ** (sigma_bits - 1) - 1
    S_q = (S_r / (S_scale / qmax)).round().clamp(-qmax, qmax).to(torch.int8)

    return {
        'U_packed': pack_ternary(U_q),
        'U_scale': U_scale,
        'U_shape': list(U_r.shape),
        'Vt_packed': pack_ternary(Vt_q),
        'Vt_scale': Vt_scale,
        'Vt_shape': list(Vt_r.shape),
        'S': S_q,
        'S_scale': S_scale,
        'rank': rank,
        'sigma_bits': sigma_bits
    }

def compute_bpw(q, orig_params):
    u_bits = q['U_packed'].numel() * 8 * 0.625
    vt_bits = q['Vt_packed'].numel() * 8 * 0.625
    s_bits = q['S'].numel() * q['sigma_bits']
    return (u_bits + vt_bits + s_bits + 32) / orig_params

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_svd_70.pt')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='SVD energy threshold (lower = more compression)')
    parser.add_argument('--sigma-bits', type=int, default=2)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
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

    print(f"Found {len(weight_keys)} weights")
    print(f"Energy threshold: {args.threshold} ({int(args.threshold*100)}%)")
    print(f"Sigma bits: {args.sigma_bits}")

    quantized = {}
    stats = {'total_original': 0, 'total_bits': 0, 'ranks': [], 'bpws': []}
    dtype_map = {'F16': np.float16, 'BF16': np.float16, 'F32': np.float32}

    for idx, (key, info) in enumerate(weight_keys):
        begin, end = info['data_offsets']
        numpy_dtype = dtype_map.get(info['dtype'], np.float32)

        with open(safetensor_path, 'rb') as f:
            f.seek(8 + hs + begin)
            data = f.read(end - begin)

        W = torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape']).float()
        m, n = W.shape
        orig_params = m * n
        stats['total_original'] += orig_params

        # SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        cum_energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()
        rank = (cum_energy < args.threshold).sum().item() + 1
        rank = min(rank, min(m, n))

        # Quantize
        q = quantize_svd(W, rank, args.sigma_bits)
        q['original_shape'] = list(info['shape'])
        q['key'] = key
        quantized[idx] = q

        bpw = compute_bpw(q, orig_params)
        stats['bpws'].append(bpw)
        stats['ranks'].append(rank)
        stats['total_bits'] += q['U_packed'].numel() * 8 + q['Vt_packed'].numel() * 8 + q['S'].numel() * args.sigma_bits + 32

        if idx % 50 == 0 or idx < 5:
            print(f"Layer {idx}: rank={rank:4d} ({100*rank/min(m,n):.1f}% of full), bpw={bpw:.4f}, shape=[{m},{n}]")

        del W, U, S, Vt, data
        gc.collect()

    stats['avg_bpw'] = np.mean(stats['bpws'])
    stats['avg_rank'] = np.mean(stats['ranks'])
    stats['compression'] = stats['total_original'] * 16 / stats['total_bits']

    print(f"\nResults:")
    print(f"  Avg rank: {stats['avg_rank']:.0f}")
    print(f"  Avg bits/weight: {stats['avg_bpw']:.4f}")
    print(f"  Compression: {stats['compression']:.1f}x")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': 'svd_ternary',
        'config': {'threshold': args.threshold, 'sigma_bits': args.sigma_bits}
    }, args.output)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    main()