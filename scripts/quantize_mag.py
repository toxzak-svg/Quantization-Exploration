"""
Fast Magnitude-Aware Quantization
No SVD - uses per-tensor/per-channel quantization with importance weighting
Much faster than SVD-based approaches
"""

import torch
import numpy as np
from typing import Dict
import json
import struct
from pathlib import Path
import os
import gc

def quantize_tensor_magnitude(W, num_bits, per_channel=True):
    """Magnitude-aware quantization without SVD"""
    orig_shape = W.shape
    W_flat = W.view(-1)

    if per_channel:
        # Per-output-channel (row-wise) - better quality
        m = orig_shape[0]
        w = W_flat.view(m, -1)
        max_vals = w.abs().max(dim=1).values
        max_vals[max_vals == 0] = 1.0
        scale = max_vals / (2 ** (num_bits - 1) - 1)
        q = (w / scale.unsqueeze(1)).round().clamp(-(2**(num_bits-1)), 2**(num_bits-1)-1)
        return {
            'q': q.to(torch.int8).flatten(),
            'scale': scale,
            'num_bits': num_bits,
            'per_channel': True,
            'orig_shape': orig_shape
        }
    else:
        # Per-tensor
        max_val = W_flat.abs().max()
        if max_val == 0:
            max_val = 1.0
        scale = max_val / (2 ** (num_bits - 1) - 1)
        q = (W_flat / scale).round().clamp(-(2**(num_bits-1)), 2**(num_bits-1)-1)
        return {
            'q': q.to(torch.int8),
            'scale': scale.item(),
            'num_bits': num_bits,
            'per_channel': False,
            'orig_shape': orig_shape
        }

def compute_bpw(q_entry, orig_params):
    if q_entry['per_channel']:
        total_bits = q_entry['q'].numel() * q_entry['num_bits'] + q_entry['scale'].numel() * 16
    else:
        total_bits = q_entry['q'].numel() * q_entry['num_bits'] + 16
    return total_bits / orig_params

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_magq.pt')
    parser.add_argument('--high-bits', type=int, default=4, help='Bits for important layers')
    parser.add_argument('--low-bits', type=int, default=2, help='Bits for less important layers')
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
    print(f"Using {args.high_bits}-bit for important, {args.low_bits}-bit for others")

    quantized = {}
    stats = {'total_original': 0, 'total_bits': 0, 'bits_used': {}}
    dtype_map = {'F16': np.float16, 'BF16': np.float16, 'F32': np.float32}

    for idx, (key, info) in enumerate(weight_keys):
        begin, end = info['data_offsets']
        numpy_dtype = dtype_map.get(info['dtype'], np.float32)

        with open(safetensor_path, 'rb') as f:
            f.seek(8 + hs + begin)
            data = f.read(end - begin)

        W = torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape']).float()
        orig_params = W.numel()
        stats['total_original'] += orig_params

        # Determine bits based on layer importance
        layer_idx = idx
        if layer_idx < 5:
            num_bits = args.high_bits
            importance = 'high'
        elif layer_idx < 50:
            num_bits = max(args.high_bits - 1, args.low_bits + 1)
            importance = 'medium'
        else:
            num_bits = args.low_bits
            importance = 'low'

        stats['bits_used'][importance] = stats['bits_used'].get(importance, 0) + orig_params

        # Quantize per-channel for better quality
        q_entry = quantize_tensor_magnitude(W, num_bits, per_channel=True)
        q_entry['key'] = key

        quantized[idx] = q_entry

        bpw = compute_bpw(q_entry, orig_params)
        stats['total_bits'] += q_entry['q'].numel() * num_bits + q_entry['scale'].numel() * 16

        if idx % 50 == 0 or idx < 5:
            print(f"Layer {idx}: {num_bits}-bit ({importance}), bpw={bpw:.4f}, shape={list(info['shape'])}")

        del W, data
        gc.collect()

    stats['avg_bpw'] = stats['total_bits'] / stats['total_original']
    stats['compression'] = stats['total_original'] * 16 / stats['total_bits']

    print(f"\nResults:")
    print(f"  Avg bits/weight: {stats['avg_bpw']:.4f}")
    print(f"  Compression: {stats['compression']:.1f}x")
    print(f"  Bits distribution: {stats['bits_used']}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': 'magnitude_quantization',
        'config': {'high_bits': args.high_bits, 'low_bits': args.low_bits}
    }, args.output)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    main()