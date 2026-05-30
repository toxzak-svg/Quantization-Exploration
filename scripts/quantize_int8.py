"""
Conservative INT8 Quantization for Gemma
- Per-channel quantization (better quality)
- Only 4x compression but minimal perplexity impact
- Goal: PPL increase < 2 points
"""

import torch
import numpy as np
from typing import Dict
import json
import struct
from pathlib import Path
import os
import gc

def quantize_int8_perchannel(W):
    """INT8 per-channel quantization with proper scales"""
    orig_shape = W.shape
    W_flat = W.view(W.shape[0], -1)

    # Per-output-channel scales
    max_vals = W_flat.abs().max(dim=1).values
    max_vals[max_vals == 0] = 1.0
    scales = max_vals / 127.0

    # Quantize
    q = (W_flat / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

    return {
        'q': q,
        'scales': scales,
        'orig_shape': orig_shape
    }

def dequantize_int8(q_entry):
    """Dequantize INT8"""
    q = q_entry['q'].float()
    scales = q_entry['scales']
    m, n = q_entry['orig_shape']
    q = q.view(m, n)
    return q * scales.unsqueeze(1)

def compute_bpw(orig_params):
    # INT8: 8 bits per weight, plus scales (1 scale per row, stored as float16 = 2 bytes)
    data_bits = orig_params * 8
    scale_bits = q_entry['scales'].numel() * 16
    total_bits = data_bits + scale_bits
    return total_bits / orig_params

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_int8.pt')
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

    quantized = {}
    stats = {'total_original': 0, 'total_bits': 0}
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

        # Quantize
        q_entry = quantize_int8_perchannel(W)
        q_entry['key'] = key

        quantized[idx] = q_entry

        # Calculate bits
        data_bits = q_entry['q'].numel() * 8
        scale_bits = q_entry['scales'].numel() * 16
        stats['total_bits'] += data_bits + scale_bits

        if idx % 50 == 0 or idx < 5:
            bpw = (data_bits + scale_bits) / orig_params
            print(f"Layer {idx}: shape={list(info['shape'])}, bpw={bpw:.4f}")

        del W, data
        gc.collect()

    stats['avg_bpw'] = stats['total_bits'] / stats['total_original']
    stats['compression'] = stats['total_original'] * 16 / stats['total_bits']

    print(f"\nResults:")
    print(f"  Avg bits/weight: {stats['avg_bpw']:.4f}")
    print(f"  Compression: {stats['compression']:.1f}x")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': 'int8_perchannel'
    }, args.output)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    main()