"""
Aggressive Sub-Bit Quantization for Gemma
1-bit for most layers, 2-3 bits for critical layers
Target: ~0.7-1.0 bits/weight range
"""

import torch
import numpy as np
from typing import Dict
import json
import struct
from pathlib import Path
import os
import gc

def quantize_ternary(W, scale=None):
    """Simple ternary quantization (±1, 0)"""
    if scale is None:
        scale = W.abs().max()
    if scale == 0:
        scale = 1.0
    q = (W / scale).round().clamp(-1, 1)
    return q.to(torch.int8), scale.item()

def quantize_ternary_pack(W):
    """Ternary quantization with bit-packing (5 values -> 8 bits)"""
    scale = W.abs().max()
    if scale == 0:
        scale = 1.0
    q = (W / scale).round().clamp(-1, 1)

    # Pack: -1->0, 0->1, 1->2
    encoded = (q + 1).to(torch.uint8)
    n = encoded.numel()
    pad = (5 - n % 5) % 5
    if pad:
        encoded = torch.cat([encoded.flatten(), torch.zeros(pad, dtype=torch.uint8)])

    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=encoded.device)
    packed = (encoded.reshape(-1, 5).to(torch.int32) * weights).sum(dim=1).to(torch.uint8)

    return {
        'packed': packed,
        'scale': scale.item(),
        'orig_shape': list(W.shape),
        'num_packed': packed.numel()
    }

def compute_bpw_from_packed(packed_entry, orig_params):
    # 5 values packed into 8 bits = 0.625 bpw for data
    data_bits = packed_entry['num_packed'] * 8 * 0.625
    scale_bits = 16  # scale as float16
    total_bits = data_bits + scale_bits
    return total_bits / orig_params

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--output', default='quantized/gemma_ternary.pt')
    parser.add_argument('--critical-bits', type=int, default=2, help='Bits for first 5 layers')
    parser.add_argument('--standard-bits', type=int, default=1, help='Bits for most layers')
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
    print(f"Critical layers: {args.critical_bits}-bit, Standard layers: {args.standard_bits}-bit ternary")

    quantized = {}
    stats = {'total_original': 0, 'total_bits': 0, 'layer_bits': {}}
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

        # Determine precision based on layer importance
        # Critical: first 5 layers = most important for maintaining quality
        # Standard: everything else
        if idx < 5:
            num_bits = args.critical_bits
        else:
            num_bits = args.standard_bits

        stats['layer_bits'][idx] = num_bits

        # Ternary quantization with packing
        packed = quantize_ternary_pack(W)

        quantized[idx] = {
            'packed': packed['packed'],
            'scale': packed['scale'],
            'orig_shape': packed['orig_shape'],
            'key': key,
            'num_bits': num_bits
        }

        bpw = compute_bpw_from_packed(packed, orig_params)
        stats['total_bits'] += packed['num_packed'] * 8 + 16  # data + scale

        if idx % 50 == 0 or idx < 5:
            print(f"Layer {idx}: {num_bits}-bit ternary, bpw={bpw:.4f}, shape={list(info['shape'])}")

        del W, data, packed
        gc.collect()

    stats['avg_bpw'] = stats['total_bits'] / stats['total_original']
    stats['compression'] = stats['total_original'] * 16 / stats['total_bits']

    # Count layers by bit usage
    bit_counts = {}
    for _, b in stats['layer_bits'].items():
        bit_counts[b] = bit_counts.get(b, 0) + 1

    print(f"\nResults:")
    print(f"  Avg bits/weight: {stats['avg_bpw']:.4f}")
    print(f"  Compression: {stats['compression']:.1f}x")
    print(f"  Layer distribution: {bit_counts}")

    # Estimate perplexity impact (rough guide)
    if stats['avg_bpw'] < 0.8:
        print(f"\n  WARNING: Very aggressive quantization (<1 bpw)")
        print(f"  Expected perplexity: Significant degradation likely")
        print(f"  Consider using this only if size is critical")
    elif stats['avg_bpw'] < 1.5:
        print(f"\n  Moderate compression - some quality loss expected")
    else:
        print(f"\n  Conservative compression - minimal quality impact expected")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'quantized': quantized,
        'stats': stats,
        'method': 'ternary_aggressive',
        'config': {'critical_bits': args.critical_bits, 'standard_bits': args.standard_bits}
    }, args.output)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    main()