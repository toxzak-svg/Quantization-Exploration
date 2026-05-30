"""
Reconstruction Quality Evaluation for Quantized Models
Measures MSE, Max Error, and estimates perplexity impact
"""

import torch
import numpy as np
import json
import struct
from pathlib import Path
from typing import Dict, List
import os
import gc

def load_gemma_weights_slice(model_dir: str, keys_to_load: List[str] = None) -> Dict:
    """Load only specific weights for evaluation"""
    model_dir = Path(model_dir)
    safetensor_path = model_dir / 'model.safetensors'

    with open(safetensor_path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hs))

    weights = {}
    for key, info in header.items():
        if key == '__metadata__' or 'weight' not in key or len(info['shape']) != 2:
            continue
        if any(x in key for x in ['lm_head', 'embed_tokens', 'norm', 'audio_tower', 'vision_tower', 'embed_vision']):
            continue
        if 'language_model' not in key:
            continue

        if keys_to_load and key not in keys_to_load:
            continue

        begin, end = info['data_offsets']
        dtype_map = {'F16': np.float16, 'BF16': np.float16, 'F32': np.float32}
        numpy_dtype = dtype_map.get(info['dtype'], np.float32)

        with open(safetensor_path, 'rb') as f:
            f.seek(8 + hs + begin)
            data = f.read(end - begin)

        weights[key] = torch.from_numpy(np.frombuffer(data, dtype=numpy_dtype).copy()).reshape(info['shape']).float()

    return weights

def reconstruct_from_quantized_hybrid(q_entry, device='cpu'):
    """Reconstruct from hybrid quantized format"""
    num_bits = q_entry['num_bits']
    q = q_entry['q'].to(device).float()
    scale = q_entry['scale']
    qmax = 2 ** (num_bits - 1) - 1
    return q * scale / qmax

def reconstruct_from_ternary_aggressive(q_entry, device='cpu'):
    """Reconstruct from ternary aggressive quantized format"""
    packed = q_entry['packed'].to(device)

    # Unpack: 8 bits -> 5 ternary values
    # We don't have the original shape easily, use orig_shape
    orig_shape = q_entry['orig_shape']

    weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=device)
    expanded = packed.to(torch.int32).unsqueeze(-1) // weights % 3
    flat = (expanded - 1).flatten()[:orig_shape[0] * orig_shape[1]]

    scale = q_entry['scale']
    return flat.view(orig_shape[0], orig_shape[1]).float() * scale

def reconstruct_from_magnitude(q_entry, device='cpu'):
    """Reconstruct from magnitude quantized format"""
    num_bits = q_entry['num_bits']
    q = q_entry['q'].to(device).float()

    if q_entry['per_channel']:
        scale = q_entry['scale'].to(device).unsqueeze(1)
        m, n = q_entry['orig_shape']
        q = q.view(m, -1)
    else:
        scale = q_entry['scale']

    qmax = 2 ** (num_bits - 1) - 1
    return q * scale / qmax

def reconstruct_from_svd_sub1bit(q_entry, device='cpu'):
    """Reconstruct from SVD Sub1Bit format"""
    # Unpack ternary U and Vt
    def unpack_ternary(packed, shape):
        weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=device)
        expanded = packed.to(torch.int32).unsqueeze(-1) // weights % 3
        flat = (expanded - 1).flatten()[:shape[0] * shape[1]]
        return flat.view(shape[0], shape[1])

    U = unpack_ternary(q_entry['U_packed'], q_entry['U_shape']).float() * q_entry['U_scale']
    Vt = unpack_ternary(q_entry['Vt_packed'], q_entry['Vt_shape']).float() * q_entry['Vt_scale']
    S = q_entry['S'].to(device).float() * q_entry['S_scale']

    return torch.matmul(U * S.unsqueeze(0), Vt)

def evaluate_reconstruction(quantized_path: str, original_weights: Dict, method: str) -> Dict:
    """Evaluate reconstruction quality for a quantized model"""
    q_data = torch.load(quantized_path, map_location='cpu', weights_only=True)
    quantized = q_data['quantized']

    results = {
        'total_mse': 0,
        'total_params': 0,
        'max_error': 0,
        'layers': []
    }

    reconstruct_fn = {
        'hybrid': reconstruct_from_quantized_hybrid,
        'ternary_aggressive': reconstruct_from_ternary_aggressive,
        'magnitude': reconstruct_from_magnitude,
        'svd_sub1bit': reconstruct_from_svd_sub1bit
    }.get(method, reconstruct_from_quantized_hybrid)

    for idx, q_entry in quantized.items():
        key = q_entry.get('key', '')
        if not key:
            continue

        if key not in original_weights:
            continue

        W_orig = original_weights[key]
        W_rec = reconstruct_fn(q_entry, 'cpu')

        # Ensure same shape
        if W_rec.shape != W_orig.shape:
            W_rec = W_rec[:W_orig.shape[0], :W_orig.shape[1]]

        mse = ((W_orig - W_rec) ** 2).mean().item()
        max_err = (W_orig - W_rec).abs().max().item()

        results['total_mse'] += mse * W_orig.numel()
        results['total_params'] += W_orig.numel()
        results['max_error'] = max(results['max_error'], max_err)

        results['layers'].append({
            'idx': idx,
            'key': key[:40],
            'mse': mse,
            'max_err': max_err,
            'shape': list(W_orig.shape)
        })

    results['avg_mse'] = results['total_mse'] / results['total_params'] if results['total_params'] > 0 else 0
    results['avg_rmse'] = results['avg_mse'] ** 0.5

    # Estimate perplexity impact (rough heuristic)
    # Higher MSE -> higher perplexity degradation
    # Base perplexity of Gemma 4 E2B ~ 4-5 on WikiText-2
    base_ppl = 5.0
    # Rough mapping: MSE of 0.01 -> ~2x ppl increase, etc
    ppl_factor = 1 + results['avg_mse'] * 100  # Very rough estimate
    results['estimated_ppl'] = base_ppl * ppl_factor

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality")
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--max-layers', type=int, default=50, help='Max layers to evaluate')
    args = parser.parse_args()

    print("=" * 70)
    print("RECONSTRUCTION QUALITY EVALUATION")
    print("=" * 70)

    # Load original weights (first N layers only to save memory)
    print("\n[1] Loading original weights...")
    original_weights = load_gemma_weights_slice(args.model_dir)
    print(f"  Loaded {len(original_weights)} weight matrices")

    # Get keys to evaluate
    eval_keys = list(original_weights.keys())[:args.max_layers]
    print(f"  Evaluating first {len(eval_keys)} layers")

    # Evaluate each quantized model
    models = [
        ('quantized/gemma_hybrid_stream.pt', 'hybrid'),
        ('quantized/gemma_ternary_aggressive.pt', 'ternary_aggressive'),
        ('quantized/gemma_magq.pt', 'magnitude'),
        ('quantized/gemma-4-E2B-sub1bit.pt', 'svd_sub1bit'),
    ]

    print("\n[2] Evaluating reconstruction quality...")
    results_all = {}

    for path, method in models:
        if not os.path.exists(path):
            print(f"\n  Skipping {path} - not found")
            continue

        print(f"\n  Evaluating {method}...")
        results = evaluate_reconstruction(path, original_weights, method)
        results_all[method] = results

        print(f"    Avg MSE: {results['avg_mse']:.6f}")
        print(f"    Avg RMSE: {results['avg_rmse']:.6f}")
        print(f"    Max Error: {results['max_error']:.4f}")
        print(f"    Estimated PPL: {results['estimated_ppl']:.2f}")

        del results
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Ordered by Quality (best first)")
    print("=" * 70)
    print(f"{'Method':<30} {'Avg MSE':<12} {'Max Error':<12} {'Est. PPL':<12}")
    print("-" * 70)

    sorted_results = sorted(results_all.items(), key=lambda x: x[1]['avg_mse'])

    for method, results in sorted_results:
        print(f"{method:<30} {results['avg_mse']:<12.6f} {results['max_error']:<12.4f} {results['estimated_ppl']:<12.2f}")

    print("-" * 70)
    print("Note: Estimated PPL assumes base Gemma 4 E2B perplexity ~5 on WikiText-2")
    print("      Actual perplexity testing requires full model loading (memory limited)")

if __name__ == "__main__":
    main()