"""
Better Reconstruction Quality Evaluation
"""

import torch
import numpy as np
import json
import struct
from pathlib import Path
from typing import Dict, Tuple
import os
import gc

def load_single_weight(model_dir: str, key: str) -> torch.Tensor:
    """Load a single weight by key"""
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

def test_layer_reconstruction(layer_idx: int, q_entry: Dict, original_key: str, method: str) -> Dict:
    """Test reconstruction for a single layer"""
    W_orig = load_single_weight('models/gemma-4-E2B', original_key)

    # Reconstruct based on method
    if method == 'hybrid':
        num_bits = q_entry['num_bits']
        q = q_entry['q'].float()
        scale = q_entry['scale']
        qmax = 2 ** (num_bits - 1) - 1
        W_rec = q * scale / qmax

    elif method == 'magnitude':
        num_bits = q_entry['num_bits']
        q = q_entry['q'].float()
        if q_entry['per_channel']:
            scale = q_entry['scale'].unsqueeze(1)
        else:
            scale = q_entry['scale']
        qmax = 2 ** (num_bits - 1) - 1
        W_rec = q * scale / qmax

    elif method == 'ternary_aggressive':
        packed = q_entry['packed']
        weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
        expanded = packed.to(torch.int32).unsqueeze(-1) // weights % 3
        flat = (expanded - 1).flatten()[:q_entry['orig_shape'][0] * q_entry['orig_shape'][1]]
        W_rec = flat.view(q_entry['orig_shape'][0], q_entry['orig_shape'][1]).float() * q_entry['scale']

    elif method == 'svd_sub1bit':
        def unpack_ternary(packed, shape):
            weights = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
            expanded = packed.to(torch.int32).unsqueeze(-1) // weights % 3
            flat = (expanded - 1).flatten()[:shape[0] * shape[1]]
            return flat.view(shape[0], shape[1]).float()

        U = unpack_ternary(q_entry['U_packed'], q_entry['U_shape']) * q_entry['U_scale']
        Vt = unpack_ternary(q_entry['Vt_packed'], q_entry['Vt_shape']) * q_entry['Vt_scale']
        S = q_entry['S'].float() * q_entry['S_scale']
        W_rec = torch.matmul(U * S.unsqueeze(0), Vt)
    else:
        return None

    # Ensure shapes match
    if W_rec.shape != W_orig.shape:
        W_rec = W_rec[:W_orig.shape[0], :W_orig.shape[1]]

    mse = ((W_orig - W_rec) ** 2).mean().item()
    max_err = (W_orig - W_rec).abs().max().item()

    # Relative error
    rel_err = (W_orig - W_rec).abs().mean().item() / (W_orig.abs().mean().item() + 1e-8)

    return {
        'mse': mse,
        'max_err': max_err,
        'rel_err': rel_err,
        'shape': list(W_orig.shape),
        'key': original_key
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--max-layers', type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("RECONSTRUCTION QUALITY EVALUATION")
    print("=" * 70)

    models = [
        ('quantized/gemma-4-E2B-sub1bit.pt', 'svd_sub1bit'),
        ('quantized/gemma_magq.pt', 'magnitude'),
        ('quantized/gemma_ternary_aggressive.pt', 'ternary_aggressive'),
        ('quantized/gemma_hybrid_stream.pt', 'hybrid'),
    ]

    # Test on specific layers
    test_layers = [0, 1, 2, 3, 4, 10, 50, 100, 150, 200]

    results_all = {}

    for path, method in models:
        if not os.path.exists(path):
            print(f"\nSkipping {path} - not found")
            continue

        print(f"\n{method}:")
        print("-" * 50)

        q_data = torch.load(path, map_location='cpu', weights_only=True)
        quantized = q_data['quantized']

        total_mse = 0
        total_params = 0
        max_mse = 0
        worst_layer = None

        for layer_idx in test_layers:
            if layer_idx >= len(quantized):
                continue

            q_entry = quantized[layer_idx]
            key = q_entry.get('key', '')

            if not key:
                continue

            try:
                result = test_layer_reconstruction(layer_idx, q_entry, key, method)

                total_mse += result['mse'] * result['shape'][0] * result['shape'][1]
                total_params += result['shape'][0] * result['shape'][1]

                if result['mse'] > max_mse:
                    max_mse = result['mse']
                    worst_layer = layer_idx

                print(f"  Layer {layer_idx:3d} ({result['shape'][0]:5d}x{result['shape'][1]:5d}): MSE={result['mse']:.6f}, MaxErr={result['max_err']:.4f}, RelErr={result['rel_err']:.4f}")

            except Exception as e:
                print(f"  Layer {layer_idx}: Error - {e}")

        if total_params > 0:
            avg_mse = total_mse / total_params
            print(f"\n  Overall Avg MSE: {avg_mse:.6f}")
            print(f"  Worst layer: {worst_layer} (MSE={max_mse:.6f})")

            # Estimate perplexity
            base_ppl = 5.0
            # Very rough: each 0.01 MSE roughly doubles perplexity at high error rates
            ppl_estimate = base_ppl * (1 + avg_mse * 50)
            print(f"  Estimated PPL: {ppl_estimate:.1f}")

            results_all[method] = {
                'avg_mse': avg_mse,
                'max_mse': max_mse,
                'worst_layer': worst_layer,
                'est_ppl': ppl_estimate
            }

    print("\n" + "=" * 70)
    print("SUMMARY (ordered by quality)")
    print("=" * 70)

    sorted_results = sorted(results_all.items(), key=lambda x: x[1]['avg_mse'])

    print(f"{'Method':<25} {'Avg MSE':<12} {'Est PPL':<12}")
    print("-" * 50)
    for method, results in sorted_results:
        print(f"{method:<25} {results['avg_mse']:<12.6f} {results['est_ppl']:<12.1f}")

    print("-" * 50)
    print("Note: Testing on selected layers (0-4 important, 10-200 regular)")
    print("Base Gemma 4 E2B perplexity on WikiText-2: ~5.0")
    print("Target: PPL <= 10.5")

if __name__ == "__main__":
    main()