import torch
import os
import gc
import sys

os.chdir(r'C:\Users\Zwmar\projects\sub1quant')
sys.path.insert(0, r'.\src')

from quantize import SubOneBitQuantizer

def quantize_binary(x):
    scale = x.abs().max()
    if scale == 0:
        scale = 1.0
    normalized = x / scale
    binary = torch.sign(normalized)
    binary[binary == 0] = 1
    return binary.to(torch.int8), scale.item()

def quantize_sigma(sigma, num_bits=2):
    max_val = sigma.abs().max()
    if max_val == 0:
        max_val = 1.0
    scale = max_val / (2 ** (num_bits - 1) - 1)
    normalized = (sigma / scale).round().clamp(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return normalized.to(torch.int8), scale.item()

print("=" * 60)
print("Re-quantizing all layers with rank=16 binary encoding")
print("=" * 60)

device = 'cpu'
max_rank = 16

quantizer = SubOneBitQuantizer(r'.\models\llama-2-7b-hf', device=device)
quantizer.load_model_weights()

output_dir = r'.\quantized\rank16_full'
os.makedirs(output_dir, exist_ok=True)

total_bits = 0
total_factor_params = 0

for idx in range(quantizer.total_weights):
    W = quantizer.load_weight(idx)
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)

    r = min(max_rank, len(S))
    U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]

    q_U, scale_U = quantize_binary(U_r)
    q_S, scale_S = quantize_sigma(S_r)
    q_Vt, scale_Vt = quantize_binary(Vt_r)

    n_U = U_r.shape[0] * U_r.shape[1]
    n_S = S_r.shape[0]
    n_Vt = Vt_r.shape[0] * Vt_r.shape[1]

    bits = n_U * 0.5 + n_S * 2 + n_Vt * 0.5
    total_bits += bits
    total_factor_params += n_U + n_S + n_Vt

    entry = {
        'U': q_U.numpy(),
        'S': q_S.numpy(),
        'Vt': q_Vt.numpy(),
        'U_shape': list(U_r.shape),
        'S_shape': list(S_r.shape),
        'Vt_shape': list(Vt_r.shape),
        'rank': r,
        'original_shape': list(W.shape),
        'scale_U': scale_U,
        'scale_S': scale_S,
        'scale_Vt': scale_Vt,
    }

    torch.save(entry, f'{output_dir}\\layer_{idx:04d}.pt')

    if idx % 50 == 0 or idx < 5:
        energy = float((S[:r]**2).sum() / (S**2).sum())
        bpp = bits / (n_U + n_S + n_Vt)
        print(f'Layer {idx:3d}: shape={W.shape}, energy@rank={energy:.4f}, bits/param={bpp:.3f}')

    del U, S, Vt, W, U_r, S_r, Vt_r, q_U, q_S, q_Vt
    gc.collect()

print(f'\n=== Summary ===')
print(f'Layers processed: {quantizer.total_weights}')
print(f'Total bits: {total_bits:,}')
print(f'Avg bits/param: {total_bits/total_factor_params:.4f}')
print(f'Estimated size: {total_bits/8/1024/1024:.2f} MB')

torch.save({
    'quantized': {i: torch.load(f'{output_dir}\\layer_{i:04d}.pt') for i in range(quantizer.total_weights)},
    'config': {'max_rank': max_rank, 'encoding': 'binary'}
}, f'{output_dir}\\model.pt')

print(f'\nSaved to {output_dir}\\model.pt')