import torch, os

os.chdir(r'C:\Users\Zwmar\projects\sub1quant')

# Check existing checkpoint ranks
for i in [0, 1, 55, 111]:
    ckpt = torch.load(f'quantized/checkpoints/layer_{i:04d}.pt', weights_only=False)
    print(f'Layer {i:03d}: rank={ckpt["rank"]}, original_shape={ckpt["original_shape"]}, energy={ckpt["energy_captured"]:.4f}')

print()

# Calculate total size if we had 224 layers all at rank 16
# The checkpoint folder has 224 layers but only some are at low rank
import glob
layers = sorted(glob.glob('quantized/checkpoints/layer_*.pt'))
print(f'Total checkpoint files: {len(layers)}')

# Sample a few to understand the distribution
total_params = 0
total_factor_params = 0
total_bits = 0

for path in layers[:10]:
    ckpt = torch.load(path, weights_only=False)
    r = ckpt['rank']
    orig_shape = ckpt['original_shape']
    m, n = orig_shape

    factor_params = m * r + r + r * n
    bits = m * r * 0.5 + r * 2 + r * n * 0.5

    total_params += m * n
    total_factor_params += factor_params
    total_bits += bits

print(f'Sample of 10 layers:')
print(f'  Original params: {total_params:,}')
print(f'  Factor params @ rank 16: {total_factor_params:,}')
print(f'  Total bits: {total_bits:,}')
print(f'  Avg bits/param: {total_bits/total_factor_params:.4f}')
print(f'  Size: {total_bits/8/1024:.1f} KB')