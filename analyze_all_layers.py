import torch, os, glob

os.chdir(r'C:\Users\Zwmar\projects\sub1quant')

layers = sorted(glob.glob('quantized/checkpoints/layer_*.pt'))
print(f'Total checkpoint files: {len(layers)}')

# Check rank distribution across all 224 layers
ranks = []
for path in layers:
    ckpt = torch.load(path, weights_only=False)
    ranks.append(ckpt['rank'])

print(f'\nRank distribution:')
print(f'  Min rank: {min(ranks)}')
print(f'  Max rank: {max(ranks)}')
print(f'  Avg rank: {sum(ranks)/len(ranks):.0f}')
print(f'  Low rank (≤32): {sum(1 for r in ranks if r <= 32)}')
print(f'  High rank (>32): {sum(1 for r in ranks if r > 32)}')

# Show first 20 and last 5
print(f'\nFirst 20 layers:')
for i, path in enumerate(layers[:20]):
    ckpt = torch.load(path, weights_only=False)
    print(f'  {i:3d}: rank={ckpt["rank"]:5d}, shape={ckpt["original_shape"]}')

# Calculate total bits for FULL model at rank 16 vs current
print(f'\n=== Size Analysis ===')
total_params = 0
total_bits_rank16 = 0
total_bits_current = 0

for path in layers:
    ckpt = torch.load(path, weights_only=False)
    orig_shape = ckpt['original_shape']
    r = ckpt['rank']
    m, n = orig_shape

    orig_params = m * n
    total_params += orig_params

    # Bits at rank 16
    bits_rank16 = m * 16 * 0.5 + 16 * 2 + 16 * n * 0.5
    total_bits_rank16 += bits_rank16

    # Bits at current rank
    bits_current = m * r * 0.5 + r * 2 + r * n * 0.5
    total_bits_current += bits_current

print(f'Total model params: {total_params:,}')
print(f'At rank 16 everywhere: {total_bits_rank16/8/1024/1024:.2f} MB')
print(f'At current ranks: {total_bits_current/8/1024/1024:.2f} MB')