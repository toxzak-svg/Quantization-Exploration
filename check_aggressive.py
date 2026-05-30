import torch

# Check ternary aggressive structure
data = torch.load('quantized/gemma_ternary_aggressive.pt', map_location='cpu', weights_only=True)
q = data.get('quantized', {})

print('Keys in first entry:')
idx = list(q.keys())[0]
print(f'Layer {idx}:')
for k in q[idx].keys():
    v = q[idx][k]
    if hasattr(v, 'numel'):
        print(f'  {k}: {type(v).__name__} {v.shape}')
    else:
        print(f'  {k}: {type(v).__name__} = {v}')

# Calculate total size
total = 0
for idx, entry in q.items():
    for k, v in entry.items():
        if hasattr(v, 'numel'):
            total += v.numel() * v.element_size()

print(f'\nTotal: {total/1e6:.1f} MB')

# Check method
print(f'\nMethod: {data.get("method", "N/A")}')