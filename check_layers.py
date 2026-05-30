import torch

data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = data.get('quantized', {})

# Check layer types
layers_with_keys = [(idx, q[idx].get('key', '')) for idx in q.keys()]
print(f'Total entries: {len(layers_with_keys)}')

# Check keys
for idx, key in list(layers_with_keys)[:10]:
    print(f'  Layer {idx}: {key}')