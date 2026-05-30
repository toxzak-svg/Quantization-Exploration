import torch
import json

config_path = r'C:\Users\Zwmar\projects\sub1quant\models\llama-2-7b-hf\config.json'
with open(config_path) as f:
    config = json.load(f)
print('Model Config:')
for k, v in config.items():
    print(f'  {k}: {v}')

pt_path = r'C:\Users\Zwmar\projects\sub1quant\llama-2-7b-sub1bit.gguf.pt'
d = torch.load(pt_path, map_location='cpu', weights_only=True)

quantized = d['quantized']
print(f'\nQuantized entries: {len(quantized)}')
print(f'Config keys: {list(d.keys())}')

for idx in [0, 55]:
    q = quantized[idx]
    print(f'\nEntry {idx}:')
    for k, v in q.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}')
        else:
            print(f'  {k}: {v}')

if 'weight_keys' in d:
    print(f'\nWeight keys ({len(d["weight_keys"])}):')
    for name, shape in d['weight_keys'][:10]:
        print(f'  {name}: {shape}')

# Check the U_packed values more carefully
q0 = quantized[0]
U_packed = q0['U_packed']
print(f'\nU_packed[0]: shape={U_packed.shape}')
print(f'First 20 bytes: {U_packed[:20].tolist()}')
print(f'Unique values: {U_packed.unique()[:20].tolist()}')
