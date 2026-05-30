import torch
from collections import Counter

pt_path = r'C:\Users\Zwmar\projects\sub1quant\llama-2-7b-sub1bit.gguf.pt'
d = torch.load(pt_path, map_location='cpu', weights_only=True)

q = d['quantized']
print(f'Total quantized entries: {len(q)}')
print(f'Keys in pt: {list(d.keys())}')

if 'weight_keys' in d:
    wk = d['weight_keys']
    print(f'Weight keys count: {len(wk)}')
    for i, (name, shape) in enumerate(wk):
        print(f'  [{i}] {name}: {shape}')
else:
    print('No weight_keys found')

print()
shape_counts = Counter()
for idx, entry in q.items():
    shape_counts[tuple(entry['original_shape'])] += 1
print('Original shape distribution:')
for shape, count in shape_counts.most_common():
    print(f'  {shape}: {count}')

# Check if U/Vt data looks like actual quantized ternary data
q0 = q[0]
U_packed = q0['U_packed']
print(f'\nU_packed[0]: shape={U_packed.shape}, dtype={U_packed.dtype}')
print(f'First 30 bytes: {U_packed[:30].tolist()}')
print(f'Unique values count: {len(U_packed.unique())}')
uv = U_packed.unique().tolist()
print(f'First 20 unique values: {sorted(uv)[:20]}')

Vt_packed = q0['Vt_packed'] 
print(f'\nVt_packed[0]: shape={Vt_packed.shape}, dtype={Vt_packed.dtype}')
print(f'First 30 bytes: {Vt_packed[:30].tolist()}')
print(f'Unique values count: {len(Vt_packed.unique())}')
vv = Vt_packed.unique().tolist()
print(f'First 20 unique values: {sorted(vv)[:20]}')

S = q0['S']
print(f'\nS[0]: shape={S.shape}, dtype={S.dtype}')
print(f'First 30 bytes: {S[:30].tolist()}')
print(f'Unique values: {S.unique().tolist()}')
