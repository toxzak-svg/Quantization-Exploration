import torch

# Check gemma-4-E2B-sub1bit.pt structure more closely
data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = data.get('quantized', {})

print(f'Total layers: {len(q)}')

# Sum up all packed data
total_packed = 0
total_U_elements = 0
total_Vt_elements = 0

for idx, entry in q.items():
    u_packed = entry['U_packed'].numel()
    vt_packed = entry['Vt_packed'].numel()
    total_packed += u_packed + vt_packed
    total_U_elements += entry['U_shape'][0] * entry['U_shape'][1]
    total_Vt_elements += entry['Vt_shape'][0] * entry['Vt_shape'][1]

print(f'Total U elements: {total_U_elements:,}')
print(f'Total Vt elements: {total_Vt_elements:,}')
print(f'Total packed bytes: {total_packed:,} ({total_packed/1e6:.1f} MB)')

# Original fp16 would be
orig_fp16 = (total_U_elements + total_Vt_elements) * 2
print(f'Original fp16: {orig_fp16/1e6:.1f} MB')
print(f'Compression: {orig_fp16 / total_packed:.1f}x')

# Now estimate GGUF overhead
# GGUF needs: scales (4 bytes * 5 * 316 layers), header, tensor metadata
scales = len(q) * 5 * 4  # U_scale, Vt_scale, S_scale (3) + maybe rank
gguf_overhead = scales + 1024 * 1024  # ~1MB for headers/metadata
print(f'Estimated GGUF size: {(total_packed + gguf_overhead)/1e6:.1f} MB')

# Check config
print(f'\nConfig: {data.get("config", {})}')
print(f'Stats: {data.get("stats", {})}')