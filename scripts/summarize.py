import torch

q = torch.load('quantized/gemma_hybrid_stream.pt', weights_only=True)
stats = q['stats']
print('=== Quantization Summary ===')
print('Method:', q['method'])
print('Layers:', stats['layer_count'])
print('Average bits/weight:', round(stats['avg_bpw'], 4))
print('Compression vs FP16:', round(stats['compression'], 1), 'x')
print('Bits distribution:', stats['bits_distribution'])

# Compare with original SVD approach
q2 = torch.load('quantized/gemma-4-E2B-sub1bit.pt', weights_only=True)
print()
print('=== Original SVD Sub1Bit (90% threshold) ===')
print('Layers:', len(q2['quantized']))
# Calculate avg bpw
total_bits = 0
total_orig = 0
for e in q2['quantized'].values():
    orig = e['original_shape'][0] * e['original_shape'][1]
    total_orig += orig
    u_bits = e['U_packed'].numel() * 8 * 0.625
    s_bits = e['S'].numel() * 2
    vt_bits = e['Vt_packed'].numel() * 8 * 0.625
    total_bits += u_bits + s_bits + vt_bits
print('Average bits/weight:', round(total_bits / total_orig, 4))
print('Compression:', round(total_orig * 16 / total_bits, 1), 'x')