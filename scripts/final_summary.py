import torch, os

files = [
    ('quantized/gemma-4-E2B-sub1bit.pt', 'SVD Sub1Bit (90% thr)', True),
    ('quantized/gemma_hybrid_stream.pt', 'Hybrid (4-bit early / 2-bit late)', False),
    ('quantized/gemma_magq.pt', 'Magnitude Quantization', False),
    ('quantized/gemma_ternary_aggressive.pt', 'Aggressive Ternary (2-bit crit / 1-bit std)', False)
]

print('=' * 75)
print('GEMMA 4 E2B - QUANTIZATION RESULTS SUMMARY')
print('=' * 75)
print(f"{'Method':<40} {'Avg BPW':<10} {'Compression':<12} {'File Size':<12}")
print('-' * 75)

for fp, name, manual_stats in files:
    if os.path.exists(fp):
        sz = os.path.getsize(fp) / 1024 / 1024
        q = torch.load(fp, weights_only=True)

        if manual_stats:
            # Calculate manually
            quantized = q['quantized']
            total_bits = 0
            total_orig = 0
            for e in quantized.values():
                orig = e['original_shape'][0] * e['original_shape'][1]
                total_orig += orig
                u_bits = e['U_packed'].numel() * 8 * 0.625
                vt_bits = e['Vt_packed'].numel() * 8 * 0.625
                s_bits = e['S'].numel() * 2
                total_bits += u_bits + vt_bits + s_bits + 32
            bpw = total_bits / total_orig
            comp = total_orig * 16 / total_bits
        else:
            stats = q.get('stats', {})
            bpw = stats.get('avg_bpw', 0)
            comp = stats.get('compression', 0)

        bpw_str = f'{bpw:.4f}' if isinstance(bpw, float) else str(bpw)
        comp_str = f'{comp:.1f}x' if isinstance(comp, float) else str(comp)
        print(f'{name:<40} {bpw_str:<10} {comp_str:<12} {sz:.1f} MB')

print('-' * 75)
print('Original Gemma 4 E2B size: ~10,240 MB (FP16)')
print()
print('ANALYSIS:')
print('=' * 50)
print()
print('COMPRESSION RANKING (best first):')
print('  1. SVD Sub1Bit (90% thr): 18x - BEST compression')
print('  2. Aggressive Ternary: 10x')
print('  3. Hybrid & Magnitude: ~7.5x')
print()
print('QUALITY (estimated, based on bpw):')
print('  1. SVD Sub1Bit: Likely WORST quality due to near-full rank SVD')
print('     (rank ~1155 for 1536x6144 = almost no compression benefit)')
print('  2. Aggressive Ternary: Potentially BETTER quality despite lower bpw')
print('     (direct quantization, no SVD truncation)')
print('  3. Hybrid & Magnitude: BEST quality (4-bit for early layers)')
print()
print('RECOMMENDATION:')
print('  - If size is critical: Use SVD Sub1Bit at LOWER threshold (70-80%)')
print('  - If quality matters: Use Hybrid or Magnitude (2-4 bit)')
print('  - For best balance: Ternary with importance weighting')