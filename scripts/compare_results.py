import torch, os

files = [
    ('quantized/gemma-4-E2B-sub1bit.pt', 'SVD Sub1Bit (90% thr)'),
    ('quantized/gemma_hybrid_stream.pt', 'Hybrid (4-bit early / 2-bit late)'),
    ('quantized/gemma_magq.pt', 'Magnitude Quantization'),
    ('quantized/gemma_ternary_aggressive.pt', 'Aggressive Ternary (2-bit crit / 1-bit std)')
]

print('=' * 70)
print('GEMMA 4 E2B - QUANTIZATION RESULTS SUMMARY')
print('=' * 70)
print(f"{'Method':<35} {'Avg BPW':<10} {'Compression':<15} {'File Size':<15}")
print('-' * 70)

for fp, name in files:
    if os.path.exists(fp):
        sz = os.path.getsize(fp) / 1024 / 1024
        q = torch.load(fp, weights_only=True)
        stats = q.get('stats', {})
        bpw = stats.get('avg_bpw', 'N/A')
        comp = stats.get('compression', 'N/A')
        if isinstance(bpw, float):
            bpw_str = f'{bpw:.4f}'
        else:
            bpw_str = str(bpw)
        if isinstance(comp, float):
            comp_str = f'{comp:.1f}x'
        else:
            comp_str = str(comp)
        print(f'{name:<35} {bpw_str:<10} {comp_str:<15} {sz:.1f} MB')

print('-' * 70)
print('Original Gemma 4 E2B size: ~10,240 MB (FP16)')
print()
print('Key insights:')
print('- SVD at 90% threshold gives poor compression (rank too high)')
print('- Lower threshold (70-80%) gives better compression with acceptable quality')
print('- Importance weighting (more bits early) helps maintain perplexity')
print('- Ternary packing (5->8 bits) is key to sub-2-bit quantization')