import torch

# Check all quantized files
files = [
    'quantized/gemma-4-E2B-sub1bit.pt',
    'quantized/gemma_ternary_aggressive.pt',
]

for f in files:
    try:
        data = torch.load(f, map_location='cpu', weights_only=True)
        q = data.get('quantized', {})
        stats = data.get('stats', {})
        config = data.get('config', {})

        print(f'{f}:')
        print(f'  Layers: {len(q)}')
        if 'avg_bpw' in stats:
            print(f'  Avg bits/weight: {stats["avg_bpw"]:.2f}')
        if 'compression' in stats:
            print(f'  Compression: {stats["compression"]:.1f}x')

        # Get first layer to see dimensions
        if q:
            idx = list(q.keys())[0]
            e = q[idx]
            print(f'  Layer {idx} U_shape: {e.get("U_shape", "N/A")}')
            print(f'  Layer {idx} Vt_shape: {e.get("Vt_shape", "N/A")}')

        # Calc packed size
        total_packed = sum(entry['U_packed'].numel() + entry['Vt_packed'].numel() for entry in q.values())
        print(f'  Total packed: {total_packed/1e6:.1f} MB')
        print()
    except Exception as ex:
        print(f'{f}: ERROR - {ex}')
        print()