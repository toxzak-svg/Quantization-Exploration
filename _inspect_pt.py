"""Inspect the quantized .pt structure without loading everything into memory."""
import torch, json, sys
PT = "/content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt"
print(f"loading {PT} (CPU map, weights_only=True)…")
data = torch.load(PT, map_location="cpu", weights_only=False)
print(f"top-level keys: {list(data.keys())}")
quantized = data.get("quantized")
print(f"quantized entries: {len(quantized) if isinstance(quantized, dict) else 'N/A'}")
if isinstance(quantized, dict):
    first3 = list(quantized.keys())[:3]
    print(f"first 3 keys: {first3}")
    e0 = quantized[first3[0]]
    print(f"sample entry keys: {sorted(e0.keys()) if isinstance(e0, dict) else type(e0)}")
    print(f"sample entry format tag: {e0.get('format') if isinstance(e0, dict) else None}")
    # Count format types
    from collections import Counter
    fmts = Counter()
    for k, v in quantized.items():
        if isinstance(v, dict):
            fmts[v.get("format", "(none)")] += 1
    print(f"format distribution: {dict(fmts)}")
print("\nweight_keys (first 5):")
wk = data.get("weight_keys") or []
if isinstance(wk, list):
    for w in wk[:5]:
        print(f"  {w}")
else:
    print(f"  type: {type(wk)}")
print(f"\nfull top-level metadata:")
for k, v in data.items():
    if k == "quantized":
        continue
    print(f"  {k}: {v if not isinstance(v, (list, dict)) else f'<{type(v).__name__} len={len(v)}>'}")
