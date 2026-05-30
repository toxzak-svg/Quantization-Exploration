# Gemma 4 E2B Quantization - Full Experiment Plan

## Goal
Get the best compression vs quality tradeoff for Gemma 4 E2B with sub-1-bit quantization, targeting PPL < 10.5 on WikiText-2.

---

## Methods to Test

### 1. SVD Proper (Low Threshold)
**Innovation:** Use 50-60% energy threshold instead of 90% to get actual compression.

```bash
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_50.pt --threshold 0.50
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_60.pt --threshold 0.60
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_70.pt --threshold 0.70
```

### 2. Ternary Aggressive (1-bit standard, 2-bit critical)
**Innovation:** Direct ternary quantization without SVD, importance-weighted bits.

```bash
python scripts/quantize_ternary_aggressive.py --model-dir models/gemma-4-E2B --output quantized/gemma_ternary_aggressive.pt --critical-bits 2 --standard-bits 1
```

### 3. INT8 Per-Channel (Conservative)
**Innovation:** Simple but effective - per-channel scales, minimal quality loss.

```bash
python scripts/quantize_int8.py --model-dir models/gemma-4-E2B --output quantized/gemma_int8.pt
```

### 4. Learned Transform + Quantization (Most Innovative)
**Innovation:** Learn a rotation/transformation that makes weights more compressible before quantization.

```bash
python scripts/train_gemma_transform.py --model-dir models/gemma-4-E2B --output quantized/gemma_transforms.pt --epochs 10 --codebook-dim 64
```

---

## Evaluation Script

After running all quantizations, evaluate reconstruction quality:

```bash
python scripts/eval_reconstruction.py --model-dir models/gemma-4-E2B --max-layers 50
```

---

## Files to Run (in order)

1. **quantize_svd_proper_v2.py** - SVD with different thresholds (50%, 60%, 70%)
2. **quantize_ternary_aggressive.py** - Ternary approach
3. **quantize_int8.py** - Conservative INT8
4. **train_gemma_transform.py** - Learnable transforms (takes longest)

---

## Expected Results

| Method | Target BPW | Expected PPL | Notes |
|--------|-----------|-------------|-------|
| SVD 50% | ~0.4 | 8-12 | Aggressive, near target |
| SVD 60% | ~0.5 | 6-10 | Good balance |
| SVD 70% | ~0.7 | 5-8 | Conservative |
| Ternary Aggressive | 1.6 | 50-100 | Direct quantization |
| INT8 | 8.0 | 6-8 | Minimal loss |
| Learned Transform | ? | TBD | Most innovative |

---

## Key Insight

**The 90% threshold was too high** - it gives near-full-rank SVD (~96% of max rank), which means:
- Almost no compression from SVD
- Ternary quantization on high-rank factors adds noise
- Result: broken reconstruction

**Lower thresholds (50-70%) force actual dimensionality reduction**, giving real compression benefits while keeping reconstruction quality acceptable.

---

## To Run After Restart

```powershell
# 1. SVD at 50%
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_50.pt --threshold 0.50

# 2. SVD at 60%
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_60.pt --threshold 0.60

# 3. SVD at 70%
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_70.pt --threshold 0.70

# 4. Ternary aggressive
python scripts/quantize_ternary_aggressive.py --model-dir models/gemma-4-E2B --output quantized/gemma_ternary_aggressive.pt --critical-bits 2 --standard-bits 1

# 5. INT8
python scripts/quantize_int8.py --model-dir models/gemma-4-E2B --output quantized/gemma_int8.pt

# Then evaluate
python scripts/eval_reconstruction.py --model-dir models/gemma-4-E2B --max-layers 50
```