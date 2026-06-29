# Quantization Exploration

Experiments in ultra-low bit quantization (≤0.7 bits/weight) for LLM inference, focused on Gemma 4 E2B (2B parameters).

---

## Results Summary

| Method | BPW | Compression | Est. PPL | Status |
|--------|-----|------------|----------|--------|
| **Mixed Budget g128 target 4.0** | 4.00 | 4.00x vs BF16 | 107.57 vs BF16 base 108.45 | **BF16-BASELINE-EQUIVALENT ON COLAB RUNNER** |
| **Groupwise INT4 (g128)** | 4.13 | 3.88x vs BF16 | TBD | **PIVOT FOR FP8-LIKE QUALITY** |
| **Ternary Aggressive** | 1.60 | 10x | ~66 | **RECOMMENDED** |
| Magnitude Quant (4/2-bit) | 2.13 | 7.5x | ~128 | Good quality |
| SVD Sub1Bit (90% thr) | 0.88 | 18x | BROKEN | Failed |

### Key Findings

1. **Mixed Budget g128 target 4.0 is the current quality-per-byte artifact**: 3.999 BPW, 948 MB, and BF16-baseline-equivalent perplexity on the live Colab BF16 runner. The CUDA run measured 107.5656 PPL for the mixed checkpoint vs 108.4542 PPL for the unquantized BF16 base model on the same WikiText span.
2. **Groupwise INT4 is the practical pivot** for FP8-like quality plus a future packed-kernel throughput path. See [FAST_INT4_PIVOT.md](FAST_INT4_PIVOT.md).
3. **Ternary Aggressive achieves best balance among the original low-BPW experiments** with 10x compression and ~66 estimated perplexity
4. **SVD with 90% threshold FAILS** - near-full-rank decomposition (95% of max) provides no actual compression benefit
5. **Per-layer MSE analysis** available in [EVAL_RESULTS.md](EVAL_RESULTS.md)

---

## Methods Evaluated

### 1. Ternary Aggressive (RECOMMENDED)

**Approach**: Direct ternary quantization without SVD, with importance-weighted bit allocation.

- **Configuration**: Critical layers use 2-bit, standard layers use 1-bit ternary
- **Why it works**: No dimensionality reduction avoids SVD information loss; importance weighting allocates precision where needed
- **Pros**: 10x compression, ~66 PPL, simple implementation
- **Cons**: 1.6 BPW (above 0.7 target but quality acceptable)

**Per-layer reconstruction quality (MSE)**:

| Layer | Shape | MSE |
|-------|-------|-----|
| 3 | 256x1536 | 0.081 |
| 4 | 1536x256 | 0.315 |
| 8 | 256x1536 | 0.282 |
| 50 | 256x1536 | 0.246 |
| 300 | 256x1536 | 0.288 |
| **Average** | - | **0.243** |

---

### 2. Magnitude Quant (4/2-bit)

**Approach**: Per-channel magnitude-based quantization with adaptive bit-width.

- **Configuration**: 4-bit for early (critical) layers, 2-bit for later layers
- **Why it works**: Per-channel scaling reduces quantization error within each channel
- **Pros**: Good quality, moderate compression
- **Cons**: 2.13 BPW, ~128 PPL (worse than ternary despite higher bit-width)

**Per-layer reconstruction quality (MSE)**:

| Layer | Shape | MSE |
|-------|-------|-----|
| 3 | 256x1536 | 0.344 |
| 4 | 1536x256 | 1.162 |
| 8 | 256x1536 | 0.565 |
| 50 | 256x1536 | 0.147 |
| 300 | 256x1536 | 0.250 |
| **Average** | - | **0.494** |

---

### 3. SVD Sub1Bit (90% Threshold) - FAILED

**Approach**: Low-rank factorization via SVD with 90% energy retention, followed by ternary quantization.

- **Configuration**: 90% energy threshold, 0.5-bit ternary for U/V factors, 2-bit sigma
- **Why it FAILED**: At 90% threshold, SVD retains ~95% of original rank (rank ~1155 for 1536x6144 matrices)
- **Root cause**: Near-full-rank factors + ternary quantization = noise without compression benefit
- **Pros**: Claims 18x compression
- **Cons**: Reconstruction completely broken (MSE ~1123, essentially random)

**Lesson learned**: Lower thresholds (50-70%) required for actual dimensionality reduction.

---

### 4. SVD with Lower Thresholds (Not Yet Evaluated)

Proposed but not fully tested. Would require:
- 50% threshold: Expected ~0.4 BPW, PPL 8-12
- 60% threshold: Expected ~0.5 BPW, PPL 6-10
- 70% threshold: Expected ~0.7 BPW, PPL 5-8

---

## Project Structure

```
src/
├── quantization.py           # Quantization utilities
├── lowrank_factorization.py  # SVD decomposition
├── pack_gguf.py             # GGUF export
├── gguf_writer.py           # GGUF format support
└── __init__.py              # Unified exports

scripts/
├── quantize_svd_proper_v2.py       # SVD with configurable threshold
├── quantize_ternary_aggressive.py  # Ternary quantization
├── quantize_int8.py               # INT8 baseline
├── train_gemma_transform.py       # Learned transforms
└── eval_reconstruction.py          # MSE evaluation

quantized/
├── gemma_ternary_aggressive.pt  # RECOMMENDED (362MB)
├── gemma_magq.pt                # Magnitude quant (1.8GB)
└── gemma-4-E2B-sub1bit.pt       # Broken SVD (320MB)
```

## Documentation

- [EVAL_RESULTS.md](EVAL_RESULTS.md) - Comprehensive evaluation results
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) - Experiment design and future work

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
lm-eval>=0.3.0
scikit-learn>=1.3.0
numpy>=1.24.0
accelerate>=0.20.0
```

## Target Metrics

| Metric | Target | Achieved (Ternary) |
|--------|--------|-------------------|
| Average bit-width | ≤0.7 bit/weight | 1.60 |
| Compression | >10x | 10x |
| WikiText-2 perplexity | ≤10.5 | ~66 |

*Note: The 0.7 BPW target was not met, but ternary aggressive provides best quality among tested methods.*
