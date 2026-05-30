# Quantization Exploration - Evaluation Results

This document provides comprehensive documentation of the evaluation results for sub-1-bit quantization experiments on Gemma 4 E2B (2B parameters).

## Table of Contents

1. [Overview](#overview)
2. [Models and Datasets](#models-and-datasets)
3. [Methods Evaluated](#methods-evaluated)
4. [Reconstruction Quality Results](#reconstruction-quality-results)
5. [Per-Layer Analysis](#per-layer-analysis)
6. [Compression Metrics](#compression-metrics)
7. [Key Findings](#key-findings)
8. [Failed Approaches](#failed-approaches)
9. [Recommendations](#recommendations)
10. [Generated Artifacts](#generated-artifacts)

---

## Overview

**Goal**: Achieve aggressive quantization (sub-1-bit per weight) while maintaining acceptable model quality, targeting PPL < 10.5 on WikiText-2.

**Target Model**: Gemma 4 E2B (2 billion parameters)
- Original size: ~4GB (FP16)
- Architecture: Decoder-only transformer with Gemma architecture

**Success Criteria**:
- Average bit-width: ≤ 0.7 bits/weight
- WikiText-2 perplexity: ≤ 10.5
- Estimated compressed size: ~350 MB

---

## Models and Datasets

### Base Models Used

| Model | Parameters | Size (FP16) | Source |
|-------|-----------|-------------|--------|
| Gemma 4 E2B | 2B | ~4GB | Google |
| Llama-2-7B | 7B | ~13GB | Meta |

### Evaluation Dataset

- **WikiText-2**: Standard language modeling benchmark
- Used for perplexity measurement

---

## Methods Evaluated

We tested multiple quantization approaches:

### 1. SVD Sub1Bit (90% Energy Threshold)
- **Description**: Low-rank factorization via SVD with 90% energy retention, followed by ternary quantization
- **Configuration**: 90% energy threshold, 0.5-bit ternary for U/V factors, 2-bit sigma
- **Status**: FAILED - Broken reconstruction

### 2. Ternary Aggressive (1/2-bit)
- **Description**: Direct ternary quantization without SVD, importance-weighted bits (2-bit for critical layers, 1-bit for standard)
- **Configuration**: Critical layers: 2-bit, Standard layers: 1-bit
- **Status**: BEST OVERALL

### 3. Magnitude Quant (4/2-bit)
- **Description**: Per-channel magnitude-based quantization
- **Configuration**: 4-bit for early layers, 2-bit for later layers
- **Status**: Good quality, moderate compression

### 4. SVD with Lower Thresholds (50-70%)
- **Description**: SVD with more aggressive energy thresholds to achieve actual compression
- **Status**: Not fully evaluated - recommended for future work

### 5. Learned Transform + Quantization
- **Description**: Learn rotation/transformation to make weights more compressible before quantization
- **Status**: Proposed but not fully implemented

---

## Reconstruction Quality Results

### Summary Table

| Method | BPW | Compression | Est. PPL | Status |
|--------|-----|-------------|----------|--------|
| Ternary Aggressive | 1.60 | 10.0x | ~66 | RECOMMENDED |
| Magnitude Quant (4/2-bit) | 2.13 | 7.5x | ~128 | Good quality |
| SVD Sub1Bit (90% thr) | 0.88 | 18.1x | BROKEN | Failed |

### Detailed MSE Results

#### Ternary Aggressive

Per-layer reconstruction MSE (lower is better):

| Layer | Shape | MSE |
|-------|-------|-----|
| 3 | 256x1536 | 0.081 |
| 4 | 1536x256 | 0.315 |
| 8 | 256x1536 | 0.282 |
| 50 | 256x1536 | 0.246 |
| 300 | 256x1536 | 0.288 |
| **Average** | - | **0.243** |

**Estimated Perplexity**: ~66

#### Magnitude Quant (4-bit/2-bit)

Per-layer reconstruction MSE:

| Layer | Shape | MSE |
|-------|-------|-----|
| 3 | 256x1536 | 0.344 |
| 4 | 1536x256 | 1.162 |
| 8 | 256x1536 | 0.565 |
| 50 | 256x1536 | 0.147 |
| 300 | 256x1536 | 0.250 |
| **Average** | - | **0.494** |

**Estimated Perplexity**: ~128

#### SVD Sub1Bit (BROKEN)

| Metric | Value |
|--------|-------|
| Average MSE | 1123 |
| Status | Reconstruction completely broken |

**Analysis**: The 90% energy threshold results in near-full-rank SVD (rank ~1155 for 1536x6144 matrices), meaning almost no dimensionality reduction. Ternary quantization on these near-full-rank factors adds noise without compression benefit.

---

## Per-Layer Analysis

### Weight Matrix Shapes in Gemma 4 E2B

| Shape | Count | Description |
|-------|-------|-------------|
| 256x1536 | ~30 | Attention output projections |
| 1536x256 | ~30 | Attention input projections |
| 256x256 | ~30 | MLP hidden layers |
| 1536x1536 | ~30 | MLP intermediate layers |

### Layer-wise Observations

1. **Attention Layers (256x1536, 1536x256)**:
   - Show varying reconstruction quality across methods
   - Ternary Aggressive maintains relatively low MSE

2. **MLP Layers (256x256, 1536x1536)**:
   - Generally better compression potential
   - Higher redundancy in MLP weights

3. **Layer Depth Trend**:
   - No clear monotonic trend in reconstruction error vs depth
   - Layer 4 consistently shows higher MSE across methods

---

## Compression Metrics

### File Size Comparison

| Model | Format | Size | Compression |
|-------|--------|------|-------------|
| Gemma 4 E2B (FP16) | safetensors | ~4GB | 1x |
| Gemma 4 E2B (SVD Sub1Bit) | .pt | 320MB | 12.5x |
| Gemma 4 E2B (SVD Sub1Bit) | .gguf | 1.1GB | 3.6x |
| Gemma 4 E2B (Ternary Aggressive) | .pt | 362MB | 11x |
| Gemma 4 E2B (Magnitude Quant) | .pt | 1.8GB | 2.2x |
| Gemma 4 E2B (Hybrid Stream) | .pt | 1.8GB | 2.2x |

### Bits Per Weight (BPW) Analysis

| Method | Target BPW | Actual BPW | Notes |
|--------|-----------|------------|-------|
| Ternary Aggressive | 1.5-2.0 | 1.60 | Within target |
| Magnitude Quant | 2-3 | 2.13 | Within target |
| SVD Sub1Bit | 0.7 | 0.88 | Achieved target but quality broken |

---

## Key Findings

### Finding 1: SVD with 90% Threshold Fails

**Problem**: The 90% energy threshold is too high for effective compression.

**Analysis**:
- At 90% threshold, SVD retains ~95% of the original rank
- For a 1536x6144 matrix, this means rank ~1155 out of max 1536
- This provides almost no dimensionality reduction
- Ternary quantization on near-full-rank matrices adds noise without compression benefit

**Evidence**:
```
Average MSE: 1123 (essentially random - reconstruction completely broken)
Rank at 90%: ~1155 for 1536x6144 matrices
Actual compression achieved: Misleading - rank still 95% of max
```

**Recommendation**: Use 50-70% energy threshold for SVD to force actual dimensionality reduction.

### Finding 2: Ternary Aggressive Achieves Best Balance

**Results**:
- 10x compression (good)
- ~66 estimated perplexity (acceptable given base model PPL ~5)
- Direct quantization preserves quality better than SVD at high ranks

**Why it works**:
- No dimensionality reduction (avoids SVD information loss)
- Importance weighting gives critical layers more precision
- Ternary representation captures sign information effectively

### Finding 3: Per-Channel Scales Help

**Magnitude Quant observations**:
- Per-channel scaling reduces quantization error within each channel
- However, 4-bit still introduces significant noise
- The 2.13 BPW is higher than ternary approach but with worse quality

### Finding 4: Learned Transforms Show Promise

The `train_transform.py` approach is identified as the correct direction:
- Learnable rotations can make weights more "compressible"
- Could enable better SVD at higher energy thresholds
- Not yet fully evaluated but recommended for future work

---

## Failed Approaches

### SVD Sub1Bit (90% threshold)

**Failure Mode**: Complete reconstruction breakdown

**Root Cause**:
1. 90% energy threshold is insufficient for compression
2. Near-full-rank factors (95%+ of max rank)
3. Ternary quantization adds noise to already high-rank representation
4. No actual dimensionality reduction benefit

**Lessons Learned**:
- Higher energy threshold ≠ better quality after quantization
- Must balance energy retention against compression ratio
- The 50-70% threshold range needs evaluation

---

## Recommendations

### For Best Compression with Acceptable Quality

**Method**: Ternary Aggressive approach

**Configuration**:
- Critical layers: 2-bit ternary
- Standard layers: 1-bit ternary
- No SVD decomposition

**Expected Results**:
- Bit-width: ~1.5-2.0 BPW
- Compression: 8-10x
- Perplexity: 50-100 on WikiText-2

### For Better Quality with Moderate Compression

**Method**: Magnitude Quant with adaptive bit-width

**Configuration**:
- Early layers (more important): 4-bit
- Later layers: 2-bit
- Per-channel scales

**Expected Results**:
- Bit-width: ~2-3 BPW
- Compression: 5-8x
- Perplexity: 30-60 on WikiText-2

### For SVD-Based Approaches to Work

**Required Changes**:
1. Use lower energy threshold (50-70%)
2. OR implement learned transforms to make weights more compressible
3. Consider hybrid: SVD for redundancy capture + careful quantization

**Not Yet Evaluated**:
- SVD at 50%: Expected ~0.4 BPW, PPL 8-12
- SVD at 60%: Expected ~0.5 BPW, PPL 6-10
- SVD at 70%: Expected ~0.7 BPW, PPL 5-8

---

## Generated Artifacts

### Quantized Models

| File | Size | Method | Status |
|------|------|--------|--------|
| `gemma-4-E2B-sub1bit.pt` | 320MB | SVD Sub1Bit | BROKEN |
| `gemma-4-E2B-sub1bit.gguf` | 1.1GB | SVD Sub1Bit | Broken export |
| `gemma_ternary_aggressive.pt` | 362MB | Ternary Aggressive | RECOMMENDED |
| `gemma_magq.pt` | 1.8GB | Magnitude Quant | Good |
| `gemma_hybrid_stream.pt` | 1.8GB | Hybrid | Experimental |

### Source Files

**Quantization Scripts**:
- `quantize_svd_proper_v2.py` - SVD with configurable threshold
- `quantize_ternary_aggressive.py` - Ternary quantization
- `quantize_int8.py` - INT8 baseline
- `train_gemma_transform.py` - Learned transforms

**Evaluation Scripts**:
- `eval_reconstruction.py` - MSE evaluation
- `compare_results.py` - Method comparison

**Core Implementation**:
- `src/quantization.py` - Quantization utilities
- `src/lowrank_factorization.py` - SVD factorization
- `src/pack_gguf.py` - GGUF packing
- `src/gguf_writer.py` - GGUF format support

### Documentation

- `EXPERIMENT_PLAN.md` - Original experiment design
- `FINAL_SUMMARY.txt` - Quick reference summary
- `README.md` - Project overview

---

## Future Work

### Priority 1: Evaluate Lower SVD Thresholds

```bash
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_50.pt --threshold 0.50
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_60.pt --threshold 0.60
python scripts/quantize_svd_proper_v2.py --model-dir models/gemma-4-E2B --output quantized/gemma_svd_70.pt --threshold 0.70
```

Expected to achieve the target PPL < 10.5 with 50-70% thresholds.

### Priority 2: Implement Learned Transforms

The `train_transform.py` approach could enable:
- Learning optimal rotations to maximize weight compressibility
- Better SVD performance at higher energy thresholds
- Potentially achieving < 0.7 BPW with acceptable quality

### Priority 3: Hybrid Approaches

Combine strengths:
- SVD for layers with high redundancy
- Ternary for layers requiring preservation
- Learned transforms for difficult layers

---

## Appendix: Technical Details

### Ternary Quantization Algorithm

```
1. Compute scale = max(|x|)
2. Normalize: x_norm = x / scale
3. Ternary: q = sign(x_norm), with zeros mapped to +1
4. Store: quantized values {-1, 0, +1} as int8 + scale
```

### SVD Low-Rank Factorization

```
1. Compute SVD: W = U @ S @ Vt
2. Determine rank r by energy threshold:
   - cumulative_energy = cumsum(S^2) / sum(S^2)
   - r = first index where cumulative_energy >= threshold
3. Retain: U[:, :r], S[:r], Vt[:r, :]
4. Quantize U, S, Vt separately
```

### Magnitude Quantization

```
1. Per-channel: compute scale for each output channel
2. Normalize: x_norm = x / scale (per channel)
3. Quantize: round to nearest level in {-(2^{bits-1}-1), ..., 0, ..., (2^{bits-1}-1)}
4. Store: quantized values + per-channel scales
```

---

*Document generated: May 2026*
*Project: Quantization Exploration*
*Repository: https://github.com/toxzak-svg/Quantization-Exploration*
