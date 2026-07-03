# Fast INT4 Pivot

The original sub-1-bit SVD path is not the right route to FP8-like perplexity. The 90 percent SVD artifact is real, but the repo evaluation notes mark it as broken. The practical pivot is a quality-first low-bit path:

- Use packed row-wise group INT4 weights with per-group scales.
- Evaluate real perplexity through `scripts/eval_quantized.py`.
- Add or export to a real packed INT4 inference backend before claiming throughput gains.

## Why this pivot

Ternary and failed SVD checkpoints are storage experiments. They do not currently provide a fast inference kernel, and their quality is too far from FP8. Groupwise INT4 keeps much more information while still cutting BF16 weight bandwidth by about 3.8x at group size 128.

This does not preserve the original sub-1-bit target. It preserves the project goal that matters more for a usable release: similar quality to FP8 with a path to higher throughput.

## Build

Smoke test the first two matrices:

```powershell
.\.venv\Scripts\python.exe scripts\quantize_groupwise_int4.py `
  --model-dir models\gemma-4-E2B `
  --output quantized\gemma_groupwise_int4_smoke.pt `
  --group-size 128 `
  --max-layers 2
```

Build the full checkpoint when there is enough disk space:

```powershell
.\.venv\Scripts\python.exe scripts\quantize_groupwise_int4.py `
  --model-dir models\gemma-4-E2B `
  --output quantized\gemma_groupwise_int4_g128.pt `
  --group-size 128
```

Expected full artifact size is multi-GB. Check disk first.

## Evaluate

Run reconstruction comparison:

```powershell
.\.venv\Scripts\python.exe scripts\eval_reconstruction.py --model-dir models\gemma-4-E2B
```

Run publication-grade base-vs-INT4 perplexity. This writes compact JSON with
the full token count, chunk count, PPL values, comparison ratio, checkpoint
stats, and exact quantized-weight application counts:

```powershell
.\.venv\Scripts\python.exe scripts\run_publication_ppl.py `
  --label groupwise_int4_g128_full `
  --quantized-pt quantized\gemma_groupwise_int4_g128.pt `
  --model-dir models\gemma-4-E2B `
  --wikitext data\wiki.test.txt `
  --tokens all `
  --device cuda `
  --max-length 512 `
  --stride 512 `
  --expect-replaced 276 `
  --expect-skipped-shared-kv 40 `
  --output eval_results\groupwise_int4_g128_full_ppl.json
```

Colab equivalent from `/content/sub1quant`:

```bash
python -u scripts/run_publication_ppl.py \
  --label groupwise_int4_g128_full \
  --quantized-pt /content/sub1quant/quantized/gemma_groupwise_int4_g128.pt \
  --model-dir /content/sub1quant/models/gemma-4-E2B \
  --wikitext /content/sub1quant/data/wiki.test.txt \
  --tokens all \
  --device cuda \
  --max-length 512 \
  --stride 512 \
  --expect-replaced 276 \
  --expect-skipped-shared-kv 40 \
  --output /content/sub1quant/eval_results/groupwise_int4_g128_full_ppl.json
```

Use `--device cpu` only for debugging. It will be slow.

## Throughput

The new checkpoint is packed and kernel-ready, but the current evaluator reconstructs weights into dense tensors for correctness. That proves quality, not speed.

To beat FP8 throughput, the next implementation step is one of:

1. Export this format to an existing INT4 runtime such as a GGUF Q4 family format.
2. Add a CUDA or Triton kernel that multiplies activations by `packed_int4` plus per-group scales without dense dequantization.
3. Route through a maintained INT4 weight-only backend and keep this repo as the quantization/evaluation harness.

Do not claim higher throughput until one of those backends is benchmarked against an FP8 baseline on the target GPU.

## Error-budget residual iteration

The next sub-4-bit candidate is an INT2 base plus a 1-bit residual and a sparse error-budget side channel. The side channel stores the largest remaining residual corrections per group as index plus signed INT4 correction codes.

Build a smoke artifact:

```powershell
.\.venv\Scripts\python.exe scripts\quantize_error_budget_residual.py `
  --model-dir models\gemma-4-E2B `
  --output quantized\gemma_int2_error_budget_g128_k8_smoke.pt `
  --group-size 128 `
  --outliers-per-group 8 `
  --max-layers 4
```

Scan reconstruction tradeoffs:

```powershell
.\.venv\Scripts\python.exe scripts\scan_error_budget_residual.py `
  --model-dir models\gemma-4-E2B `
  --group-size 128 `
  --outliers-per-group 8 `
  --max-layers 4
```

Current local 4-layer scan:

| Method | BPW | Weighted RMSE | Compression vs BF16 |
|--------|-----|---------------|---------------------|
| INT2 base | 2.1250 | 0.010457 | 7.53x |
| INT2 + 1-bit residual | 3.2500 | 0.005689 | 4.92x |
| INT2 + 1-bit residual + g128/k8 side channel | 4.0625 | 0.004022 | 3.94x |
| Groupwise INT4 g128 | 4.1250 | 0.002821 | 3.88x |

The side channel is doing useful work: it roughly halves the weighted MSE of the plain 1-bit residual while staying slightly smaller than g128 INT4. It does not yet beat INT4 reconstruction quality, so it is not the final "beats existing 4-bit" result.

The best below-INT4 g256 variant tested so far was g256/k18:

| Method | BPW | Weighted RMSE | Compression vs BF16 |
|--------|-----|---------------|---------------------|
| INT2 + 1-bit residual + g256/k18 side channel | 4.0313 | 0.004073 | 3.97x |
| Groupwise INT4 g256 | 4.0625 | 0.003100 | 3.94x |

That is smaller than g256 INT4 and better than the binary residual, but still behind INT4 reconstruction quality.

## Mixed budget iteration

Uniform sub-4-bit residual formats still trail INT4. A better next step is mixed budgeting: compute several candidate formats per matrix, then greedily allocate extra bits where they buy the largest weighted error reduction.

Run the mixed-budget scan:

```powershell
.\.venv\Scripts\python.exe scripts\scan_mixed_budget.py `
  --model-dir models\gemma-4-E2B `
  --group-size 128 `
  --outlier-options 4,8 `
  --max-layers 8 `
  --output eval_results\mixed_budget_scan_8layers_g128.json
```

Run a stricter 4.0 BPW target:

```powershell
.\.venv\Scripts\python.exe scripts\scan_mixed_budget.py `
  --model-dir models\gemma-4-E2B `
  --group-size 128 `
  --outlier-options 4,8 `
  --max-layers 8 `
  --target-bpw 4.0 `
  --output eval_results\mixed_budget_scan_8layers_g128_target4p0.json
```

Current 8-matrix scan:

| Method | BPW | Weighted RMSE | Compression vs BF16 |
|--------|-----|---------------|---------------------|
| Uniform groupwise INT4 g128 | 4.1250 | 0.002982 | 3.88x |
| Mixed budget, near-INT4 target | 4.1030 | 0.002982 | 3.90x |
| Mixed budget, 4.0 BPW target | 3.9959 | 0.003163 | 4.00x |

The near-INT4 target found cheap savings by downgrading low-impact matrices while preserving reconstruction almost exactly. The 4.0 BPW target selected one large `int2_error_budget_k4` matrix, one tiny `int2_base` matrix, and INT4 for the rest. This is the most promising direction so far, but it is still a reconstruction result. A real claim needs either activation-weighted calibration or short perplexity evaluation of a material mixed artifact.

Collect activation weights for a calibrated mixed-budget pass:

```powershell
.\.venv\Scripts\python.exe scripts\collect_activation_weights.py `
  --model-dir models\gemma-4-E2B `
  --wikitext data\wiki.test.txt `
  --tokens 32768 `
  --output eval_results\activation_weights_gemma4.json `
  --stats-output eval_results\activation_stats_gemma4.json `
  --progress-output eval_results\activation_weights_progress.json `
  --save-every-chunks 1
```

The collector writes stats, normalized activation weights, and progress JSON
after every chunk. On Colab, use mounted Drive paths for these three outputs if
you need the calibration state to survive a runtime disconnect or reset.

Full-surface scan:

```powershell
.\.venv\Scripts\python.exe scripts\scan_mixed_budget.py `
  --model-dir models\gemma-4-E2B `
  --group-size 128 `
  --outlier-options 4,8 `
  --target-bpw 4.0 `
  --activation-weights eval_results\activation_weights_gemma4.json `
  --resume-jsonl eval_results\mixed_budget_scan_full_g128_target4p0.layers.jsonl `
  --output eval_results\mixed_budget_scan_full_g128_target4p0.json
```

Run a cross-layer-MI-biased full-surface scan after producing an MI report with
`scripts/run_cross_layer_mi_colab.py`:

```powershell
.\.venv\Scripts\python.exe scripts\scan_mixed_budget.py `
  --model-dir models\gemma-4-E2B `
  --group-size 128 `
  --outlier-options 4,8 `
  --target-bpw 4.0 `
  --activation-weights eval_results\activation_weights_gemma4.json `
  --mi-report eval_results\cross_layer_mi_colab_delta.json `
  --mi-prior 1.0 `
  --resume-jsonl eval_results\mixed_budget_scan_full_g128_target4p0_mi.layers.jsonl `
  --output eval_results\mixed_budget_scan_full_g128_target4p0_mi.json
```

Use `--mi-prior 0.0` for the reconstruction-only baseline and `--mi-prior 1.0`
as the first MI-biased candidate. Larger values make MI dominate upgrade order;
do not publish a larger-prior result until it beats the baseline scan in real
perplexity, not just layer-ranking plausibility.

On Colab, put `--resume-jsonl` on mounted Drive or another persistent path when
possible, for example `/content/drive/MyDrive/sub1quant_saves/...layers.jsonl`.
The scanner appends and fsyncs one completed layer record at a time, so rerunning
the same command after a disconnect reuses completed layer candidates instead of
starting over.

Build the selected mixed checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\quantize_mixed_budget.py `
  --scan-json eval_results\mixed_budget_scan_full_g128_target4p0.json `
  --model-dir models\gemma-4-E2B `
  --checkpoint-dir quantized\mixed_budget_full_g128_target4p0_shards `
  --output quantized\gemma_mixed_budget_full_g128_target4p0.pt
```

On Colab, point `--checkpoint-dir` at mounted Drive if the run may be interrupted.
Each layer shard is written through a temporary file and then atomically renamed.
Rerunning the same build command validates and reuses matching shards before
assembling the final `.pt` checkpoint.

Current full-surface reconstruction result:

| Method | Layers | BPW | Weighted RMSE | Compression vs BF16 | Size |
|--------|--------|-----|---------------|---------------------|------|
| Uniform groupwise INT4 g128 | 316 | 4.1250 | 0.002849 | 3.88x | 932.44 MiB on Colab L4 |
| Mixed budget full g128 target 4.0 | 316 | 3.9990 | 0.002947 | 4.00x | 948 MB |

The full mixed artifact uses `301` groupwise INT4 matrices, `14` INT2+binary-residual matrices, and `1` INT2+error-budget-k4 matrix. This is a real full-model storage artifact and the best current quality-per-byte result in the repo.

Live Colab CUDA perplexity, 2026-06-29:

| Run | Runtime dtype | BPW | WikiText tokens | Chunks | PPL |
|-----|---------------|-----|-----------------|--------|-----|
| Unquantized `google/gemma-4-E2B` base | BF16 | 16.00 | 292282 | 571 | 108.4542 |
| Mixed budget full g128 target 4.0 | BF16 dense eval after applying quantized weights | 3.9990 | 292282 | 571 | 107.5656 |

Uniform groupwise INT4 full perplexity is not yet in the repo. The exact gate
is now `eval_results/groupwise_int4_g128_full_ppl.json` from
`scripts/run_publication_ppl.py`; publish only after that JSON records the full
`292282`-token / `571`-chunk run and `276` replaced weights with `40`
shared-KV skips.

This supports a narrow claim: BF16-baseline-equivalent perplexity on this exact Gemma4/WikiText/Colab runner at about 4.00 BPW. It is not an FP16 result, not an FP8 comparison, and not a throughput result because the evaluator reconstructs/applies weights into a normal dense model for correctness.

## Current local smoke result

On the first two Gemma language-model matrices with group size 128:

- Average BPW: 4.1250
- Compression vs BF16: 3.88x
- Weighted MSE: 0.000007
- Weighted RMSE: 0.002638
- Weighted mean absolute error: 0.002103

This is only a smoke result, not full-model perplexity.
