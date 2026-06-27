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

Run real perplexity:

```powershell
.\.venv\Scripts\python.exe scripts\eval_quantized.py `
  --quantized-pt quantized\gemma_groupwise_int4_g128.pt `
  --model-dir models\gemma-4-E2B `
  --wikitext data\wiki.test.txt `
  --device cuda `
  --max-length 512 `
  --stride 512
```

Use `--device cpu` only for debugging. It will be slow.

## Throughput

The new checkpoint is packed and kernel-ready, but the current evaluator reconstructs weights into dense tensors for correctness. That proves quality, not speed.

To beat FP8 throughput, the next implementation step is one of:

1. Export this format to an existing INT4 runtime such as a GGUF Q4 family format.
2. Add a CUDA or Triton kernel that multiplies activations by `packed_int4` plus per-group scales without dense dequantization.
3. Route through a maintained INT4 weight-only backend and keep this repo as the quantization/evaluation harness.

Do not claim higher throughput until one of those backends is benchmarked against an FP8 baseline on the target GPU.

## Current local smoke result

On the first two Gemma language-model matrices with group size 128:

- Average BPW: 4.1250
- Compression vs BF16: 3.88x
- Weighted MSE: 0.000007
- Weighted RMSE: 0.002638
- Weighted mean absolute error: 0.002103

This is only a smoke result, not full-model perplexity.
