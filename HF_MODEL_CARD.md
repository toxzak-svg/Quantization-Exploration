---
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
base_model:
- google/gemma-4-E2B
tags:
- quantization
- sub-4-bit
- sub1quant
- int4
- int2
- gemma
- gemma4
- wikitext
datasets:
- wikitext
---

# sub1quant mixed-budget Gemma 4 E2B artifacts

This repository contains the mixed-budget sub-4-bit artifact from `sub1quant`.
The base model is not mirrored here; download `google/gemma-4-E2B` separately.

## Current artifact

| File | Method | Avg BPW | Size |
|------|--------|--------:|-----:|
| `quantized/gemma_mixed_budget_full_g128_target4p0.pt` | mixed budget, g128, target 4.0 BPW | 3.9990 | 948 MB |

The checkpoint contains 316 language-model weight tensors:

| Format | Count |
|--------|------:|
| Groupwise INT4 | 301 |
| INT2 + binary residual | 14 |
| INT2 + error-budget k4 side channel | 1 |

## Live Colab evaluation

Run date: 2026-06-29

Hardware/runtime: NVIDIA L4, CUDA, dense BF16 evaluation after applying the quantized weights.

| Run | Runtime dtype | WikiText tokens | Chunks | PPL |
|-----|---------------|----------------:|-------:|----:|
| Unquantized `google/gemma-4-E2B` base | BF16 | 292,282 | 571 | 108.4542 |
| Mixed budget full g128 target 4.0 | BF16 dense eval after applying quantized weights | 292,282 | 571 | 107.5656 |

This supports a narrow claim: BF16-baseline-equivalent perplexity on this exact Gemma4/WikiText/Colab runner at about 4.00 BPW. It is not an FP16 result, not an FP8 comparison, and not a throughput result. The current evaluator reconstructs/applies weights into a normal dense model for correctness.

Result files:

- `eval_results/mixed_budget_full_g128_target4p0_ppl_live.json`
- `eval_results/base_full_ppl_live.json`
- `eval_results/mixed_budget_live_colab_comparison.json`
- `eval_results/mixed_budget_scan_full_g128_target4p0.json`

## Reproduce

```bash
pip install "transformers>=5.5.0" torch accelerate safetensors huggingface_hub

python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B', local_dir='./models/gemma-4-E2B')"

python scripts/limited_ppl_bench.py \
  --label mixed_budget_full_g128_target4p0 \
  --model-dir models/gemma-4-E2B \
  --wikitext data/wiki.test.txt \
  --quantized-pt quantized/gemma_mixed_budget_full_g128_target4p0.pt \
  --tokens 1000000000 \
  --max-length 512 \
  --stride 512 \
  --device cuda \
  --output eval_results/mixed_budget_full_g128_target4p0_ppl_live.json
```

## License

The quantization code and metadata in this repository are Apache-2.0. The base model remains governed by Google's Gemma license.
