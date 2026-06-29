---
license: apache-2.0
tags:
  - quantization
  - sub-4-bit
  - gemma
---

# sub1quant — sub-4-bit quantization artifacts for gemma-4-E2B

This repo holds the **mixed-budget sub-4-bit quantization artifacts** produced by the
`sub1quant` project. The base model (`google/gemma-4-E2B`) is **not** mirrored here —
pull it separately from Hugging Face.

## Contents

```
quantized/
  gemma_mixed_budget_full_g128_target4p0.pt    # 948 MB, 316 language_model weight tensors
                                                # 301 groupwise_int4 + 14 int2_binary_residual
                                                # + 1 int2_error_budget_residual
                                                # avg BPW ≈ 4.0  (vs BF16 ≈ 16 BPW)
eval_results/
  mixed_budget_full_g128_target4p0_ppl_colab.json   # perplexity on wikitext test
  mixed_budget_scan_full_g128_target4p0.json        # full-surface reconstruction scan
  error_budget_residual_*.json                      # earlier int2+residual scan results
src/                                                 # quantization/dequant primitives
scripts/                                             # quantize_mixed_budget, eval_quantized, ...
test_perplexity.py                                   # entry-point for perplexity eval
data/wiki.test.txt                                   # wikitext-103 test, ~287k tokens
```

## Perplexity (latest)

| format | BPW | perplexity | chunks | tokens | status |
|--------|----:|-----------:|-------:|-------:|--------|
| mixed_budget_full_g128_target4p0 | 4.00 | **107.2452** | 571 | 292282 | FAIL (>10.5) |

Run on `NVIDIA L4` bf16→fp16, `gemma-4-E2B` from `google/gemma-4-E2B`, wikitext test
(stride=512, max_length=512). Result file:
`eval_results/mixed_budget_full_g128_target4p0_ppl_colab.json`.

The 107 perplexity is materially higher than a working sub-4-bit quant on a 2B
model — treat it as a measurement, not a quality claim. Reconstruction RMSE alone
(see scan JSONs) does not predict this number.

## Reproducing the perplexity eval

```bash
# install
pip install "transformers>=5.5.0" torch accelerate safetensors

# pull the base model (NOT in this repo)
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B', local_dir='./models/gemma-4-E2B')"

# run
python test_perplexity.py \
  --model models/gemma-4-E2B \
  --quantized quantized/gemma_mixed_budget_full_g128_target4p0.pt \
  --wikitext data/wiki.test.txt \
  --device cuda \
  --max-length 512 --stride 512
```

## License

The base model is governed by Google's Gemma license. The quantization
artifacts in this repo are released under Apache-2.0.
