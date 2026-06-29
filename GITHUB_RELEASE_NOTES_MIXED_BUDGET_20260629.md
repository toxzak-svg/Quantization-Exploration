# Mixed Budget g128 Target 4.0 - Live Colab Result

Release date: 2026-06-29

This release packages the current best `sub1quant` quality-per-byte artifact and its live Colab evaluation outputs.

## Artifact

- `gemma_mixed_budget_full_g128_target4p0.pt`
- Size: 948,181,931 bytes
- SHA-256: `c4337f598433a909e3895a3d3b47f2093dbdba2a1ed4e71502f5549e233f6326`
- Average BPW: 3.9989887990043558
- Method mix: 301 groupwise INT4, 14 INT2 + binary residual, 1 INT2 + error-budget k4

## Live Colab evaluation

Hardware/runtime: NVIDIA L4, CUDA, dense BF16 evaluation after applying quantized weights.

| Run | Runtime dtype | WikiText tokens | Chunks | PPL |
|-----|---------------|----------------:|-------:|----:|
| Unquantized `google/gemma-4-E2B` base | BF16 | 292,282 | 571 | 108.4542 |
| Mixed budget full g128 target 4.0 | BF16 dense eval after applying quantized weights | 292,282 | 571 | 107.5656 |

Claim scope: BF16-baseline-equivalent perplexity on this exact Gemma4/WikiText/Colab runner at about 4.00 BPW. This is not an FP16 result, not an FP8 comparison, and not a throughput result.

## Included result assets

- `mixed_budget_full_g128_target4p0_ppl_live.json`
- `base_full_ppl_live.json`
- `mixed_budget_live_colab_comparison.json`
- `mixed_budget_scan_full_g128_target4p0.json`
