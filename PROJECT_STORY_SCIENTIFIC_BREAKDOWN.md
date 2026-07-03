# sub1quant Scientific Project Story

Chronological breakdown of the project rationale, bugs, fixes, findings, and
current evidence state. This is a scientific decision narrative, not a private
scratchpad: it records the hypotheses that were tested, the reasons the project
pivoted, and the evidence that does or does not support each claim.

## Executive status

The project began as an aggressive sub-1-bit quantization attempt for Gemma 4
E2B. The original SVD-based route did produce small artifacts, but the quality
evidence showed the method was broken. The useful project story then pivoted
from "sub-1-bit at any cost" to "quality-per-byte near 4 BPW, with a route to
packed runtime speed later."

The strongest current result is the mixed-budget g128 target-4.0 artifact:

| Item | Value |
| --- | ---: |
| Model | `google/gemma-4-E2B` |
| Artifact | `gemma_mixed_budget_full_g128_target4p0.pt` |
| Average BPW | 3.9989887990043558 |
| Size | 948,181,931 bytes |
| Method mix | 301 groupwise INT4, 14 INT2 + binary residual, 1 INT2 + error-budget k4 |
| Live base PPL | 108.4542 |
| Live mixed-budget PPL | 107.5656 |
| Tokens / chunks | 292,282 / 571 |
| Runtime | CUDA BF16 dense evaluation after applying quantized weights |

The supported claim is narrow: BF16-baseline-equivalent perplexity on this exact
Gemma4/WikiText/Colab runner at about 4.00 BPW. It is not an FP16 result, not an
FP8 comparison, and not a throughput result, because the evaluator reconstructs
or applies weights into a dense model for correctness.

## Phase 0: project framing and target

The initial target was ultra-low bit quantization, specifically less than or
equal to 0.7 bits per weight, with acceptable WikiText-2 perplexity. The project
was centered on Gemma 4 E2B and asked a high-risk research question:

Can a 2B-parameter language model be compressed to sub-1-bit-per-weight storage
while preserving useful language-model quality?

The early success criteria in `EVAL_RESULTS.md` were:

| Criterion | Target |
| --- | ---: |
| Average bit width | <= 0.7 bits/weight |
| WikiText-2 perplexity | <= 10.5 |
| Estimated compressed size | about 350 MB |

The key scientific risk was that a low storage number could be meaningless if
the representation destroyed the model. That risk became the central lesson of
the first phase.

## Phase 1: SVD sub-1-bit path

### Hypothesis

The first major hypothesis was that low-rank factorization plus ternary
quantization could compress weight matrices below 1 BPW while keeping enough
signal for inference. The planned mechanism was:

1. Factorize a matrix with SVD.
2. Keep enough singular energy, initially around 90 percent.
3. Quantize the retained factors aggressively.
4. Package the result as a much smaller checkpoint or GGUF.

The thought process was straightforward: if transformer weights were highly
compressible in a low-rank basis, storing low-rank factors could beat direct
per-weight quantization.

### Bug and failure

The SVD path failed because the energy threshold did not produce real low rank.
At 90 percent energy retention, the retained rank was still near full rank:
roughly 1155 for a 1536 by 6144 matrix, or about 95 percent of the maximum
rank. That meant the method paid the cost of factorization and quantization
noise without getting the intended dimensionality reduction.

The repository evaluation notes record the SVD sub-1-bit artifact as broken:

| SVD metric | Result |
| --- | ---: |
| Claimed/target BPW | about 0.88 |
| Average MSE | about 1123 |
| Status | Reconstruction completely broken |

### Scientific finding

High singular-energy retention is not automatically compatible with aggressive
post-factor quantization. For this model and implementation, the SVD route
created a compact artifact but not a usable model. The storage result was real;
the quality result was not.

This forced a methodological separation that remains important:

| Category | Meaning |
| --- | --- |
| Small artifact exists | The compressor emitted a file with fewer stored bits |
| Reconstruction is acceptable | Weight reconstruction error is bounded |
| Perplexity is acceptable | The applied model preserves language-model behavior |
| Runtime is faster | A packed backend avoids dense dequantization and wins on target hardware |

The SVD path only satisfied the first category.

## Phase 2: direct low-bit baselines

### Hypothesis

After SVD broke reconstruction, the next question was whether direct
quantization could preserve more structure than low-rank factorization. The two
main paths were:

1. Ternary aggressive quantization with layer importance.
2. Magnitude/per-channel quantization with mixed 4-bit and 2-bit choices.

The decision rationale was that removing SVD might avoid the large information
loss introduced by quantizing near-full-rank factors.

### Findings

The early evaluation table favored direct ternary over the SVD artifact:

| Method | BPW | Compression | Estimated PPL | Status |
| --- | ---: | ---: | ---: | --- |
| Ternary aggressive | 1.60 | 10.0x | about 66 | Best early low-BPW balance |
| Magnitude 4/2-bit | 2.13 | 7.5x | about 128 | Moderate quality |
| SVD sub1bit 90 percent | 0.88 | 18.1x | broken | Failed |

The important result was not that ternary solved the project. It did not meet
the original <= 0.7 BPW target, and the estimated PPL was still far from a
strong model-quality claim. The real finding was negative but useful:

Direct quantization was less destructive than the SVD route, but the original
sub-1-bit target was not yet compatible with usable quality.

## Phase 3: practical INT4 pivot

### Hypothesis

The project then pivoted toward a practical question:

Can the repo produce FP8-like or baseline-like quality at lower storage cost
than BF16, while preserving a plausible future path to packed throughput?

This moved groupwise INT4 into the role of control group and practical anchor.
Groupwise INT4 g128 was not the novel invention, but it was the necessary
quality baseline.

### Decision rationale

The old goal, "sub-1-bit," was too strict relative to the measured quality
failure. The revised goal was more useful:

1. Establish a quality-preserving low-bit control.
2. Search below or around INT4 for quality-per-byte gains.
3. Avoid claiming throughput until there is a packed kernel or maintained
   backend.

### Finding

The Colab L4 reconstruction run recorded groupwise INT4 g128 at:

| Metric | Groupwise INT4 g128 |
| --- | ---: |
| Size | 932.44 MiB |
| BPW | 4.125 |
| Weighted RMSE | 0.002849 |
| Compression vs BF16 | 3.88x |

This became the control. The repo explicitly documents that the current
evaluator reconstructs into dense tensors, proving quality rather than speed.

## Phase 4: INT2 plus binary residual and error-budget side channel

### Hypothesis

The next scientific idea was to combine a very cheap INT2 base with a small
residual signal. The expectation was that most weights could be represented
coarsely, while a low-cost residual or sparse side channel repaired the largest
errors.

Tested variants included:

1. INT2 base.
2. INT2 plus 1-bit binary residual.
3. INT2 plus 1-bit residual plus sparse error-budget side channel.
4. Groupwise INT4 control.

### Findings

The local 4-layer scan in `FAST_INT4_PIVOT.md` showed:

| Method | BPW | Weighted RMSE | Compression vs BF16 |
| --- | ---: | ---: | ---: |
| INT2 base | 2.1250 | 0.010457 | 7.53x |
| INT2 + 1-bit residual | 3.2500 | 0.005689 | 4.92x |
| INT2 + residual + g128/k8 side channel | 4.0625 | 0.004022 | 3.94x |
| Groupwise INT4 g128 | 4.1250 | 0.002821 | 3.88x |

The side channel worked in the limited sense that it reduced error relative to
the plain residual. It did not beat INT4 reconstruction quality.

The Colab L4 artifact comparison sharpened the same conclusion:

| Artifact | Size | BPW | Weighted RMSE |
| --- | ---: | ---: | ---: |
| Groupwise INT4 g128 | 977,735,147 bytes | 4.125 | 0.0040125 |
| INT2 + binary residual g128 | 770,587,063 bytes | 3.25 | 0.0081481 |

The residual artifact was about 207,148,084 bytes smaller, roughly 21.2 percent
smaller than INT4, but its weighted RMSE was about 2.03x higher. The correct
scientific wording is therefore "smaller than INT4," not "better than INT4."

### Bugs and caveats

The limited PPL and throughput smokes were not publication-grade:

1. The base PPL in the smoke was abnormally high.
2. 40 expected shared-KV-style checkpoint keys were not applied by the loaded
   model state map.
3. Throughput used dequantized BF16 forward after applying checkpoints, not a
   packed low-bit kernel.

This identified a recurring load/evaluation risk: a quantized artifact can be
valid as a stored checkpoint and still fail as an applied model if key mapping
or runtime integration is incomplete.

## Phase 5: mixed-budget allocation

### Hypothesis

Uniform formats were too blunt. Some matrices could be downgraded cheaply while
others needed INT4. The mixed-budget hypothesis was:

Allocate precision per matrix according to weighted reconstruction benefit,
rather than forcing every matrix into the same format.

Candidate methods per matrix included groupwise INT4, INT2 binary residual, and
INT2 error-budget variants. The allocator greedily selected lower-cost formats
where the error increase was small, while preserving a target average budget.

### Early scan

The 8-matrix scan showed the idea was plausible:

| Method | BPW | Weighted RMSE | Compression vs BF16 |
| --- | ---: | ---: | ---: |
| Uniform groupwise INT4 g128 | 4.1250 | 0.002982 | 3.88x |
| Mixed budget near-INT4 target | 4.1030 | 0.002982 | 3.90x |
| Mixed budget 4.0 BPW target | 3.9959 | 0.003163 | 4.00x |

This was still reconstruction evidence, not model-quality evidence.

### Full-surface scan

The full scan covered 316 language-model weight tensors:

| Method | Layers | BPW | Weighted RMSE | Compression vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| Uniform groupwise INT4 g128 | 316 | 4.1250 | 0.0028486063 | 3.8788x |
| Mixed budget g128 target 4.0 | 316 | 3.9989887990 | 0.0029466952 | 4.0010x |

The selected method counts were:

| Format | Count |
| --- | ---: |
| Groupwise INT4 | 301 |
| INT2 + binary residual | 14 |
| INT2 + error-budget k4 | 1 |

### Live PPL finding

The live Colab CUDA BF16 run changed the status of the mixed-budget artifact
from reconstruction-plausible to model-quality-plausible:

| Run | Runtime dtype | Tokens | Chunks | PPL |
| --- | --- | ---: | ---: | ---: |
| Unquantized Gemma 4 E2B base | BF16 | 292,282 | 571 | 108.4542 |
| Mixed budget target 4.0 | BF16 dense eval after applying quantized weights | 292,282 | 571 | 107.5656 |

The mixed run was 0.8886 PPL lower than the base on that runner. That should
not be overinterpreted as "better than base"; the responsible interpretation is
baseline-equivalent within this exact evaluation path.

### Bug fixed: baseline dtype ambiguity

There was a publication-facing ambiguity about whether the baseline equivalent
was FP16. The benchmark path used BF16 on CUDA:

`torch.bfloat16 if device == "cuda" else torch.float32`

The correct public claim was updated to BF16-baseline-equivalent, not FP16.

## Phase 6: publishing and claim control

### Published assets

The mixed-budget result was pushed to both Hugging Face and GitHub release
surfaces. The publish-facing files include:

1. `eval_results/mixed_budget_full_g128_target4p0_ppl_live.json`
2. `eval_results/base_full_ppl_live.json`
3. `eval_results/mixed_budget_live_colab_comparison.json`
4. `eval_results/mixed_budget_scan_full_g128_target4p0.json`
5. `GITHUB_RELEASE_NOTES_MIXED_BUDGET_20260629.md`
6. `HF_MODEL_CARD.md`

The release notes and model card both state the narrow scope: BF16 baseline
equivalence on the live Colab runner, not FP16, FP8, or throughput.

### Scientific value

This was the first project point where a result could be described as a real
full-model quality artifact rather than just a helper test or reconstruction
scan.

The remaining caveat is still major: the artifact is evaluated through dense
application of quantized weights. It is not yet an end-to-end low-bit inference
runtime.

## Phase 7: Colab-disconnect-safe workflow

### Problem

The full model scans and builds are long enough that Colab disconnects can
waste significant work. The user explicitly required that everything save in
case Colab disconnects.

### Fixes

The project added persistence at the long-running boundaries:

| Area | Persistence mechanism |
| --- | --- |
| Mixed-budget scan | `--resume-jsonl`, one completed layer record at a time |
| Checkpoint build | `--checkpoint-dir`, per-layer shard files with validation |
| Activation calibration | periodic weights/stats/progress JSON writes |
| Cross-layer MI | activation cache, progress JSON, atomic report writes |
| Colab notebook | Drive-backed `GDRIVE_RESULTS_DIR` for live outputs |

The scan resume behavior is keyed by layer `key`; reruns emit records with
`"resumed": true` when a saved layer is reused. The checkpoint builder reuses
`layer_00000.pt`-style shards when the key, method, and group size match.

### Bugs found during persistence work

1. Temporary smoke artifacts were left behind because cleanup was blocked by
   approval constraints. They should be treated as verification scratch, not
   scientific result files.
2. `pytest` was not installed in the repo venv; the reliable test command is
   `python -m unittest discover -s tests -p 'test_*.py'`.
3. Windows process/CLR instability later affected notebook validation, so
   smaller validation steps are safer on this machine.

## Phase 8: SIGMA experiment foundation

### Hypothesis

SIGMA explored whether a learned sketch/generator representation could encode
weight blocks more efficiently than direct scalar quantization. The core idea:

1. Split a 2D weight tensor into fixed-size blocks.
2. Store signs and normalized magnitudes.
3. Bucket sign patterns through a random or learned sketch.
4. Train a small generator conditioned on bucket and tau to reconstruct block
   magnitudes.
5. Choose tau and scale per block to minimize reconstruction error.

The scientific rationale was that low-rank or shared generative structure might
capture recurring block-level magnitude patterns better than a uniform scalar
format.

### Bugs in the prototype

The initial notebook-builder direction had drifted:

1. Missing imports.
2. Undefined config references.
3. A learned-sketch path that was not differentiable in the way the notebook
   intended.
4. Large inline notebook logic that could diverge from tested repo code.

### Fix

The SIGMA code was moved into a tested module:

`src/sigma_ablation.py`

The notebook builder now consumes the canonical module source instead of
maintaining a separate, stale inline implementation.

### Evidence

Regression tests cover:

1. Block creation, sign preservation, and magnitude normalization.
2. Learned sketch execution without autograd failure.
3. Tiny-layer generator quantization shape checks.
4. Variance-ratio and BPW sanity.
5. Notebook generation sanity.

The SIGMA work should be understood as a foundation, not a final result. It has
local correctness tests and a generated notebook, but it has not yet produced a
full-model artifact beating the mixed-budget result.

## Phase 9: cross-layer mutual information

### Hypothesis

The next allocation hypothesis was that reconstruction error alone may choose
the wrong layers. A layer can have modest weight-MSE impact but high downstream
information value. Cross-layer mutual information attempts to score layers by
how strongly their activations carry information used by downstream layers.

The implemented approximation is:

1. Capture calibration hidden states.
2. Estimate cross-layer dependence using HSIC or RFF-HSIC.
3. Aggregate per-layer scores over a local horizon.
4. Convert scores into bit preferences.
5. Feed those preferences into the mixed-budget scan through `--mi-report` and
   `--mi-prior`.

### Residual-stream bug

The first conceptual bug is that transformer residual streams make raw
activation MI misleading. If `x_{l+1} = x_l + subblock(x_l)`, then raw
`I(x_l; x_{l+k})` mostly sees the shared residual stream, not the specific
information contributed by the layer.

The fix was conditional/delta MI:

`x_l - x_{l-1}`

This removes the direct residual pass-through and estimates dependence among
the layer contributions. The code exposes this as `conditioning="delta"`.

### Implementation and tests

The cross-layer MI module includes:

1. HSIC estimator.
2. Random Fourier Feature HSIC estimator.
3. MINE estimator for publication-grade pair estimates.
4. Residual delta conditioning.
5. Conditional RFF-HSIC.
6. MI-to-bit allocation.
7. Sigma-score comparison.

The tests use synthetic chains with known coupling structure. They verify that
the estimators recover near-vs-far dependence, that RFF tracks RBF in ranking,
that conditional MI removes shared residual confounding, and that GPU tests can
run when CUDA is available.

### Local MI result

The local synthetic MI report is not a full Gemma result, but it validates the
plumbing:

| Metric | Value |
| --- | ---: |
| Layers | 12 |
| Target average BPW | 4.0 |
| Average allocated bits | 4.0197 |
| MI min | 0.01046 |
| MI max | 0.03108 |
| MI mean | 0.02254 |
| Kendall tau vs sigma | -0.4848 |
| Calibration tokens | 256 |

The negative Kendall tau is scientifically interesting: MI and sigma are not
ranking layers the same way on that synthetic/local setup. It is not yet proof
that MI improves real model perplexity.

## Phase 10: current Colab notebook and MI-biased breakthrough path

### Goal

The current notebook direction is an end-to-end Colab workflow for the current
eval path, not the older release notebook. It is intended to:

1. Load secrets and mount Drive.
2. Clone or prepare the repo.
3. Check GPU/runtime state.
4. Capture calibration activations.
5. Run cross-layer MI, including delta conditioning.
6. Run MI-biased mixed-budget scan.
7. Build the MI-biased checkpoint.
8. Save progress, cache, JSONL scan records, shards, and final artifacts into
   `GDRIVE_RESULTS_DIR`.

### Bugs fixed

1. The earlier comprehensive notebook validation hit Windows process/CLR
   instability, not a confirmed notebook logic failure.
2. The newest builder now writes generated notebooks with LF newlines so
   `git diff --check` does not fail on notebook line endings.
3. The MI report now writes `layer_indices`, avoiding the bug where MI scores
   could be mapped to the wrong transformer layer.
4. The MI runner now writes progress JSON and uses atomic cache/report writes.
5. The Colab wrapper exposes `--progress-output`.

### Current local verification

The latest local test command was:

`python -m unittest discover -s tests -p 'test_*.py'`

Result:

`Ran 84 tests in 24.068s`

`OK (skipped=5)`

The skipped tests are CUDA-dependent checks on the local CPU-only environment.

## Consolidated bug chronology

| Phase | Bug or risk | Fix or current status |
| --- | --- | --- |
| SVD sub1bit | 90 percent energy kept near-full rank and destroyed reconstruction after ternary quantization | Marked as failed; pivoted away |
| Early docs | Storage/compression looked better than model quality justified | Claim scope separated by artifact, reconstruction, PPL, and throughput |
| Direct ternary | Better than SVD but still far from original quality target | Retained as historical low-BPW baseline, not final result |
| Residual artifact | Smaller than INT4 but about 2.03x higher weighted RMSE | Treated as smaller-than-INT4 experiment, not a win |
| PPL smoke | Abnormally high base PPL and 40 missing shared-KV-style keys | Not used as publication-grade evidence |
| Throughput smoke | Dense BF16 dequantized forward, not packed kernel | No throughput claim allowed |
| Baseline wording | User asked if baseline equivalent was FP16 | Corrected to CUDA BF16 |
| Colab disconnect risk | Long scans/builds lost work if runtime reset | Added JSONL resumes, shard checkpoints, progress JSON, Drive-backed notebook outputs |
| SIGMA notebook | Inline code had missing imports, undefined config, and non-differentiable learned sketch path | Moved core into tested module |
| Notebook generation | Nested triple quotes caused syntax problems during generated notebook build | Changed generated string quoting |
| Windows validation | `CreateProcessAsUserW failed: 1455` and CLR failures disrupted validation | Prefer smaller validation chunks |
| Cross-layer MI | Raw activation MI over-credited residual stream | Added delta/conditional MI |
| MI scan integration | MI score arrays could be ambiguous without layer ids | Added `layer_indices` and mapped by transformer layer key |
| Current cleanup | Some `_smoke_*` files remain as scratch artifacts | Ignore or clean separately with explicit approval |

## Consolidated findings

1. Sub-1-bit SVD achieved a small artifact but failed reconstruction badly.
2. Direct ternary was the best early low-BPW baseline, but not a high-quality
   release answer.
3. Groupwise INT4 is the necessary control, not the invention.
4. INT2 plus residual is smaller than INT4, but current reconstruction quality
   trails INT4.
5. Mixed budgeting is the strongest demonstrated quality-per-byte path.
6. The mixed-budget target-4.0 artifact has a real full-model PPL result on a
   live CUDA BF16 runner.
7. The result is baseline-equivalent in that runner, not a speed result.
8. Activation weighting and cross-layer MI are the right next scientific
   levers because pure weight-MSE may mis-rank layer importance.
9. Delta-conditioned MI is necessary because transformer residual streams
   otherwise confound raw activation dependence.
10. The project is now in the "promising applied quantization method" zone,
    not the "paper-grade breakthrough proven" zone.

## Current claim boundaries

| Claim | Status |
| --- | --- |
| A sub-1-bit Gemma artifact exists | Historical artifact exists, but quality failed |
| Direct ternary is viable at 1.6 BPW | Supported only as early estimated/limited evidence |
| INT2 residual is smaller than INT4 | Supported |
| INT2 residual beats INT4 | Not supported |
| Mixed budget target 4.0 is BF16-baseline-equivalent on live Colab runner | Supported |
| Mixed budget beats FP8 | Not tested |
| Mixed budget is faster than BF16/FP8 | Not tested |
| Cross-layer MI improves real PPL | Not yet tested |
| Colab path is now disconnect-aware | Locally implemented and tested; full live run still needed |

## Next scientific experiment

The next decisive experiment is not another local helper test. It is a full
Colab run of the regenerated `notebook/cross_layer_mi_colab.ipynb` with
`GDRIVE_RESULTS_DIR` set, producing:

1. Real Gemma calibration activation cache.
2. Delta-conditioned cross-layer MI report.
3. MI-biased full-surface mixed-budget scan with `--mi-prior 1.0`.
4. MI-biased checkpoint shards and final checkpoint.
5. Full PPL comparison against the existing mixed-budget target-4.0 baseline.

The decision rule should be:

| Outcome | Interpretation |
| --- | --- |
| MI-biased PPL improves or matches at lower BPW | Real breakthrough candidate |
| MI-biased PPL matches at same BPW but changes layer choices | Interesting allocation evidence, needs more ablation |
| MI-biased PPL regresses | MI prior is not useful yet, or the proxy is miscalibrated |
| Scan improves reconstruction only | Not enough; PPL is the gate |
| PPL improves but throughput remains dense | Quality result only; still no speed claim |

## Publication framing

The strongest honest story today is:

The project disproved its first sub-1-bit SVD route, established INT4 as the
quality control, found that uniform sub-4-bit residual formats trail INT4, and
then produced a mixed-budget full-model artifact at about 4.00 BPW with
BF16-baseline-equivalent perplexity on a live CUDA BF16 WikiText runner. The
current research frontier is whether activation-weighted and delta-conditioned
cross-layer-MI allocation can improve the mixed-budget layer choices enough to
produce a real quality-per-byte gain beyond the existing target-4.0 artifact.

That is a strong engineering and Substack story. It is not yet a complete paper
claim because the method still lacks:

1. Dense BF16/FP16/FP8 and mature INT4 baseline matrix across more tasks.
2. A packed runtime or kernel benchmark.
3. Repeated full-model PPL runs for the MI-biased variants.
4. Ablations separating reconstruction weighting, activation weighting, and MI
   prior effects.
5. Clear evidence that the improvement transfers beyond one model and one
   WikiText runner.

