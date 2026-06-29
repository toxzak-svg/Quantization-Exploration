"""Final HF push: build the staging dir, create repo, upload_folder, verify.

Stage:
  /tmp/_hf_stage/
    README.md                                (custom — describes what's here)
    quantized/
      gemma_mixed_budget_full_g128_target4p0.pt
    eval_results/
      mixed_budget_full_g128_target4p0_ppl_colab.json
      mixed_budget_scan_full_g128_target4p0.json
      mixed_budget_scan_8layers_g128_target4p0.json
      mixed_budget_scan_4layers_g128.json
      mixed_budget_scan_8layers_g128.json
      error_budget_residual_20_layers.json
      error_budget_residual_20_layers_packed.json
      error_budget_residual_colab_l4_summary.json
      error_budget_residual_scan_4layers_g256_k16.json
      error_budget_residual_scan_4layers_g256_k18.json
      error_budget_residual_scan_4layers_k16.json
      error_budget_residual_scan_4layers_k8.json
      error_budget_residual_scan_8layers.json
      error_budget_residual_smoke.json
    src/
      __init__.py, error_budget_residual.py, groupwise_int4.py, quantization.py,
      mixed_budget.py, lowrank_factorization.py, gguf_writer.py, pack_gguf.py,
      Sub1BitLLM.py, ...
    scripts/
      eval_quantized.py, quantize_mixed_budget.py, limited_ppl_bench.py, ...
    test_perplexity.py
    data/
      wiki.test.txt
"""
import os, sys, json, shutil, traceback

# Pull token from the file we uploaded
tok_path = "/root/.hf_token"
with open(tok_path) as f:
    HF_TOKEN = f.read().strip()
os.chmod(tok_path, 0o600)
os.environ["HF_TOKEN"] = HF_TOKEN
print(f"HF_TOKEN loaded, len={len(HF_TOKEN)}")

# 1. Identify user
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
me = api.whoami()
USER = me["name"]
print(f"authenticated as: {USER}")

REPO = f"{USER}/sub1quant"
print(f"target repo: {REPO}")

# 2. Build stage
STAGE = "/tmp/_hf_stage"
if os.path.isdir(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE + "/quantized", exist_ok=True)
os.makedirs(STAGE + "/eval_results", exist_ok=True)
os.makedirs(STAGE + "/src", exist_ok=True)
os.makedirs(STAGE + "/scripts", exist_ok=True)
os.makedirs(STAGE + "/data", exist_ok=True)

# Quantized — only the latest artifact
src_pt = "/content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt"
shutil.copy2(src_pt, STAGE + "/quantized/" + os.path.basename(src_pt))
print(f"  copied {os.path.basename(src_pt)}  {os.path.getsize(STAGE+'/quantized/'+os.path.basename(src_pt)):,} bytes")

# Eval results — ALL JSONs from the eval_results dir on the kernel
src_ev = "/content/sub1quant/eval_results"
for fn in sorted(os.listdir(src_ev)):
    if fn.endswith(".json"):
        shutil.copy2(os.path.join(src_ev, fn), STAGE + "/eval_results/" + fn)
print(f"  copied {len(os.listdir(STAGE+'/eval_results'))} json files to eval_results/")

# Local eval_results JSONs on the local machine (the original scan JSONs)
local_ev = r"C:\Users\Zwmar\projects\sub1quant\eval_results"
# Note: we can't access Windows path from the kernel. Skip; eval_results/ is what
# the kernel has from this session. That's actually all we need.

# src/
src_src = "/content/sub1quant/src"
for fn in os.listdir(src_src):
    src = os.path.join(src_src, fn)
    if os.path.isfile(src) and fn.endswith(".py"):
        shutil.copy2(src, STAGE + "/src/" + fn)
print(f"  copied {len([f for f in os.listdir(STAGE+'/src') if f.endswith('.py')])} py files to src/")

# scripts/
src_scripts = "/content/sub1quant/scripts"
for fn in os.listdir(src_scripts):
    src = os.path.join(src_scripts, fn)
    if os.path.isfile(src) and fn.endswith(".py"):
        shutil.copy2(src, STAGE + "/scripts/" + fn)
print(f"  copied {len([f for f in os.listdir(STAGE+'/scripts') if f.endswith('.py')])} py files to scripts/")

# test_perplexity.py
shutil.copy2("/content/sub1quant/test_perplexity.py", STAGE + "/test_perplexity.py")

# data/
shutil.copy2("/content/sub1quant/data/wiki.test.txt", STAGE + "/data/wiki.test.txt")
print(f"  copied data/wiki.test.txt")

# README
README = """# sub1quant — sub-4-bit quantization artifacts for gemma-4-E2B

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
python test_perplexity.py \\
  --model models/gemma-4-E2B \\
  --quantized quantized/gemma_mixed_budget_full_g128_target4p0.pt \\
  --wikitext data/wiki.test.txt \\
  --device cuda \\
  --max-length 512 --stride 512
```

## License

The base model is governed by Google's Gemma license. The quantization
artifacts in this repo are released under Apache-2.0.
"""
with open(STAGE + "/README.md", "w") as f:
    f.write(README)
print(f"  wrote README.md")

# 3. Summary
print("\n=== stage summary ===")
total = 0
for root, _, files in os.walk(STAGE):
    for f in files:
        full = os.path.join(root, f)
        sz = os.path.getsize(full)
        total += sz
        rel = os.path.relpath(full, STAGE)
        if sz > 1_000_000:
            print(f"  {rel}  {sz/1024/1024:.2f} MB")
        else:
            print(f"  {rel}  {sz:,} bytes")
print(f"TOTAL: {total/1024/1024:.2f} MB  ({total:,} bytes)")

# 4. Create repo (private)
print(f"\n=== creating {REPO} (private, model) ===")
url = api.create_repo(repo_id=REPO, repo_type="model", private=True, exist_ok=True)
print(f"  URL: {url}")

# 5. upload_folder
print(f"\n=== upload_folder ===")
try:
    upload_url = api.upload_folder(
        folder_path=STAGE,
        repo_id=REPO,
        repo_type="model",
        commit_message="Initial upload: mixed-budget sub-4-bit artifacts + perplexity result",
        ignore_patterns=["**/_pycache__/**", "**/__pycache__", "**/.ipynb_checkpoints"],
    )
    print(f"  committed. Files now live in {REPO}")
except Exception as e:
    print(f"  upload_folder FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(0)

# 6. Verify
print(f"\n=== verifying upload ===")
files = api.list_repo_files(repo_id=REPO, repo_type="model")
print(f"  {len(files)} files at HEAD:")
for f in sorted(files):
    print(f"    {f}")

# 7. Cleanup
try:
    os.remove(tok_path)
    print(f"\nremoved {tok_path}")
except Exception as e:
    print(f"couldn't remove {tok_path}: {e}")

print("\nDONE")
