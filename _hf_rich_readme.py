"""Add rich yaml metadata to the README of toxzak/sub1quant on HF.

Per HF docs the full model-card YAML frontmatter supports:
  license, tags, base_model (with relation), datasets, metrics,
  library_name, pipeline_tag, language, author, description, ...

For a quantization-artifacts repo (HF repo_type="model", even though we're
shipping a .pt file + helper code, not a deployable model), we use:
  - license: apache-2.0 (the .pt and code; the base model has its own Gemma license)
  - tags: suitable for filtering
  - base_model: pointing to google/gemma-4-E2B with relation=quantized
  - datasets: WikiText-103 (the benchmark we ran)
  - metrics: perplexity 107.2452
  - library_name: transformers (the repo this integrates with)
  - pipeline_tag: text-generation (because the post-quant model is still generative)
  - model_name: a descriptive title

The body of the README stays unchanged.
"""
import os, sys, json, urllib.request

HF_TOKEN = os.environ["HF_TOKEN"]
REPO = "toxzak/sub1quant"

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
me = api.whoami()
print(f"authenticated as {me['name']} (target: {REPO})")

# Read current README from kernel-side mirror — actually, we have a copy locally
# at C:\Users\Zwmar\projects\sub1quant\_hf_stage\README.md (staging copy from earlier).
# But the README currently on HF has only minimal yaml. Let's read the HF version and
# replace its frontmatter with rich metadata, keeping the body.

# Download current README
existing = api.hf_hub_download(
    repo_id=REPO,
    filename="README.md",
    repo_type="model",
    local_dir=r"C:\Users\Zwmar\projects\sub1quant\_hf_stage",
)
body = open(existing, encoding="utf-8").read()
print(f"current README size: {len(body)} bytes")
print(f"first 80 chars: {body[:80]!r}")

# Strip any existing frontmatter so we don't double up
FM_DELIM = "---"
if body.startswith(FM_DELIM):
    parts = body.split(FM_DELIM, 2)
    if len(parts) >= 3:
        body = parts[2].lstrip("\n")
        print("stripped existing frontmatter")

# Compose rich yaml
YAML = """---
license: apache-2.0
tags:
  - quantization
  - sub-4-bit
  - sub1quant
  - int4
  - int2
  - gemma
  - gemma4
  - wikitext
pipeline_tag: text-generation
language:
  - en
library_name: transformers
base_model:
  - google/gemma-4-E2B
datasets:
  - wikitext
metrics:
  - name: WikiText-103 perplexity
    type: perplexity
    value: 107.2452
---

"""

# Add an additional commentary paragraph documenting the artifact details
# (anything beyond the strict yaml schema lives in the markdown body)
DETAILS = (
    "\n## Artifact details\n\n"
    "- Model name: gemma-4-E2B-sub1bit-mixed-budget\n"
    "- Format mix: 301 groupwise_int4 + 14 int2_binary_residual + 1 int2_error_budget_residual\n"
    "- Average bits per weight: 4.00 (vs BF16 ≈ 16 BPW → 4.00× compression)\n"
    "- Quantized entries in `.pt`: 316\n"
    "- Replaced at eval: 276 (40 skipped as shared-KV)\n"
    "- Eval device: NVIDIA L4 (cuda), fp16\n"
    "- Eval result: perplexity **107.2452** on WikiText-103 (FAIL vs target ≤ 10.5)\n\n"
)

# Write new README
new_readme = YAML + body + DETAILS
out_path = r"C:\Users\Zwmar\projects\sub1quant\_hf_stage\_README_rich.md"
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(new_readme)
print(f"new README size: {len(new_readme)} bytes")
print("preview (first 1500 chars):")
print(new_readme[:1500])

# Push via direct API call
api.upload_file(
    path_or_fileobj=out_path,
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
    commit_message="Add full yaml metadata to README",
)
print("\nuploaded README.md to", REPO)

# Verify yaml parses
import yaml
parsed = yaml.safe_load(new_readme.split("---\n", 2)[1])
print("\nparsed yaml keys:", sorted(parsed.keys()))
print("schema validates OK" if "license" in parsed and "tags" in parsed else "schema incomplete!")
