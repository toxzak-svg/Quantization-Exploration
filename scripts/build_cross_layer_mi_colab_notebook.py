"""Build the cross-layer MI Colab notebook.

Generates ``notebook/cross_layer_mi_colab.ipynb`` -- a self-contained
Colab session that ties the Phase 4 GPU-accelerated cross-layer MI
pipeline together end-to-end on real Gemma 4 E2B.

What it does, top to bottom:

  1. GPU sanity check (fail-fast if no CUDA, fail-fast on a CPU-only
     PyTorch wheel).
  2. Clones the repo and installs deps.
  3. Downloads Gemma 4 E2B safetensors + wikitext-2 calibration text
     (or accepts local paths via flags).
  4. Runs the 5 GPUCorrectnessTests from
     ``tests.test_cross_layer_mi.GPUCorrectnessTests`` to confirm the
     GPU code path matches the CPU path to within tolerance.
  5. Runs ``bench_gpu.py`` at Gemma-shape workloads (35 layers, 512 and
     2048 tokens, hidden=1536) and prints the speedup.
  6. Runs the full ``scripts/run_cross_layer_mi_colab.py`` pipeline with
     ``--device cuda`` -- this is the headline real-Gemma validation.
  7. Re-runs with ``--conditioning delta`` to show how the residual
     delta changes the per-layer ranking vs unconditional.
  8. Runs the MI-biased mixed-budget scan using the delta-conditioned
     MI report.
  9. Builds the selected mixed-budget checkpoint with resumable layer
     shards.
 10. (Optional) Copies the remaining local results to Google Drive so
     they survive the Colab session reset.

Secrets consumed:

  * ``HF_TOKEN`` -- only if downloading Gemma from HF Hub. If you
    already have a local copy at ``--model-path``, this isn't needed.
  * ``GDRIVE_RESULTS_DIR`` -- optional. Drive folder used as the live
    results/checkpoint-shard directory. If unset, repo-local
    ``eval_results/`` is used.

Runtime: T4 (16 GB) is enough for Gemma 4 E2B in bfloat16; L4 (24 GB)
gives headroom for the full matrix build at 4096 tokens.

Run from the project root::

    python scripts/build_cross_layer_mi_colab_notebook.py

Output: ``notebook/cross_layer_mi_colab.ipynb``.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebook" / "cross_layer_mi_colab.ipynb"

DEFAULT_GH_REPO = "toxzak-svg/Quantization-Exploration"
DEFAULT_MODEL_ID = "google/gemma-4-E2B"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells: list[dict] = []

# -----------------------------------------------------------------------------
# Cell 1: Title + overview
# -----------------------------------------------------------------------------
cells.append(md(
    """# Cross-layer MI on Gemma 4 E2B (GPU)

> **Phase 4 deliverable** — GPU-accelerated end-to-end validation of the
> cross-layer mutual information allocation on real Gemma 4 E2B
> calibration activations. Closes the loop on whether conditional MI
> gives a different per-layer ranking than unconditional MI (and
> therefore a different bit allocation than the per-layer sigma score).

This notebook closes the loop on the cross-layer mutual-information
allocation: it runs the full Phase 4 GPU pipeline on **real Gemma 4 E2B
calibration activations** and compares the unconditional HSIC matrix
against the conditional-MI variant that removes the residual-stream
confounder.

If conditional MI gives a meaningfully different per-layer ranking than
unconditional on the real model, the bit allocation story changes --
layers that look "important" in the unconditional matrix only because
they sit on the residual highway are down-weighted, and layers whose
weights carry information read by many downstream consumers get more
bits than the static sigma score would suggest.

## What this notebook runs

| Step | What | Expected time on T4 |
|------|------|---------------------|
| GPU sanity + repo clone | Setup | <1 min |
| Download Gemma 4 E2B safetensors | 10 GB from HF Hub | 2-5 min (or skip if local) |
| Download wikitext-2 calibration | ~250 KB | <5 s |
| `GPUCorrectnessTests` | 5 unit tests on GPU vs CPU parity | <30 s |
| `bench_gpu.py` | CPU vs GPU timing at 35×512×1536, 35×2048×1536 | <1 min |
| Full pipeline (unconditional, GPU) | One forward pass + HSIC + alloc | 3-6 min |
| Full pipeline (delta conditioning, GPU) | Same with conditioning | 3-6 min |
| Drive mirror | Copy results JSON to Drive | <30 s |

Total: ~10-20 minutes on a fresh Colab T4 / L4.

## Required Colab Secrets

Open the key icon on the left sidebar and set (only `HF_TOKEN` is
required; everything else has a sensible default):

| Secret | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | yes (if downloading) | Gemma 4 E2B is gated; accept the license on the HF model card first |
| `GDRIVE_RESULTS_DIR` | no | Drive folder (e.g. `sub1quant-results`); if set, cache/results/shards are written there directly |

If you'd rather not download the model, copy a local checkout to
`/content/gemma-4-E2B` and the download cell will skip the Hub call.
"""
))

# -----------------------------------------------------------------------------
# Cell 2: GPU sanity check (fail-fast)
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 1: GPU sanity check

This must print `GPU: <name> (<X> GB)` and `GPU matmul sanity: ok`.
Anything else means the runtime is misconfigured — don't proceed.
"""
))

cells.append(code(
    """import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "FATAL: CUDA is not available. Enable GPU: Runtime -> Change runtime type -> GPU. "
        "Re-run this cell after switching."
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

# Detect a CPU-only PyTorch wheel -- the most common reason for
# `CUDA available: False` even on a GPU runtime.
if "+cpu" in torch.__version__:
    raise SystemExit(
        f"FATAL: torch=={torch.__version__} is a CPU-only wheel. "
        "Run `!pip install --upgrade torch` in a new cell to get CUDA support, "
        "then re-run this cell."
    )

# Quick matmul sanity: catches half-configured CUDA where torch sees the
# device but cublas is missing.
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
_ = (x @ y).sum().item()
print(f"GPU matmul sanity: ok ({(x @ y).norm().item():.4f})")
"""
))

# -----------------------------------------------------------------------------
# Cell 4: Install dependencies
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 2: Install Python dependencies

Standard scientific stack + `huggingface_hub` + `datasets`. Idempotent --
skips already-installed packages.
"""
))

cells.append(code(
    """import importlib.util
import subprocess
import sys

REQUIRED = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
    "datasets": "datasets",
}
missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *missing])
else:
    print("All deps already installed.")

# Re-import torch so the cell above sees the GPU-enabled wheel in case
# it was just upgraded.
import torch
print("PyTorch after install:", torch.__version__, "CUDA:", torch.cuda.is_available())
"""
))

# -----------------------------------------------------------------------------
# Cell 4: Read secrets + clone repo
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 3: Read secrets + clone the repo

Reads `HF_TOKEN` and `GDRIVE_RESULTS_DIR` from Colab Secrets (or env
vars as a fallback). Clones the `sub1quant` repo into
`/content/sub1quant` if it's not already there.
"""
))

cells.append(code(
    f"""from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules
if IS_COLAB:
    from google.colab import userdata
else:
    userdata = None

def read_secret(name: str, default: str | None = None) -> str | None:
    if userdata is not None:
        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    return default

HF_TOKEN = read_secret("HF_TOKEN", "")
GDRIVE_RESULTS_DIR = read_secret("GDRIVE_RESULTS_DIR", "")
GH_REPO_ID = read_secret("GH_REPO_ID", "{DEFAULT_GH_REPO}")
MODEL_ID = read_secret("MODEL_ID", "{DEFAULT_MODEL_ID}")

print("IS_COLAB:", IS_COLAB)
print("MODEL_ID:", MODEL_ID)
print("GH_REPO_ID:", GH_REPO_ID)
print("HF_TOKEN configured:", bool(HF_TOKEN))
print("GDRIVE_RESULTS_DIR:", GDRIVE_RESULTS_DIR or "(unset -- using repo-local eval_results)")

REPO_DIR = Path("/content/sub1quant")
if not REPO_DIR.exists():
    # Public clone -- no auth token needed for read-only.
    clone_url = f"https://github.com/{{GH_REPO_ID}}.git"
    print("Cloning:", GH_REPO_ID)
    subprocess.check_call(["git", "clone", "--depth=1", clone_url, str(REPO_DIR)])
else:
    print("Repo already at", REPO_DIR)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("CWD:", Path.cwd())

if GDRIVE_RESULTS_DIR and IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    SUB1QUANT_SAVE_DIR = Path("/content/drive/MyDrive") / GDRIVE_RESULTS_DIR
else:
    SUB1QUANT_SAVE_DIR = REPO_DIR / "eval_results"

RESULTS_DIR = SUB1QUANT_SAVE_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("RESULTS_DIR:", RESULTS_DIR)
"""
))

# -----------------------------------------------------------------------------
# Cell 5: GPU + disk check
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 4: Disk + VRAM check

Gemma 4 E2B safetensors are ~10 GB; we need that much disk plus enough
VRAM to hold the model in bf16 (~5 GB on the GPU). This cell asserts
both and prints the available headroom.
"""
))

cells.append(code(
    """import shutil

import torch

free, total = shutil.disk_usage("/content")
print(f"Disk: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

if torch.cuda.is_available():
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    print(f"VRAM: {free_vram / 1e9:.1f} GB free / {total_vram / 1e9:.1f} GB total")

# Gemma 4 E2B safetensors are ~10 GB; we need that much disk + enough VRAM
# to hold the model in bf16 (~5 GB on the GPU).
assert free / 1e9 > 12, "Need at least 12 GB of free disk for the model + cache"
if torch.cuda.is_available():
    assert total_vram / 1e9 > 14, (
        "Need at least 14 GB VRAM (T4 is 16 GB). Switch to L4 / A100 if available."
    )
print("Sanity checks passed.")
"""
))

# -----------------------------------------------------------------------------
# Cell 6: Download model + calibration data
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 5: Download Gemma 4 E2B + wikitext-2 calibration text

Downloads the model safetensors (text-only files; vision/audio towers
are skipped to keep the artifact small) and the wikitext-2 test set to
`/content/wiki.test.txt`. If you already have a local model copy at
`/content/gemma-4-E2B`, the HF Hub call is skipped.
"""
))

cells.append(code(
    """from pathlib import Path
import os

MODEL_DIR = Path("/content/gemma-4-E2B")
CALIB_PATH = Path("/content/wiki.test.txt")

# --- model --------------------------------------------------------------------
if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.safetensors")):
    print(f"Using existing model at {{MODEL_DIR}}")
else:
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN is required to download Gemma 4 E2B from HF Hub, "
            "or copy a local model to /content/gemma-4-E2B and re-run."
        )
    from huggingface_hub import snapshot_download
    print(f"Downloading {{MODEL_ID}} -> {{MODEL_DIR}} (this takes 2-5 min)")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
        # Text-only files; vision/audio towers would explode the download.
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer*",
            "*.model",
        ],
    )
    print(f"Model downloaded to {{MODEL_DIR}}")

# --- calibration text ---------------------------------------------------------
if CALIB_PATH.exists() and CALIB_PATH.stat().st_size > 1000:
    print(f"Using existing calibration text at {{CALIB_PATH}}")
else:
    from datasets import load_dataset
    print("Downloading wikitext-2 test set ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\\n\\n".join(ds["text"])
    CALIB_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {{len(text)}} chars to {{CALIB_PATH}}")

print("Data ready.")
"""
))

# -----------------------------------------------------------------------------
# Cell 7: GPU correctness sanity (5 unit tests)
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 6: GPU code path matches CPU within tolerance

Runs the 5 `GPUCorrectnessTests` from
`tests.test_cross_layer_mi.GPUCorrectnessTests`. These check that the
GPU RFF HSIC, the GPU HSIC matrix, and the GPU conditional estimator
all match their CPU counterparts within 5-10% relative tolerance. If
any of these fail, the GPU code path has drifted and the headline
numbers downstream are not trustworthy.
"""
))

cells.append(code(
    """import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "unittest",
     "tests.test_cross_layer_mi.GPUCorrectnessTests", "-v"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise SystemExit(
        f"GPUCorrectnessTests FAILED (rc={{result.returncode}}). "
        "Do not proceed -- the GPU code path is not within tolerance of the CPU path."
    )
print("All 5 GPUCorrectnessTests passed -- GPU code path matches CPU within tolerance.")
"""
))

# -----------------------------------------------------------------------------
# Cell 8: GPU bench at Gemma workload sizes
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 7: CPU vs GPU bench at Gemma workload sizes

Times `hsic_matrix` and `estimate_hsic_conditional_rff` on CPU and GPU
at two realistic Gemma-shape sizes (35 layers, 512/2048 tokens,
hidden=1536). The expected headline is ~25-40x speedup on the matrix
build at 35×512×1536, and a smaller but still meaningful speedup on
the pair-level conditional estimator.
"""
))

cells.append(code(
    """import json
import subprocess
import sys
from pathlib import Path

result = subprocess.run(
    [sys.executable, "bench_gpu.py",
     "--sizes", "35,512,1536;35,2048,1536",
     "--sub-sample", "2048",
     "--output", str(RESULTS_DIR / "gpu_bench_colab.json")],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise SystemExit(f"bench_gpu.py failed (rc={result.returncode})")

bench = json.loads((RESULTS_DIR / "gpu_bench_colab.json").read_text())
print("\\n=== Headline speedup ===")
for r in bench["results"]:
    if r.get("speedup", 0) > 0:
        print(
            f"  {r['n_layers']}x{r['n_tokens']}x{r['hidden']}: "
            f"{r['cpu_ms']:.0f} ms CPU -> {r['gpu_ms']:.1f} ms GPU "
            f"({r['speedup']:.1f}x; max rel err {r.get('max_rel_err_vs_cpu', 0):.3f})"
        )
"""
))

# Hmm -- `json` isn't imported in that cell. Let me fix that by importing json
# explicitly. The next cell appends a cleaner version below; this old
# cells[7] overwrite is intentionally removed because cell indices have
# shifted (the markdown dividers inserted between every step changed
# what `cells[7]` refers to).
# cells[7] = code(...)   # deprecated -- see the Step 7 bench cell above

# -----------------------------------------------------------------------------
# Cell 9: Headline pipeline -- unconditional MI on GPU
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 8: Real Gemma pipeline (unconditional MI, GPU)

Runs `scripts/run_cross_layer_mi_colab.py` with `--device cuda`. This is
the headline deliverable: a single forward pass through Gemma 4 E2B,
HSIC matrix build on the captured per-layer activations, and bit
allocation at the configured target BPW (4.0). Output goes to
`RESULTS_DIR / "cross_layer_mi_colab.json"`.

Calibration activations are cached at `RESULTS_DIR / "calib_activations.pt"`,
so the conditional-MI run in step 10 reuses them and only re-runs the
delta preprocessing + matrix build.
"""
))

cells.append(code(
    """import subprocess
import sys
from pathlib import Path

# Use sub-sample=2048 to keep the matrix build snappy on T4. The full
# calibration batch is 4-8x slower; bump this if you're on L4 / A100.
cmd = [
    sys.executable, "scripts/run_cross_layer_mi_colab.py",
    "--model-path", str(MODEL_DIR),
    "--calib-data", str(CALIB_PATH),
    "--calib-tokens", "2048",
    "--cache", str(RESULTS_DIR / "calib_activations.pt"),
    "--output", str(RESULTS_DIR / "cross_layer_mi_colab.json"),
    "--progress-output", str(RESULTS_DIR / "cross_layer_mi_colab.progress.json"),
    "--horizon", "4",
    "--target-bpw", "4.0",
    "--bits-min", "1.5",
    "--bits-max", "8.0",
    "--sub-sample", "2048",
    "--device", "cuda",
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise SystemExit(f"unconditional pipeline failed (rc={result.returncode})")

REPORT_UNCOND = RESULTS_DIR / "cross_layer_mi_colab.json"
print(f"Report written to {REPORT_UNCOND} ({REPORT_UNCOND.stat().st_size} bytes)")
"""
))

# -----------------------------------------------------------------------------
# Cell 10: Render unconditional MI summary table
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 9: Render the unconditional MI summary table

Reads the JSON report and shows it as a markdown table -- per-layer MI
score, bit allocation, sigma score, and the sigma rank. The Kendall
tau between MI and sigma rankings at the bottom is the headline
"does cross-layer give a different answer than per-layer?" number.
"""
))

cells.append(code(
    """import json
from pathlib import Path
from IPython.display import display, Markdown

report = json.loads((RESULTS_DIR / "cross_layer_mi_colab.json").read_text())

n = report["n_layers"]
mi_scores = report["mi_scores"]
sigma_scores = report["sigma_scores"]
bits = report["bits_per_layer"]

# Rank by MI descending.
mi_order = sorted(range(n), key=lambda i: -mi_scores[i])
sigma_rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -sigma_scores[i]))}

rows = ["| rank | layer | mi_score | bits | sigma_score | sigma_rank |",
        "|------|-------|----------|------|-------------|------------|"]
for rank, i in enumerate(mi_order):
    rows.append(
        f"| {rank:4d} | {i:5d} | {mi_scores[i]:8.4f} | {bits[i]:5.2f} "
        f"| {sigma_scores[i]:8.4f} | {sigma_rank[i]:10d} |"
    )

display(Markdown("### Unconditional MI (HSIC-RFF, horizon=4) on Gemma 4 E2B\\n\\n"
                 + "\\n".join(rows) +
                 f"\\n\\n**Layers:** {n} &nbsp; "
                 f"**Avg bits:** {report['avg_bits']:.2f} (target {report['target_avg_bpw']}) &nbsp; "
                 f"**MI vs sigma Kendall tau:** {report['kendall_tau_vs_sigma']:+.4f}\\n\\n"
                 f"Method tag: `{report['method']}`"))
"""
))

# -----------------------------------------------------------------------------
# Cell 11: Conditional MI (delta conditioning)
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 10: Conditional MI (`conditioning=delta`)

Re-runs the pipeline with `--conditioning delta`. This preprocesses
each layer's activations to its per-position delta from the previous
layer, removing the residual-stream confounder by construction. The
HSIC matrix then approximates `I(subblock_l; subblock_{l+k} | x_{l-1})`
instead of `I(x_l; x_{l+k})`.

If the per-layer ranking changes meaningfully between this and step 9,
the unconditional matrix is partly measuring how the residual stream
threads through the network -- which is a property of the architecture,
not the weights' quantization sensitivity.
"""
))

cells.append(code(
    """import subprocess
import sys
from pathlib import Path

cmd = [
    sys.executable, "scripts/run_cross_layer_mi_colab.py",
    "--model-path", str(MODEL_DIR),
    "--calib-data", str(CALIB_PATH),
    "--calib-tokens", "2048",
    "--cache", str(RESULTS_DIR / "calib_activations.pt"),  # reuse cache
    "--output", str(RESULTS_DIR / "cross_layer_mi_colab_delta.json"),
    "--progress-output", str(RESULTS_DIR / "cross_layer_mi_colab_delta.progress.json"),
    "--horizon", "4",
    "--target-bpw", "4.0",
    "--bits-min", "1.5",
    "--bits-max", "8.0",
    "--sub-sample", "2048",
    "--device", "cuda",
    "--conditioning", "delta",
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise SystemExit(f"delta-conditioning pipeline failed (rc={result.returncode})")

REPORT_COND = RESULTS_DIR / "cross_layer_mi_colab_delta.json"
print(f"Delta-conditioned report written to {REPORT_COND}")
"""
))

# -----------------------------------------------------------------------------
# Cell 12: Compare unconditional vs conditional rankings
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 11: Compare unconditional vs conditional rankings

Reads both JSON reports and shows the Kendall tau between the two MI
rankings, the top-10 layers by each, and the per-layer bit allocation
delta. The expected outcome is that the top-10 lists differ in at
least 1-3 positions; an identical top-10 means the residual-stream
confounder isn't material on this model and conditional MI isn't
adding new information.
"""
))

cells.append(code(
    """import json
from pathlib import Path
from IPython.display import display, Markdown

uncond = json.loads((RESULTS_DIR / "cross_layer_mi_colab.json").read_text())
cond = json.loads((RESULTS_DIR / "cross_layer_mi_colab_delta.json").read_text())

n = uncond["n_layers"]
assert cond["n_layers"] == n, "layer count mismatch between unconditional and conditional"

# Kendall tau between the two MI rankings.
def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                conc += 1
            elif da * db < 0:
                disc += 1
    return (conc - disc) / max(1, conc + disc)

tau = kendall_tau(uncond["mi_scores"], cond["mi_scores"])

# Top-10 layers by each.
uncond_top = sorted(range(n), key=lambda i: -uncond["mi_scores"][i])[:10]
cond_top = sorted(range(n), key=lambda i: -cond["mi_scores"][i])[:10]
overlap = set(uncond_top) & set(cond_top)

# Per-layer bit delta (conditional - unconditional).
bits_u = uncond["bits_per_layer"]
bits_c = cond["bits_per_layer"]
bit_delta = [bits_c[i] - bits_u[i] for i in range(n)]
biggest_movers = sorted(range(n), key=lambda i: -abs(bit_delta[i]))[:10]

rows = ["| rank | layer | mi_uncond | mi_cond | bits_uncond | bits_cond | delta_bits |",
        "|------|-------|-----------|---------|-------------|-----------|------------|"]
for rank, i in enumerate(biggest_movers):
    rows.append(
        f"| {rank:4d} | {i:5d} | {uncond['mi_scores'][i]:8.4f} | {cond['mi_scores'][i]:8.4f} "
        f"| {bits_u[i]:5.2f} | {bits_c[i]:5.2f} "
        f"| {bit_delta[i]:+10.3f} |"
    )

display(Markdown("### Unconditional vs conditional (delta) MI on Gemma 4 E2B\\n\\n"
                 f"**Kendall tau between MI rankings:** {tau:+.4f}\\n\\n"
                 f"**Top-10 unconditional:** {uncond_top}\\n\\n"
                 f"**Top-10 conditional:**     {cond_top}\\n\\n"
                 f"**Overlap:** {len(overlap)}/10 layers "
                 f"({'identical top-10' if len(overlap) == 10 else 'rankings differ -- residual-stream confounder is material'})\\n\\n"
                 "### Biggest bit movers\\n\\n"
                 + "\\n".join(rows) +
                 "\\n\\nIf the top-10 overlap is <10, the delta-conditioning has "
                 "removed information from the residual stream -- that's the point."))
"""
))

# -----------------------------------------------------------------------------
# Cell 13: MI-biased mixed-budget scan
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 12: MI-biased mixed-budget scan

Uses the delta-conditioned MI report as a layer prior for the mixed-budget
allocator. This is the first candidate that actually links the cross-layer MI
signal to a storage artifact plan. The scan is resumable: completed layer
candidates append to `mixed_budget_scan_full_g128_target4p0_mi.layers.jsonl`.
"""
))

cells.append(code(
    """import subprocess
import sys
from pathlib import Path

MI_SCAN_JSON = RESULTS_DIR / "mixed_budget_scan_full_g128_target4p0_mi.json"
MI_SCAN_RESUME = RESULTS_DIR / "mixed_budget_scan_full_g128_target4p0_mi.layers.jsonl"
ACT_WEIGHTS = RESULTS_DIR / "activation_weights_gemma4.json"

cmd = [
    sys.executable, "scripts/scan_mixed_budget.py",
    "--model-dir", str(MODEL_DIR),
    "--group-size", "128",
    "--outlier-options", "4,8",
    "--target-bpw", "4.0",
    "--mi-report", str(RESULTS_DIR / "cross_layer_mi_colab_delta.json"),
    "--mi-prior", "1.0",
    "--resume-jsonl", str(MI_SCAN_RESUME),
    "--output", str(MI_SCAN_JSON),
]
if ACT_WEIGHTS.exists():
    cmd.extend(["--activation-weights", str(ACT_WEIGHTS)])

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise SystemExit(f"MI-biased mixed-budget scan failed (rc={result.returncode})")
print(f"MI-biased scan written to {MI_SCAN_JSON} ({MI_SCAN_JSON.stat().st_size} bytes)")
"""
))

# -----------------------------------------------------------------------------
# Cell 14: Build MI-biased mixed-budget checkpoint
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 13: Build the MI-biased mixed-budget checkpoint

Builds the checkpoint selected by the MI-biased scan. Layer shards are written
as individual `.pt` files under `RESULTS_DIR / "mixed_budget_mi_shards"` and
reused on rerun, so a Colab disconnect does not force the build to start over.
"""
))

cells.append(code(
    """import subprocess
import sys
from pathlib import Path

MI_CKPT = RESULTS_DIR / "gemma_mixed_budget_full_g128_target4p0_mi.pt"
MI_SHARDS = RESULTS_DIR / "mixed_budget_mi_shards"

cmd = [
    sys.executable, "scripts/quantize_mixed_budget.py",
    "--scan-json", str(RESULTS_DIR / "mixed_budget_scan_full_g128_target4p0_mi.json"),
    "--model-dir", str(MODEL_DIR),
    "--checkpoint-dir", str(MI_SHARDS),
    "--output", str(MI_CKPT),
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("--- STDERR ---")
    print(result.stderr)
    raise SystemExit(f"MI-biased checkpoint build failed (rc={result.returncode})")
print(f"MI-biased checkpoint written to {MI_CKPT} ({MI_CKPT.stat().st_size} bytes)")
"""
))

# -----------------------------------------------------------------------------
# Cell 15: Persist results to Drive (optional)
# -----------------------------------------------------------------------------
cells.append(md(
    """## Step 14: Mirror repo-local results to Google Drive (optional)

If `GDRIVE_RESULTS_DIR` is unset, this cell is skipped. If it is set, most
artifacts were already written directly to Drive through `RESULTS_DIR`; this
cell also mirrors any remaining repo-local `eval_results/` JSON/PT files.
"""
))

cells.append(code(
    """import shutil
from pathlib import Path

if not GDRIVE_RESULTS_DIR:
    print("GDRIVE_RESULTS_DIR not set -- skipping Drive mirror.")
else:
    target = RESULTS_DIR
    for src in Path("eval_results").glob("*.json"):
        if src.resolve() != (target / src.name).resolve():
            shutil.copy2(src, target / src.name)
    for src in Path("eval_results").glob("*.pt"):
        if src.resolve() != (target / src.name).resolve():
            shutil.copy2(src, target / src.name)
    print(f"Mirrored repo-local eval_results/ to {target}")
    print("Files in results folder:")
    for p in sorted(target.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
"""
))

# -----------------------------------------------------------------------------
# Cell 14: Wrap-up + where to look next
# -----------------------------------------------------------------------------
cells.append(md(
    """## What you have now

After running all cells you have:

* `eval_results/cross_layer_mi_colab.json` -- unconditional MI on real Gemma 4 E2B (35 layers, 2048 calib tokens), per-layer `mi_scores`, `bits`, `kendall_tau_vs_sigma`, `method` tag.
* `eval_results/cross_layer_mi_colab_delta.json` -- same pipeline with `--conditioning delta`. The two rankings should differ; if they don't, the residual-stream confounder isn't material on this model.
* `eval_results/gpu_bench_colab.json` -- CPU vs GPU timing at two Gemma-shape workloads.
* `mixed_budget_scan_full_g128_target4p0_mi.json` -- MI-biased mixed-budget allocation plan.
* `gemma_mixed_budget_full_g128_target4p0_mi.pt` -- MI-biased checkpoint if the build step completed.
* `calib_activations.pt` -- cached calibration activations (reused across the unconditional + conditional runs, so the conditional step is ~free beyond the delta + matrix build).

## Where to look next

* **Headline claim**: "Cross-layer MI gives a different bit allocation than per-layer sigma on Gemma 4 E2B." The Kendall tau between MI and sigma ranking should be **negative** (above ~0.6 in absolute value) for that claim to hold. The runbook's `reports/baseline/cross_layer_mi_report.json` on CPU shows what to compare against.
* **Conditional claim**: "After removing the residual stream via `delta`-conditioning, the top-10 most-important layers change." The overlap should be **<10/10**. If it's exactly 10, the per-layer static signal already absorbed the residual stream and the conditional MI is no longer providing new information.
* **Speedup claim**: GPU matrix build should be **~25-40x** faster than CPU on the 35×512×1536 workload. The `gpu_bench_colab.json` file has the raw numbers.

## If anything failed

* `CUDA available: False` after switching the runtime -- you probably got a CPU-only PyTorch wheel. Run the install cell again, then the GPU sanity cell.
* `GPUCorrectnessTests FAILED` -- the GPU code path has drifted from the CPU path. Don't ship the numbers; file an issue with the failing cell output.
* `OutOfMemoryError` during the matrix build -- lower `--sub-sample` from 2048 to 1024, or reduce `--calib-tokens` to 1024. T4 is tight on the 4096-token build.
* The downloads hang on HF Hub -- check `HF_TOKEN` is set and you've accepted the Gemma license at https://huggingface.co/google/gemma-4-E2B.
"""
))


def build_notebook() -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
                "collapsed_sections": [],
            },
            "title": "Cross-layer MI on Gemma 4 E2B (GPU)",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Wrote {NOTEBOOK_PATH} ({NOTEBOOK_PATH.stat().st_size} bytes)")
    print(f"  {n_code} code cells, {n_md} markdown cells")


if __name__ == "__main__":
    main()
