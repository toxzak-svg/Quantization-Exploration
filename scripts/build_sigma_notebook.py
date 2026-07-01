"""Build the SIGMA sketch ablation Colab notebook.

This script generates a self-contained Jupyter notebook that:

  1. Reads HF_TOKEN / GH_TOKEN / GH_REPO_ID / HF_REPO_ID / MODEL_ID from
     Colab Secrets (or environment variables as a fallback).
  2. Clones the sub1quant repo from GitHub.
  3. Downloads the model weights from HF Hub (or falls back to local).
  4. Runs the SIGMA sketch ablation kill-test on layers[15].mlp.down_proj:
     three sketch families (random Hadamard, learned binary, learned
     continuous) compared on within-bucket magnitude variance and
     reconstruction MSE.
  5. Saves the ablation results locally (small JSON + plot) and pushes
     everything to Hugging Face Hub.
  6. Commits a results summary back to the GitHub repo.

The notebook is intentionally self-contained: the SIGMA primitive, the
sketch families, and the generator training live in one inline module
cell. The notebook does NOT depend on the scripts/ tree at runtime, only
on the repo for config / tokenizer / model.

Run this builder from the project root:

    python scripts/build_sigma_notebook.py

The output is written to notebook/sigma_sketch_ablation.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebook" / "sigma_sketch_ablation.ipynb"
DEFAULT_GH_REPO = "toxzak-svg/Quantization-Exploration"
DEFAULT_HF_REPO = "toxzak-svg/sub1quant-sigma-ablation"
DEFAULT_MODEL_ID = "google/gemma-4-E2B"
HF_UPLOAD_THRESHOLD_MB = 50  # files smaller than this stay local too
SIGMA_MODULE_SOURCE = (REPO_ROOT / "src" / "sigma_ablation.py").read_text(encoding="utf-8")


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
# Cell 1: Title + instructions
# -----------------------------------------------------------------------------
cells.append(md(
    """# SIGMA sketch ablation â€” Colab runner

This notebook validates the core hypothesis behind **SIGMA** (Sign-Indexed
Generative Magnitude Attractors): *does the sign mask of a weight block
carry enough structure to be a useful index into a magnitude manifold?*

If the answer is no, the rest of the primitive is engineering on top of
a bad idea. If the answer is yes, we have a real alternative to
AQLM / VPTQ / QuIP# vector quantization.

## What it does

1. Loads a single target layer from Gemma-4-E2B (`layers[15].mlp.down_proj`, shape 1536x12288, ~295k blocks of 64 weights).
2. Runs three sketch families on the sign mask:
   - **random_hadamard** â€” fixed orthogonal projection, baseline.
   - **learned_binary** â€” projection trained via straight-through estimator to minimize within-bucket magnitude variance.
   - **learned_continuous** â€” same objective, but the projection itself is continuous.
3. For each sketch family:
   - Measures **within-bucket variance ratio** (the kill-test number â€” <0.3 viable, >0.6 dead).
   - Trains a low-rank generative magnitude model (U, V, R factors + softplus).
   - Quantizes the layer and measures reconstruction MSE.
4. Saves the table + plot locally, pushes to HF Hub, commits a summary back to the repo.

## Required Colab Secrets

Open the key icon on the left sidebar and set:

| Secret | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | yes | Download gated model + push to Hub |
| `GH_TOKEN` | yes | Push results back to your GitHub repo |
| `MODEL_ID` | no | HF repo id; default `google/gemma-4-E2B` |
| `GH_REPO_ID` | no | default `toxzak-svg/Quantization-Exploration` |
| `HF_REPO_ID` | no | default `toxzak-svg/sub1quant-sigma-ablation` |

Runtime: at least L4 (24 GB VRAM). The model is ~9.5 GB safetensors; the
target layer weights alone are ~36 MB, easily handled on any Colab GPU.
"""
))

# -----------------------------------------------------------------------------
# Cell 2: Install dependencies
# -----------------------------------------------------------------------------
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
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "datasets": "datasets",
}
missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *missing])
else:
    print("All deps already installed.")
"""
))

# -----------------------------------------------------------------------------
# Cell 3: Read secrets + clone repo
# -----------------------------------------------------------------------------
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

HF_TOKEN = read_secret("HF_TOKEN")
GH_TOKEN = read_secret("GH_TOKEN")
MODEL_ID = read_secret("MODEL_ID", "{DEFAULT_MODEL_ID}")
GH_REPO_ID = read_secret("GH_REPO_ID", "{DEFAULT_GH_REPO}")
HF_REPO_ID = read_secret("HF_REPO_ID", "{DEFAULT_HF_REPO}")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required (set it in Colab Secrets or HF_TOKEN env var).")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN is required (set it in Colab Secrets or GH_TOKEN env var).")

print("IS_COLAB:", IS_COLAB)
print("MODEL_ID:", MODEL_ID)
print("GH_REPO_ID:", GH_REPO_ID)
print("HF_REPO_ID:", HF_REPO_ID)
print("HF_TOKEN configured:", bool(HF_TOKEN))
print("GH_TOKEN configured:", bool(GH_TOKEN))

REPO_DIR = Path("/content/sub1quant")
if not REPO_DIR.exists():
    clone_url = f"https://{{GH_TOKEN}}@github.com/{{GH_REPO_ID}}.git"
    print("Cloning:", GH_REPO_ID)
    subprocess.check_call(["git", "clone", "--depth=1", clone_url, str(REPO_DIR)])
else:
    print("Repo already at", REPO_DIR)

%cd {{REPO_DIR}}
print("CWD:", Path.cwd())
"""
))

# -----------------------------------------------------------------------------
# Cell 4: GPU + disk check
# -----------------------------------------------------------------------------
cells.append(code(
    """import shutil
import subprocess
import torch

print("=" * 60)
print("RUNTIME ENVIRONMENT")
print("=" * 60)
print(f"PyTorch:     {torch.__version__}")
print(f"CUDA:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM total:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"VRAM free:   {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
print(f"Disk free:   {shutil.disk_usage('/content').free / 1e9:.1f} GB / {shutil.disk_usage('/content').total / 1e9:.1f} GB")
print(f"CPU count:   {subprocess.check_output(['nproc']).decode().strip()}")
"""
))

# -----------------------------------------------------------------------------
# Cell 5: Load model + extract target layer
# -----------------------------------------------------------------------------
cells.append(code(
    """import time
import torch
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

LOCAL_MODEL_PATH = Path("/content/sub1quant/models/gemma-4-E2B")
LOCAL_WEIGHT_FILE = LOCAL_MODEL_PATH / "model.safetensors"

print("=" * 60)
print(f"LOADING MODEL: {MODEL_ID}")
print("=" * 60)
t0 = time.time()

if LOCAL_WEIGHT_FILE.exists():
    model_path = str(LOCAL_MODEL_PATH)
    print(f"Using local model at {model_path}")
else:
    print(f"Downloading from HuggingFace: {MODEL_ID}")
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        token=HF_TOKEN,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
        max_workers=4,
    )
    print(f"Downloaded to {model_path}")

print(f"Loading weights into memory (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Extract target layer.
TARGET_LAYER_IDX = 15
TARGET_MODULE = "mlp.down_proj"
target_layer = model.language_model.layers[TARGET_LAYER_IDX].mlp.down_proj

W = target_layer.weight.data.float().cpu()
print(f"Layer {TARGET_LAYER_IDX}.{TARGET_MODULE} weight shape: {tuple(W.shape)}")
print(f"Load + extract took {time.time() - t0:.1f}s")
print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
"""
))

# -----------------------------------------------------------------------------
# Cell 6: Calibration pass â€” capture inputs to target layer
# -----------------------------------------------------------------------------
cells.append(code(
    """import torch
from datasets import load_dataset
from pathlib import Path
from safetensors.torch import save_file

CALIB_OUT = Path("/content/sigma_ablation/calibration")
CALIB_OUT.mkdir(parents=True, exist_ok=True)

CACHE_FILE = CALIB_OUT / "calib_input.safetensors"

print("=" * 60)
print("CALIBRATION PASS â€” capturing layer inputs")
print("=" * 60)

if CACHE_FILE.exists():
    print(f"Loading cached calibration input from {CACHE_FILE}")
    calib = load_file(str(CACHE_FILE))
    X_calib = calib["X_calib"]
else:
    print("Loading WikiText-2 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\\n\\n".join([t for t in ds["text"] if t.strip()])[:300_000]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    captured = {}
    def hook(module, inp, out):
        captured["input"] = inp[0].detach()

    handle = target_layer.register_forward_hook(hook)
    print(f"Forward pass on {inputs.input_ids.shape[1]} tokens...")
    with torch.no_grad():
        model(**inputs, use_cache=False)
    handle.remove()

    X_calib = captured["input"].float().cpu()
    print(f"Captured input shape: {tuple(X_calib.shape)}")
    save_file({"X_calib": X_calib}, str(CACHE_FILE))
    print(f"Cached to {CACHE_FILE}")

print(f"VRAM after calibration: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Free the model now that we have the layer weights + calibration input.
print("Freeing full model to free VRAM...")
del model
torch.cuda.empty_cache()
print(f"VRAM after model free: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
"""
))

# -----------------------------------------------------------------------------
# Cell 7: Section header â€” The SIGMA primitive
# -----------------------------------------------------------------------------
cells.append(md(
    """## The SIGMA primitive (inline module)

Everything below this cell is the actual primitive: three sketch families,
the generative magnitude model, the per-block quantize/dequant, and the
layer-level ablation helpers. Self-contained â€” no imports from `scripts/`.
"""
))

# -----------------------------------------------------------------------------
# Cell 8: SIGMA module (the actual code)
# -----------------------------------------------------------------------------
cells.append(code(SIGMA_MODULE_SOURCE + """

_default_sigma_config = SIGMAConfig()
print("SIGMA module loaded.")
print("  sketch families:    random_hadamard, learned_binary, learned_continuous")
print(f"  generator:          low-rank bucket/tau factors + softplus, rank={_default_sigma_config.rank}, taus={_default_sigma_config.n_taus}")
print(f"  block size B={_default_sigma_config.block_size}, buckets={1 << _default_sigma_config.n_bits}")
print(f"  bit budget (no residuals): {sigma_bpw(295000, 1 << _default_sigma_config.n_bits, _default_sigma_config.n_taus, _default_sigma_config.rank, _default_sigma_config.block_size):.3f} bpw")
"""))
# -----------------------------------------------------------------------------
# Cell 9: Section header â€” The kill-test ablation
# -----------------------------------------------------------------------------
cells.append(md(
    """## The kill-test ablation

This is the experiment that decides whether the SIGMA primitive is worth
building. For each of the three sketch families we measure:

1. **Within-bucket variance ratio** â€” the headline number. If the sign mask
   carries magnitude information, blocks in the same bucket should have
   similar magnitudes, and this ratio is small.
2. **Reconstruction MSE** â€” does the learned generator actually recover the
   block given just sign + bucket + scale + tau?
3. **Activation-weighted MSE** â€” same, but weighted by the calibration
   input norms. This is what actually matters for downstream PPL.
"""
))

# -----------------------------------------------------------------------------
# Cell 10: Run the ablation
# -----------------------------------------------------------------------------
cells.append(code(
    """import json
import time
from pathlib import Path
from tqdm.notebook import tqdm
import torch

OUT_DIR = Path("/content/sigma_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

B = 64
N_BITS = 8
N_BUCKETS = 1 << N_BITS
N_TAUS = 8
RANK = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

signs, mags, blocks, stats = make_blocks(W, block_size=B)
print(f"Blocks: {stats.n_blocks:,}, B={B}, total weights: {stats.total_elements:,}")
print(f"Sign ones fraction: {signs.float().mean().item():.4f}")
print(f"Mean magnitude (normalized): {mags.mean().item():.4f}")

sketch_fns = [
    ("random_hadamard", lambda s, m: sketch_random_hadamard(s, n_bits=N_BITS, seed=42)),
    ("learned_binary",   lambda s, m: sketch_learned_binary(s, m, n_bits=N_BITS, steps=250)),
    ("learned_continuous", lambda s, m: sketch_learned_continuous(s, m, n_bits=N_BITS, steps=250)),
]

results: dict = {}
budget_bpw = sigma_bpw(stats.n_blocks, N_BUCKETS, N_TAUS, RANK, B)
print(f"SIGMA bit budget (no residuals): {budget_bpw:.3f} bpw\\n")

for name, sketch_fn in tqdm(sketch_fns, desc="Sketch families"):
    print()
    print("=" * 60)
    print(f"SKETCH: {name}")
    print("=" * 60)
    t0 = time.time()

    buckets = sketch_fn(signs, mags)
    unique_buckets = int(buckets.unique().numel())
    ratio = within_bucket_variance_ratio(mags, buckets, N_BUCKETS)
    print(f"Within-bucket var / total var: {ratio:.4f}")
    print(f"Unique buckets used: {unique_buckets} / {N_BUCKETS}")
    print(f"Sketch phase: {time.time() - t0:.1f}s")

    print("Training generator...")
    t0 = time.time()
    G = train_generator(
        signs, blocks, buckets,
        config=SIGMAConfig(block_size=B, n_bits=N_BITS, n_taus=N_TAUS, rank=RANK),
        lr=1e-3, steps=400, device=DEVICE,
    )
    print(f"Generator training: {time.time() - t0:.1f}s")

    print("Quantizing layer...")
    t0 = time.time()
    recon, tau_used, alpha_used = quantize_layer(
        signs, blocks, buckets, G, n_taus=N_TAUS, device=DEVICE,
    )
    print(f"Quantization: {time.time() - t0:.1f}s")

    block_mse = ((blocks - recon) ** 2).mean().item()
    W_hat = recon.reshape(W.shape)
    act_weighted = activation_weighted_mse(W, W_hat, X_calib, block_size=B, device=DEVICE)

    tau_hist = torch.bincount(tau_used, minlength=N_TAUS).tolist()
    print(f"Reconstruction MSE:           {block_mse:.6e}")
    print(f"Activation-weighted MSE:      {act_weighted:.6e}")
    print(f"Tau histogram:                {tau_hist}")
    print(f"Bit budget for this config:   {budget_bpw:.3f} bpw")

    results[name] = {
        "within_bucket_variance_ratio": ratio,
        "unique_buckets_used": unique_buckets,
        "reconstruction_mse": block_mse,
        "activation_weighted_mse": act_weighted,
        "tau_histogram": tau_hist,
        "generator_train_seconds": time.time() - t0,
        "bit_budget_bpw": budget_bpw,
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(
            {
                "config": {
                    "B": B, "n_bits": N_BITS, "n_buckets": N_BUCKETS,
                    "n_taus": N_TAUS, "rank": RANK,
                    "layer": f"layers[{TARGET_LAYER_IDX}].mlp.down_proj",
                    "weight_shape": list(W.shape),
                    "device": DEVICE,
                },
                "results": results,
            },
            f, indent=2,
        )
    print(f"\\nSaved partial results to {OUT_DIR / 'results.json'}")

print()
print("=" * 60)
print("ABLATION COMPLETE")
print("=" * 60)
"""
))

# -----------------------------------------------------------------------------
# Cell 11: Results table
# -----------------------------------------------------------------------------
cells.append(code(
    """import json
from pathlib import Path

OUT_DIR = Path("/content/sigma_ablation")
data = json.loads((OUT_DIR / "results.json").read_text())
results = data["results"]
cfg = data["config"]

print()
print("=" * 88)
print("SIGMA SKETCH ABLATION RESULTS")
print("=" * 88)
print(f"Layer: {cfg['layer']}, shape {tuple(cfg['weight_shape'])}")
print(f"B={cfg['B']}, sketch_bits={cfg['n_bits']}, buckets={cfg['n_buckets']}, "
      f"taus={cfg['n_taus']}, rank={cfg['rank']}")
print(f"Device: {cfg['device']}")
print()
hdr = f"{'Sketch':<22} {'Var ratio':>10} {'Recon MSE':>12} {'ActW MSE':>12} {'Buckets':>8} {'Tau-mode':>9}"
print(hdr)
print("-" * len(hdr))
for name, r in results.items():
    tau_hist = r["tau_histogram"]
    tau_mode = max(range(len(tau_hist)), key=lambda i: tau_hist[i])
    print(f"{name:<22} {r['within_bucket_variance_ratio']:>10.4f} "
          f"{r['reconstruction_mse']:>12.4e} "
          f"{r['activation_weighted_mse']:>12.4e} "
          f"{r['unique_buckets_used']:>8d} "
          f"{tau_mode:>9d}")
print()
print("INTERPRETATION")
print("-" * 88)
print("Within-bucket var ratio:")
print("  < 0.3  sign topology clusters magnitudes well. SIGMA viable.")
print("  0.3-0.6  marginal. SIGMA needs stronger generator or learned sketch.")
print("  > 0.6  primitive needs rethink; signs do not predict magnitudes here.")
print()
print("Learned vs random:")
print("  If learned sketch >> random sketch, sketch is doing real work.")
print("  If they tie, the generator is doing all the heavy lifting.")
print("  If random >> learned, learned projection is collapsing into a bad local minimum.")
"""
))

# -----------------------------------------------------------------------------
# Cell 12: Plot
# -----------------------------------------------------------------------------
cells.append(code(
    """import json
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path("/content/sigma_ablation")
data = json.loads((OUT_DIR / "results.json").read_text())
results = data["results"]

names = list(results.keys())
ratios = [results[n]["within_bucket_variance_ratio"] for n in names]
mses = [results[n]["reconstruction_mse"] for n in names]
awmses = [results[n]["activation_weighted_mse"] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(names, ratios, color=["#3b82f6", "#10b981", "#f59e0b"])
axes[0].axhline(0.3, color="green", linestyle="--", alpha=0.7, label="viable threshold (0.3)")
axes[0].axhline(0.6, color="red", linestyle="--", alpha=0.7, label="kill threshold (0.6)")
axes[0].set_ylabel("Within-bucket var / total var")
axes[0].set_title("Sign-magnitude correlation\n(lower = signs predict magnitudes)")
axes[0].legend(fontsize=8)
axes[0].tick_params(axis="x", rotation=15)
axes[0].set_ylim(0, max(ratios) * 1.2 if max(ratios) > 0 else 1.0)

axes[1].bar(names, mses, color=["#3b82f6", "#10b981", "#f59e0b"])
axes[1].set_ylabel("Reconstruction MSE")
axes[1].set_title("Per-weight MSE\n(lower = better reconstruction)")
axes[1].tick_params(axis="x", rotation=15)
axes[1].set_yscale("log")

axes[2].bar(names, awmses, color=["#3b82f6", "#10b981", "#f59e0b"])
axes[2].set_ylabel("Activation-weighted MSE")
axes[2].set_title("Act-weighted MSE\n(lower = better downstream PPL)")
axes[2].tick_params(axis="x", rotation=15)
axes[2].set_yscale("log")

plt.suptitle("SIGMA sketch ablation â€” Gemma-4-E2B layers[15].mlp.down_proj", y=1.02)
plt.tight_layout()

plot_path = OUT_DIR / "sigma_ablation.png"
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved {plot_path} ({plot_path.stat().st_size / 1024:.1f} KB)")
"""
))

# -----------------------------------------------------------------------------
# Cell 13: Section header â€” Save artifacts
# -----------------------------------------------------------------------------
cells.append(md(
    """## Save artifacts

Local-first: small stuff stays in `/content/sigma_ablation/`. Anything
above 50 MB gets pushed to Hugging Face only (the local /content is wiped
on Colab disconnect).

Then a single GitHub commit puts the summary back in
`eval_results/sigma_ablation/` of the cloned repo.
"""
))

# -----------------------------------------------------------------------------
# Cell 14: Push to Hugging Face
# -----------------------------------------------------------------------------
cells.append(code(
    """import shutil
import time
from datetime import datetime, timezone
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

OUT_DIR = Path("/content/sigma_ablation")

# Local inventory
local_files = sorted([p for p in OUT_DIR.rglob("*") if p.is_file()])
print("=" * 60)
print("LOCAL ARTIFACTS")
print("=" * 60)
total_mb = 0
for p in local_files:
    sz = p.stat().st_size / (1024 * 1024)
    total_mb += sz
    print(f"  {p.relative_to(OUT_DIR):40} {sz:>10.2f} MB")
print(f"  {'TOTAL':40} {total_mb:>10.2f} MB")

if total_mb > 50:
    print(f"Total > 50 MB; uploading only the small files locally and pushing full folder to HF.")
else:
    print(f"Total under threshold; full folder will go to HF Hub as a single commit.")

print()
print("=" * 60)
print("HUGGING FACE HUB UPLOAD")
print("=" * 60)
api = HfApi(token=HF_TOKEN)
create_repo(HF_REPO_ID, token=HF_TOKEN, repo_type="model", exist_ok=True, private=False)
print(f"Repo ready: https://huggingface.co/{HF_REPO_ID}")

commit_msg = f"SIGMA sketch ablation {datetime.now(timezone.utc).isoformat()}"
upload_folder(
    folder_path=str(OUT_DIR),
    repo_id=HF_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message=commit_msg,
)
print(f"Pushed: {commit_msg}")
print(f"URL:    https://huggingface.co/{HF_REPO_ID}/tree/main")
"""
))

# -----------------------------------------------------------------------------
# Cell 15: GitHub commit
# -----------------------------------------------------------------------------
cells.append(code(
    """import shutil
import subprocess
from pathlib import Path

REPO_DIR = Path("/content/sub1quant")
TARGET_DIR = REPO_DIR / "eval_results" / "sigma_ablation"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("/content/sigma_ablation")

print("=" * 60)
print("GITHUB COMMIT")
print("=" * 60)
print(f"Target: {TARGET_DIR.relative_to(REPO_DIR)}")

# Copy artifacts to the repo
copied = []
for src in sorted(OUT_DIR.iterdir()):
    if src.is_file():
        dst = TARGET_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(dst.name)
        print(f"  copied {src.name} -> {dst.relative_to(REPO_DIR)}")

# Add a small README in the results folder so the commit has context.
readme = TARGET_DIR / "README.md"
readme.write_text(
    f'''# SIGMA sketch ablation

Auto-generated by `notebook/sigma_sketch_ablation.ipynb`.

- Layer: `layers[15].mlp.down_proj` (Gemma-4-E2B, {tuple(W.shape)})
- Block size: B=64
- Sketch bits: {N_BITS}
- Generator: rank={RANK}, taus={N_TAUS}

See `results.json` for the full table and `sigma_ablation.png` for the plot.

Interpretation rule of thumb: within-bucket variance ratio < 0.3 means the
sign mask is a useful index into a magnitude manifold and SIGMA is worth
building. > 0.6 means the primitive is dead and a different angle is
needed.
''',
    encoding="utf-8",
)
print(f"  wrote {readme.relative_to(REPO_DIR)}")

# Git add / commit / push
subprocess.check_call(["git", "config", "user.email", "colab-sigma@local"], cwd=REPO_DIR)
subprocess.check_call(["git", "config", "user.name", "Colab SIGMA Ablation"], cwd=REPO_DIR)
subprocess.check_call(["git", "add", "eval_results/sigma_ablation/"], cwd=REPO_DIR)

status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=REPO_DIR, capture_output=True, text=True,
)
if not status.stdout.strip():
    print("No changes to commit.")
else:
    print("Changes staged:")
    print(status.stdout)

    msg = f"sigma: sketch ablation results ({len(copied)} files)"
    commit = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    print(commit.stdout)
    if commit.returncode != 0:
        print("Commit failed:")
        print(commit.stderr)

    push = subprocess.run(
        ["git", "push", f"https://{GH_TOKEN}@github.com/{GH_REPO_ID}.git", "HEAD:main"],
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    print(push.stdout)
    if push.returncode != 0:
        print("Push failed:")
        print(push.stderr)
    else:
        print(f"Pushed to https://github.com/{GH_REPO_ID}/tree/main/eval_results/sigma_ablation")
"""
))

# -----------------------------------------------------------------------------
# Cell 16: Summary + next steps
# -----------------------------------------------------------------------------
cells.append(code(
    """import json
from pathlib import Path

OUT_DIR = Path("/content/sigma_ablation")
data = json.loads((OUT_DIR / "results.json").read_text())
results = data["results"]
cfg = data["config"]

print("=" * 88)
print("FINAL SUMMARY")
print("=" * 88)
print()
print("Artifacts:")
print(f"  Local: /content/sigma_ablation/")
print(f"  HF:    https://huggingface.co/{HF_REPO_ID}")
print(f"  GH:    https://github.com/{GH_REPO_ID}/tree/main/eval_results/sigma_ablation")
print()
print("Headline numbers:")
for name, r in results.items():
    verdict = "VIABLE" if r["within_bucket_variance_ratio"] < 0.3 else ("MARGINAL" if r["within_bucket_variance_ratio"] < 0.6 else "KILLED")
    print(f"  {name:<22} var_ratio={r['within_bucket_variance_ratio']:.4f}  recon_mse={r['reconstruction_mse']:.4e}  -> {verdict}")
print()
print("Next steps depending on outcome:")
print("  If all three sketches kill (>0.6):")
print("    -> SIGMA primitive is dead in this form. Try activation-conditional decoder,")
print("       or different generative factor (per-channel scale, learned rotation).")
print("  If learned sketches beat random:")
print("    -> The sketch is doing real work. Build the full SIGMA quantizer.")
print("  If learned == random:")
print("    -> Generator is doing all the work. SIGMA reduces to AQLM with sign-derived ID;")
print("       novelty is thin, but the engineering might still be publishable.")
print("  If random beats learned:")
print("    -> Learned projection is collapsing; fix training (longer, different lr, beta-anneal).")
"""
))

# -----------------------------------------------------------------------------
# Build the notebook JSON
# -----------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "collapsed_sections": [],
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
print(f"Cells: {len(cells)}")
print(f"Size:  {NOTEBOOK_PATH.stat().st_size / 1024:.1f} KB")

