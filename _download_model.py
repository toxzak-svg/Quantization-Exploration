"""Download google/gemma-4-E2B to /content/sub1quant/models/gemma-4-E2B on the kernel.

Uses huggingface_hub.snapshot_download so the structure matches the local
models/gemma-4-E2B folder.

Accepts a HF_TOKEN env var if set, but tries with anonymous first (gated models
will fail).
"""
import os, sys, time, traceback, json

REPO = os.environ.get("HF_REPO", "google/gemma-4-E2B")
LOCAL = os.environ.get("LOCAL_DIR", "/content/sub1quant/models/gemma-4-E2B")
TOKEN = os.environ.get("HF_TOKEN") or None

print(f"REPO   = {REPO}")
print(f"LOCAL  = {LOCAL}")
print(f"TOKEN  = {'set' if TOKEN else '(none, anonymous)'}")

# Make sure hf_hub_download works in 5.12.0-era transformers world
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    sys.exit(0)

os.makedirs(LOCAL, exist_ok=True)
os.makedirs("/content/sub1quant/models", exist_ok=True)

t0 = time.time()
try:
    path = snapshot_download(
        repo_id=REPO,
        local_dir=LOCAL,
        local_dir_use_symlinks=False,
        token=TOKEN,
        max_workers=4,
    )
    print(f"DOWNLOAD OK in {time.time()-t0:.1f}s -> {path}")
except Exception as e:
    print(f"DOWNLOAD FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\nDirectory listing:")
    if os.path.isdir(LOCAL):
        for n in sorted(os.listdir(LOCAL)):
            full = os.path.join(LOCAL, n)
            sz = os.path.getsize(full) if os.path.isfile(full) else "<dir>"
            print(f"  {n}\t{sz}")
    sys.exit(0)

# Verify
print("\nFiles at LOCAL:")
total = 0
for n in sorted(os.listdir(LOCAL)):
    full = os.path.join(LOCAL, n)
    if os.path.isfile(full):
        sz = os.path.getsize(full)
        total += sz
        print(f"  {n}\t{sz:,} bytes")
print(f"TOTAL: {total:,} bytes  ({total/1e9:.2f} GB)")
