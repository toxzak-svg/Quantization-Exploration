"""Daemon-style model download that survives the bridge HTTP closure.

Writes its progress to /content/sub1quant/_download_state.json every chunk.
Stays alive after the bridge HTTP request returns. Use ps and tail to monitor.
"""
import os, sys, json, time, traceback

LOG = "/content/sub1quant/_download.log"
STATE = "/content/sub1quant/_download_state.json"
LOCAL = "/content/sub1quant/models/gemma-4-E2B"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def write_state(**kw):
    s = dict(kw)
    s["updated_at"] = time.time()
    with open(STATE, "w") as f:
        json.dump(s, f)

log("START _daemon_download.py")
write_state(status="starting", repo="google/gemma-4-E2B", local=LOCAL)
os.makedirs(os.path.dirname(LOCAL), exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)

try:
    from huggingface_hub import snapshot_download
    log("imported snapshot_download")
except Exception as e:
    log(f"huggingface_hub import FAILED: {e}")
    write_state(status="err", error=f"import fail: {e}")
    sys.exit(0)

t0 = time.time()
write_state(status="downloading")
try:
    path = snapshot_download(
        repo_id="google/gemma-4-E2B",
        local_dir=LOCAL,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
        tqdm_class=None,
    )
    log(f"DOWNLOAD OK in {time.time()-t0:.1f}s -> {path}")
    files = sorted(os.listdir(LOCAL))
    total = 0
    for n in files:
        full = os.path.join(LOCAL, n)
        if os.path.isfile(full):
            sz = os.path.getsize(full)
            total += sz
            log(f"  {n}  {sz:,} bytes")
    log(f"TOTAL: {total:,} bytes ({total/1e9:.2f} GB)")
    write_state(status="done", elapsed=time.time()-t0, total_bytes=total, files=files)
except Exception as e:
    log(f"DOWNLOAD FAILED: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    write_state(status="err", error=f"{type(e).__name__}: {e}")
