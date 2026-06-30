"""Background launcher for sub1quant experiment on colab.

Spawns three long-running subprocesses and exits immediately:
  1. Model download (gemma-4-E2B, ~10GB)
  2. WikiText-2 download
  3. (none yet - experiment runs in foreground after model lands)

Each subprocess writes to /content/<name>.log so you can poll progress.
"""
import os
import subprocess
import time
from pathlib import Path

os.makedirs("/content/models", exist_ok=True)
os.makedirs("/content/sub1quant/data", exist_ok=True)


def spawn(cmd: str, log_path: str) -> int:
    """Spawn a nohup'd subprocess writing to log_path. Returns pid."""
    print(f"Launching: {cmd} -> {log_path}", flush=True)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    pid = subprocess.Popen(
        ["bash", "-c", f"nohup {cmd} > {log_path} 2>&1 & echo $!"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, _ = pid.communicate(timeout=10)
    return int(out.strip())


# 1. Model download
model_pid = spawn(
    "python3 -u -c \"from huggingface_hub import snapshot_download; "
    "snapshot_download(repo_type='model', repo_id='google/gemma-4-E2B', "
    "local_dir='/content/models/gemma-4-E2B', max_workers=4)\"",
    "/content/model_download.log",
)
print(f"Model download PID: {model_pid}")

# 2. WikiText-2 download
wt_pid = spawn(
    "python3 -u -c \"from datasets import load_dataset; "
    "ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test'); "
    "open('/content/sub1quant/data/wiki.test.txt', 'w').write('\\n'.join(ds['text'])); "
    "print('wikitext saved', len(ds), 'rows')\"",
    "/content/wikitext_download.log",
)
print(f"WikiText PID: {wt_pid}")

print("Both background jobs launched. Poll with:")
print("  !ls -lh /content/models/gemma-4-E2B/")
print("  !tail -n 20 /content/model_download.log")
print("  !tail -n 20 /content/wikitext_download.log")
print("  !ls -la /content/sub1quant/data/")
