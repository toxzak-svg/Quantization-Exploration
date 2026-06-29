"""Spawn the model download as a detached subprocess on the colab kernel.
Then return immediately so the bridge HTTP call doesn't time out.
"""
import os, subprocess, sys

LOG = "/content/sub1quant/_download.log"
CMD = [sys.executable, "-u", "/content/sub1quant/_download_model.py"]

os.makedirs("/content/sub1quant/models", exist_ok=True)
f = open(LOG, "ab", buffering=0)
proc = subprocess.Popen(
    CMD,
    stdout=f,
    stderr=f,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
print(f"OK spawned pid={proc.pid}")
print(f"LOG  = {LOG}")
