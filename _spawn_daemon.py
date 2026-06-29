"""Spawn the daemon download via nohup so it survives when /exec returns.
Reads HF_TOKEN from kernel env if present.
"""
import os, subprocess, sys
LOG = "/content/sub1quant/_download.log"
SCRIPT = "/content/sub1quant/_daemon_download.py"
os.makedirs("/content/sub1quant/models", exist_ok=True)

# Build env, inherit HF_TOKEN if present
env = os.environ.copy()

# Detach with nohup so HTTP closure doesn't kill it
cmd = f"nohup python -u {SCRIPT} > {LOG} 2>&1 & echo $!"
proc = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
pid = proc.stdout.strip().split("\n")[-1]
print(f"OK pid={pid}  cmd='{cmd}'")
print(f"LOG={LOG}")
print(f"STATE=/content/sub1quant/_download_state.json")
