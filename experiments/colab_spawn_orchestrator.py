"""Spawn the act_weighted_pareto_scan orchestrator in background on colab.

Writes PID + log path so it can be polled.
"""
import os
import subprocess
import time

os.chdir("/content/sub1quant")
log_path = "/content/pareto_scan.log"
pid_path = "/content/pareto_scan.pid"

if os.path.exists(pid_path):
    old_pid = int(open(pid_path).read().strip())
    print(f"Killing old orchestrator pid={old_pid}")
    os.system(f"kill -9 {old_pid} 2>/dev/null || true")
    time.sleep(2)

cmd = (
    "python3 -u /content/sub1quant/experiments/act_weighted_pareto_scan.py "
    "--model-dir /content/models/gemma-4-E2B "
    "--wikitext /content/sub1quant/data/wiki.test.txt "
    "--out-dir /content/sub1quant/eval_results/act_weighted_scan "
    "--targets 3.75,3.5,3.25,3.0,2.75,2.5 "
    "--group-size 128 "
    "--act-tokens 32768 "
    "--act-max-length 512 "
    "--act-stride 512 "
    "--device cuda"
)
print(f"Launching: {cmd}")
print(f"Log: {log_path}")
print(f"PID file: {pid_path}")

with open(log_path, "w") as logf:
    proc = subprocess.Popen(
        cmd.split(), stdout=logf, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    with open(pid_path, "w") as pf:
        pf.write(str(proc.pid))
    print(f"Orchestrator pid={proc.pid}")
print("Done spawning. Poll:")
print(f"  !ps -p $(cat {pid_path})")
print(f"  !tail -n 30 {log_path}")
print(f"  !ls /content/sub1quant/eval_results/act_weighted_scan/")
