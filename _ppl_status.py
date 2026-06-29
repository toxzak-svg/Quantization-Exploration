"""Check ppl run progress: tail the log and report key state."""
import os, json, time, subprocess
LOG = "/content/sub1quant/_ppl.log"
print(f"=== {LOG} ===")
if not os.path.exists(LOG):
    print("(no log yet)")
else:
    with open(LOG, "rb") as f:
        data = f.read()
    print(f"(size: {len(data)} bytes)")
    # only last ~2KB
    print(data[-3000:].decode(errors="replace"))

print("=== PS ===")
try:
    print(subprocess.check_output(["bash", "-c", "ps -ef | grep -E 'python|test_perplexity' | grep -v grep | head -10"], text=True))
except Exception as e:
    print(f"err: {e}")
