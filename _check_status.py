"""Status check on the kernel — log, files, running processes."""
import os, subprocess, time
print("CWD:", os.getcwd())
print("---")
print("DISK:")
print(subprocess.check_output(["df", "-h", "/content"], text=True))
print("---")
print("LIST /content/sub1quant:")
print(subprocess.check_output(["ls", "-la", "/content/sub1quant"], text=True))
print("---")
print("LIST /content/sub1quant/models:")
try:
    print(subprocess.check_output(["ls", "-la", "/content/sub1quant/models"], text=True))
except Exception as e:
    print(f"  err: {e}")
print("---")
print("LIST /content/sub1quant/models/gemma-4-E2B (if any):")
try:
    print(subprocess.check_output(["ls", "-la", "/content/sub1quant/models/gemma-4-E2B"], text=True))
except Exception as e:
    print(f"  err: {e}")
print("---")
print("PS:")
print(subprocess.check_output(["bash", "-c", "ps -ef | grep -E 'python|wget|curl' | grep -v grep | head -30"], text=True))
print("---")
print("TAIL /content/sub1quant/_download.log:")
try:
    print(subprocess.check_output(["tail", "-n", "50", "/content/sub1quant/_download.log"], text=True))
    print("--end--")
except Exception as e:
    print(f"  err: {e}")
print("---")
print(f"now: {time.time():.0f}")
