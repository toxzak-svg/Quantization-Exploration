"""Find any HF auth state on the kernel."""
import os, subprocess
# Check env
print("=== env vars ===")
for k in os.environ:
    if "HF" in k.upper() or "HUGG" in k.upper() or "HUGGING" in k.upper():
        v = os.environ[k]
        vmask = v[:8] + "..." if len(v) > 12 else v
        print(f"  {k} = {vmask}  (len {len(v)})")

# Check saved HF tokens
print("\n=== saved tokens ===")
try:
    out = subprocess.check_output(["ls", "-la", "/root/.huggingface/"], text=True)
    print(out)
    for f in os.listdir("/root/.huggingface/"):
        full = "/root/.huggingface/" + f
        if os.path.isfile(full):
            sz = os.path.getsize(full)
            print(f"  {f}  {sz} bytes")
except Exception as e:
    print(f"  err: {e}")

print("\n=== HF Hub default cache ===")
hf_cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
print(f"  HF_HOME = {hf_cache}")
if os.path.isdir(hf_cache):
    print("  exists")

# Check if cookies/token file
for p in ["/root/.cache/huggingface/token", "/root/.huggingface/token", "/root/.cache/huggingface/credentials/token"]:
    if os.path.exists(p):
        print(f"\nTOKEN FILE: {p}")
        try:
            with open(p) as f:
                v = f.read().strip()
                print(f"  contents: {v[:8]}... (len {len(v)})")
        except Exception as e:
            print(f"  err: {e}")
