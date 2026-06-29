"""Check what's on the kernel in /tmp/_chunks."""
import os, glob
d = "/tmp/_chunks"
if os.path.isdir(d):
    files = sorted(os.listdir(d), key=lambda n: int(n.split("_")[1].split(".")[0]))
    n = len(files)
    total = sum(os.path.getsize(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f)))
    print(f"chunks on kernel: {n} files, {total/1024/1024:.1f} MB total")
    for f in files[-5:]:
        full = os.path.join(d, f)
        if os.path.isfile(full):
            sz = os.path.getsize(full)
            print(f"  {f}  {sz:,} bytes ({sz/1024/1024:.1f} MB)")
else:
    print(f"no chunks dir at {d}")
