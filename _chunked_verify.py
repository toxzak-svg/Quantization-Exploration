import os, subprocess, hashlib, sys
chunks_dir = "/tmp/_chunks"
final_path = "/content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt"
os.makedirs(os.path.dirname(final_path), exist_ok=True)
files = sorted(os.listdir(chunks_dir), key=lambda n: int(n.split("_")[2].split(".")[0]))
print("chunks to cat:", files)
total = 0
with open(final_path, "wb") as out:
    for n in files:
        p = os.path.join(chunks_dir, n)
        sz = os.path.getsize(p)
        total += sz
        with open(p, "rb") as f:
            while True:
                block = f.read(8 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
        print(f"  appended {n} ({sz:,} bytes)")
print("reassembled", total, "bytes to", final_path)
# Verify hash
h = hashlib.sha256()
with open(final_path, "rb") as f:
    while True:
        block = f.read(8 * 1024 * 1024)
        if not block:
            break
        h.update(block)
print("sha256:", h.hexdigest())
print("expected:", "c4337f598433a909e3895a3d3b47f2093dbdba2a1ed4e71502f5549e233f6326")
print("MATCH" if h.hexdigest() == "c4337f598433a909e3895a3d3b47f2093dbdba2a1ed4e71502f5549e233f6326" else "MISMATCH")
