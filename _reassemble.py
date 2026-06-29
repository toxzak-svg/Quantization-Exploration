"""Reassemble chunks on the kernel into the final .pt and verify SHA256."""
import os, hashlib, shutil

CHUNKS_DIR = "/tmp/_chunks"
FINAL = "/content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt"
EXPECTED_SHA = "c4337f598433a909e3895a3d3b47f2093dbdba2a1ed4e71502f5549e233f6326"

files = sorted(os.listdir(CHUNKS_DIR), key=lambda n: int(n.split("_")[1].split(".")[0]))
print(f"chunks to cat: {len(files)}")
total_expected = sum(os.path.getsize(os.path.join(CHUNKS_DIR, f)) for f in files if os.path.isfile(os.path.join(CHUNKS_DIR, f)))
print(f"total expected: {total_expected:,} bytes ({total_expected/1024/1024:.2f} MB)")

os.makedirs(os.path.dirname(FINAL), exist_ok=True)
wrote = 0
h = hashlib.sha256()
with open(FINAL, "wb") as out:
    for f in files:
        full = os.path.join(CHUNKS_DIR, f)
        if not os.path.isfile(full):
            print(f"  MISSING {f}")
            continue
        with open(full, "rb") as fp:
            while True:
                block = fp.read(8 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
                h.update(block)
                wrote += len(block)
        print(f"  appended {f}  {os.path.getsize(full):,}")

print(f"\nreassembled {wrote:,} bytes -> {FINAL}")
print(f"sha256: {h.hexdigest()}")
print(f"expect: {EXPECTED_SHA}")
print("MATCH" if h.hexdigest() == EXPECTED_SHA else "MISMATCH")
print(f"final size: {os.path.getsize(FINAL):,}")
