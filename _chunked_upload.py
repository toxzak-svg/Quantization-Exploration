r"""Split the .pt into 32MB chunks and upload each via the bridge, then reassemble.

Usage: python _chunked_upload.py <local_pt> <remote_dir>
  local_pt:   e.g. C:\Users\Zwmar\projects\sub1quant\quantized\gemma_mixed_budget_full_g128_target4p0.pt
  remote_dir: e.g. /content/sub1quant/quantized
"""
import os, sys, math, time, json, hashlib, urllib.request, urllib.error

BASE = "https://prime-apparently-types-explanation.trycloudflare.com"
TOKEN = "pNWps4V97_opsqamUqTwvr8HK4JI_bDd"

CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB
MAX_TRIES = 3

LOCAL_PT = sys.argv[1]
REMOTE_DIR = sys.argv[2]
LOCAL_SIZE = os.path.getsize(LOCAL_PT)
LOCAL_SHA = hashlib.sha256(open(LOCAL_PT, "rb").read()).hexdigest()
NCHUNKS = math.ceil(LOCAL_SIZE / CHUNK_SIZE)

print(f"local   : {LOCAL_PT}")
print(f"remote  : {REMOTE_DIR}")
print(f"size    : {LOCAL_SIZE:,} bytes  ({LOCAL_SIZE/1024/1024:.2f} MB)")
print(f"sha256  : {LOCAL_SHA}")
print(f"chunks  : {NCHUNKS}  x {CHUNK_SIZE//1024//1024} MB")

# We'll temporarily name chunks in /tmp/_chunked_upload/
REMOTE_TMP = "/tmp/_chunks"
FINAL_REMOTE = os.path.join(REMOTE_DIR, os.path.basename(LOCAL_PT)).replace("\\", "/")

# Write the reassembly + verification script (it's tiny)
verify_script = f"""import os, subprocess, hashlib, sys
chunks_dir = "{REMOTE_TMP}"
final_path = "{FINAL_REMOTE}"
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
        print(f"  appended {{n}} ({{sz:,}} bytes)")
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
print("expected:", "{LOCAL_SHA}")
print("MATCH" if h.hexdigest() == "{LOCAL_SHA}" else "MISMATCH")
"""
# Write verify script to local tmpfile for upload
VERIFY_PATH = os.path.join(os.path.dirname(__file__), "_chunked_verify.py")
with open(VERIFY_PATH, "w") as f:
    f.write(verify_script)
print(f"wrote {VERIFY_PATH}")

# Step 1: ask kernel to clear any prior /tmp/_chunks
def exec_py(code, timeout=30):
    body = json.dumps({"code": code}).encode()
    req = urllib.request.Request(f"{BASE}/exec", method="POST",
                                  headers={"X-Bridge-Token": TOKEN, "Content-Type": "application/json"},
                                  data=body)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
print("\n=== prep kernel ===")
print(json.dumps(exec_py(f"import os, shutil; os.makedirs('{REMOTE_TMP}', exist_ok=True); print('ready')"), indent=2))
print()

# Step 2: upload chunks
chunks = []
with open(LOCAL_PT, "rb") as f:
    idx = 0
    while True:
        block = f.read(CHUNK_SIZE)
        if not block:
            break
        idx += 1
        remote_chunk = f"{REMOTE_TMP}/chunk_{idx:03d}.bin"
        chunks.append(remote_chunk)
        for attempt in range(1, MAX_TRIES + 1):
            boundary = "----MavisChunkBoundary7e3f1a02"
            head = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="path"\r\n\r\n'
                f"{remote_chunk}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="chunk_{idx:03d}.bin"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            body = head + block + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(f"{BASE}/upload", data=body, method="POST",
                                          headers={"X-Bridge-Token": TOKEN,
                                                   "Content-Type": f"multipart/form-data; boundary={boundary}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read())
                if resp.get("ok"):
                    sz = len(block)
                    elapsed = time.time() - t0
                    rate = sz / elapsed / 1e6 if elapsed > 0 else 0
                    print(f"  chunk {idx}/{NCHUNKS}  {sz/1024/1024:.0f}MB  {elapsed:.1f}s  {rate:.1f}MB/s  OK")
                    break
                else:
                    print(f"  chunk {idx} server-rejected: {resp}")
            except urllib.error.HTTPError as e:
                print(f"  chunk {idx} HTTP {e.code} after {time.time()-t0:.1f}s: {e.read()[:200].decode()}")
            except Exception as e:
                print(f"  chunk {idx} ERR after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
            if attempt < MAX_TRIES:
                time.sleep(3 * attempt)
        else:
            print(f"chunk {idx} failed all attempts; aborting")
            sys.exit(1)

print(f"\n=== uploaded {len(chunks)} chunks ===")

# Step 3: upload the verify script as a tiny /upload
print("\n=== upload verify script ===")
print(json.dumps(upload_text_or_multipart(f"{REMOTE_DIR}/_chunked_verify.py", open(VERIFY_PATH, "rb").read()), indent=2))

# Step 4: kick off verification+reassembly on the kernel
print("\n=== run verify on kernel ===")
reassembly_cmd = (
    f"import sys; sys.path.insert(0, '{REMOTE_DIR}'); "
    f"exec(open('{REMOTE_DIR}/_chunked_verify.py').read())"
)
resp = exec_py(reassembly_cmd, timeout=120)
print("STDOUT:", resp.get("stdout"))
print("STDERR:", resp.get("stderr"))
print("ELAPSED:", resp.get("elapsed_s"))
