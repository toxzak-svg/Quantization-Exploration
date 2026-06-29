"""Upload the 948MB quantized .pt via the bridge with retries and progress.

Single multipart form, single big blob. Reports progress every 5s to stderr.
"""
import urllib.request, urllib.error, json, time, os, sys

BASE = "https://prime-apparently-types-explanation.trycloudflare.com"
TOKEN = "pNWps4V97_opsqamUqTwvr8HK4JI_bDd"
LOCAL = sys.argv[1]
REMOTE = sys.argv[2]
MAX_ATTEMPTS = 3

print(f"local  = {LOCAL}")
print(f"remote = {REMOTE}")
print(f"size   = {os.path.getsize(LOCAL):,} bytes ({os.path.getsize(LOCAL)/1024/1024:.1f} MB)")

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n--- attempt {attempt}/{MAX_ATTEMPTS} ---")
    boundary = "----MavisBridgeBoundary7e3f1a02"
    fname = os.path.basename(LOCAL)
    with open(LOCAL, "rb") as f:
        file_bytes = f.read()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="path"\r\n\r\n'
        f"{REMOTE}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body = head + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    del file_bytes
    req = urllib.request.Request(
        f"{BASE}/upload",
        data=body,
        headers={
            "X-Bridge-Token": TOKEN,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        # Long timeout — cloudflared can be slow on big bodies
        with urllib.request.urlopen(req, timeout=900) as r:
            raw = r.read()
        elapsed = time.time() - t0
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw[:500].decode(errors="replace")}
        print(f"OK in {elapsed:.1f}s")
        print(json.dumps(data, indent=2))
        if data.get("ok") is True:
            sys.exit(0)
        print(f"server returned ok=false: {data}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        elapsed = time.time() - t0
        print(f"HTTP {e.code} after {elapsed:.1f}s: {body[:300]}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERR after {elapsed:.1f}s: {type(e).__name__}: {e}")
    if attempt < MAX_ATTEMPTS:
        wait = 10 * attempt
        print(f"retry in {wait}s")
        time.sleep(wait)
print("all attempts failed")
sys.exit(1)
