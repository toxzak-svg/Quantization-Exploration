"""Upload the user's HF_TOKEN from Windows env to the kernel securely."""
import os, urllib.request

TOKEN = os.environ['HF_TOKEN']
boundary = "----MavisBridgeBoundary7e3f1a02"
body_str = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="path"\r\n\r\n'
    f"/root/.hf_token\r\n"
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="_token"\r\n'
    f"Content-Type: application/octet-stream\r\n\r\n"
    f"{TOKEN}"
    f"\r\n--{boundary}--\r\n"
)
body = body_str.encode()

req = urllib.request.Request(
    "https://prime-apparently-types-explanation.trycloudflare.com/upload",
    data=body,
    method="POST",
    headers={
        "X-Bridge-Token": "pNWps4V97_opsqamUqTwvr8HK4JI_bDd",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    },
)
with urllib.request.urlopen(req, timeout=30) as r:
    print(r.read().decode())
