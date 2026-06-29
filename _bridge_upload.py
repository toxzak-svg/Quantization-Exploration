"""Tiny client for the colab_bridge Flask app.
Uses multipart/form-data upload for arbitrary files, /exec for code.
Usage:
  python _bridge_upload.py upload <remote_path> <local_path>
  python _bridge_upload.py exec <code_file>      # runs stdlib exec via the /exec endpoint
  python _bridge_upload.py ipyexec <code_file>    # run code, print stdout/stderr
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

BASE = os.getenv("BRIDGE_URL", "https://prime-apparently-types-explanation.trycloudflare.com")
TOKEN = os.getenv("BRIDGE_TOKEN", "pNWps4V97_opsqamUqTwvr8HK4JI_bDd")


def _hdr(extra: dict | None = None) -> dict:
    h = {"X-Bridge-Token": TOKEN}
    if extra:
        h.update(extra)
    return h


def upload(local: Path, remote: str) -> dict:
    """Multipart upload of `local` to `remote` (absolute path on the kernel)."""
    boundary = "----MavisBridgeBoundary7e3f1a02"
    file_bytes = local.read_bytes()
    fname = local.name
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="path"\r\n\r\n'
        f"{remote}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/upload",
        data=body,
        headers=_hdr({"Content-Type": f"multipart/form-data; boundary={boundary}"}),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"upload HTTP {e.code}: {e.read()!r}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:500].decode(errors="replace")}


def upload_text(remote: str, content: str) -> dict:
    """Upload via JSON `content` field (for small text files)."""
    body = json.dumps({"path": remote, "content": content}).encode()
    req = urllib.request.Request(
        f"{BASE}/upload",
        data=body,
        headers=_hdr({"Content-Type": "application/json"}),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"upload_text HTTP {e.code}: {e.read()!r}") from e


def exec_py(code: str, timeout: int = 600) -> dict:
    body = json.dumps({"code": code}).encode()
    req = urllib.request.Request(
        f"{BASE}/exec",
        data=body,
        headers=_hdr({"Content-Type": "application/json"}),
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"ok": False, "raw": raw[:500].decode(errors="replace")}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"ok": False, "stderr": body, "elapsed_s": time.time() - started, "error": f"HTTP {e.code}"}
    data["client_elapsed_s"] = round(time.time() - started, 3)
    return data


def exec_file(path: Path, timeout: int = 600) -> dict:
    code = path.read_text(encoding="utf-8")
    return exec_py(code, timeout=timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("upload")
    p_up.add_argument("remote")
    p_up.add_argument("local")
    p_txt = sub.add_parser("upload_text")
    p_txt.add_argument("remote")
    p_txt.add_argument("content")
    p_ex = sub.add_parser("exec")
    p_ex.add_argument("path")
    p_ex.add_argument("--timeout", type=int, default=600)
    p_ix = sub.add_parser("ipyexec")
    p_ix.add_argument("path")
    p_ix.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.cmd == "upload":
        r = upload(Path(args.local), args.remote)
        print(json.dumps(r, indent=2))
    elif args.cmd == "upload_text":
        r = upload_text(args.remote, args.content)
        print(json.dumps(r, indent=2))
    elif args.cmd == "exec":
        r = exec_file(Path(args.path), args.timeout)
        print(json.dumps({k: r[k] for k in ("ok", "stdout", "stderr", "error", "elapsed_s", "client_elapsed_s")}, indent=2))
    elif args.cmd == "ipyexec":
        r = exec_file(Path(args.path), args.timeout)
        if r.get("stdout"):
            print("STDOUT:")
            print(r["stdout"])
        if r.get("stderr"):
            print("STDERR:")
            print(r["stderr"])
        if r.get("error"):
            print("ERROR:", r["error"])
        print(f"ok={r.get('ok')} elapsed={r.get('elapsed_s')}s client={r.get('client_elapsed_s')}s")
