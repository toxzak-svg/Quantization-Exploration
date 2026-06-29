"""HF probe: simplest possible test of token + whoami."""
import os, sys, json
tok = os.environ.get("HF_TOKEN")
print("HF_TOKEN present:", bool(tok))
if tok:
    print("  starts with:", tok[:8], "  length:", len(tok))

from huggingface_hub import HfApi
api = HfApi(token=tok)
try:
    me = api.whoami()
    print("user:", me.get("name"))
except Exception as e:
    print("whoami err:", type(e).__name__, str(e)[:300])
