"""Probe HF_TOKEN: confirm it's set, callable works, and identify the user.
Then create the private repo toxzak/sub1quant if it doesn't exist."""
import os, sys, json
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
if os.environ.get("HF_TOKEN"):
    tok = os.environ["HF_TOKEN"]
    masked = tok[:4] + "..." + tok[-4:] if len(tok) > 12 else "<short>"
    print(f"  masked: {masked}")

try:
    from huggingface_hub import HfApi, whoami
except ImportError as e:
    print(f"huggingface_hub import failed: {e}")
    sys.exit(0)

api = HfApi(token=os.environ.get("HF_TOKEN"))
try:
    me = whoami(token=os.environ.get("HF_TOKEN"))
    print(f"whoami user: {me.get('name')}  fullname={me.get('fullname')}")
except Exception as e:
    print(f"whoami failed: {type(e).__name__}: {e}")
    sys.exit(0)

REPO = "toxzak/sub1quant"
print(f"\nrepo target: {REPO}")
print(f"existing repo info:")
try:
    info = api.repo_info(repo_id=REPO, repo_type="model")
    print(f"  type=model  visibility={'private' if info.private else 'public'}  downloads={info.downloads}")
    print(f"  last_modified={info.last_modified}")
except Exception as e:
    print(f"  not found or err: {type(e).__name__}: {e}")

print("\ncreate if missing:")
try:
    url = api.create_repo(
        repo_id=REPO,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    print(f"  repo URL: {url}")
except Exception as e:
    print(f"  create_repo err: {type(e).__name__}: {e}")

# Re-check
try:
    info = api.repo_info(repo_id=REPO, repo_type="model")
    print(f"\nfinal: type=model  visibility={'private' if info.private else 'public'}")
except Exception as e:
    print(f"re-check err: {e}")
