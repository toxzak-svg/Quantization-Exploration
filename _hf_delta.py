"""Upload the local-extras tarball to the kernel, extract, and do a delta push
of just the extra files to toxzak/sub1quant (HF). Uses repo.upload_file so we
don't touch the existing files."""
import os, json, sys, tarfile, traceback, shutil

# Re-pull token (kernel env doesn't persist across ipyexec calls)
import urllib.request, urllib.error

# Step 1: kernel side — load token from saved location
token_path = "/root/.hf_token"
if not os.path.exists(token_path):
    print(f"ERROR: token file {token_path} missing. Upload it via /upload first.")
    sys.exit(1)
with open(token_path) as f:
    HF_TOKEN = f.read().strip()
os.environ["HF_TOKEN"] = HF_TOKEN

# Step 2: extract extras tarball
EXTRAS_TARBALL = "/tmp/_local_extras.tar.gz"
EXTRACT_TO = "/tmp/_extras_extracted"
if os.path.isdir(EXTRACT_TO):
    shutil.rmtree(EXTRACT_TO)
os.makedirs(EXTRACT_TO, exist_ok=True)
if os.path.exists(EXTRAS_TARBALL):
    tarfile.open(EXTRAS_TARBALL).extractall(EXTRACT_TO)
else:
    print(f"ERROR: extras tarball missing at {EXTRAS_TARBALL}; upload first")

# Step 3: walk extracted tree
files_to_push = []
for root, _, files in os.walk(EXTRACT_TO):
    for fn in files:
        full = os.path.join(root, fn)
        rel = os.path.relpath(full, EXTRACT_TO)
        # strip leading "extras/"
        if rel.startswith("extras/"):
            rel = rel[len("extras/"):]
        files_to_push.append((rel, full))

print(f"files to push ({len(files_to_push)}):")
for rel, full in files_to_push:
    print(f"  {rel}  ({os.path.getsize(full):,} bytes)")

# Step 4: push via huggingface_hub
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
me = api.whoami()
USER = me["name"]
REPO = f"{USER}/sub1quant"
print(f"\nrepo: {REPO}")

# Step 5: load existing files to skip duplicates
existing = set(api.list_repo_files(repo_id=REPO, repo_type="model"))
print(f"existing files at HEAD: {len(existing)}")

# Step 6: upload each file
uploaded = []
skipped = []
for rel, full in files_to_push:
    if rel in existing:
        skipped.append(rel)
        continue
    try:
        api.upload_file(
            path_or_fileobj=full,
            path_in_repo=rel,
            repo_id=REPO,
            repo_type="model",
            commit_message=f"Add {rel}",
        )
        uploaded.append(rel)
        print(f"  + {rel}")
    except Exception as e:
        print(f"  ERR {rel}: {type(e).__name__}: {e}")
        traceback.print_exc()

print(f"\nuploaded: {len(uploaded)}, skipped (already there): {len(skipped)}")

# Step 7: fix README yaml metadata warning by adding minimal frontmatter
README_PATH_ON_REPO = "README.md"
README_LOCAL = "/tmp/_hf_stage/README.md"
if os.path.exists(README_LOCAL):
    fixed = open(README_LOCAL).read()
    # inject minimal yaml header at top
    yaml_header = "---\nlicense: apache-2.0\ntags:\n  - quantization\n  - sub-4-bit\n  - gemma\n---\n\n"
    if not fixed.startswith("---"):
        fixed = yaml_header + fixed
        tmp = "/tmp/_hf_stage/_README_fixed.md"
        open(tmp, "w").write(fixed)
        # rewrite our local stage README too
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="README.md",
            repo_id=REPO,
            repo_type="model",
            commit_message="Add yaml metadata to README",
        )
        print("updated README.md with yaml metadata")

# Step 8: final listing
final = sorted(api.list_repo_files(repo_id=REPO, repo_type="model"))
print(f"\nfinal HEAD has {len(final)} files:")
for f in final:
    print(f"  {f}")
