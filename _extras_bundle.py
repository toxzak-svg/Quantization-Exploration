"""Bundle the local eval_results JSONs + markdown docs into a tarball and upload to HF."""
import os, json, tarfile, urllib.request

LOCAL = r"C:\Users\Zwmar\projects\sub1quant\eval_results"
DOCS_DIR = r"C:\Users\Zwmar\projects\sub1quant"
TARBALL = r"C:\Users\Zwmar\projects\sub1quant\_local_extras.tar.gz"

# Files we want to add
EXTRAS = []
for fn in sorted(os.listdir(LOCAL)):
    full = os.path.join(LOCAL, fn)
    if os.path.isfile(full) and fn.endswith(".json"):
        EXTRAS.append(("eval_results/" + fn, full))

# Top-level docs
for fn in ["FAST_INT4_PIVOT.md", "README.md", "CHANGELOG.md"]:
    full = os.path.join(DOCS_DIR, fn)
    if os.path.isfile(full):
        EXTRAS.append((fn, full))

print(f"files to add: {len(EXTRAS)}")
for arc, full in EXTRAS:
    print(f"  {arc}  ({os.path.getsize(full)} bytes)  <- {full}")

# Make tarball
with tarfile.open(TARBALL, "w:gz", compresslevel=6) as tar:
    for arc, full in EXTRAS:
        tar.add(full, arcname="extras/" + arc)
print(f"\nwrote {TARBALL}  ({os.path.getsize(TARBALL)} bytes)")

# Show how it'd look uploaded (just print contents)
print("contents:")
with tarfile.open(TARBALL) as tar:
    for n in tar.getnames():
        print(f"  {n}")
