"""Build a slim zip of the sub1quant project, excluding heavy artifacts."""
import os
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Zwmar\projects\sub1quant")
OUT = ROOT / "tools" / "sub1quant_slim.zip"

EXCLUDE_DIRS = {".venv", ".git", "__pycache__", ".cache", "models", "quantized",
                "eval_results", "tools", "_hf_stage", ".harness", "node_modules",
                "llama.cpp", ".opencode", "dist", "build"}
EXCLUDE_GLOBS_SUFFIX = {".pyc", ".safetensors", ".gguf", ".parquet", ".pt",
                        ".bin", ".idx", ".index"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}

if OUT.exists():
    OUT.unlink()
OUT.parent.mkdir(parents=True, exist_ok=True)


def should_skip(path: Path) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    return any(path.name.endswith(s) for s in EXCLUDE_GLOBS_SUFFIX)


count = 0
total = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        rel = path.relative_to(ROOT)
        zf.write(path, rel)
        count += 1
        total += path.stat().st_size
        if count % 50 == 0:
            print(f"  {count} files, {total/1e6:.1f} MB raw", flush=True)

print(f"Wrote {OUT}")
print(f"Files: {count}")
print(f"Raw size: {total/1e6:.1f} MB")
print(f"Zip size: {OUT.stat().st_size/1e6:.1f} MB")
