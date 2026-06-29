"""Build an eval bundle (tar.gz) of everything we need on the colab kernel
to run scripts/eval_quantized.py against the gemma-4-E2B model.

Excludes checkpoints, the base model, .git, __pycache__, etc.
The bundle extracts directly into /content/sub1quant on the kernel.
"""
from __future__ import annotations
import os
import tarfile
from pathlib import Path

ROOT = Path(r"C:\Users\Zwmar\projects\sub1quant")
OUT = ROOT / "_eval_bundle.tar.gz"

# Files / dirs we MUST have
REQUIRED = [
    "scripts/eval_quantized.py",
    "scripts/quantize_mixed_budget.py",
    "scripts/limited_ppl_bench.py",
    "test_perplexity.py",
    # Full src/ — needed because src/__init__.py does wildcard imports
    "src/__init__.py",
    "src/Sub1BitLLM.py",
    "src/error_budget_residual.py",
    "src/groupwise_int4.py",
    "src/quantization.py",
    "src/mixed_budget.py",
    "src/lowrank_factorization.py",
    "src/gguf_writer.py",
    "src/pack_gguf.py",
    "data/wiki.test.txt",
] 

# Skip anything in these directories or matching these patterns
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "node_modules", ".idea",
    "models", "quantized", "eval_results", "checkpoints", "llama.cpp",
    "&&", "mkdir",
}
EXCLUDE_SUFFIX = {".pyc", ".pt", ".gguf", ".bin", ".safetensors", ".zip"}
EXCLUDE_PREFIX = (".",)  # dotfiles/dot-dirs (keep .gitattributes only at root)


def _is_excluded(rel: Path) -> bool:
    parts = rel.parts
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return True
    name = rel.name
    if any(name.endswith(suf) for suf in EXCLUDE_SUFFIX):
        return True
    # Skip dotfiles at any level except .gitattributes at the root
    if rel != Path(name) and name.startswith("."):
        return True
    return False


def main() -> None:
    members: list[tuple[Path, str]] = []
    for rel_str in REQUIRED:
        rel = Path(rel_str)
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(src)
        if src.is_dir():
            for sub in src.rglob("*"):
                if sub.is_dir():
                    continue
                if _is_excluded(sub.relative_to(ROOT)):
                    continue
                members.append((sub, str(sub.relative_to(ROOT)).replace(os.sep, "/")))
        else:
            members.append((src, rel_str))

    OUT.unlink(missing_ok=True)
    with tarfile.open(OUT, "w:gz", compresslevel=6) as tar:
        for src, name in members:
            tar.add(str(src), arcname=f"sub1quant/{name}")

    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print("contents:")
    with tarfile.open(OUT, "r:gz") as tar:
        for n in tar.getnames():
            print(f"  {n}")


if __name__ == "__main__":
    main()
