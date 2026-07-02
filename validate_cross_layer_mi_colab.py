"""Smoke-test the cross-layer MI Colab notebook.

Validates that:
  1. The notebook JSON parses (nbformat=4).
  2. Every cell has the required fields.
  3. Every code cell compiles to valid Python bytecode (catches
     syntax errors that would brick the notebook at runtime).
  4. The expected section dividers are present in the markdown cells.
  5. Required CLI surface is referenced in the code cells (fails fast
     if we accidentally drop a step).
"""
from __future__ import annotations

import json
import pathlib
import py_compile
import sys

NB_PATH = pathlib.Path("notebook/cross_layer_mi_colab.ipynb")

REQUIRED_MARKDOWN_SNIPPETS = [
    "# Cross-layer MI on Gemma 4 E2B",
    "Step 1: GPU sanity check",
    "Step 2: Install Python dependencies",
    "Step 3: Read secrets + clone the repo",
    "Step 4: Disk + VRAM check",
    "Step 5: Download Gemma 4 E2B",
    "Step 6: GPU code path matches CPU",
    "Step 7: CPU vs GPU bench",
    "Step 8: Real Gemma pipeline",
    "Step 9: Render the unconditional MI summary table",
    "Step 10: Conditional MI",
    "Step 11: Compare unconditional vs conditional rankings",
    "Step 12: Mirror results to Google Drive",
]

REQUIRED_CODE_SNIPPETS = [
    "torch.cuda.is_available()",
    "snapshot_download",
    "GPUCorrectnessTests",
    "bench_gpu.py",
    "run_cross_layer_mi_colab.py",
    '"--device"',
    '"cuda"',
    '"--conditioning"',
    '"delta"',
    "kendall_tau_vs_sigma",
    "mi_scores",
    "bits",
    "/content/drive",
]


def main() -> int:
    if not NB_PATH.exists():
        print(f"FAIL: {NB_PATH} does not exist")
        return 1
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        print(f"FAIL: expected nbformat=4, got {nb.get('nbformat')}")
        return 1

    cells = nb["cells"]
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"cells: {len(cells)} ({n_code} code, {n_md} md)")

    failures = []

    # 1. Structure
    for i, c in enumerate(cells):
        for f in ("cell_type", "source", "metadata"):
            if f not in c:
                failures.append(f"cell {i} missing field {f!r}")
    if n_code < 8:
        failures.append(f"expected at least 8 code cells, got {n_code}")
    if n_md < 8:
        failures.append(f"expected at least 8 markdown cells, got {n_md}")

    # 2. Compile each code cell.
    print("\n--- Compile check ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        for i, c in enumerate(cells):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            # py_compile on Py 3.14 doesn't take long in-memory source
            # cleanly; write to a temp file.
            tmp_py = tmpdir / f"cell_{i:02d}.py"
            tmp_py.write_text(src, encoding="utf-8")
            try:
                py_compile.compile(str(tmp_py), doraise=True)
                first = src.strip().split("\n", 1)[0][:80]
                print(f"  cell {i:2d}: ok ({first!r})")
            except py_compile.PyCompileError as e:
                failures.append(f"cell {i} syntax error: {e}")
                print(f"  cell {i:2d}: SYNTAX ERROR")
                print(f"    {e}")

    # 3. Required markdown snippets.
    md_text = "\n".join(
        "".join(c.get("source", [])) for c in cells if c["cell_type"] == "markdown"
    )
    print("\n--- Markdown snippet coverage ---")
    for snippet in REQUIRED_MARKDOWN_SNIPPETS:
        ok = snippet in md_text
        print(f"  {'ok ' if ok else 'MISS'} {snippet!r}")
        if not ok:
            failures.append(f"markdown missing: {snippet!r}")

    # 4. Required code snippets.
    code_text = "\n".join(
        "".join(c.get("source", [])) for c in cells if c["cell_type"] == "code"
    )
    print("\n--- Code snippet coverage ---")
    for snippet in REQUIRED_CODE_SNIPPETS:
        ok = snippet in code_text
        print(f"  {'ok ' if ok else 'MISS'} {snippet!r}")
        if not ok:
            failures.append(f"code missing: {snippet!r}")

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: {NB_PATH.name} is structurally valid, all snippets present, "
          f"all {n_code} code cells compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())