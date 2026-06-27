"""Validate the Hugging Face/GitHub release notebook structure.

This is intentionally lightweight: it catches regression back to demo-only
cells without executing a multi-GB artifact upload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("colab_hf_github_pipeline.ipynb")

FORBIDDEN_SNIPPETS = [
    "demo_tensor",
    "torch.randn",
    "Customize this cell",
    "Add demo tensor",
    "Example: create",
]

REQUIRED_SNIPPETS = [
    "gemma_ternary_aggressive.pt",
    "gemma_magq.pt",
    "gemma_hybrid_stream.pt",
    "gemma-4-E2B-sub1bit.pt",
    "release_manifest.json",
    "README.md",
    "upload_folder",
    "create_github_release",
    "upload_release_asset",
    "HF_REPO_ID",
    "GH_REPO_ID",
    "RELEASE_TAG",
    "sha256",
    "Ternary Aggressive",
    "SVD Sub1Bit (90% threshold)",
    "BROKEN",
    "RECOMMENDED",
    "INCLUDE_FAILED_SVD_ARTIFACTS",
]


def flatten_sources(notebook_path: Path) -> tuple[list[dict], str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    return cells, text


def validate(notebook_path: Path = NOTEBOOK) -> list[str]:
    cells, text = flatten_sources(notebook_path)
    failures: list[str] = []

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    if len(code_cells) < 8:
        failures.append(f"expected at least 8 code cells, found {len(code_cells)}")
    if len(markdown_cells) < 5:
        failures.append(f"expected at least 5 markdown cells, found {len(markdown_cells)}")

    for snippet in FORBIDDEN_SNIPPETS:
        if snippet.lower() in text.lower():
            failures.append(f"forbidden demo snippet still present: {snippet!r}")

    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            failures.append(f"required release snippet missing: {snippet!r}")

    return failures


def main() -> int:
    failures = validate()
    if failures:
        print("Notebook release validation failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Notebook release validation passed: {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
