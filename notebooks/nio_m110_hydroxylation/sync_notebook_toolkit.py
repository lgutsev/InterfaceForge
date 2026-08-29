#!/usr/bin/env python3
"""Synchronize the notebook's inlined toolkit from the reviewable Python source."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "nio_hydroxylation_utils.py"
NOTEBOOK = HERE / "NiO_m110_hydroxylation.ipynb"
TOOLKIT_CELLS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
INSTALL_HINT = (
    '# install once if needed:\n'
    '# %pip install "ase>=3.22,<4" "pymatgen>=2024.1" "numpy>=1.24" '
    '"pandas>=2.0" "matplotlib>=3.7" jupyterlab\n\n'
)


def source_lines(text: str) -> list[str]:
    """Use nbformat's conventional line-preserving source representation."""
    return text.splitlines(keepends=True)


def split_toolkit(text: str) -> list[str]:
    header = re.compile(r"(?m)^# ={70,}\n# [^\n]+\n# ={70,}\n")
    starts = [match.start() for match in header.finditer(text)]
    chunks = [text[: starts[0]]]
    chunks.extend(
        text[start:end]
        for start, end in zip(starts, starts[1:] + [len(text)], strict=True)
    )
    if len(chunks) != len(TOOLKIT_CELLS):
        raise RuntimeError(
            f"expected {len(TOOLKIT_CELLS)} toolkit chunks, found {len(chunks)}; "
            "update TOOLKIT_CELLS if sections changed"
        )
    chunks[0] = INSTALL_HINT + chunks[0]
    return chunks


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    chunks = split_toolkit(SOURCE.read_text(encoding="utf-8"))
    for cell_index, chunk in zip(TOOLKIT_CELLS, chunks, strict=True):
        cell = payload["cells"][cell_index]
        if cell["cell_type"] != "code":
            raise RuntimeError(f"cell {cell_index} is not a code cell")
        cell["source"] = source_lines(chunk)
    # Never commit stale manifests/diagnostics from an older generator version.
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    NOTEBOOK.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
