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


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    """Apply one deterministic notebook workflow edit and fail if it drifts."""
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} marker, found {count}")
    return text.replace(old, new)


def synchronize_workflow_cells(payload: dict) -> None:
    """Keep chemistry audits outside the inlined toolkit reproducible too."""
    for cell in payload["cells"]:
        source = "".join(cell.get("source", []))
        if ("def export_ligand_case(" in source
                and "contact = chemical_contact_diagnostic" not in source):
            source = replace_once(
                source,
                "    ligand_slab_gap = _min_gap(struct.positions[n_slab:], struct.positions[:n_slab], struct.cell)\n",
                "    ligand_slab_gap = _min_gap(struct.positions[n_slab:], struct.positions[:n_slab], struct.cell)\n"
                "    contact = chemical_contact_diagnostic(struct, n_slab)\n"
                "    if not contact.ok:\n"
                "        raise ValueError(\n"
                "            f\"refusing to export chemically compressed {contact.ligand_symbol}--\"\n"
                "            f\"{contact.slab_symbol} contact: {contact.distance:.3f} Å < \"\n"
                "            f\"{contact.cutoff:.3f} Å\")\n",
                label="contact audit insertion",
            )
            source = replace_once(
                source,
                "        contact_tilt_deg=contact_tilt,\n",
                "        contact_tilt_deg=contact_tilt,\n"
                "        chemical_contact_min_margin=round(contact.min_margin, 4),\n"
                "        chemical_contact_limiting_pair=(\n"
                "            f\"{contact.ligand_symbol}--{contact.slab_symbol}\"),\n",
                label="contact provenance insertion",
            )
            source = replace_once(
                source,
                "                anchor_ni_o=round(anchor_ni_o, 3), ligand_slab_gap=round(ligand_slab_gap, 3),\n",
                "                anchor_ni_o=round(anchor_ni_o, 3), ligand_slab_gap=round(ligand_slab_gap, 3),\n"
                "                chemical_contact_margin=round(contact.min_margin, 3),\n"
                "                limiting_contact=(\n"
                "                    f\"{contact.ligand_symbol}--{contact.slab_symbol}\"),\n",
                label="contact manifest insertion",
            )
            source = replace_once(
                source,
                "                ok=(clashes == 0 and ligand_slab_gap >= 1.25\n"
                "                    and anchor_ni_o <= 2.40 and vac[\"fits_vacuum\"]))\n",
                "                ok=(contact.ok and clashes == 0\n"
                "                    and anchor_ni_o <= 2.40 and vac[\"fits_vacuum\"]))\n",
                label="chemistry-aware ok criterion",
            )
            source = replace_once(
                source,
                "        if not r[\"fits_vacuum\"]: why.append(f\"vacuum {r['vacuum_gap']} Å\")\n",
                "        if r[\"chemical_contact_margin\"] < 0:\n"
                "            why.append(f\"compressed {r['limiting_contact']} contact\")\n"
                "        if not r[\"fits_vacuum\"]: why.append(f\"vacuum {r['vacuum_gap']} Å\")\n",
                label="contact failure report",
            )
            cell["source"] = source_lines(source)
        elif "checks overlaps + vacuum" in source:
            source = source.replace(
                "checks overlaps + vacuum",
                "checks species-aware contacts + overlaps + vacuum",
            )
            cell["source"] = source_lines(source)


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    chunks = split_toolkit(SOURCE.read_text(encoding="utf-8"))
    for cell_index, chunk in zip(TOOLKIT_CELLS, chunks, strict=True):
        cell = payload["cells"][cell_index]
        if cell["cell_type"] != "code":
            raise RuntimeError(f"cell {cell_index} is not a code cell")
        cell["source"] = source_lines(chunk)
    synchronize_workflow_cells(payload)
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
