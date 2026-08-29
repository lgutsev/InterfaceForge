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
        if ("BATCH_CASES     = [(0.0" in source
                and "bindable_ligand_case_grid" not in source):
            source = replace_once(
                source,
                "BATCH_CASES     = [(0.0,  \"\",          \"\"),        # pristine (no OH) + ligand -> OH0/\n"
                "                   (0.50, \"clustered\", \"capped\"),\n"
                "                   (0.50, \"scattered\", \"dissoc\"),\n"
                "                   (0.75, \"clustered\", \"capped\")]\n",
                "# Complete production grid for every surface that retains a bare Ni.\n"
                "BATCH_CASES = bindable_ligand_case_grid(FRACTIONS)\n"
                "# Full-OH references have no free Ni anchor; passivating them requires an\n"
                "# explicit OH-displacement reaction and is not neutral molecular docking.\n"
                "BATCH_REFERENCE_ONLY = [(1.0, \"full\", motif) for motif in (\"capped\", \"dissoc\")]\n",
                label="complete ligand batch grid",
            )
            source = replace_once(
                source,
                "    return pd.DataFrame(out)\n",
                "    expected = len(SURFACES) * len(BATCH_LIGANDS) * sum(\n"
                "        1 if frac == 0.0 else len(BATCH_POSITIONS)\n"
                "        for frac, _pattern, _motif in BATCH_CASES)\n"
                "    if len(out) != expected:\n"
                "        raise RuntimeError(f\"incomplete ligand grid: generated {len(out)}/{expected} cases\")\n"
                "    return pd.DataFrame(out)\n",
                label="batch completeness assertion",
            )
            source = replace_once(
                source,
                "if RUN_BATCH:\n",
                "if BATCH_REFERENCE_ONLY:\n"
                "    print(\"Reference-only full-OH cases (no bare Ni anchor):\", BATCH_REFERENCE_ONLY)\n"
                "\n"
                "if RUN_BATCH:\n",
                label="full coverage explanation",
            )
            cell["source"] = source_lines(source)
        if ("Keep the lists short and skim the table first." in source
                and "complete bindable" not in source):
            source = source.replace(
                "Keep the lists short and skim the table first.",
                "The default covers both bare-Ni chemisorption and separately labelled "
                "surface-OH...O=P molecular adsorption on the fully hydroxylated surfaces.",
            )
            cell["source"] = source_lines(source)
        if "Fully hydroxylated references remain ligand-free" in source:
            source = source.replace(
                "The default is the complete bindable surface grid. Fully hydroxylated "
                "references remain ligand-free because every exposed Ni is already occupied; "
                "they require a separate, stoichiometrically explicit OH-displacement model.",
                "The default covers both bare-Ni chemisorption and separately labelled "
                "surface-OH...O=P molecular adsorption on the fully hydroxylated surfaces.",
            )
            cell["source"] = source_lines(source)
        if ("def dock_ligand(" in source
                and "def dock_hbonded_ligand(" not in source):
            source = replace_once(
                source,
                "# each ligand case carries its full context (so it knows which OH-type folder it belongs in)\n",
                "def dock_hbonded_ligand(built, mol, anchor, donor_o, donor_h):\n"
                "    if isinstance(LIGAND_CELL_C, (int, float)):\n"
                "        built = extend_vacuum(built, float(LIGAND_CELL_C))\n"
                "    oriented = orient_phosphonate(mol, anchor)\n"
                "    struct, contact_tilt = place_hbonded_ligand(\n"
                "        built, oriented, anchor, donor_o, donor_h,\n"
                "        h_o_distance=D_HBOND_HO, anchor_o_index=binding_oxygen(anchor))\n"
                "    if LIGAND_CELL_C == \"auto\":\n"
                "        slab_lo = float(struct.positions[:len(built), 2].min())\n"
                "        ligand_hi = float(struct.positions[len(built):, 2].max())\n"
                "        needed_c = ligand_hi - slab_lo + LIGAND_PERIODIC_GAP\n"
                "        struct = extend_vacuum(struct, float(np.ceil(needed_c)))\n"
                "    return apply_freeze(struct), contact_tilt, ligand_tilt_deg(oriented, anchor)\n"
                "\n"
                "# each ligand case carries its full context (so it knows which OH-type folder it belongs in)\n",
                label="hydrogen-bond docking helper",
            )
            cell["source"] = source_lines(source)
        if ("def export_ligand_case(" in source
                and "binding_mode=\"ni_o\"" not in source):
            source = replace_once(
                source,
                "                       contact_tilt=0.0, tilt=0.0):\n",
                "                       contact_tilt=0.0, tilt=0.0, binding_mode=\"ni_o\",\n"
                "                       donor_o=None, donor_h=None):\n",
                label="binding-mode export signature",
            )
            source = replace_once(
                source,
                "    name = case_name(fraction, pattern, motif, ligand=ligand, anchor_position=pos_name,\n",
                "    hbond_h_o = hbond_angle = None\n"
                "    if binding_mode == \"hbond\":\n"
                "        if donor_o is None or donor_h is None:\n"
                "            raise ValueError(\"hbond export requires donor_o and donor_h\")\n"
                "        hbond_h_o, hbond_angle = hbond_geometry(\n"
                "            np.asarray(donor_o), np.asarray(donor_h),\n"
                "            struct.positions[bind_o_global], struct.cell)\n"
                "        binding_ok = (1.45 <= hbond_h_o <= 2.20\n"
                "                      and hbond_angle >= 150.0 and anchor_ni_o >= 2.80)\n"
                "    elif binding_mode == \"ni_o\":\n"
                "        binding_ok = anchor_ni_o <= 2.40\n"
                "    else:\n"
                "        raise ValueError(f\"unknown binding mode {binding_mode!r}\")\n"
                "    name = case_name(fraction, pattern, motif, ligand=ligand, anchor_position=pos_name,\n",
                label="binding-mode geometry validation",
            )
            source = replace_once(
                source,
                "        anchor_position=pos_name, chosen_phosphonate_P_index=anchor.p_index,\n",
                "        anchor_position=pos_name, binding_mode=binding_mode,\n"
                "        chosen_phosphonate_P_index=anchor.p_index,\n",
                label="binding-mode provenance",
            )
            source = replace_once(
                source,
                "        backbone_C_index=anchor.c_index, oh_height=OH_HEIGHT,\n",
                "        backbone_C_index=anchor.c_index,\n"
                "        ni_o_target=(OH_HEIGHT if binding_mode == \"ni_o\" else None),\n"
                "        hbond_h_o_target=(D_HBOND_HO if binding_mode == \"hbond\" else None),\n"
                "        hbond_h_o=(round(hbond_h_o, 4) if hbond_h_o is not None else None),\n"
                "        hbond_angle=(round(hbond_angle, 3) if hbond_angle is not None else None),\n",
                label="binding-mode metrics provenance",
            )
            source = replace_once(
                source,
                "                anchor_position=pos_name, chosen_P_index=anchor.p_index, n_atoms=len(struct),\n",
                "                anchor_position=pos_name, binding_mode=binding_mode,\n"
                "                chosen_P_index=anchor.p_index, n_atoms=len(struct),\n",
                label="binding-mode manifest",
            )
            source = replace_once(
                source,
                "                anchor_ni_o=round(anchor_ni_o, 3), ligand_slab_gap=round(ligand_slab_gap, 3),\n",
                "                anchor_ni_o=round(anchor_ni_o, 3),\n"
                "                hbond_h_o=(round(hbond_h_o, 3) if hbond_h_o is not None else None),\n"
                "                hbond_angle=(round(hbond_angle, 1) if hbond_angle is not None else None),\n"
                "                binding_geometry_ok=binding_ok,\n"
                "                ligand_slab_gap=round(ligand_slab_gap, 3),\n",
                label="binding-mode manifest metrics",
            )
            source = replace_once(
                source,
                "                ok=(contact.ok and clashes == 0\n"
                "                    and anchor_ni_o <= 2.40 and vac[\"fits_vacuum\"]))\n",
                "                ok=(contact.ok and clashes == 0\n"
                "                    and binding_ok and vac[\"fits_vacuum\"]))\n",
                label="binding-mode ok criterion",
            )
            source = replace_once(
                source,
                "        if not r[\"fits_vacuum\"]: why.append(f\"vacuum {r['vacuum_gap']} Å\")\n",
                "        if not r[\"binding_geometry_ok\"]:\n"
                "            why.append(f\"invalid {r['binding_mode']} binding geometry\")\n"
                "        if not r[\"fits_vacuum\"]: why.append(f\"vacuum {r['vacuum_gap']} Å\")\n",
                label="binding-mode failure report",
            )
            cell["source"] = source_lines(source)
        if ("BATCH_REFERENCE_ONLY" in source
                and "BATCH_HBOND_CASES" not in source):
            source = source.replace(
                "# Full-OH references have no free Ni anchor; passivating them requires an\n"
                "# explicit OH-displacement reaction and is not neutral molecular docking.\n"
                "BATCH_REFERENCE_ONLY = [(1.0, \"full\", motif) for motif in (\"capped\", \"dissoc\")]\n",
                "# Full-OH cases use molecular surface-OH...O=P adsorption, separately\n"
                "# labelled and validated from the bare-Ni chemisorption cases above.\n"
                "BATCH_HBOND_CASES = hydrogen_bonded_ligand_case_grid(FRACTIONS)\n",
            )
            source = replace_once(
                source,
                "    expected = len(SURFACES) * len(BATCH_LIGANDS) * sum(\n",
                "        for frac, pat, mot in BATCH_HBOND_CASES:\n"
                "            s = select_sites(sm, frac, pat, seed=BATCH_SEED)\n"
                "            builder = build_capped_hydroxide if mot == \"capped\" else build_dissociated_pair\n"
                "            hxs = builder(sm, s, d_ni_o=D_NI_O)\n"
                "            assign_hydrogens(hxs, b.cell, d_oh=D_O_H_, default_tilt_deg=H_TILT_DEG)\n"
                "            donor_i = expose_hbond_donor(hxs, d_oh=D_O_H_)\n"
                "            donor_o = hxs[donor_i].o_pos.copy()\n"
                "            donor_h = hxs[donor_i].h_pos.copy()\n"
                "            slab = assemble(b, hxs)\n"
                "            for lig in BATCH_LIGANDS:\n"
                "                m = load_molecule(MOLECULE_FILES[lig])\n"
                "                anc = find_phosphonate_anchor(m)\n"
                "                st, contact_tilt, tilt = dock_hbonded_ligand(\n"
                "                    slab, m, anc, donor_o, donor_h)\n"
                "                out.append(export_ligand_case(\n"
                "                    st, n_slab=len(slab), surface_file=surf_file,\n"
                "                    surface_name=surf_name, ligand=lig,\n"
                "                    mol_path=MOLECULE_FILES[lig], anchor=anc, fraction=frac,\n"
                "                    pattern=pat, motif=mot, pos_name=\"hbond\",\n"
                "                    target_xy=donor_o[:2], sites=s,\n"
                "                    contact_tilt=contact_tilt, tilt=tilt,\n"
                "                    binding_mode=\"hbond\", donor_o=donor_o, donor_h=donor_h))\n"
                "    expected = len(SURFACES) * len(BATCH_LIGANDS) * sum(\n",
                label="full-OH hydrogen-bond loop",
            )
            source = replace_once(
                source,
                "        for frac, _pattern, _motif in BATCH_CASES)\n",
                "        for frac, _pattern, _motif in BATCH_CASES)\n"
                "    expected += len(SURFACES) * len(BATCH_LIGANDS) * len(BATCH_HBOND_CASES)\n",
                label="hydrogen-bond completeness assertion",
            )
            source = source.replace(
                "if BATCH_REFERENCE_ONLY:\n"
                "    print(\"Reference-only full-OH cases (no bare Ni anchor):\", BATCH_REFERENCE_ONLY)\n",
                "if BATCH_HBOND_CASES:\n"
                "    print(\"Full-OH molecular H-bond cases:\", BATCH_HBOND_CASES)\n",
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
