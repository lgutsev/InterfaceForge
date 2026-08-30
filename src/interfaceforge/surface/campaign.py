"""Declarative reactive-surface campaign planning and export."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..errors import SafetyError
from .geometry import _ase, analyze_surface, freeze_bottom_layers, frozen_indices, read_structure
from .magnetism import assign_superexchange_afm, compact_magmom
from .reactions import ReactiveState, dock_phosphonate, enumerate_reactive_states


def _resolved(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_surface_campaign(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("version") != 1:
        raise ValueError("surface campaign requires version: 1")
    for key in ("surface", "states"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"surface campaign requires a {key}: mapping")
    surface = payload["surface"]
    if not all(surface.get(key) for key in ("structure", "metal", "anion")):
        raise ValueError("surface requires structure, metal, and anion")
    states = payload["states"]
    if not states.get("coverages") or not states.get("motifs"):
        raise ValueError("states requires nonempty coverages and motifs")
    root = source.parent
    payload["_source"] = source
    payload["_root"] = root
    surface["structure"] = _resolved(root, surface["structure"])
    export = payload.setdefault("export", {})
    export["output"] = _resolved(root, export.get("output", "generated_surface_campaign"))
    if export.get("vasp_template"):
        export["vasp_template"] = _resolved(root, export["vasp_template"])
    for adsorbate in payload.get("adsorbates", []):
        if not isinstance(adsorbate, dict) or not adsorbate.get("name") or not adsorbate.get("file"):
            raise ValueError("each adsorbate requires name and file")
        adsorbate["file"] = _resolved(root, adsorbate["file"])
    return payload


def _prepare_base(config: dict[str, Any]):
    surface = config["surface"]
    base = read_structure(surface["structure"])
    freeze = config.get("freeze", {})
    if freeze.get("mode", "inherit") == "bottom-layers":
        base = freeze_bottom_layers(
            base,
            int(freeze.get("count", 1)),
            tolerance=float(freeze.get("tolerance", 0.60)),
        )
    elif freeze.get("mode", "inherit") == "inherit":
        if freeze.get("required", True) and not len(frozen_indices(base)):
            raise ValueError("surface has no frozen atoms; use freeze.mode: bottom-layers or required: false")
    elif freeze.get("mode") == "none":
        base.set_constraint()
    else:
        raise ValueError("freeze.mode must be inherit, bottom-layers, or none")

    magnetism = config.get("magnetism", {"mode": "inherit"})
    mode = magnetism.get("mode", "inherit")
    magnetic_report: dict[str, Any]
    if mode == "superexchange":
        magnetic_report = assign_superexchange_afm(
            base,
            magnetic_species=magnetism.get("magnetic_species", surface["metal"]),
            bridge_species=magnetism.get("bridge_species", surface["anion"]),
            moment=float(magnetism.get("moment", 2.0)),
            bond_cutoff=float(magnetism.get("bond_cutoff", surface.get("coordination_cutoff", 2.7))),
            linear_dot=float(magnetism.get("linear_dot", -0.85)),
            require_balanced=bool(magnetism.get("require_balanced", True)),
        )
    elif mode == "inherit":
        moments = np.asarray(base.get_initial_magnetic_moments(), dtype=float)
        if magnetism.get("required", True) and not np.any(moments):
            raise ValueError("magnetism.mode=inherit but the structure carries no initial moments")
        magnetic_report = {
            "mode": "inherit",
            "nonzero_moments": int(np.sum(np.abs(moments) > 0)),
            "net_moment": float(np.sum(moments)),
            "magmom": compact_magmom(moments) if np.any(moments) else None,
        }
    elif mode == "none":
        magnetic_report = {"mode": "none"}
    else:
        raise ValueError("magnetism.mode must be inherit, superexchange, or none")
    return base, magnetic_report


def _states(config: dict[str, Any], base):
    surface = config["surface"]
    analysis = analyze_surface(
        base,
        metal=surface["metal"],
        anion=surface["anion"],
        coordination_cutoff=float(surface.get("coordination_cutoff", 2.7)),
        bulk_coordination=int(surface.get("bulk_coordination", 6)),
        top_tolerance=float(surface.get("top_tolerance", 0.8)),
    )
    state_config = config["states"]
    states = enumerate_reactive_states(
        base,
        analysis,
        coverages=[float(value) for value in state_config["coverages"]],
        arrangements=list(state_config.get("arrangements", ["clustered", "scattered"])),
        motifs=list(state_config["motifs"]),
        seed=int(state_config.get("seed", 1)),
        parameters=dict(state_config.get("parameters", {})),
        name_prefix=str(config.get("name", "surface")),
    )
    return analysis, states


def _decorated_states(config: dict[str, Any], analysis, states: list[ReactiveState]):
    decorated: list[tuple[ReactiveState, Any, dict[str, Any], str]] = []
    skipped: list[dict[str, str]] = []
    for adsorbate in config.get("adsorbates", []):
        modes = adsorbate.get("modes", ["direct"])
        for state in states:
            for mode_config in modes:
                mode = mode_config if isinstance(mode_config, str) else mode_config.get("mode")
                options = (
                    {}
                    if isinstance(mode_config, str)
                    else {key: value for key, value in mode_config.items() if key not in {"mode", "coverages"}}
                )
                coverage_filter = None if isinstance(mode_config, str) else mode_config.get("coverages")
                if coverage_filter is not None and not any(
                    np.isclose(state.coverage, float(value)) for value in coverage_filter
                ):
                    continue
                try:
                    atoms, docking = dock_phosphonate(
                        state,
                        analysis,
                        adsorbate["file"],
                        mode=str(mode),
                        **options,
                    )
                except ValueError as exc:
                    skipped.append(
                        {"state": state.name, "adsorbate": adsorbate["name"], "mode": str(mode), "reason": str(exc)}
                    )
                    continue
                token = re.sub(r"[^A-Za-z0-9]+", "", str(adsorbate["name"]))
                name = f"{state.name}_{token}_{mode}"
                decorated.append((state, atoms, docking, name))
    return decorated, skipped


def _state_graph(states: list[ReactiveState], decorated) -> dict[str, Any]:
    nodes = [
        {
            "id": state.name,
            "kind": "surface-state",
            "coverage": state.coverage,
            "arrangement": state.arrangement,
            "motif": state.motif,
            "source_equivalents": state.provenance.get("source_equivalents"),
        }
        for state in states
    ]
    edges: list[dict[str, Any]] = []
    clean = next((state for state in states if np.isclose(state.coverage, 0.0)), None)
    for state in states:
        if clean is not None and state is not clean:
            previous = [
                candidate
                for candidate in states
                if candidate.coverage < state.coverage
                and candidate.motif == state.motif
                and (
                    candidate.arrangement == state.arrangement
                    or state.arrangement == "full"
                    or np.isclose(candidate.coverage, 0.0)
                )
            ]
            maximum_coverage = max((candidate.coverage for candidate in previous), default=clean.coverage)
            parents = [candidate for candidate in previous if np.isclose(candidate.coverage, maximum_coverage)] or [
                clean
            ]
            for parent in parents:
                edges.append(
                    {
                        "source": parent.name,
                        "target": state.name,
                        "reaction": "increase-coverage",
                        "delta_coverage": round(state.coverage - parent.coverage, 8),
                    }
                )
        if state.motif == "dissociated_water":
            terminal = next(
                (
                    candidate
                    for candidate in states
                    if np.isclose(candidate.coverage, state.coverage)
                    and candidate.arrangement == state.arrangement
                    and candidate.motif == "terminal_hydroxyl"
                ),
                None,
            )
            if terminal is not None:
                edges.append(
                    {
                        "source": terminal.name,
                        "target": state.name,
                        "reaction": "protonate-lattice-oxygen",
                        "protons": len(state.selected_sites),
                    }
                )
    for parent, _atoms, docking, name in decorated:
        nodes.append(
            {
                "id": name,
                "kind": "adsorbate-state",
                "coverage": parent.coverage,
                "arrangement": parent.arrangement,
                "motif": parent.motif,
                "binding_mode": docking["mode"],
            }
        )
        edges.append(
            {
                "source": parent.name,
                "target": name,
                "reaction": "bind-adsorbate",
                "binding_mode": docking["mode"],
            }
        )
    return {"schema": "interfaceforge.reactive-surface-graph/v1", "nodes": nodes, "edges": edges}


def plan_surface_campaign(path: str | Path) -> dict[str, Any]:
    config = load_surface_campaign(path)
    base, magnetic = _prepare_base(config)
    analysis, states = _states(config, base)
    decorated, skipped = _decorated_states(config, analysis, states)
    graph = _state_graph(states, decorated)
    return {
        "campaign": config.get("name", "surface"),
        "config": str(config["_source"]),
        "surface": analysis.to_dict(),
        "magnetism": magnetic,
        "freeze": {
            "frozen_atoms": int(len(frozen_indices(base))),
            "total_atoms": len(base),
        },
        "state_count": len(states),
        "decorated_state_count": len(decorated),
        "states": [
            {
                "name": state.name,
                "coverage": state.coverage,
                "arrangement": state.arrangement,
                "motif": state.motif,
                "selected_sites": state.selected_sites,
                "atoms": len(state.atoms),
                "source_equivalents": state.provenance.get("source_equivalents"),
                "initial_geometry_audit": state.provenance.get("initial_geometry_audit"),
            }
            for state in states
        ],
        "decorated_states": [
            {"name": name, "parent": state.name, **docking, "atoms": len(atoms)}
            for state, atoms, docking, name in decorated
        ],
        "skipped_decorations": skipped,
        "state_graph": graph,
        "output": str(config["export"]["output"]),
    }


def _replace_incar_value(lines: list[str], key: str, value: str) -> list[str]:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
    output: list[str] = []
    replaced = False
    for line in lines:
        if pattern.match(line):
            if not replaced:
                output.append(f"{key} = {value}")
                replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key} = {value}")
    return output


def _template_ldau_map(text: str) -> dict[str, tuple[str, str, str]]:
    species_match = re.search(r"species order\s+(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if not species_match:
        return {}
    species = species_match.group(1).strip().split()
    rows = {}
    values: dict[str, list[str]] = {}
    for key in ("LDAUL", "LDAUU", "LDAUJ"):
        match = re.search(rf"^\s*{key}\s*=\s*(.+)$", text, re.MULTILINE)
        if match:
            values[key] = match.group(1).split()
    if set(values) != {"LDAUL", "LDAUU", "LDAUJ"}:
        return {}
    for index, element in enumerate(species):
        if all(index < len(values[key]) for key in values):
            rows[element] = tuple(values[key][index] for key in ("LDAUL", "LDAUU", "LDAUJ"))
    return rows


def _write_vasp_inputs(folder: Path, atoms, config: dict[str, Any], *, name: str, decorated: bool) -> dict[str, Any]:
    template = config["export"].get("vasp_template")
    if template is None:
        return {"template": None, "potcar": "not generated"}
    template = Path(template)
    if not template.is_dir() or not (template / "INCAR").is_file():
        raise ValueError(f"VASP template must contain INCAR: {template}")
    for filename in ("KPOINTS", "runvasp.sh", "run.slurm", "run.sh"):
        source = template / filename
        if source.is_file():
            shutil.copy2(source, folder / filename)
    text = (template / "INCAR").read_text(encoding="utf-8")
    lines = text.splitlines()
    moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    lines = _replace_incar_value(lines, "SYSTEM", name)
    if np.any(moments):
        lines = _replace_incar_value(lines, "MAGMOM", compact_magmom(moments))
    lines = _replace_incar_value(lines, "ISIF", "2")
    lines = _replace_incar_value(lines, "LDIPOL", ".TRUE." if decorated else ".FALSE.")
    if decorated:
        lines = _replace_incar_value(lines, "IDIPOL", "3")
        scaled = atoms.get_center_of_mass(scaled=True)
        lines = _replace_incar_value(lines, "DIPOL", f"0.5 0.5 {scaled[2] % 1.0:.8f}")

    elements = list(dict.fromkeys(atoms.get_chemical_symbols()))
    ldau = _template_ldau_map(text)
    if ldau:
        defaults = ("-1", "0.0", "0.0")
        for column, key in enumerate(("LDAUL", "LDAUU", "LDAUJ")):
            lines = _replace_incar_value(
                lines,
                key,
                " ".join(ldau.get(element, defaults)[column] for element in elements),
            )
    for key, value in config["export"].get("incar_overrides", {}).items():
        lines = _replace_incar_value(lines, str(key).upper(), str(value))
    (folder / "INCAR").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"template": str(template), "species_order": elements, "potcar": "not generated"}


def _ordered(atoms):
    symbols = np.asarray(atoms.get_chemical_symbols())
    return atoms[np.argsort(symbols, kind="stable")]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_leaf(
    output: Path,
    state: ReactiveState,
    atoms,
    config: dict[str, Any],
    magnetic: dict[str, Any],
    *,
    name: str,
    docking: dict[str, Any] | None,
    force: bool,
) -> dict[str, Any]:
    folder = output / f"OH{int(round(100 * state.coverage))}" / name
    if folder.exists() and any(folder.iterdir()) and not force:
        raise SafetyError(f"surface run already exists: {folder}; pass --force to update known files")
    folder.mkdir(parents=True, exist_ok=True)
    ordered = _ordered(atoms)
    poscar = folder / "POSCAR"
    _ase()["write"](poscar, ordered, format="vasp", direct=True, sort=False)
    _ase()["write"](folder / "structure.extxyz", ordered)
    inputs = _write_vasp_inputs(folder, ordered, config, name=name, decorated=docking is not None)
    frozen = frozen_indices(ordered)
    provenance = {
        "schema": "interfaceforge.reactive-surface/v1",
        "campaign": config.get("name", "surface"),
        "name": name,
        "parent_state": state.name,
        "coverage": state.coverage,
        "arrangement": state.arrangement,
        "motif": state.motif,
        "selected_sites": state.selected_sites,
        "occupied_sites": state.occupied_sites,
        "source_equivalents": state.provenance.get("source_equivalents"),
        "initial_geometry_audit": state.provenance.get("initial_geometry_audit"),
        "docking": docking,
        "magnetism": magnetic,
        "n_atoms": len(ordered),
        "frozen_atoms": frozen.tolist(),
        "poscar_sha256": _sha256(poscar),
        "vasp_inputs": inputs,
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "run_dir": str(folder), **provenance}


def build_surface_campaign(path: str | Path, *, force: bool = False) -> dict[str, Any]:
    config = load_surface_campaign(path)
    base, magnetic = _prepare_base(config)
    analysis, states = _states(config, base)
    decorated, skipped = _decorated_states(config, analysis, states)
    graph = _state_graph(states, decorated)
    output = Path(config["export"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for state in states:
        rows.append(
            _write_leaf(output, state, state.atoms, config, magnetic, name=state.name, docking=None, force=force)
        )
    for parent, atoms, docking, name in decorated:
        rows.append(_write_leaf(output, parent, atoms, config, magnetic, name=name, docking=docking, force=force))

    manifest = output / "manifest.csv"
    columns = ["name", "run_dir", "parent_state", "coverage", "arrangement", "motif", "n_atoms", "docking_mode"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "run_dir": row["run_dir"],
                    "parent_state": row["parent_state"],
                    "coverage": row["coverage"],
                    "arrangement": row["arrangement"],
                    "motif": row["motif"],
                    "n_atoms": row["n_atoms"],
                    "docking_mode": (row.get("docking") or {}).get("mode", ""),
                }
            )
    summary = {
        "campaign": config.get("name", "surface"),
        "output": str(output),
        "manifest": str(manifest),
        "surface": analysis.to_dict(),
        "magnetism": magnetic,
        "state_count": len(states),
        "decorated_state_count": len(decorated),
        "run_count": len(rows),
        "skipped_decorations": skipped,
        "state_graph": str(output / "state_graph.json"),
    }
    (output / "state_graph.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    (output / "campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
