"""Post-relaxation chemistry and spin classification for surface campaigns."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import _ase, mic_delta
from .magnetism import audit_magnetization, parse_outcar_magnetization


def _energy(outcar: Path) -> float | None:
    if not outcar.is_file():
        return None
    values = re.findall(
        r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)",
        outcar.read_text(encoding="utf-8", errors="ignore"),
    )
    return float(values[-1]) if values else None


def _finished(outcar: Path) -> bool:
    if not outcar.is_file():
        return False
    tail = outcar.read_text(encoding="utf-8", errors="ignore")[-250_000:]
    return "General timing and accounting informations" in tail


def classify_surface_structure(atoms, *, metal: str = "Ni") -> dict[str, Any]:
    """Classify robust local bonds without relying on original atom indices."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    positions = atoms.positions
    oxygen = np.where(symbols == "O")[0]
    hydrogen = np.where(symbols == "H")[0]
    phosphorus = np.where(symbols == "P")[0]
    metals = np.where(symbols == metal)[0]
    oh_distances: list[float] = []
    for index in hydrogen:
        if len(oxygen):
            oh_distances.append(
                float(np.linalg.norm(mic_delta(positions[oxygen], positions[index], atoms.cell), axis=1).min())
            )
    bonded_h = int(sum(distance <= 1.25 for distance in oh_distances))
    detached_h = len(hydrogen) - bonded_h

    phosphonate_mode = None
    closest_anchor_metal = None
    if len(phosphorus) and len(oxygen) and len(metals):
        anchor_oxygen: set[int] = set()
        for p_index in phosphorus:
            distances = np.linalg.norm(mic_delta(positions[oxygen], positions[p_index], atoms.cell), axis=1)
            anchor_oxygen.update(int(oxygen[i]) for i in np.where(distances <= 2.05)[0])
        if anchor_oxygen:
            matrix = np.linalg.norm(
                mic_delta(
                    positions[np.asarray(sorted(anchor_oxygen))][:, None, :],
                    positions[metals][None, :, :],
                    atoms.cell,
                ),
                axis=-1,
            )
            closest_anchor_metal = float(matrix.min())
            phosphonate_mode = "metal-bound" if closest_anchor_metal <= 2.55 else "non-chemisorbed"

    all_distances = np.linalg.norm(mic_delta(positions[:, None, :], positions[None, :, :], atoms.cell), axis=-1)
    np.fill_diagonal(all_distances, np.inf)
    minimum_distance = float(all_distances.min()) if len(atoms) > 1 else None
    status = "PASS"
    issues: list[str] = []
    if detached_h:
        status = "CHECK"
        issues.append(f"{detached_h} H atoms lack an O-H bond <=1.25 A")
    if minimum_distance is not None and minimum_distance < 0.65:
        status = "CHECK"
        issues.append(f"minimum interatomic distance is {minimum_distance:.3f} A")
    return {
        "status": status,
        "formula": atoms.get_chemical_formula(),
        "o_h_bonds": bonded_h,
        "detached_h": detached_h,
        "minimum_distance_a": round(minimum_distance, 6) if minimum_distance is not None else None,
        "phosphonate_mode": phosphonate_mode,
        "closest_anchor_o_metal_a": round(closest_anchor_metal, 6) if closest_anchor_metal is not None else None,
        "issues": issues,
    }


def _read_atoms(run: Path, names: tuple[str, ...]):
    for name in names:
        path = run / name
        if path.is_file():
            try:
                return _ase()["read"](path)
            except Exception:
                continue
    raise ValueError(f"no readable structure in {run}")


def audit_surface_runs(root: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    provenance_files = sorted(root.rglob("provenance.json"))
    rows: list[dict[str, Any]] = []
    for provenance_path in provenance_files:
        run = provenance_path.parent
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        initial_atoms = _read_atoms(run, ("structure.extxyz", "POSCAR", "CONTCAR"))
        final_atoms = _read_atoms(run, ("CONTCAR", "structure.extxyz", "POSCAR"))
        outcar = run / "OUTCAR"
        metal = provenance.get("magnetism", {}).get("magnetic_species", "Ni")
        chemistry = classify_surface_structure(final_atoms, metal=metal)
        spin = audit_magnetization(
            initial_atoms,
            parse_outcar_magnetization(outcar),
            magnetic_species=metal,
        )
        rows.append(
            {
                "name": provenance.get("name", run.name),
                "run_dir": str(run),
                "finished": _finished(outcar),
                "energy_ev": _energy(outcar),
                "coverage": provenance.get("coverage"),
                "initial_motif": provenance.get("motif"),
                "initial_binding": (provenance.get("docking") or {}).get("mode"),
                "chemical_status": chemistry["status"],
                "classified_binding": chemistry["phosphonate_mode"],
                "detached_h": chemistry["detached_h"],
                "spin_status": spin["status"],
                "chemistry": chemistry,
                "spin": spin,
            }
        )
    payload = {
        "root": str(root),
        "run_count": len(rows),
        "finished": sum(bool(row["finished"]) for row in rows),
        "chemistry_checks": sum(row["chemical_status"] != "PASS" for row in rows),
        "spin_checks": sum(row["spin_status"] not in {"PASS", "NOT_APPLICABLE"} for row in rows),
        "runs": rows,
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".csv":
            fields = [
                "name",
                "run_dir",
                "finished",
                "energy_ev",
                "coverage",
                "initial_motif",
                "initial_binding",
                "chemical_status",
                "classified_binding",
                "detached_h",
                "spin_status",
            ]
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in fields})
        else:
            destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["output"] = str(destination.resolve())
    return payload
