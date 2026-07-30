"""ASE-backed, command-line-friendly VASP geometry preparation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError


def _ase() -> dict[str, Any]:
    try:
        from ase.build import add_vacuum, sort, surface
        from ase.constraints import FixAtoms
        from ase.io import read, write
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required for geometry operations. Install with: "
            "pip install 'interfaceforge[vasp]'"
        ) from exc
    return {
        "add_vacuum": add_vacuum,
        "sort": sort,
        "surface": surface,
        "FixAtoms": FixAtoms,
        "read": read,
        "write": write,
    }


def _output_guard(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing structure: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def convert_structure(
    source: str | Path,
    output: str | Path,
    *,
    cell_from: str | Path | None = None,
    cell: Sequence[float] | None = None,
    center: bool = False,
    sort_atoms: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Convert a structure, optionally transplanting a trusted periodic cell."""

    ase = _ase()
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    _output_guard(output_path, force)
    atoms = ase["read"](str(source_path))
    if cell_from:
        reference = ase["read"](str(Path(cell_from).resolve()))
        atoms.set_cell(reference.cell)
        atoms.set_pbc(reference.pbc)
    elif cell:
        if len(cell) not in {3, 6, 9}:
            raise ValueError("--cell needs 3 lengths, 6 cell parameters, or 9 matrix values")
        values = [float(item) for item in cell]
        if len(values) == 3:
            atoms.set_cell(values)
        elif len(values) == 6:
            from ase.geometry import cellpar_to_cell

            atoms.set_cell(cellpar_to_cell(values))
        else:
            atoms.set_cell(np.asarray(values).reshape(3, 3))
        atoms.set_pbc(True)
    if center:
        atoms.center()
    if sort_atoms:
        atoms = ase["sort"](atoms)
    ase["write"](str(output_path), atoms, format="vasp", direct=True, vasp5=True)
    return structure_summary(atoms, source=source_path, output=output_path)


def build_supercell(
    source: str | Path,
    output: str | Path,
    repeat: Sequence[int],
    *,
    vacuum: float | None = None,
    axis: int = 2,
    sort_atoms: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    ase = _ase()
    if len(repeat) != 3 or any(int(value) < 1 for value in repeat):
        raise ValueError("--repeat needs three positive integers")
    atoms = ase["read"](str(Path(source).resolve())) * tuple(int(value) for value in repeat)
    if vacuum is not None:
        atoms.center(vacuum=float(vacuum), axis=int(axis))
    if sort_atoms:
        atoms = ase["sort"](atoms)
    output_path = Path(output).resolve()
    _output_guard(output_path, force)
    ase["write"](str(output_path), atoms, format="vasp", direct=True, vasp5=True)
    return structure_summary(atoms, source=Path(source).resolve(), output=output_path)


def build_slab(
    source: str | Path,
    output: str | Path,
    miller: Sequence[int],
    layers: int,
    *,
    repeat: Sequence[int] = (1, 1, 1),
    vacuum: float = 15.0,
    periodic: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Build a Miller-index slab with explicit layer, repeat, and vacuum choices."""

    ase = _ase()
    if len(miller) != 3:
        raise ValueError("--miller needs h k l")
    if layers < 1:
        raise ValueError("--layers must be positive")
    bulk = ase["read"](str(Path(source).resolve()))
    slab = ase["surface"](bulk, tuple(int(value) for value in miller), layers, periodic=periodic)
    slab *= tuple(int(value) for value in repeat)
    slab.center(vacuum=float(vacuum), axis=2)
    slab = ase["sort"](slab)
    output_path = Path(output).resolve()
    _output_guard(output_path, force)
    ase["write"](str(output_path), slab, format="vasp", direct=True, vasp5=True)
    summary = structure_summary(slab, source=Path(source).resolve(), output=output_path)
    summary.update({"miller": list(miller), "layers": layers, "repeat": list(repeat), "vacuum_a": vacuum})
    return summary


def freeze_structure(
    source: str | Path,
    output: str | Path,
    *,
    axis: str = "z",
    lower: float | None = None,
    upper: float | None = None,
    region: str = "outside",
    elements: Sequence[str] = (),
    append: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write VASP Selective Dynamics constraints from spatial or elemental rules."""

    ase = _ase()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    if region not in {"inside", "outside"}:
        raise ValueError("region must be inside or outside")
    if lower is None and upper is None and not elements:
        raise ValueError("Provide at least one bound or element")

    atoms = ase["read"](str(Path(source).resolve()))
    coordinates = atoms.positions[:, axis_index]
    spatial = np.ones(len(atoms), dtype=bool) if region == "inside" else np.zeros(len(atoms), dtype=bool)
    if lower is not None or upper is not None:
        inside = np.ones(len(atoms), dtype=bool)
        if lower is not None:
            inside &= coordinates >= float(lower)
        if upper is not None:
            inside &= coordinates <= float(upper)
        spatial = inside if region == "inside" else ~inside
    selected_elements = {str(item) for item in elements}
    elemental = np.asarray(
        [atom.symbol in selected_elements for atom in atoms], dtype=bool
    )
    fixed = spatial | elemental

    if append:
        for constraint in getattr(atoms, "constraints", []):
            try:
                fixed[np.asarray(constraint.get_indices(), dtype=int)] = True
            except (AttributeError, TypeError, ValueError):
                continue
    atoms.set_constraint(ase["FixAtoms"](mask=fixed))
    output_path = Path(output).resolve()
    _output_guard(output_path, force)
    ase["write"](str(output_path), atoms, format="vasp", direct=True, vasp5=True)
    summary = structure_summary(atoms, source=Path(source).resolve(), output=output_path)
    summary.update(
        {
            "fixed_atoms": int(fixed.sum()),
            "mobile_atoms": int(len(atoms) - fixed.sum()),
            "axis": axis,
            "lower": lower,
            "upper": upper,
            "region": region,
            "elements": sorted(selected_elements),
        }
    )
    return summary


def clean_duplicates(
    source: str | Path,
    output: str | Path,
    *,
    cutoff: float = 0.5,
    force: bool = False,
) -> dict[str, Any]:
    """Remove later atoms from pairs closer than a conservative cutoff."""

    ase = _ase()
    if cutoff <= 0:
        raise ValueError("--cutoff must be positive")
    atoms = ase["read"](str(Path(source).resolve()))
    distances = atoms.get_all_distances(mic=True)
    remove: set[int] = set()
    close_pairs: list[dict[str, Any]] = []
    for left in range(len(atoms)):
        if left in remove:
            continue
        for right in range(left + 1, len(atoms)):
            if right in remove:
                continue
            distance = float(distances[left, right])
            if distance < cutoff:
                remove.add(right)
                close_pairs.append(
                    {
                        "kept": left,
                        "removed": right,
                        "distance_a": distance,
                        "kept_element": atoms[left].symbol,
                        "removed_element": atoms[right].symbol,
                    }
                )
    if remove:
        del atoms[sorted(remove)]
    output_path = Path(output).resolve()
    _output_guard(output_path, force)
    ase["write"](str(output_path), atoms, format="vasp", direct=True, vasp5=True)
    summary = structure_summary(atoms, source=Path(source).resolve(), output=output_path)
    summary.update({"cutoff_a": cutoff, "removed_atoms": len(remove), "close_pairs": close_pairs})
    return summary


def structure_summary(
    atoms_or_path: Any,
    *,
    source: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    ase = _ase()
    atoms = ase["read"](str(Path(atoms_or_path).resolve())) if isinstance(atoms_or_path, (str, Path)) else atoms_or_path
    minimum = None
    pair = None
    if len(atoms) > 1:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        flat = int(np.argmin(distances))
        left, right = np.unravel_index(flat, distances.shape)
        minimum = float(distances[left, right])
        pair = [int(left), int(right)]
    fixed: set[int] = set()
    for constraint in getattr(atoms, "constraints", []):
        try:
            fixed.update(int(index) for index in constraint.get_indices())
        except (AttributeError, TypeError, ValueError):
            continue
    summary = {
        "source": str(source) if source else None,
        "output": str(output) if output else None,
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "cell_a": np.asarray(atoms.cell.array).tolist(),
        "volume_a3": float(atoms.get_volume()),
        "pbc": [bool(value) for value in atoms.pbc],
        "minimum_distance_a": minimum,
        "minimum_distance_pair": pair,
        "fixed_atoms": len(fixed),
        "mobile_atoms": len(atoms) - len(fixed),
        "elements": list(dict.fromkeys(atoms.get_chemical_symbols())),
    }
    return summary


def write_summary(summary: dict[str, Any], path: str | Path | None) -> None:
    if path:
        Path(path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
