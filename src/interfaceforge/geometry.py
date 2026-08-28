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


_AXIS_INDEX = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}


def _axis_index(axis: int | str) -> int:
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError(f"axis index must be 0, 1, or 2; got {axis}")
        return axis
    key = str(axis).strip().lower()
    if key not in _AXIS_INDEX:
        raise ValueError(f"axis must be a/b/c or 0/1/2; got {axis!r}")
    return _AXIS_INDEX[key]


def _vacuum_along(atoms: Any, ax: int) -> dict[str, Any] | None:
    """Vacuum around a contiguous slab+adsorbate stack along lattice vector ``ax``.

    The stack is the single largest contiguous band of atoms along the axis
    (modulo periodicity); ``vacuum_a`` is the remaining gap, i.e. the real
    distance from the top of the stack, through the periodic vacuum, to the
    bottom of the stack's own image. This is frame-independent: it does not
    matter where the slab sits in the cell or which face the adsorbate is
    on. ``box_gap_below_a`` / ``box_gap_above_a`` are the empty space between
    the cell faces (frac 0 and 1) and the stack -- purely cosmetic (where
    the atoms sit in the box), never the physical constraint.
    """

    length = float(np.linalg.norm(np.asarray(atoms.cell.array)[ax]))
    if length == 0.0 or len(atoms) == 0:
        return None
    raw = np.sort(np.mod(atoms.get_scaled_positions(wrap=False)[:, ax], 1.0))
    if len(raw) == 1:
        lo = hi = float(raw[0])
        wrapped = False
    else:
        gaps = np.append(np.diff(raw), (raw[0] + 1.0) - raw[-1])
        widest = int(np.argmax(gaps))
        wrapped = widest != len(gaps) - 1
        rotated = np.sort(np.mod(raw + (1.0 - raw[widest + 1]), 1.0)) if wrapped else raw
        lo, hi = float(rotated[0]), float(rotated[-1])
    span = hi - lo
    return {
        "axis_length_a": length,
        "slab_span_a": span * length,
        "vacuum_a": (1.0 - span) * length,
        "box_gap_below_a": float(raw[0]) * length,
        "box_gap_above_a": (1.0 - float(raw[-1])) * length,
        "wrapped": wrapped,
    }


def slab_vacuum(atoms_or_path: Any, *, axis: int | str = "auto") -> dict[str, Any]:
    """Vacuum between a slab (plus any adsorbate) and its periodic image.

    ``axis="auto"`` picks the lattice vector with the most vacuum (the
    surface normal for a slab). ``vacuum_a`` is the one number that matters:
    the gap from the top of the tallest adsorbate, through the periodic
    vacuum on both sides of the centred slab, to the slab's own image.
    """

    ase = _ase()
    atoms = (
        ase["read"](str(Path(atoms_or_path).resolve()))
        if isinstance(atoms_or_path, (str, Path))
        else atoms_or_path
    )
    if str(axis) == "auto":
        candidates = [
            (report, index)
            for index in range(3)
            if (report := _vacuum_along(atoms, index)) is not None
        ]
        if not candidates:
            raise SafetyError("Structure has no cell; cannot measure vacuum")
        report, ax = max(candidates, key=lambda item: item[0]["vacuum_a"])
    else:
        ax = _axis_index(axis)
        report = _vacuum_along(atoms, ax)
        if report is None:
            raise SafetyError(f"Lattice vector {'abc'[ax]} has zero length")
    return {"axis": "abc"[ax], **report}


def extend_slab_vacuum(
    source: str | Path,
    output: str | Path,
    *,
    axis: int | str = "auto",
    vacuum: float = 15.0,
    recenter: bool = True,
    sort_atoms: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Grow the normal cell vector so the slab-to-image gap is ``vacuum`` Å.

    Only empty space is added along the surface normal -- every atomic
    position (and every bond) is unchanged, so a relaxed CONTCAR can be
    stretched and reused directly. ``recenter`` (default) then translates
    the whole stack so it sits tidily inside ``[0, c)``; that is cosmetic
    and does not change ``vacuum_a``. A shorter cell is left alone.
    """

    if vacuum <= 0:
        raise ValueError("--extend vacuum must be positive")
    ase = _ase()
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    _output_guard(output_path, force)
    atoms = ase["read"](str(source_path))
    ax = (
        _axis_index(axis)
        if str(axis) != "auto"
        else int("abc".index(slab_vacuum(atoms, axis="auto")["axis"]))
    )
    before = _vacuum_along(atoms, ax)
    if before is None:
        raise SafetyError(f"Lattice vector {'abc'[ax]} has zero length")

    cell = np.asarray(atoms.cell.array).copy()
    unit = cell[ax] / np.linalg.norm(cell[ax])
    new_length = max(before["axis_length_a"], before["slab_span_a"] + float(vacuum))
    cell[ax] = unit * new_length
    atoms.set_cell(cell, scale_atoms=False)
    if recenter:
        atoms.center(axis=ax)

    if sort_atoms:
        atoms = ase["sort"](atoms)
    ase["write"](str(output_path), atoms, format="vasp", direct=True, vasp5=True)
    after = _vacuum_along(atoms, ax)
    summary = structure_summary(atoms, source=source_path, output=output_path)
    summary.update(
        {
            "axis": "abc"[ax],
            "vacuum_target_a": float(vacuum),
            "recentred": bool(recenter),
            "vacuum_before_a": round(before["vacuum_a"], 2),
            "vacuum_after_a": round(after["vacuum_a"], 2),
            "axis_length_before_a": round(before["axis_length_a"], 2),
            "axis_length_after_a": round(after["axis_length_a"], 2),
        }
    )
    return summary


_VACUUM_BATCH_NAMES = ("POSCAR", "CONTCAR")
_VACUUM_BATCH_SKIP = ("archive", "backup")


def _discover_structures(root: Path) -> list[Path]:
    found: list[Path] = []
    for name in _VACUUM_BATCH_NAMES:
        for path in sorted(root.rglob(name)):
            parts = {p.lower() for p in path.relative_to(root).parts}
            if parts & set(_VACUUM_BATCH_SKIP) or any(p.startswith("x") for p in parts):
                continue
            # one file per directory: POSCAR wins over CONTCAR
            if name == "CONTCAR" and (path.parent / "POSCAR").is_file():
                continue
            found.append(path)
    return sorted(found)


def batch_slab_vacuum(
    root: str | Path,
    *,
    axis: int | str = "auto",
    min_vacuum: float = 12.0,
    extend: float | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Audit every slab under a tree; with ``extend`` plan (or, if ``execute``,
    apply) an in-place cell stretch for each one below ``min_vacuum``."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise SafetyError(f"Not a directory: {root_path}")
    structures = _discover_structures(root_path)
    if not structures:
        raise SafetyError(f"No POSCAR/CONTCAR found below {root_path}")

    rows: list[dict[str, Any]] = []
    for path in structures:
        report = slab_vacuum(path, axis=axis)
        thin = report["vacuum_a"] < min_vacuum
        row: dict[str, Any] = {
            "path": str(path.relative_to(root_path)),
            "axis": report["axis"],
            "slab_span_a": round(report["slab_span_a"], 2),
            "vacuum_a": round(report["vacuum_a"], 2),
            "status": "THIN" if thin else "PASS",
        }
        if extend is not None and thin:
            row["would_be_a"] = round(max(report["vacuum_a"], float(extend)), 2)
            if execute:
                extend_slab_vacuum(path, path, axis=axis, vacuum=extend, force=True)
                row["vacuum_a"] = row.pop("would_be_a")
                row["extended"] = True
        rows.append(row)

    mode = "audit" if extend is None else ("extended" if execute else "dry-run")
    return {
        "root": str(root_path),
        "mode": mode,
        "min_vacuum_a": float(min_vacuum),
        "structures": len(rows),
        "thin": sum(r["status"] == "THIN" for r in rows),
        "extended": sum(r.get("extended", False) for r in rows),
        "rows": rows,
    }


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
