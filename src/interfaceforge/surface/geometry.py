"""Geometry primitives shared by reactive-surface workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InterfaceForgeError


def _ase() -> dict[str, Any]:
    try:
        from ase import Atoms
        from ase.constraints import FixAtoms
        from ase.io import read, write
    except ImportError as exc:  # pragma: no cover - exercised without optional deps
        raise InterfaceForgeError(
            "Reactive surface operations require ASE. Install InterfaceForge with the 'vasp' extra."
        ) from exc
    return {"Atoms": Atoms, "FixAtoms": FixAtoms, "read": read, "write": write}


def read_structure(path: str | Path):
    atoms = _ase()["read"](Path(path))
    if not atoms.cell.rank:
        raise ValueError(f"{path} has no periodic cell")
    atoms.pbc = (True, True, False)
    return atoms


def read_molecule(path: str | Path):
    atoms = _ase()["read"](Path(path))
    atoms.set_cell(None)
    atoms.pbc = False
    atoms.set_constraint()
    atoms.center()
    return atoms


def mic_delta(p: np.ndarray, q: np.ndarray, cell) -> np.ndarray:
    """Minimum-image ``p-q`` with wrapping confined to the surface plane."""
    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    delta = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    fractional = delta @ np.linalg.inv(matrix)
    fractional[..., :2] -= np.round(fractional[..., :2])
    return fractional @ matrix


def inplane_distance_matrix(atoms, indices: Sequence[int]) -> np.ndarray:
    pos = atoms.positions[np.asarray(indices, dtype=int)]
    delta = mic_delta(pos[:, None, :], pos[None, :, :], atoms.cell)
    delta[..., 2] = 0.0
    return np.linalg.norm(delta, axis=-1)


def frozen_indices(atoms) -> np.ndarray:
    frozen: set[int] = set()
    for constraint in getattr(atoms, "constraints", []):
        values = getattr(constraint, "index", getattr(constraint, "a", None))
        if values is not None:
            frozen.update(int(i) for i in np.atleast_1d(values))
    return np.asarray(sorted(frozen), dtype=int)


def detect_layers(atoms, *, axis: int = 2, tolerance: float = 0.60) -> list[np.ndarray]:
    order = np.argsort(atoms.positions[:, axis])
    layers: list[list[int]] = []
    for index in order:
        if not layers:
            layers.append([int(index)])
            continue
        center = float(np.mean(atoms.positions[layers[-1], axis]))
        if abs(float(atoms.positions[index, axis]) - center) <= tolerance:
            layers[-1].append(int(index))
        else:
            layers.append([int(index)])
    return [np.asarray(layer, dtype=int) for layer in layers]


def freeze_bottom_layers(atoms, count: int, *, axis: int = 2, tolerance: float = 0.60):
    if count < 0:
        raise ValueError("frozen layer count cannot be negative")
    out = atoms.copy()
    if count == 0:
        out.set_constraint()
        return out
    layers = detect_layers(out, axis=axis, tolerance=tolerance)
    if count >= len(layers):
        raise ValueError(f"cannot freeze {count} of {len(layers)} detected layers")
    indices = np.concatenate(layers[:count])
    out.set_constraint(_ase()["FixAtoms"](indices=indices))
    return out


@dataclass(frozen=True)
class SurfaceAnalysis:
    metal: str
    anion: str
    metal_indices: tuple[int, ...]
    anion_indices: tuple[int, ...]
    exposed_indices: tuple[int, ...]
    exposed_z: float
    coordination: dict[int, int]
    site_labels: dict[int, int]

    @property
    def n_exposed(self) -> int:
        return len(self.exposed_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metal": self.metal,
            "anion": self.anion,
            "n_metal": len(self.metal_indices),
            "n_anion": len(self.anion_indices),
            "n_exposed": self.n_exposed,
            "exposed_indices": list(self.exposed_indices),
            "exposed_z_a": round(self.exposed_z, 6),
            "coordination": {str(k): v for k, v in self.coordination.items()},
            "site_labels": {str(k): v for k, v in self.site_labels.items()},
        }


def _site_labels(atoms, indices: Sequence[int], tolerance: float) -> dict[int, int]:
    """Best-effort crystallographic equivalence labels with a safe fallback."""
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = AseAtomsAdaptor.get_structure(atoms)
        dataset = SpacegroupAnalyzer(structure, symprec=tolerance).get_symmetry_dataset()
        equivalent = dataset["equivalent_atoms"]
        return {int(i): int(equivalent[i]) for i in indices}
    except Exception:
        return {int(i): int(i) for i in indices}


def analyze_surface(
    atoms_or_path,
    *,
    metal: str,
    anion: str = "O",
    coordination_cutoff: float = 2.7,
    bulk_coordination: int = 6,
    top_tolerance: float = 0.8,
    symmetry_tolerance: float = 0.1,
) -> SurfaceAnalysis:
    """Identify under-coordinated metal sites on the upper slab surface."""
    atoms = read_structure(atoms_or_path) if isinstance(atoms_or_path, (str, Path)) else atoms_or_path
    symbols = np.asarray(atoms.get_chemical_symbols())
    metal_indices = np.where(symbols == metal)[0]
    anion_indices = np.where(symbols == anion)[0]
    if not len(metal_indices) or not len(anion_indices):
        raise ValueError(f"surface must contain both {metal} and {anion}")

    z = atoms.positions[:, 2]
    upper_midpoint = float(np.median(z[metal_indices]))
    exposed_z = float(z[metal_indices].max())
    coordination: dict[int, int] = {}
    for index in metal_indices:
        distances = np.linalg.norm(
            mic_delta(atoms.positions[anion_indices], atoms.positions[index], atoms.cell), axis=1
        )
        coordination[int(index)] = int(np.sum(distances <= coordination_cutoff))

    exposed = sorted(
        int(index)
        for index in metal_indices
        if z[index] >= upper_midpoint
        and (z[index] >= exposed_z - top_tolerance or coordination[int(index)] < bulk_coordination)
    )
    if not exposed:
        raise ValueError("no exposed upper-surface metal sites were identified")
    labels = _site_labels(atoms, exposed, symmetry_tolerance)
    return SurfaceAnalysis(
        metal=metal,
        anion=anion,
        metal_indices=tuple(map(int, metal_indices)),
        anion_indices=tuple(map(int, anion_indices)),
        exposed_indices=tuple(exposed),
        exposed_z=exposed_z,
        coordination=coordination,
        site_labels=labels,
    )


def select_sites(atoms, exposed: Sequence[int], fraction: float, arrangement: str, seed: int = 0) -> list[int]:
    n = int(round(float(fraction) * len(exposed)))
    if n <= 0:
        return []
    if n >= len(exposed):
        return sorted(map(int, exposed))
    matrix = inplane_distance_matrix(atoms, exposed)
    rng = np.random.default_rng(seed)
    if arrangement not in {"clustered", "scattered"}:
        raise ValueError("arrangement must be clustered or scattered")
    chosen = [int(rng.integers(len(exposed)))]
    while len(chosen) < n:
        remaining = [i for i in range(len(exposed)) if i not in chosen]
        nearest = [min(matrix[i, j] for j in chosen) for i in remaining]
        pick = int(np.argmin(nearest) if arrangement == "clustered" else np.argmax(nearest))
        chosen.append(remaining[pick])
    return sorted(int(exposed[i]) for i in chosen)


def extend_atoms(base, extra):
    constraints = list(getattr(base, "constraints", []))
    moments = np.asarray(base.get_initial_magnetic_moments(), dtype=float)
    out = base.copy()
    out += extra
    if constraints:
        out.set_constraint(constraints)
    if np.any(moments):
        out.set_initial_magnetic_moments(np.concatenate([moments, np.zeros(len(extra))]))
    out.pbc = (True, True, False)
    return out


def periodic_self_image_gap(positions: np.ndarray, cell, *, images: int = 2) -> float:
    """Minimum distance between an object and its in-plane periodic copies."""
    positions = np.asarray(positions, dtype=float)
    if not len(positions):
        raise ValueError("cannot measure the image gap of an empty object")
    a, b = np.asarray(cell, dtype=float)[:2]
    best = np.inf
    for ia in range(-images, images + 1):
        for ib in range(-images, images + 1):
            if ia == 0 and ib == 0:
                continue
            shifted = positions + ia * a + ib * b
            best = min(
                best,
                float(np.linalg.norm(positions[:, None] - shifted[None, :], axis=-1).min()),
            )
    return best
