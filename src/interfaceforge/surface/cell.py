"""Cost-aware in-plane supercell selection for decorated magnetic slabs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..errors import SafetyError
from .geometry import _ase, freeze_bottom_layers, periodic_self_image_gap, read_molecule, read_structure


def _hnf_matrices(max_multiplier: int):
    """Yield unique 2-D Hermite-normal-form surface supercells."""
    for determinant in range(1, max_multiplier + 1):
        for a in range(1, determinant + 1):
            if determinant % a:
                continue
            d = determinant // a
            for b in range(d):
                yield np.asarray([[a, b], [0, d]], dtype=int)


def _translation_lengths(cell2: np.ndarray, shell: int = 2) -> list[float]:
    values: list[float] = []
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            if i == 0 and j == 0:
                continue
            values.append(float(np.linalg.norm(i * cell2[0] + j * cell2[1])))
    return values


def _parity_compatible(matrix: np.ndarray, translation_parity: tuple[int, int] | None) -> bool:
    if translation_parity is None:
        return True
    phase = np.asarray(translation_parity, dtype=int)
    return bool(np.all((matrix @ phase) % 2 == 0))


def _rotated_gap(molecule, cell, samples: int) -> float:
    """Worst periodic-image clearance over sampled azimuthal orientations."""
    positions = molecule.positions - molecule.positions.mean(axis=0)
    best = np.inf
    for index in range(max(1, samples)):
        angle = 2.0 * np.pi * index / max(1, samples)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        best = min(best, periodic_self_image_gap(positions @ rotation.T, cell))
    return float(best)


def optimize_surface_cell(
    slab_path: str | Path,
    *,
    adsorbate_path: str | Path | None = None,
    min_multiplier: int = 1,
    max_multiplier: int = 30,
    max_atoms: int | None = None,
    min_translation: float = 0.0,
    min_image_gap: float = 3.5,
    max_aspect: float = 2.0,
    translation_parity: tuple[int, int] | None = None,
    orientation_samples: int = 12,
    frozen_bottom_layers: int | None = None,
    output: str | Path | None = None,
    force: bool = False,
    top: int = 20,
) -> dict[str, Any]:
    """Find the smallest useful in-plane supercell under physical constraints.

    ``translation_parity`` describes a collinear AFM phase on the primitive
    surface basis.  For example ``(1, 0)`` means translation by the first
    primitive vector flips the spin, so both rows of an admissible supercell
    matrix must contain an even first coefficient.
    """
    if min_multiplier < 1 or max_multiplier < min_multiplier:
        raise ValueError("invalid multiplier range")
    if top < 1:
        raise ValueError("top must be positive")
    slab = read_structure(slab_path)
    molecule = read_molecule(adsorbate_path) if adsorbate_path else None
    if molecule is not None:
        # A raw phosphonic-acid XYZ is usually stored lying in an arbitrary
        # molecular frame.  Cell clearance must be evaluated in its intended
        # surface-head/body-up frame, otherwise the long molecular axis is
        # incorrectly treated as lateral footprint and the optimizer selects
        # an unnecessarily large slab.
        try:
            from .reactions import _orient_phosphonate, _phosphonate_anchor

            phosphorus, oxygens, carbon, _binding_oxygen = _phosphonate_anchor(molecule)
            molecule = _orient_phosphonate(molecule, phosphorus, oxygens, carbon)
        except ValueError:
            pass
    base_cell = np.asarray(slab.cell, dtype=float)
    candidates: list[dict[str, Any]] = []

    for matrix2 in _hnf_matrices(max_multiplier):
        multiplier = int(round(np.linalg.det(matrix2)))
        if multiplier < min_multiplier:
            continue
        atoms = len(slab) * multiplier
        if max_atoms is not None and atoms > max_atoms:
            continue
        if not _parity_compatible(matrix2, translation_parity):
            continue
        surface_cell = matrix2 @ base_cell[:2]
        lengths = [float(np.linalg.norm(v)) for v in surface_cell]
        shortest = min(_translation_lengths(surface_cell))
        aspect = max(lengths) / min(lengths)
        if shortest + 1e-8 < min_translation or aspect > max_aspect:
            continue
        cell = base_cell.copy()
        cell[:2] = surface_cell
        gap = _rotated_gap(molecule, cell, orientation_samples) if molecule is not None else shortest
        if molecule is not None and gap + 1e-8 < min_image_gap:
            continue
        dot = float(np.dot(surface_cell[0], surface_cell[1]) / (lengths[0] * lengths[1]))
        angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
        score = (
            atoms,
            round(aspect, 12),
            round(abs(angle - 90.0), 12),
            -round(gap, 12),
        )
        candidates.append(
            {
                "matrix": matrix2.tolist(),
                "multiplier": multiplier,
                "atoms": atoms,
                "lengths_a": [round(value, 6) for value in lengths],
                "angle_deg": round(angle, 6),
                "aspect": round(aspect, 6),
                "shortest_translation_a": round(shortest, 6),
                "adsorbate_image_gap_a": round(gap, 6) if molecule is not None else None,
                "afm_parity_compatible": True,
                "_score": score,
            }
        )

    candidates.sort(key=lambda row: row["_score"])
    if not candidates:
        raise ValueError("no surface cell satisfies the atom, aspect, clearance, and magnetic constraints")
    for row in candidates:
        row.pop("_score", None)
    best = candidates[0]

    written = None
    if output is not None:
        destination = Path(output)
        if destination.exists() and not force:
            raise SafetyError(f"output already exists: {destination}; pass --force to replace it")
        from ase.build import make_supercell

        matrix3 = np.eye(3, dtype=int)
        matrix3[:2, :2] = np.asarray(best["matrix"], dtype=int)
        built = make_supercell(slab, matrix3)
        built.pbc = (True, True, False)
        if frozen_bottom_layers is not None:
            built = freeze_bottom_layers(built, frozen_bottom_layers)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ase()["write"](destination, built, format="vasp", direct=True, sort=False)
        written = str(destination.resolve())

    return {
        "source": str(Path(slab_path).resolve()),
        "adsorbate": str(Path(adsorbate_path).resolve()) if adsorbate_path else None,
        "constraints": {
            "min_multiplier": min_multiplier,
            "max_multiplier": max_multiplier,
            "max_atoms": max_atoms,
            "min_translation_a": min_translation,
            "min_image_gap_a": min_image_gap,
            "max_aspect": max_aspect,
            "translation_parity": list(translation_parity) if translation_parity else None,
            "orientation_samples": orientation_samples,
        },
        "best": best,
        "candidates": candidates[:top],
        "candidate_count": len(candidates),
        "written": written,
    }
