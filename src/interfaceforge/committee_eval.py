"""Backend-agnostic committee evaluation on canonical, frame-identified test data.

Every backend writes the same prediction file per committee member::

    predictions.npz: frame_ids (str[n]), natoms (int[n]), energy (float[n], eV),
                     forces (float[sum(natoms), 3], eV/A, atom order of the frame)

and this module aligns them to the canonical reference frames *by frame_id*,
then reports per-member errors, the committee mean, and committee
disagreement (spread). Nothing here calls the spread an uncertainty: without a
separate calibration study it is only a measure of how much the members
disagree.

Energy normalisation: every energy error is ``(E_pred - E_DFT) / N_atoms`` in
meV/atom using total energies against the same ``REF_energy`` labels every
backend was trained on; no per-backend or per-system reference shift is
applied (the ``centered`` column removes each system's mean offset and is
reported separately).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SafetyError
from .state import sha256_file, utc_now

SPREAD_NOTE = (
    "Committee spread is reported as member disagreement (population standard deviation "
    "across members). It is NOT a calibrated uncertainty; no calibration procedure has "
    "been validated for these committees."
)
ENERGY_NOTE = (
    "Energy error = (E_pred - E_DFT) / N_atoms in meV/atom on total energies against the "
    "canonical REF_energy labels (ASE energy(sigma->0)); no reference shift. "
    "'centered' removes each system's mean energy offset."
)
FORCE_NOTE = (
    "Force errors use raw predicted forces (constraints never applied) against raw DFT "
    "forces, per Cartesian component, all atoms; '_mobile' restricts to move_mask == 1."
)


def read_reference(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical reference frames (extxyz) with identity and mobility."""

    from ase.io import iread

    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, atoms in enumerate(iread(str(path), index=":")):
        frame_id = str(atoms.info.get("frame_id") or f"{Path(path).name}:{index}")
        if frame_id in seen:
            raise SafetyError(f"Duplicate reference frame_id {frame_id} in {path}")
        seen.add(frame_id)
        mask = np.ones(len(atoms), dtype=bool)
        for constraint in atoms.constraints:
            try:
                mask[np.asarray(constraint.get_indices(), dtype=int)] = False
            except (AttributeError, TypeError, ValueError):
                continue
        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            raise SafetyError(f"Reference frame {frame_id} lacks REF_energy/REF_forces")
        frames.append(
            {
                "frame_id": frame_id,
                "system": str(atoms.info.get("IF_leaf") or atoms.info.get("source_run") or "unknown"),
                "natoms": len(atoms),
                "energy": float(atoms.info["REF_energy"]),
                "forces": np.asarray(atoms.arrays["REF_forces"], dtype=np.float64),
                "mobile": mask,
                "stage": atoms.info.get("IF_stage"),
                "temperature_k": atoms.info.get("IF_temperature_k"),
                "case": atoms.info.get("IF_case"),
                "ligand": atoms.info.get("IF_ligand"),
                "coverage_pct": atoms.info.get("IF_coverage_pct"),
            }
        )
    if not frames:
        raise SafetyError(f"No reference frames in {path}")
    return frames


def load_predictions(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise SafetyError(f"Missing prediction file: {source}")
    with np.load(source, allow_pickle=False) as data:
        missing = [key for key in ("frame_ids", "natoms", "energy", "forces") if key not in data]
        if missing:
            raise SafetyError(f"{source} lacks {missing}")
        frame_ids = [str(value) for value in data["frame_ids"]]
        natoms = np.asarray(data["natoms"], dtype=int)
        energy = np.asarray(data["energy"], dtype=np.float64)
        forces = np.asarray(data["forces"], dtype=np.float64)
    if not (len(frame_ids) == len(natoms) == len(energy)):
        raise SafetyError(f"{source}: inconsistent prediction lengths")
    if forces.shape != (int(natoms.sum()), 3):
        raise SafetyError(f"{source}: forces shape {forces.shape} != ({int(natoms.sum())}, 3)")
    if len(set(frame_ids)) != len(frame_ids):
        raise SafetyError(f"{source}: duplicate frame_ids")
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    by_id = {
        frame_id: (float(energy[index]), forces[offsets[index] : offsets[index + 1]], int(natoms[index]))
        for index, frame_id in enumerate(frame_ids)
    }
    return {"path": str(source), "sha256": sha256_file(source), "by_id": by_id}


def _align(
    reference: Sequence[dict[str, Any]], prediction: dict[str, Any], label: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    by_id = prediction["by_id"]
    wanted = [frame["frame_id"] for frame in reference]
    missing = [frame_id for frame_id in wanted if frame_id not in by_id]
    if missing:
        raise SafetyError(f"{label}: {len(missing)} reference frame(s) missing from predictions, e.g. {missing[:3]}")
    extra = sorted(set(by_id) - set(wanted))
    if extra:
        raise SafetyError(f"{label}: predictions contain {len(extra)} frame(s) not in the reference, e.g. {extra[:3]}")
    energies = np.empty(len(reference))
    forces: list[np.ndarray] = []
    for index, frame in enumerate(reference):
        energy, frame_forces, natoms = by_id[frame["frame_id"]]
        if natoms != frame["natoms"] or frame_forces.shape != frame["forces"].shape:
            raise SafetyError(f"{label}: atom count mismatch for {frame['frame_id']}")
        if not math.isfinite(energy) or not np.isfinite(frame_forces).all():
            raise SafetyError(f"{label}: non-finite prediction for {frame['frame_id']}")
        energies[index] = energy
        forces.append(frame_forces)
    return energies, forces


def _metrics(
    reference: Sequence[dict[str, Any]], energies: np.ndarray, forces: Sequence[np.ndarray]
) -> dict[str, float]:
    natoms = np.asarray([frame["natoms"] for frame in reference], dtype=float)
    ref_energy = np.asarray([frame["energy"] for frame in reference])
    energy_error = (energies - ref_energy) / natoms
    systems: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(reference):
        systems[frame["system"]].append(index)
    centered = np.concatenate([energy_error[idx] - energy_error[idx].mean() for idx in systems.values()])
    force_error = np.concatenate([pred - frame["forces"] for pred, frame in zip(forces, reference, strict=True)])
    mobile = np.concatenate([frame["mobile"] for frame in reference])
    mobile_error = force_error[mobile]
    result = {
        "frames": float(len(reference)),
        "atoms": float(natoms.sum()),
        "energy_mae_mev_per_atom": float(np.mean(np.abs(energy_error)) * 1000.0),
        "energy_rmse_mev_per_atom": float(np.sqrt(np.mean(energy_error**2)) * 1000.0),
        "energy_centered_rmse_mev_per_atom": float(np.sqrt(np.mean(centered**2)) * 1000.0),
        "energy_mean_error_mev_per_atom": float(np.mean(energy_error) * 1000.0),
        "force_mae_mev_per_angstrom": float(np.mean(np.abs(force_error)) * 1000.0),
        "force_rmse_mev_per_angstrom": float(np.sqrt(np.mean(force_error**2)) * 1000.0),
        "force_mae_mobile_mev_per_angstrom": float(np.mean(np.abs(mobile_error)) * 1000.0)
        if mobile_error.size
        else math.nan,
        "force_rmse_mobile_mev_per_angstrom": float(np.sqrt(np.mean(mobile_error**2)) * 1000.0)
        if mobile_error.size
        else math.nan,
    }
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_committee(
    reference_path: str | Path,
    members: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    backend: str,
) -> dict[str, Any]:
    """Evaluate committee members on identical canonical frames.

    ``members``: ``[{"label": "model_000", "seed": 11, "predictions": path}, ...]``.
    """

    if not members:
        raise SafetyError("Committee evaluation needs at least one member")
    labels = [str(member["label"]) for member in members]
    if len(set(labels)) != len(labels):
        raise SafetyError(f"Duplicate committee member labels: {labels}")
    reference = read_reference(reference_path)
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    member_energy: list[np.ndarray] = []
    member_forces: list[list[np.ndarray]] = []
    provenance: list[dict[str, Any]] = []
    for member in members:
        prediction = load_predictions(member["predictions"])
        energies, forces = _align(reference, prediction, str(member["label"]))
        member_energy.append(energies)
        member_forces.append(forces)
        provenance.append(
            {
                "label": member["label"],
                "seed": member.get("seed"),
                "predictions": prediction["path"],
                "sha256": prediction["sha256"],
            }
        )

    natoms = np.asarray([frame["natoms"] for frame in reference], dtype=float)
    ref_energy = np.asarray([frame["energy"] for frame in reference])
    energy_matrix = np.vstack(member_energy)  # [members, frames]
    mean_energy = energy_matrix.mean(axis=0)
    energy_spread = energy_matrix.std(axis=0) / natoms  # eV/atom
    mean_forces = [np.mean([member[index] for member in member_forces], axis=0) for index in range(len(reference))]
    force_std = [np.std([member[index] for member in member_forces], axis=0) for index in range(len(reference))]
    disagreement = [np.linalg.norm(values, axis=1) for values in force_std]  # per atom, eV/A

    per_model: list[dict[str, Any]] = []
    for member, energies, forces in zip(members, member_energy, member_forces, strict=True):
        per_model.append(
            {
                "backend": backend,
                "model": member["label"],
                "seed": member.get("seed"),
                **_metrics(reference, energies, forces),
            }
        )
    per_model.append(
        {
            "backend": backend,
            "model": "ensemble_mean",
            "seed": "committee",
            **_metrics(reference, mean_energy, mean_forces),
        }
    )

    per_frame: list[dict[str, Any]] = []
    for index, frame in enumerate(reference):
        row: dict[str, Any] = {
            "frame_id": frame["frame_id"],
            "system": frame["system"],
            "stage": frame["stage"],
            "temperature_k": frame["temperature_k"],
            "case": frame["case"],
            "ligand": frame["ligand"],
            "coverage_pct": frame["coverage_pct"],
            "natoms": frame["natoms"],
            "ref_energy_per_atom_ev": ref_energy[index] / natoms[index],
        }
        for label, energies in zip(labels, member_energy, strict=True):
            row[f"{label}_energy_per_atom_ev"] = energies[index] / natoms[index]
        row["ensemble_energy_per_atom_ev"] = mean_energy[index] / natoms[index]
        row["ensemble_energy_error_mev_per_atom"] = (mean_energy[index] - ref_energy[index]) / natoms[index] * 1000.0
        row["energy_spread_mev_per_atom"] = energy_spread[index] * 1000.0
        for label, forces in zip(labels, member_forces, strict=True):
            row[f"{label}_force_rmse_mev_per_angstrom"] = float(
                np.sqrt(np.mean((forces[index] - frame["forces"]) ** 2)) * 1000.0
            )
        row["ensemble_force_rmse_mev_per_angstrom"] = float(
            np.sqrt(np.mean((mean_forces[index] - frame["forces"]) ** 2)) * 1000.0
        )
        row["force_disagreement_mean_mev_per_angstrom"] = float(disagreement[index].mean() * 1000.0)
        row["force_disagreement_max_mev_per_angstrom"] = float(disagreement[index].max() * 1000.0)
        per_frame.append(row)

    per_system: list[dict[str, Any]] = []
    systems: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(reference):
        systems[frame["system"]].append(index)
    for system, indices in sorted(systems.items()):
        subset = [reference[index] for index in indices]
        for label, energies, forces in zip(labels, member_energy, member_forces, strict=True):
            per_system.append(
                {
                    "backend": backend,
                    "system": system,
                    "model": label,
                    **_metrics(subset, energies[indices], [forces[index] for index in indices]),
                }
            )
        per_system.append(
            {
                "backend": backend,
                "system": system,
                "model": "ensemble_mean",
                **_metrics(subset, mean_energy[indices], [mean_forces[index] for index in indices]),
                "energy_spread_mean_mev_per_atom": float(energy_spread[indices].mean() * 1000.0),
                "force_disagreement_mean_mev_per_angstrom": float(
                    np.mean(np.concatenate([disagreement[index] for index in indices])) * 1000.0
                ),
            }
        )

    ensemble_error = np.abs(mean_energy - ref_energy) / natoms
    frame_force_error = np.asarray([row["ensemble_force_rmse_mev_per_angstrom"] for row in per_frame])
    frame_disagreement = np.asarray([row["force_disagreement_mean_mev_per_angstrom"] for row in per_frame])

    def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
        if len(first) < 3 or np.std(first) == 0 or np.std(second) == 0:
            return None
        return float(np.corrcoef(first, second)[0, 1])

    np.savez_compressed(
        out / "predictions_aligned.npz",
        frame_ids=np.asarray([frame["frame_id"] for frame in reference]),
        natoms=natoms.astype(int),
        ref_energy=ref_energy,
        member_labels=np.asarray(labels),
        member_energy=energy_matrix,
        ref_forces=np.concatenate([frame["forces"] for frame in reference]),
        member_forces=np.stack([np.concatenate(forces) for forces in member_forces]),
        mobile=np.concatenate([frame["mobile"] for frame in reference]),
    )
    _write_csv(out / "per_frame.csv", per_frame)
    _write_csv(out / "per_model.csv", per_model)
    _write_csv(out / "per_system.csv", per_system)
    summary = {
        "schema_version": 1,
        "artifact_type": "committee_evaluation",
        "created_at": utc_now(),
        "backend": backend,
        "reference": {
            "path": str(Path(reference_path).resolve()),
            "sha256": sha256_file(reference_path),
            "frames": len(reference),
            "systems": len(systems),
        },
        "members": provenance,
        "per_model": per_model,
        "ensemble": next(row for row in per_model if row["model"] == "ensemble_mean"),
        "disagreement": {
            "energy_spread_mean_mev_per_atom": float(energy_spread.mean() * 1000.0),
            "force_disagreement_mean_mev_per_angstrom": float(frame_disagreement.mean()),
            "diagnostic_pearson_energy_spread_vs_error": _pearson(energy_spread, ensemble_error),
            "diagnostic_pearson_force_disagreement_vs_error": _pearson(frame_disagreement, frame_force_error),
            "calibrated": False,
        },
        "notes": {"spread": SPREAD_NOTE, "energy": ENERGY_NOTE, "forces": FORCE_NOTE, "stress": "not evaluated"},
        "outputs": {
            name: str(out / name)
            for name in ("per_frame.csv", "per_model.csv", "per_system.csv", "predictions_aligned.npz")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def write_predictions(
    path: str | Path,
    *,
    frame_ids: Sequence[str],
    natoms: Sequence[int],
    energy: Sequence[float],
    forces: Sequence[np.ndarray],
) -> Path:
    """Write the shared prediction format (used by tests and local evaluators)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        frame_ids=np.asarray(list(frame_ids)),
        natoms=np.asarray(list(natoms), dtype=int),
        energy=np.asarray(list(energy), dtype=np.float64),
        forces=np.concatenate([np.asarray(value, dtype=np.float64) for value in forces])
        if forces
        else np.zeros((0, 3)),
    )
    temporary.replace(target)
    return target
