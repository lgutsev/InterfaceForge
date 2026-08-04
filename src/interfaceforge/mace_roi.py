"""Region- and thermodynamic-cycle-aware MACE training support.

MACE-ROI is deliberately implemented as a derived dataset and a narrow runtime
adapter.  Canonical DFT labels remain immutable; the derived extxyz files add
per-atom force weights and optional, split-local thermodynamic-cycle metadata.
The runtime adapter then reuses MACE's own model construction, restart,
checkpointing, and evaluation code while replacing only the loss and, when
needed, the batch sampler.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import SPLITS, Campaign
from .errors import ConfigurationError, DependencyError, InterfaceForgeError, SafetyError
from .state import StateStore, sha256_file

ROI_WEIGHT_KEY = "IF_roi_weight"
ROI_MASK_KEY = "IF_roi_mask"
CYCLE_ID_KEY = "IF_cycle_id"
CYCLE_COEFFICIENT_KEY = "IF_cycle_coefficient"
CYCLE_SCALE_KEY = "IF_cycle_scale_ev"
CYCLE_SIZE_KEY = "IF_cycle_size"


def _ase_modules() -> tuple[Any, Any, Any]:
    try:
        from ase.io import read, write
        from ase.neighborlist import neighbor_list
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required for MACE-ROI dataset preparation. Install InterfaceForge "
            "with: pip install 'interfaceforge[mace-roi]'"
        ) from exc
    return read, write, neighbor_list


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _prepare_derived_output(path: Path, source: Path, *, force: bool) -> None:
    if path == source or path in source.parents or source in path.parents:
        raise SafetyError(
            "MACE-ROI output must be separate from, not replace or contain, the canonical dataset"
        )
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SafetyError(f"MACE-ROI output directory is not empty: {path}")
        if path == Path("/") or len(path.parts) < 3:
            raise SafetyError(f"Refusing broad destructive output replacement: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _configured_ranges(atoms: Any, settings: Mapping[str, Any]) -> Sequence[Any] | None:
    configured = settings.get("component_ranges", [])
    if isinstance(configured, Mapping):
        source_run = str(atoms.info.get("source_run", ""))
        matches = [
            (str(pattern), ranges)
            for pattern, ranges in configured.items()
            if fnmatch.fnmatchcase(source_run, str(pattern))
        ]
        if len(matches) > 1:
            patterns = ", ".join(pattern for pattern, _ranges in matches)
            raise SafetyError(
                f"source_run {source_run!r} matches multiple component-range patterns: "
                f"{patterns}"
            )
        # A source-specific mapping intentionally leaves ordinary bulk and
        # isolated-surface frames unweighted when they match no pattern.
        return matches[0][1] if matches else None
    return configured


def _component_ids(atoms: Any, settings: Mapping[str, Any]) -> np.ndarray:
    component_key = str(settings.get("component_key", "IF_component"))
    if component_key in atoms.arrays:
        raw_components = np.asarray(atoms.arrays[component_key])
        if raw_components.shape not in {(len(atoms),), (len(atoms), 1)}:
            raise SafetyError(
                f"{component_key} has shape {raw_components.shape}; expected ({len(atoms)},)"
            )
        try:
            numeric_components = np.asarray(raw_components, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise SafetyError(f"{component_key} must contain integer component IDs") from exc
        if (
            not np.isfinite(numeric_components).all()
            or not np.array_equal(numeric_components, np.floor(numeric_components))
            or np.any(numeric_components < 0)
        ):
            raise SafetyError(f"{component_key} must contain non-negative integer component IDs")
        return numeric_components.astype(int)

    ranges = _configured_ranges(atoms, settings)
    if ranges is None:
        return np.zeros(len(atoms), dtype=int)
    if not ranges:
        if float(settings.get("interface_multiplier", 4.0)) == 1.0:
            return np.zeros(len(atoms), dtype=int)
        raise SafetyError(
            f"No {component_key} array is present and models.mace.roi.component_ranges is empty"
        )
    components = np.full(len(atoms), -1, dtype=int)
    for component, bounds in enumerate(ranges):
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
            raise ConfigurationError(
                "models.mace.roi.component_ranges entries must be [start, stop] half-open ranges"
            )
        start, stop = (int(bounds[0]), int(bounds[1]))
        if start < 0 or stop <= start or stop > len(atoms):
            raise SafetyError(
                f"Invalid component range [{start}, {stop}) for a {len(atoms)}-atom frame"
            )
        if np.any(components[start:stop] != -1):
            raise SafetyError(f"Overlapping component range [{start}, {stop})")
        components[start:stop] = component
    missing = np.flatnonzero(components < 0)
    if missing.size:
        preview = ", ".join(map(str, missing[:8]))
        raise SafetyError(f"Component ranges do not cover atom indices: {preview}")
    if len(ranges) < 2:
        raise SafetyError("MACE-ROI needs at least two component ranges")
    return components


def compute_roi_weights(
    atoms: Any,
    components: np.ndarray,
    *,
    cutoff: float,
    interface_multiplier: float,
    shell_depth: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-one force weights and an interface-region mask.

    An atom enters the initial region of interest when it has a neighbor from a
    different immutable component. Optional shell expansion adds neighboring
    layers without assuming that the interface normal is the Cartesian z axis.
    """

    _, _, neighbor_list = _ase_modules()
    if components.shape != (len(atoms),):
        raise ValueError("components must contain one integer per atom")
    if (
        not math.isfinite(cutoff)
        or not math.isfinite(interface_multiplier)
        or cutoff <= 0
        or interface_multiplier < 1
        or shell_depth < 0
    ):
        raise ValueError("cutoff must be positive, multiplier >= 1, and shell_depth >= 0")

    senders, receivers = neighbor_list("ij", atoms, cutoff)
    senders = np.asarray(senders, dtype=int)
    receivers = np.asarray(receivers, dtype=int)
    mask = np.zeros(len(atoms), dtype=bool)
    if senders.size:
        cross = components[senders] != components[receivers]
        mask[senders[cross]] = True
        mask[receivers[cross]] = True
        for _ in range(shell_depth):
            expanded = mask.copy()
            touching = mask[senders] | mask[receivers]
            expanded[senders[touching]] = True
            expanded[receivers[touching]] = True
            mask = expanded

    weights = np.ones(len(atoms), dtype=np.float64)
    weights[mask] = float(interface_multiplier)
    weights /= float(weights.mean())
    return weights, mask.astype(np.int8)


def _read_cycle_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise SafetyError(f"MACE-ROI cycle manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "source_run", "source_frame", "cycle_id", "coefficient"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ConfigurationError(f"Cycle manifest is missing columns: {missing}")
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                split = str(row["split"]).strip().lower()
                source_run = str(row["source_run"] or "").strip()
                source_frame = int(row["source_frame"])
                cycle_id = str(row["cycle_id"] or "").strip()
                coefficient = float(row["coefficient"])
                scale = float(row.get("scale_ev") or 1.0)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Invalid numeric value in cycle manifest line {line_number}"
                ) from exc
            if split not in SPLITS:
                raise ConfigurationError(f"Unknown split {split!r} in cycle manifest line {line_number}")
            if not source_run or not cycle_id:
                raise ConfigurationError(
                    f"source_run and cycle_id are required in cycle manifest line {line_number}"
                )
            if source_frame < 0:
                raise ConfigurationError(
                    f"source_frame cannot be negative in cycle manifest line {line_number}"
                )
            if not math.isfinite(coefficient) or coefficient == 0:
                raise ConfigurationError(
                    f"Cycle coefficient must be finite and nonzero in line {line_number}"
                )
            if not math.isfinite(scale) or scale <= 0:
                raise ConfigurationError(
                    f"scale_ev must be finite and positive in cycle manifest line {line_number}"
                )
            key = (split, source_run, source_frame)
            if key in seen:
                raise ConfigurationError(f"Duplicate cycle member reference: {key}")
            seen.add(key)
            rows.append(
                {
                    "split": split,
                    "source_run": source_run,
                    "source_frame": source_frame,
                    "cycle_id": cycle_id,
                    "coefficient": coefficient,
                    "scale_ev": scale,
                }
            )
    return rows


def _composition(atoms: Any) -> Counter[str]:
    return Counter(atoms.get_chemical_symbols())


def _validate_cycle_groups(
    rows: Sequence[Mapping[str, Any]],
    frames: Mapping[tuple[str, str, int], Any],
) -> dict[tuple[str, str], int]:
    by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    split_by_name: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = (str(row["split"]), str(row["cycle_id"]))
        by_group[group].append(row)
        split_by_name[str(row["cycle_id"])].add(str(row["split"]))

    leaking = sorted(name for name, splits in split_by_name.items() if len(splits) > 1)
    if leaking:
        raise SafetyError(
            "Thermodynamic cycles may not cross train/valid/test splits; repeated cycle IDs: "
            + ", ".join(leaking)
        )

    for group, members in by_group.items():
        if len(members) < 2:
            raise SafetyError(f"Cycle {group[1]!r} in {group[0]} has fewer than two members")
        scales = {float(member["scale_ev"]) for member in members}
        if len(scales) != 1:
            raise SafetyError(f"Cycle {group[1]!r} has inconsistent scale_ev values")
        balance: defaultdict[str, float] = defaultdict(float)
        for member in members:
            frame_key = (
                str(member["split"]),
                str(member["source_run"]),
                int(member["source_frame"]),
            )
            if frame_key not in frames:
                raise SafetyError(f"Cycle member does not match an extxyz frame: {frame_key}")
            for symbol, count in _composition(frames[frame_key]).items():
                balance[symbol] += float(member["coefficient"]) * count
        unbalanced = {symbol: value for symbol, value in balance.items() if abs(value) > 1e-8}
        if unbalanced:
            raise SafetyError(
                f"Cycle {group[1]!r} is not composition-conserving: {unbalanced}"
            )

    return {group: index for index, group in enumerate(sorted(by_group))}


def prepare_mace_roi_dataset(
    campaign: Campaign,
    *,
    source_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cycle_manifest: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create reviewable MACE-ROI extxyz files without mutating canonical data."""

    mace = dict(campaign.models.get("mace", {}))
    settings = dict(mace.get("roi", {}))
    if not settings.get("enabled", False):
        raise SafetyError("models.mace.roi.enabled is false")

    source = (
        Path(source_root).expanduser().resolve()
        if source_root
        else _resolve(campaign.root, settings.get("source_dir", "datasets/canonical"))
    )
    output = (
        Path(output_root).expanduser().resolve()
        if output_root
        else _resolve(campaign.root, settings.get("output_dir", "datasets/mace_roi"))
    )
    configured_cycles = cycle_manifest if cycle_manifest is not None else settings.get("cycle_manifest")
    cycles_path = _resolve(campaign.root, configured_cycles) if configured_cycles else None
    _prepare_derived_output(output, source, force=force)
    read, write, _ = _ase_modules()

    split_frames: dict[str, list[Any]] = {}
    frame_lookup: dict[tuple[str, str, int], Any] = {}
    for split in SPLITS:
        path = source / f"{split}.extxyz"
        if not path.is_file():
            raise SafetyError(f"Missing canonical MACE split: {path}")
        loaded = read(str(path), index=":")
        frames = loaded if isinstance(loaded, list) else [loaded]
        if not frames:
            raise SafetyError(f"MACE split contains no frames: {path}")
        split_frames[split] = frames
        for atoms in frames:
            try:
                key = (split, str(atoms.info["source_run"]), int(atoms.info["source_frame"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise SafetyError(
                    f"Every MACE-ROI frame needs source_run and source_frame metadata: {path}"
                ) from exc
            if key in frame_lookup:
                raise SafetyError(f"Duplicate frame identity in canonical data: {key}")
            frame_lookup[key] = atoms

    cycle_rows = _read_cycle_rows(cycles_path)
    cycle_indices = _validate_cycle_groups(cycle_rows, frame_lookup)
    member_by_frame = {
        (str(row["split"]), str(row["source_run"]), int(row["source_frame"])): row
        for row in cycle_rows
    }
    group_sizes = Counter((str(row["split"]), str(row["cycle_id"])) for row in cycle_rows)

    cutoff = float(settings.get("cutoff", 3.5))
    multiplier = float(settings.get("interface_multiplier", 4.0))
    shell_depth = int(settings.get("shell_depth", 0))
    summary: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}
    for split, frames in split_frames.items():
        destination = output / f"{split}.extxyz"
        roi_atoms = 0
        total_atoms = 0
        cycle_members = 0
        for atoms in frames:
            components = _component_ids(atoms, settings)
            weights, mask = compute_roi_weights(
                atoms,
                components,
                cutoff=cutoff,
                interface_multiplier=multiplier,
                shell_depth=shell_depth,
            )
            atoms.arrays[ROI_WEIGHT_KEY] = weights
            atoms.arrays[ROI_MASK_KEY] = mask
            key = (split, str(atoms.info["source_run"]), int(atoms.info["source_frame"]))
            member = member_by_frame.get(key)
            if member is None:
                atoms.info[CYCLE_ID_KEY] = -1
                atoms.info[CYCLE_COEFFICIENT_KEY] = 0.0
                atoms.info[CYCLE_SCALE_KEY] = 1.0
                atoms.info[CYCLE_SIZE_KEY] = 0
            else:
                group = (split, str(member["cycle_id"]))
                atoms.info[CYCLE_ID_KEY] = cycle_indices[group]
                atoms.info[CYCLE_COEFFICIENT_KEY] = float(member["coefficient"])
                atoms.info[CYCLE_SCALE_KEY] = float(member["scale_ev"])
                atoms.info[CYCLE_SIZE_KEY] = int(group_sizes[group])
                cycle_members += 1
            write(str(destination), atoms, format="extxyz", append=destination.exists())
            roi_atoms += int(mask.sum())
            total_atoms += len(atoms)
        output_hashes[split] = sha256_file(destination)
        summary[split] = {
            "frames": len(frames),
            "atoms": total_atoms,
            "roi_atoms": roi_atoms,
            "roi_fraction": roi_atoms / total_atoms if total_atoms else 0.0,
            "cycle_members": cycle_members,
            "path": str(destination),
        }

    total_roi_atoms = sum(int(values["roi_atoms"]) for values in summary.values())
    if multiplier > 1 and total_roi_atoms == 0:
        raise SafetyError(
            "MACE-ROI found no cross-component interface atoms. Check component_ranges, "
            "source_run patterns, and cutoff; use interface_multiplier: 1 for cycle-only training."
        )

    payload = {
        "schema_version": 1,
        "method": "mace-roi",
        "source_root": str(source),
        "output_root": str(output),
        "source_hashes": {
            split: sha256_file(source / f"{split}.extxyz") for split in SPLITS
        },
        "output_hashes": output_hashes,
        "roi": {
            "weight_key": ROI_WEIGHT_KEY,
            "mask_key": ROI_MASK_KEY,
            "component_key": settings.get("component_key", "IF_component"),
            "component_ranges": settings.get("component_ranges", []),
            "cutoff_angstrom": cutoff,
            "interface_multiplier": multiplier,
            "shell_depth": shell_depth,
            "normalization": "mean weight equals one in every frame",
        },
        "cycles": {
            "manifest": str(cycles_path) if cycles_path else None,
            "groups": len(cycle_indices),
            "members": len(cycle_rows),
            "id_key": CYCLE_ID_KEY,
            "coefficient_key": CYCLE_COEFFICIENT_KEY,
            "scale_key": CYCLE_SCALE_KEY,
            "size_key": CYCLE_SIZE_KEY,
            "composition_conservation_checked": True,
            "split_isolation_checked": True,
        },
        "splits": summary,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    state = StateStore(campaign.root)
    state.event("mace_roi_prepare", **payload)
    state.artifact("mace_roi_manifest", manifest)
    return payload


def _item_scalar(item: Any, key: str, default: int = -1) -> int:
    try:
        value = item[key]
    except (KeyError, TypeError, IndexError):
        value = getattr(item, key, default)
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


class CycleBatchSampler:
    """Pack complete thermodynamic cycles into indivisible mini-batches."""

    def __init__(
        self,
        dataset: Sequence[Any],
        batch_size: int,
        *,
        cycle_id_key: str = "if_cycle_id",
        shuffle: bool = True,
        seed: int = 2026,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.cycle_id_key = cycle_id_key
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, list[int]] = defaultdict(list)
        singles: list[list[int]] = []
        for index in range(len(dataset)):
            cycle_id = _item_scalar(dataset[index], cycle_id_key)
            if cycle_id >= 0:
                groups[cycle_id].append(index)
            else:
                singles.append([index])
        self.units = [groups[key] for key in sorted(groups)] + singles

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        units = [list(unit) for unit in self.units]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(units)
        batches: list[list[int]] = []
        current: list[int] = []
        for unit in units:
            if current and len(current) + len(unit) > self.batch_size:
                batches.append(current)
                current = []
            if len(unit) > self.batch_size:
                if current:
                    batches.append(current)
                    current = []
                batches.append(unit)
            else:
                current.extend(unit)
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches())


def cycle_mse_numpy(
    reference: Sequence[float],
    predicted: Sequence[float],
    cycle_ids: Sequence[int],
    coefficients: Sequence[float],
    scales: Sequence[float],
) -> float:
    """Reference implementation used by tests and audit calculations."""

    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    ids = np.asarray(cycle_ids, dtype=int)
    coeff = np.asarray(coefficients, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if len({len(ref), len(pred), len(ids), len(coeff), len(scale)}) != 1:
        raise ValueError("Cycle-loss inputs must have equal lengths")
    if not all(np.isfinite(values).all() for values in (ref, pred, coeff, scale)):
        raise ValueError("Cycle-loss inputs must be finite")
    if np.any(scale <= 0):
        raise ValueError("Cycle scales must be positive")
    losses: list[float] = []
    for cycle_id in sorted(set(ids[ids >= 0].tolist())):
        selected = ids == cycle_id
        cycle_scales = scale[selected]
        if not np.allclose(cycle_scales, cycle_scales[0]):
            raise ValueError(f"Cycle {cycle_id} has inconsistent scales")
        residual = float(np.sum(coeff[selected] * (pred[selected] - ref[selected])))
        losses.append((residual / float(cycle_scales[0])) ** 2)
    return float(np.mean(losses)) if losses else 0.0


def _residual_metrics(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    residuals = np.asarray(values, dtype=float).reshape(-1)
    if residuals.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "max_abs": None, "bias": None}
    if not np.isfinite(residuals).all():
        raise SafetyError("Prediction residuals must be finite")
    return {
        "count": int(residuals.size),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_abs": float(np.max(np.abs(residuals))),
        "bias": float(np.mean(residuals)),
    }


def evaluate_mace_roi_predictions(
    source: str | Path,
    output: str | Path,
    *,
    reference_energy_key: str = "REF_energy",
    predicted_energy_key: str = "MACE_energy",
    reference_forces_key: str = "REF_forces",
    predicted_forces_key: str = "MACE_forces",
) -> dict[str, Any]:
    """Measure global, interface-local and cycle residuals from MACE extxyz output."""

    input_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if input_path == output_path:
        raise SafetyError("MACE-ROI evaluation output must not replace the prediction extxyz")
    if not input_path.is_file():
        raise SafetyError(f"MACE-ROI prediction file does not exist: {input_path}")
    read, _, _ = _ase_modules()
    loaded = read(str(input_path), index=":")
    frames = loaded if isinstance(loaded, list) else [loaded]
    if not frames:
        raise SafetyError(f"MACE-ROI prediction file has no frames: {input_path}")

    energy_per_atom_residuals: list[float] = []
    global_force_residuals: list[np.ndarray] = []
    roi_force_residuals: list[np.ndarray] = []
    non_roi_force_residuals: list[np.ndarray] = []
    cycle_members: defaultdict[int, list[dict[str, float | int]]] = defaultdict(list)
    total_atoms = 0
    roi_atoms = 0
    for frame_index, atoms in enumerate(frames):
        if len(atoms) == 0:
            raise SafetyError(f"Frame {frame_index} contains no atoms")
        try:
            reference_energy = float(atoms.info[reference_energy_key])
            predicted_energy = float(atoms.info[predicted_energy_key])
            reference_forces = np.asarray(atoms.arrays[reference_forces_key], dtype=float)
            predicted_forces = np.asarray(atoms.arrays[predicted_forces_key], dtype=float)
            roi_mask = np.asarray(atoms.arrays[ROI_MASK_KEY], dtype=bool).reshape(-1)
        except KeyError as exc:
            raise SafetyError(
                f"Frame {frame_index} is missing required MACE-ROI prediction key: {exc}"
            ) from exc
        if reference_forces.shape != (len(atoms), 3) or predicted_forces.shape != (
            len(atoms),
            3,
        ):
            raise SafetyError(f"Frame {frame_index} does not contain (natoms, 3) force arrays")
        if roi_mask.shape != (len(atoms),):
            raise SafetyError(f"Frame {frame_index} does not contain one ROI mask value per atom")
        if not all(
            np.isfinite(values).all()
            for values in (
                np.asarray([reference_energy, predicted_energy]),
                reference_forces,
                predicted_forces,
            )
        ):
            raise SafetyError(f"Frame {frame_index} contains non-finite labels or predictions")

        force_residual = predicted_forces - reference_forces
        global_force_residuals.append(force_residual)
        roi_force_residuals.append(force_residual[roi_mask])
        non_roi_force_residuals.append(force_residual[~roi_mask])
        energy_per_atom_residuals.append((predicted_energy - reference_energy) / len(atoms))
        total_atoms += len(atoms)
        roi_atoms += int(roi_mask.sum())

        cycle_id = int(atoms.info.get(CYCLE_ID_KEY, -1))
        if cycle_id >= 0:
            try:
                cycle_members[cycle_id].append(
                    {
                        "residual_ev": predicted_energy - reference_energy,
                        "coefficient": float(atoms.info[CYCLE_COEFFICIENT_KEY]),
                        "scale_ev": float(atoms.info[CYCLE_SCALE_KEY]),
                        "size": int(atoms.info[CYCLE_SIZE_KEY]),
                    }
                )
            except KeyError as exc:
                raise SafetyError(
                    f"Frame {frame_index} is missing required cycle metadata: {exc}"
                ) from exc

    cycle_residuals: list[float] = []
    normalized_cycle_residuals: list[float] = []
    for cycle_id, members in sorted(cycle_members.items()):
        expected_sizes = {int(member["size"]) for member in members}
        scales = {float(member["scale_ev"]) for member in members}
        if expected_sizes != {len(members)}:
            raise SafetyError(
                f"Prediction file contains an incomplete cycle {cycle_id}: "
                f"observed {len(members)}, expected {sorted(expected_sizes)}"
            )
        if len(scales) != 1 or next(iter(scales)) <= 0:
            raise SafetyError(f"Prediction file contains invalid scales for cycle {cycle_id}")
        residual = sum(
            float(member["coefficient"]) * float(member["residual_ev"])
            for member in members
        )
        scale = next(iter(scales))
        cycle_residuals.append(residual)
        normalized_cycle_residuals.append(residual / scale)

    global_force = np.concatenate(global_force_residuals, axis=0)
    roi_force = np.concatenate(roi_force_residuals, axis=0)
    non_roi_force = np.concatenate(non_roi_force_residuals, axis=0)
    payload = {
        "schema_version": 1,
        "method": "mace-roi-evaluation",
        "source": str(input_path),
        "output": str(output_path),
        "frames": len(frames),
        "atoms": total_atoms,
        "roi_atoms": roi_atoms,
        "roi_fraction": roi_atoms / total_atoms,
        "keys": {
            "reference_energy": reference_energy_key,
            "predicted_energy": predicted_energy_key,
            "reference_forces": reference_forces_key,
            "predicted_forces": predicted_forces_key,
            "roi_mask": ROI_MASK_KEY,
        },
        "energy_per_atom_ev": _residual_metrics(energy_per_atom_residuals),
        "force_component_ev_a": {
            "global": _residual_metrics(global_force),
            "roi": _residual_metrics(roi_force),
            "non_roi": _residual_metrics(non_roi_force),
        },
        "force_vector_rmse_ev_a": {
            "global": float(np.sqrt(np.mean(np.sum(np.square(global_force), axis=1)))),
            "roi": (
                float(np.sqrt(np.mean(np.sum(np.square(roi_force), axis=1))))
                if roi_force.size
                else None
            ),
            "non_roi": (
                float(np.sqrt(np.mean(np.sum(np.square(non_roi_force), axis=1))))
                if non_roi_force.size
                else None
            ),
        },
        "cycle_residual_ev": _residual_metrics(cycle_residuals),
        "cycle_residual_scaled": _residual_metrics(normalized_cycle_residuals),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def _build_region_cycle_loss(args: argparse.Namespace) -> Any:
    try:
        import torch
        import torch.distributed as dist
        from mace.modules.loss import reduce_loss, weighted_mean_squared_error_energy
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "MACE-ROI training requires mace-torch 0.3.17 and PyTorch. Install with: "
            "pip install 'interfaceforge[mace-roi]'"
        ) from exc

    class RegionCycleLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dtype = torch.get_default_dtype()
            self.register_buffer("energy_weight", torch.tensor(args.energy_weight, dtype=dtype))
            self.register_buffer("forces_weight", torch.tensor(args.forces_weight, dtype=dtype))
            self.register_buffer("cycle_weight", torch.tensor(args.if_cycle_weight, dtype=dtype))

        def forward(self, ref: Any, pred: Mapping[str, Any], ddp: bool | None = None) -> Any:
            energy = weighted_mean_squared_error_energy(ref, pred, ddp)
            counts = ref.ptr[1:] - ref.ptr[:-1]
            graph_weights = torch.repeat_interleave(ref.weight * ref.forces_weight, counts).view(-1, 1)
            roi = ref[args.if_roi_property]
            if roi.ndim == 1:
                roi = roi.unsqueeze(-1)
            if roi.shape[0] != ref["forces"].shape[0]:
                raise RuntimeError("ROI weights must contain one value per atom")
            if not torch.isfinite(roi).all() or torch.any(roi <= 0):
                raise RuntimeError("ROI weights must be finite and positive")
            force_raw = graph_weights * roi * torch.square(ref["forces"] - pred["forces"])
            forces = reduce_loss(force_raw, ddp)

            cycle = pred["energy"].new_zeros(())
            if float(self.cycle_weight) > 0:
                if ddp or (dist.is_initialized() and dist.get_world_size() > 1):
                    raise RuntimeError("Thermodynamic-cycle loss is not yet DDP compatible")
                ids = ref[args.if_cycle_id_property].reshape(-1).to(torch.long)
                coefficients = ref[args.if_cycle_coefficient_property].reshape(-1)
                scales = ref[args.if_cycle_scale_property].reshape(-1)
                sizes = ref[args.if_cycle_size_property].reshape(-1).to(torch.long)
                if not torch.isfinite(coefficients).all() or not torch.isfinite(scales).all():
                    raise RuntimeError("Cycle coefficients and scales must be finite")
                if torch.any(scales <= 0):
                    raise RuntimeError("Cycle scales must be positive")
                losses = []
                for cycle_id in torch.unique(ids[ids >= 0]):
                    selected = ids == cycle_id
                    observed = int(selected.sum().item())
                    expected = int(sizes[selected][0].item())
                    if observed != expected or not torch.all(sizes[selected] == expected):
                        raise RuntimeError(
                            f"Cycle {int(cycle_id)} was split across batches: "
                            f"observed {observed}, expected {expected}"
                        )
                    cycle_scales = scales[selected]
                    if not torch.allclose(cycle_scales, cycle_scales[0]):
                        raise RuntimeError(f"Cycle {int(cycle_id)} has inconsistent scales")
                    residual = torch.sum(
                        coefficients[selected]
                        * (pred["energy"][selected] - ref["energy"][selected])
                    )
                    losses.append(torch.square(residual / cycle_scales[0]))
                if losses:
                    cycle = torch.stack(losses).mean()
            return self.energy_weight * energy + self.forces_weight * forces + self.cycle_weight * cycle

        def __repr__(self) -> str:
            return (
                "RegionCycleLoss("
                f"energy_weight={float(self.energy_weight):.3f}, "
                f"forces_weight={float(self.forces_weight):.3f}, "
                f"cycle_weight={float(self.cycle_weight):.3f})"
            )

    return RegionCycleLoss()


def run_mace_roi_training(argv: Sequence[str] | None = None) -> None:
    """Run MACE's trainer with InterfaceForge's loss and cycle batching hooks."""

    try:
        import mace
        import torch
        from mace import tools as mace_tools
        from mace.cli import run_train as mace_run_train
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "MACE-ROI training requires mace-torch. Install with: "
            "pip install 'interfaceforge[mace-roi]'"
        ) from exc
    version_parts = [int(part) for part in re.findall(r"\d+", str(mace.__version__))[:3]]
    version = tuple((version_parts + [0, 0, 0])[:3])
    if version < (0, 3, 17) or version >= (0, 4, 0):
        raise DependencyError(
            f"MACE-ROI supports mace-torch >=0.3.17,<0.4; found {mace.__version__}"
        )

    parser = mace_tools.build_default_arg_parser()
    parser.add_argument("--if-roi-weight-key", default=ROI_WEIGHT_KEY)
    parser.add_argument("--if-cycle-id-key", default=CYCLE_ID_KEY)
    parser.add_argument("--if-cycle-coefficient-key", default=CYCLE_COEFFICIENT_KEY)
    parser.add_argument("--if-cycle-scale-key", default=CYCLE_SCALE_KEY)
    parser.add_argument("--if-cycle-size-key", default=CYCLE_SIZE_KEY)
    parser.add_argument("--if-cycle-weight", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.if_cycle_weight) or args.if_cycle_weight < 0:
        raise ConfigurationError("--if-cycle-weight must be finite and non-negative")
    if args.if_cycle_weight > 0 and args.distributed:
        raise ConfigurationError(
            "MACE-ROI cycle loss currently requires single-process training; remove --distributed"
        )

    args.if_roi_property = "if_roi_weight"
    args.if_cycle_id_property = "if_cycle_id"
    args.if_cycle_coefficient_property = "if_cycle_coefficient"
    args.if_cycle_scale_property = "if_cycle_scale"
    args.if_cycle_size_property = "if_cycle_size"

    original_update = mace_run_train.update_keyspec_from_kwargs
    original_loss_factory = mace_run_train.get_loss_fn
    original_loader: Any | None = None
    atomic_data_class = mace_run_train.data.AtomicData
    original_from_config_descriptor = atomic_data_class.__dict__["from_config"]
    original_from_config = atomic_data_class.from_config

    def update_keyspec(specification: Any, values: Mapping[str, Any]) -> Any:
        result = original_update(specification, values)
        specification.update(
            arrays_keys={args.if_roi_property: args.if_roi_weight_key},
            info_keys={
                args.if_cycle_id_property: args.if_cycle_id_key,
                args.if_cycle_coefficient_property: args.if_cycle_coefficient_key,
                args.if_cycle_scale_property: args.if_cycle_scale_key,
                args.if_cycle_size_property: args.if_cycle_size_key,
            },
        )
        return result

    mace_run_train.update_keyspec_from_kwargs = update_keyspec
    mace_run_train.get_loss_fn = lambda *_unused, **_kwargs: _build_region_cycle_loss(args)

    def from_config_with_roi(cls: Any, config: Any, *positional: Any, **keywords: Any) -> Any:
        atomic_data = original_from_config(config, *positional, **keywords)
        # MACE 0.3.x patch releases differ in whether unknown Configuration
        # properties are copied into AtomicData. Copy our numeric properties
        # explicitly so the pinned adapter does not depend on that detail.
        for property_name in (
            args.if_roi_property,
            args.if_cycle_id_property,
            args.if_cycle_coefficient_property,
            args.if_cycle_scale_property,
            args.if_cycle_size_property,
        ):
            value = config.properties.get(property_name)
            if value is None:
                continue
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype=torch.get_default_dtype())
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            atomic_data[property_name] = tensor
        return atomic_data

    atomic_data_class.from_config = classmethod(from_config_with_roi)

    if args.if_cycle_weight > 0:
        original_loader = mace_run_train.torch_geometric.dataloader.DataLoader

        class CycleAwareDataLoader(original_loader):
            def __init__(self, dataset: Sequence[Any], *loader_args: Any, **loader_kwargs: Any) -> None:
                batch_size = int(loader_kwargs.pop("batch_size", 1))
                shuffle = bool(loader_kwargs.pop("shuffle", False))
                sampler = loader_kwargs.pop("sampler", None)
                drop_last = bool(loader_kwargs.pop("drop_last", False))
                if sampler is not None:
                    raise ConfigurationError(
                        "Cycle-aware MACE-ROI batching cannot be combined with another sampler"
                    )
                del drop_last  # complete cycles are never discarded
                batch_sampler = CycleBatchSampler(
                    dataset,
                    batch_size,
                    cycle_id_key=args.if_cycle_id_property,
                    shuffle=shuffle,
                    seed=args.seed,
                )
                super().__init__(
                    dataset,
                    *loader_args,
                    batch_sampler=batch_sampler,
                    **loader_kwargs,
                )

        mace_run_train.torch_geometric.dataloader.DataLoader = CycleAwareDataLoader

    try:
        mace_run_train.run(args)
    finally:
        mace_run_train.update_keyspec_from_kwargs = original_update
        mace_run_train.get_loss_fn = original_loss_factory
        atomic_data_class.from_config = original_from_config_descriptor
        if original_loader is not None:
            mace_run_train.torch_geometric.dataloader.DataLoader = original_loader


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_mace_roi_training(argv)
    except InterfaceForgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
