"""Canonical VASP trajectory collection for MACE and DeePMD."""

from __future__ import annotations

import csv
import fnmatch
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import SPLITS, Campaign
from .errors import DependencyError, SafetyError
from .state import StateStore, sha256_file


@dataclass(frozen=True)
class SourceTrajectory:
    path: Path
    run_id: str
    category: str
    group: str


@dataclass
class Frame:
    source_index: int
    atoms: Any
    energy: float
    forces: np.ndarray
    move_mask: np.ndarray
    virial: np.ndarray | None


@dataclass
class CollectionRecord:
    source: SourceTrajectory
    assignment: str | None = None
    frames_seen: int = 0
    frames_retained: int = 0
    split_counts: dict[str, int] = field(default_factory=lambda: {split: 0 for split in SPLITS})
    guard_count: int = 0
    warnings: list[str] = field(default_factory=list)


def _ase_io() -> tuple[Any, Any]:
    try:
        from ase.io import iread, write
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required for VASP collection. Install InterfaceForge with: "
            "pip install 'interfaceforge[vasp]'"
        ) from exc
    return iread, write


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return re.sub(r"_+", "_", cleaned).strip("_") or "unnamed"


def category_key(name: str) -> str:
    """Remove a trailing temperature label to group related trajectories."""

    return re.sub(r"[-_](?:\d{2,4})(?:K)?$", "", name, flags=re.I)


def normalize_ratios(values: Sequence[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("Need train, valid, and test ratios")
    ratios = [float(item) for item in values]
    if any(item < 0 for item in ratios) or sum(ratios) <= 0:
        raise ValueError("Ratios must be non-negative and have a positive sum")
    total = sum(ratios)
    return {split: value / total for split, value in zip(SPLITS, ratios, strict=True)}


def _deduplicate_run_ids(discovered: list[SourceTrajectory]) -> list[SourceTrajectory]:
    """Disambiguate run_ids that collide after safe_name() sanitization.

    Two distinct trajectory directories (e.g. "system-A" and "system_A")
    can sanitize to the same run_id. Left alone, _write_deepmd_system()'s
    mkdir(exist_ok=False) would crash with a confusing FileExistsError, or
    worse, mix two unrelated systems' frames under one directory. The first
    occurrence of a run_id keeps it unchanged; later collisions get a
    deterministic suffix.
    """

    seen: dict[str, int] = {}
    deduplicated: list[SourceTrajectory] = []
    for source in discovered:
        count = seen.get(source.run_id, 0)
        seen[source.run_id] = count + 1
        if count == 0:
            deduplicated.append(source)
        else:
            deduplicated.append(replace(source, run_id=f"{source.run_id}__dup{count + 1}"))
    return deduplicated


def discover_outcars(
    root: str | Path,
    *,
    patterns: Sequence[str] = ("OUTCAR",),
    exclude: Sequence[str] = (
        "*/.interfaceforge/*",
        "*/archive/*",
        "*/restart_archive_*/*",
        "*/refit_archive_*/*",
        "*/stability_archive_*/*",
    ),
) -> list[SourceTrajectory]:
    """Discover generic VASP trajectories without assuming Step2_* names."""

    source_root = Path(root).resolve()
    discovered: list[SourceTrajectory] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            continue
        relative = path.relative_to(source_root).as_posix()
        wrapped = f"/{relative}"
        if any(fnmatch.fnmatch(wrapped, pattern) or fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        parent_relative = path.parent.relative_to(source_root)
        run_id = safe_name("__".join(parent_relative.parts) or path.parent.name)
        category = category_key(path.parent.name)
        # The parent-of-parent is a stable system group for ordinary
        # system/temperature or system/stage directory layouts.
        group = safe_name(path.parent.parent.name if path.parent != source_root else category)
        discovered.append(
            SourceTrajectory(path=path, run_id=run_id, category=category, group=group)
        )
    return _deduplicate_run_ids(discovered)


def assign_grouped(
    sources: Sequence[SourceTrajectory],
    ratios: Sequence[float],
    *,
    seed: int = 20260730,
) -> dict[Path, str]:
    """Assign whole trajectories while balancing category and global counts."""

    normalized = normalize_ratios(ratios)
    active = [split for split in SPLITS if normalized[split] > 0]
    if not active:
        raise ValueError("At least one split must be active")
    rng = random.Random(seed)
    by_group: dict[str, list[SourceTrajectory]] = defaultdict(list)
    for source in sources:
        by_group[source.group].append(source)

    assignment: dict[Path, str] = {}
    totals = {split: 0 for split in SPLITS}
    target = {split: normalized[split] * len(sources) for split in SPLITS}
    for group_name in sorted(by_group):
        group_sources = sorted(by_group[group_name], key=lambda item: str(item.path))
        rng.shuffle(group_sources)
        group_counts = {split: 0 for split in SPLITS}
        group_target = {split: normalized[split] * len(group_sources) for split in SPLITS}
        for source in group_sources:
            split = min(
                active,
                key=lambda name: (
                    group_counts[name] - group_target[name],
                    totals[name] - target[name],
                    SPLITS.index(name),
                ),
            )
            assignment[source.path] = split
            group_counts[split] += 1
            totals[split] += 1

    # When enough independent trajectories exist, ensure every requested split
    # has at least one. Move only a whole trajectory, preserving leakage safety.
    if len(sources) >= len(active):
        for empty in [name for name in active if totals[name] == 0]:
            donor = max(active, key=lambda name: totals[name])
            candidate = next(
                source for source in reversed(sorted(sources, key=lambda item: str(item.path)))
                if assignment[source.path] == donor
            )
            assignment[candidate.path] = empty
            totals[donor] -= 1
            totals[empty] += 1
    return assignment


def allocate_counts(total: int, ratios: Sequence[float]) -> dict[str, int]:
    normalized = normalize_ratios(ratios)
    raw = {split: total * normalized[split] for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda name: (raw[name] - counts[name], -SPLITS.index(name)), reverse=True)
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def guarded_assignments(
    n_frames: int,
    ratios: Sequence[float],
    guard_frames: int,
    *,
    rotation: int = 0,
) -> tuple[dict[int, str], set[int]]:
    """Assign contiguous split blocks separated by unused guard frames."""

    active = [split for split, value in normalize_ratios(ratios).items() if value > 0]
    guard_total = max(0, len(active) - 1) * guard_frames
    usable = n_frames - guard_total
    if usable < len(active):
        raise ValueError(
            f"{n_frames} retained frames cannot support {len(active)} splits "
            f"with {guard_frames} guard frames"
        )
    counts = allocate_counts(usable, ratios)
    order = active[rotation % len(active) :] + active[: rotation % len(active)]
    assignment: dict[int, str] = {}
    guards: set[int] = set()
    cursor = 0
    for block_index, split in enumerate(order):
        for index in range(cursor, cursor + counts[split]):
            assignment[index] = split
        cursor += counts[split]
        if block_index < len(order) - 1:
            guards.update(range(cursor, cursor + guard_frames))
            cursor += guard_frames
    return assignment, guards


def _has_constraint_source(outcar_path: Path) -> bool:
    """Report whether ASE can recover Selective Dynamics constraints.

    OUTCAR does not carry Selective Dynamics flags itself; ASE's OUTCAR
    reader only recovers them by reading a sibling CONTCAR (preferred) or
    POSCAR in the same directory. Without one of those files, every atom's
    constraint mask silently comes back empty, i.e. all-mobile.
    """

    return any((outcar_path.parent / name).is_file() for name in ("CONTCAR", "POSCAR"))


def _constraint_mask(atoms: Any) -> np.ndarray:
    mask = np.ones(len(atoms), dtype=np.int8)
    for constraint in getattr(atoms, "constraints", []):
        try:
            indices = np.asarray(constraint.get_indices(), dtype=int)
        except (AttributeError, TypeError, ValueError):
            continue
        mask[indices] = 0
    return mask


def iter_frames(source: Path, *, stride: int, include_virial: bool) -> Iterator[Frame]:
    """Stream valid VASP frames while preserving raw force labels."""

    iread, _ = _ase_io()
    for source_index, atoms in enumerate(iread(str(source), index=":")):
        if source_index % stride:
            continue
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64)
            if forces.shape != (len(atoms), 3):
                raise ValueError(f"force shape {forces.shape}")
            virial = None
            if include_virial:
                stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
                virial = -float(atoms.get_volume()) * stress
            yield Frame(
                source_index=source_index,
                atoms=atoms,
                energy=energy,
                forces=forces,
                move_mask=_constraint_mask(atoms),
                virial=virial,
            )
        except Exception:
            # Incomplete final OUTCAR frames are common. The caller records
            # retained/seen counts and can inspect warnings in the manifest.
            continue


def _discover_type_map(sources: Sequence[SourceTrajectory], configured: Sequence[str]) -> list[str]:
    if configured:
        return list(configured)
    iread, _ = _ase_io()
    symbols: list[str] = []
    for source in sources:
        try:
            atoms = next(iter(iread(str(source.path), index="0")))
        except (StopIteration, Exception):
            continue
        for symbol in atoms.get_chemical_symbols():
            if symbol not in symbols:
                symbols.append(symbol)
    if not symbols:
        raise SafetyError("Could not determine an element type map from the VASP trajectories")
    return symbols


def _write_extxyz_frame(path: Path, frame: Frame, source: SourceTrajectory, split: str) -> None:
    _, write = _ase_io()
    atoms = frame.atoms.copy()
    atoms.calc = None
    atoms.info.update(
        {
            "REF_energy": frame.energy,
            "source_run": source.run_id,
            "source_path": str(source.path),
            "source_frame": frame.source_index,
            "split": split,
        }
    )
    atoms.arrays["REF_forces"] = frame.forces.copy()
    atoms.arrays["move_mask"] = frame.move_mask.copy()
    if frame.virial is not None:
        atoms.info["REF_virial"] = frame.virial.reshape(-1)
    write(str(path), atoms, format="extxyz", append=path.exists())


def _write_deepmd_system(
    root: Path,
    source: SourceTrajectory,
    frames: Sequence[Frame],
    type_map: Sequence[str],
) -> Path:
    system = root / source.run_id
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True, exist_ok=False)
    symbols = frames[0].atoms.get_chemical_symbols()
    unknown = sorted(set(symbols) - set(type_map))
    if unknown:
        raise SafetyError(f"Type map is missing elements {unknown} for {source.path}")
    atom_types = np.asarray([type_map.index(symbol) for symbol in symbols], dtype=int)
    (system / "type.raw").write_text("\n".join(map(str, atom_types)) + "\n", encoding="utf-8")
    (system / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.asarray([frame.atoms.positions.reshape(-1) for frame in frames]))
    np.save(set_dir / "box.npy", np.asarray([frame.atoms.cell.array.reshape(-1) for frame in frames]))
    np.save(set_dir / "energy.npy", np.asarray([[frame.energy] for frame in frames]))
    np.save(set_dir / "force.npy", np.asarray([frame.forces.reshape(-1) for frame in frames]))
    if any(frame.virial is not None for frame in frames):
        np.save(set_dir / "virial.npy", np.asarray([frame.virial.reshape(-1) for frame in frames]))
    with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_frame", "source_frame", "source_path"])
        for index, frame in enumerate(frames):
            writer.writerow([index, frame.source_index, str(source.path)])
    return system


def _prepare_output(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SafetyError(f"Output directory is not empty: {path}")
        if path == Path("/") or len(path.parts) < 3:
            raise SafetyError(f"Refusing broad destructive output replacement: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_dataset(
    campaign: Campaign,
    *,
    source_root: str | Path | None = None,
    output_root: str | Path | None = None,
    force: bool = False,
    seed: int = 20260730,
) -> dict[str, Any]:
    """Create synchronized canonical extxyz and native DeePMD datasets."""

    source = Path(source_root).resolve() if source_root else campaign.root / "runs" / "vasp"
    output = Path(output_root).resolve() if output_root else campaign.root / "datasets" / "canonical"
    _prepare_output(output, force=force)
    sources = discover_outcars(source)
    if not sources:
        raise SafetyError(f"No VASP OUTCAR trajectories found below {source}")

    settings = campaign.dataset
    ratios = settings["ratios"]
    stride = int(settings["stride"])
    include_virial = bool(settings["include_virial"])
    strategy = str(settings["strategy"])
    type_map = _discover_type_map(sources, settings.get("type_map", []))
    grouped = assign_grouped(sources, ratios, seed=seed) if strategy == "grouped" else {}

    extxyz = {split: output / f"{split}.extxyz" for split in SPLITS}
    deepmd = {split: output / "deepmd" / split for split in SPLITS}
    for path in deepmd.values():
        path.mkdir(parents=True, exist_ok=True)

    records: list[CollectionRecord] = []
    global_frame_rows: list[dict[str, Any]] = []
    for trajectory_index, source_trajectory in enumerate(sources):
        record = CollectionRecord(source=source_trajectory)
        if not _has_constraint_source(source_trajectory.path):
            record.warnings.append(
                "no CONTCAR/POSCAR beside OUTCAR: Selective Dynamics constraints "
                "are unavailable, so move_mask marks every atom as mobile"
            )
        frames = list(iter_frames(source_trajectory.path, stride=stride, include_virial=include_virial))
        record.frames_seen = frames[-1].source_index + 1 if frames else 0
        record.frames_retained = len(frames)
        if not frames:
            record.warnings.append("no readable labeled frames")
            records.append(record)
            continue

        selected: dict[str, list[Frame]] = {split: [] for split in SPLITS}
        if strategy == "grouped":
            split = grouped[source_trajectory.path]
            record.assignment = split
            selected[split] = frames
        else:
            try:
                assignment, guards = guarded_assignments(
                    len(frames),
                    ratios,
                    int(settings["guard_frames"]),
                    rotation=trajectory_index,
                )
            except ValueError as exc:
                record.warnings.append(str(exc))
                records.append(record)
                continue
            record.guard_count = len(guards)
            for index, frame in enumerate(frames):
                if index in assignment:
                    selected[assignment[index]].append(frame)

        for split, split_frames in selected.items():
            if not split_frames:
                continue
            record.split_counts[split] = len(split_frames)
            for frame in split_frames:
                _write_extxyz_frame(extxyz[split], frame, source_trajectory, split)
                global_frame_rows.append(
                    {
                        "split": split,
                        "run_id": source_trajectory.run_id,
                        "source_path": str(source_trajectory.path),
                        "source_frame": frame.source_index,
                        "natoms": len(frame.atoms),
                        "energy_ev": frame.energy,
                        "mobile_atoms": int(frame.move_mask.sum()),
                        "fixed_atoms": int(len(frame.move_mask) - frame.move_mask.sum()),
                    }
                )
            _write_deepmd_system(deepmd[split], source_trajectory, split_frames, type_map)
        records.append(record)

    if not global_frame_rows:
        raise SafetyError("No labeled frames were written")

    manifest_rows = [
        {
            "run_id": record.source.run_id,
            "source_path": str(record.source.path),
            "source_sha256": sha256_file(record.source.path),
            "category": record.source.category,
            "group": record.source.group,
            "assignment": record.assignment or "guarded",
            "frames_seen": record.frames_seen,
            "frames_retained": record.frames_retained,
            "train_frames": record.split_counts["train"],
            "valid_frames": record.split_counts["valid"],
            "test_frames": record.split_counts["test"],
            "guard_frames": record.guard_count,
            "warnings": "; ".join(record.warnings),
        }
        for record in records
    ]
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(global_frame_rows[0]))
        writer.writeheader()
        writer.writerows(global_frame_rows)

    split_counts = {
        split: sum(int(row[f"{split}_frames"]) for row in manifest_rows) for split in SPLITS
    }
    payload = {
        "schema_version": 1,
        "source_root": str(source),
        "output_root": str(output),
        "strategy": strategy,
        "ratios": ratios,
        "stride": stride,
        "preserve_raw_forces": True,
        "constraints": "move_mask metadata; reference forces are unmodified",
        "include_virial": include_virial,
        "type_map": type_map,
        "trajectories": len(records),
        "frame_counts": split_counts,
        "extxyz": {split: str(path) if path.exists() else None for split, path in extxyz.items()},
        "deepmd": {split: str(path) for split, path in deepmd.items()},
    }
    manifest_json = output / "manifest.json"
    manifest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    state = StateStore(campaign.root)
    state.event("collect", **payload)
    state.artifact("dataset_manifest", manifest_json)
    return payload
