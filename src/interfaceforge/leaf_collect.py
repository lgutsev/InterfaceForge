"""Leaf-directory VASP collectors for MACE and DeePMD.

Terminal VASP folders are data sources. Their directory ancestry is retained as
provenance, and the full parent lineage is the indivisible train/valid/test
grouping unit so sibling leaves from one campaign branch never leak across
splits.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError
from .vasp_provenance import sha256_file

SPLITS = ("train", "valid", "test")
DEFAULT_EXCLUDES = (
    "*/.git/*",
    "*/.interfaceforge/*",
    "*/archive/*",
    "*/restart_archive_*/*",
    "*/refit_archive_*/*",
    "*/stability_archive_*/*",
)


@dataclass(frozen=True)
class LeafSource:
    outcar: Path
    leaf: Path
    relative_leaf: Path
    run_id: str
    heritage_parts: tuple[str, ...]
    heritage_parent: str
    heritage_key: str


@dataclass
class LeafFrame:
    source_index: int
    atoms: Any
    energy: float
    forces: np.ndarray
    move_mask: np.ndarray
    virial: np.ndarray | None


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return re.sub(r"_+", "_", cleaned).strip("_") or "unnamed"


def _normalize_ratios(values: Sequence[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("Need exactly three ratios: train valid test")
    ratios = [float(value) for value in values]
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("Ratios must be non-negative and have a positive sum")
    total = sum(ratios)
    return {name: value / total for name, value in zip(SPLITS, ratios, strict=True)}


def _is_excluded(relative: Path, patterns: Sequence[str]) -> bool:
    text = relative.as_posix()
    wrapped = f"/{text}"
    if any(
        fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(wrapped, pattern)
        for pattern in patterns
    ):
        return True
    # Match existing InterfaceForge campaign hygiene: do not harvest backup
    # branches or deliberately disabled X* branches.
    return any("backup" in part.lower() or part.startswith("X") for part in relative.parts)


def _has_child_directories(directory: Path) -> bool:
    try:
        return any(child.is_dir() for child in directory.iterdir())
    except OSError:
        return True


def _heritage_context(relative_leaf: Path, depth: int) -> tuple[str, ...]:
    """Return up to ``depth`` immediate ancestor labels for explicit metadata."""
    if depth < 1:
        raise ValueError("heritage_depth must be at least 1")
    ancestors = relative_leaf.parts[:-1]
    if not ancestors:
        return ("__root__",)
    return tuple(ancestors[-depth:])


def _parent_lineage(relative_leaf: Path) -> str:
    """Return the complete source-root-relative parent lineage for grouping."""
    ancestors = relative_leaf.parts[:-1]
    return "/".join(ancestors) if ancestors else "__root__"


def discover_leaf_outcars(
    root: str | Path,
    *,
    heritage_depth: int = 2,
    outcar_name: str = "OUTCAR",
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[LeafSource]:
    """Discover OUTCARs only in terminal/deepest directories.

    ``heritage_depth`` controls how many immediate ancestor names are repeated
    in human-readable metadata. Group identity uses the *complete* parent
    lineage, preventing unrelated branches with repeated folder names from
    being merged accidentally.
    """
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise SafetyError(f"Leaf source root is not a directory: {source_root}")

    discovered: list[LeafSource] = []
    seen_ids: dict[str, int] = {}
    candidates = sorted(source_root.rglob(outcar_name)) + sorted(
        source_root.rglob(f"{outcar_name}.gz")
    )
    for outcar in candidates:
        if not outcar.is_file():
            continue
        # Prefer an uncompressed OUTCAR when both sit in the same directory.
        if outcar.name.endswith(".gz") and outcar.with_name(outcar.name[:-3]).is_file():
            continue
        relative_outcar = outcar.relative_to(source_root)
        if _is_excluded(relative_outcar, exclude):
            continue
        leaf = outcar.parent
        if _has_child_directories(leaf):
            continue
        relative_leaf = leaf.relative_to(source_root)
        run_id_base = safe_name("__".join(relative_leaf.parts) or leaf.name)
        collision = seen_ids.get(run_id_base, 0)
        seen_ids[run_id_base] = collision + 1
        run_id = run_id_base if collision == 0 else f"{run_id_base}__dup{collision + 1}"
        heritage_parts = _heritage_context(relative_leaf, heritage_depth)
        heritage_parent = _parent_lineage(relative_leaf)
        discovered.append(
            LeafSource(
                outcar=outcar,
                leaf=leaf,
                relative_leaf=relative_leaf,
                run_id=run_id,
                heritage_parts=heritage_parts,
                heritage_parent=heritage_parent,
                heritage_key=safe_name(heritage_parent.replace("/", "__")),
            )
        )
    return discovered


def assign_heritage_groups(
    sources: Sequence[LeafSource],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 20260730,
) -> dict[Path, str]:
    """Assign whole parent-lineage groups, never individual sibling leaves."""
    normalized = _normalize_ratios(ratios)
    active = [name for name in SPLITS if normalized[name] > 0]
    if not active:
        raise ValueError("At least one split must be active")

    by_group: dict[str, list[LeafSource]] = defaultdict(list)
    for source in sources:
        by_group[source.heritage_key].append(source)

    rng = random.Random(seed)
    groups = list(by_group.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    target = {name: normalized[name] * len(sources) for name in SPLITS}
    totals = {name: 0 for name in SPLITS}
    group_split: dict[str, str] = {}

    for heritage_key, group_sources in groups:
        split = max(
            active,
            key=lambda name: (
                target[name] - totals[name],
                normalized[name],
                -SPLITS.index(name),
            ),
        )
        group_split[heritage_key] = split
        totals[split] += len(group_sources)

    # If enough independent parent lineages exist, populate every requested
    # split by moving a whole lineage rather than a correlated leaf.
    if len(groups) >= len(active):
        for empty in [name for name in active if totals[name] == 0]:
            donor = max(active, key=lambda name: totals[name] - target[name])
            donor_groups = [
                (key, items) for key, items in groups if group_split[key] == donor
            ]
            if len(donor_groups) <= 1:
                continue
            key, items = min(donor_groups, key=lambda item: len(item[1]))
            group_split[key] = empty
            totals[donor] -= len(items)
            totals[empty] += len(items)

    return {source.outcar: group_split[source.heritage_key] for source in sources}



def assign_random_frame_splits(
    frames: Sequence[LeafFrame],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 20260730,
    leaf_key: str,
) -> dict[str, list[LeafFrame]]:
    """Randomly split frames within one leaf using deterministic membership."""
    normalized = _normalize_ratios(ratios)
    shuffled = list(frames)
    random.Random(f"{seed}:{leaf_key}").shuffle(shuffled)

    exact = {name: normalized[name] * len(shuffled) for name in SPLITS}
    counts = {name: int(exact[name]) for name in SPLITS}
    remainder = len(shuffled) - sum(counts.values())
    order = sorted(
        SPLITS,
        key=lambda name: (
            exact[name] - counts[name],
            normalized[name],
            -SPLITS.index(name),
        ),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1

    partitions: dict[str, list[LeafFrame]] = {}
    cursor = 0
    for name in SPLITS:
        selected = shuffled[cursor : cursor + counts[name]]
        partitions[name] = sorted(selected, key=lambda frame: frame.source_index)
        cursor += counts[name]
    return partitions


def _frame_digest(frames: Sequence[LeafFrame]) -> str:
    membership = ",".join(str(frame.source_index) for frame in frames)
    return sha256(membership.encode("utf-8")).hexdigest()


def balance_leaf_frames(
    frames: Sequence[LeafFrame],
    target: int | None,
    *,
    seed: int,
    leaf_key: str,
) -> list[LeafFrame]:
    """Return a deterministic, time-unbiased per-leaf sample."""
    available = len(frames)
    if target is None or available == target:
        return list(frames)
    if target < 1:
        raise ValueError("frames_per_leaf must be positive")
    if available < target:
        raise SafetyError(f"only {available} readable frames; balanced target is {target}")
    return sorted(
        random.Random(f"{seed}:balance:{leaf_key}").sample(list(frames), target),
        key=lambda frame: frame.source_index,
    )


def _ase_io() -> tuple[Any, Any]:
    try:
        from ase.io import iread, write
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required for leaf collection. Install InterfaceForge with "
            "pip install 'interfaceforge[vasp]'"
        ) from exc
    return iread, write


def _constraint_mask(atoms: Any) -> np.ndarray:
    mask = np.ones(len(atoms), dtype=np.int8)
    for constraint in getattr(atoms, "constraints", []):
        try:
            indices = np.asarray(constraint.get_indices(), dtype=int)
        except (AttributeError, TypeError, ValueError):
            continue
        mask[indices] = 0
    return mask


def iter_leaf_frames(
    source: LeafSource,
    *,
    stride: int = 1,
    include_virial: bool = False,
) -> Iterable[LeafFrame]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    iread, _ = _ase_io()
    for source_index, atoms in enumerate(iread(str(source.outcar), index=":")):
        if source_index % stride:
            continue
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64)
            if forces.shape != (len(atoms), 3):
                continue
            virial = None
            if include_virial:
                stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
                virial = -float(atoms.get_volume()) * stress
            yield LeafFrame(
                source_index=source_index,
                atoms=atoms,
                energy=energy,
                forces=forces,
                move_mask=_constraint_mask(atoms),
                virial=virial,
            )
        except Exception:
            continue


def _prepare_output(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SafetyError(f"Output directory is not empty: {path}")
        if path == Path("/") or len(path.parts) < 3:
            raise SafetyError(f"Refusing broad destructive output replacement: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _discover_type_map(sources: Sequence[LeafSource], configured: Sequence[str]) -> list[str]:
    if configured:
        return list(configured)
    iread, _ = _ase_io()
    symbols: list[str] = []
    for source in sources:
        try:
            atoms = next(iter(iread(str(source.outcar), index="0")))
        except Exception:
            continue
        for symbol in atoms.get_chemical_symbols():
            if symbol not in symbols:
                symbols.append(symbol)
    if not symbols:
        raise SafetyError("Could not determine a DeePMD type map from leaf OUTCARs")
    return symbols


def _write_mace_frames(
    path: Path,
    source: LeafSource,
    split: str,
    frames: Sequence[LeafFrame],
) -> None:
    _, write = _ase_io()
    heritage_context = "/".join(source.heritage_parts)
    for frame in frames:
        atoms = frame.atoms.copy()
        atoms.calc = None
        atoms.info.update(
            {
                "REF_energy": frame.energy,
                "source_run": source.run_id,
                "source_path": str(source.outcar),
                "source_frame": frame.source_index,
                "split": split,
                "IF_leaf": source.relative_leaf.as_posix(),
                "IF_heritage": source.heritage_key,
                "IF_heritage_parent": source.heritage_parent,
                "IF_heritage_context": heritage_context,
            }
        )
        atoms.arrays["REF_forces"] = frame.forces.copy()
        atoms.arrays["move_mask"] = frame.move_mask.copy()
        if frame.virial is not None:
            atoms.info["REF_virial"] = frame.virial.reshape(-1)
        write(str(path), atoms, format="extxyz", append=path.exists())


def _safe_relative_path(relative: Path) -> Path:
    return Path(*(safe_name(part) for part in relative.parts))


def _write_deepmd_system(
    split_root: Path,
    source: LeafSource,
    split: str,
    frames: Sequence[LeafFrame],
    type_map: Sequence[str],
) -> Path:
    """Write one leaf as one DeePMD system while retaining its full lineage."""
    if not frames:
        raise ValueError("Cannot write an empty DeePMD system")

    # Preserve the complete source tree: split/<relative leaf>/set.000.
    system = split_root / _safe_relative_path(source.relative_leaf)
    if system.exists():
        system = system.parent / f"{safe_name(source.leaf.name)}__{source.run_id}"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True, exist_ok=False)

    symbols = frames[0].atoms.get_chemical_symbols()
    natoms = len(symbols)
    unknown = sorted(set(symbols) - set(type_map))
    if unknown:
        raise SafetyError(f"Type map is missing elements {unknown} for {source.outcar}")
    for frame in frames[1:]:
        if len(frame.atoms) != natoms or frame.atoms.get_chemical_symbols() != symbols:
            raise SafetyError(
                f"Atom identity/order changes within {source.outcar}; refusing an invalid "
                "single DeePMD system"
            )

    atom_types = np.asarray([type_map.index(symbol) for symbol in symbols], dtype=int)
    (system / "type.raw").write_text("\n".join(map(str, atom_types)) + "\n", encoding="utf-8")
    (system / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", np.asarray([frame.atoms.positions.reshape(-1) for frame in frames]))
    np.save(set_dir / "box.npy", np.asarray([frame.atoms.cell.array.reshape(-1) for frame in frames]))
    np.save(set_dir / "energy.npy", np.asarray([[frame.energy] for frame in frames]))
    np.save(set_dir / "force.npy", np.asarray([frame.forces.reshape(-1) for frame in frames]))
    if any(frame.virial is not None for frame in frames):
        if not all(frame.virial is not None for frame in frames):
            raise SafetyError(f"Mixed virial availability within {source.outcar}")
        np.save(set_dir / "virial.npy", np.asarray([frame.virial.reshape(-1) for frame in frames]))

    context = {
        "schema_version": 1,
        "split": split,
        "source_root_relative_leaf": source.relative_leaf.as_posix(),
        "source_outcar": str(source.outcar),
        "run_id": source.run_id,
        "heritage_key": source.heritage_key,
        "heritage_parent": source.heritage_parent,
        "heritage_context": list(source.heritage_parts),
        "frames": len(frames),
    }
    (system / "heritage.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    with (system / "frame_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "local_frame",
                "source_frame",
                "source_path",
                "relative_leaf",
                "heritage_key",
                "heritage_parent",
                "heritage_context",
            ]
        )
        for index, frame in enumerate(frames):
            writer.writerow(
                [
                    index,
                    frame.source_index,
                    str(source.outcar),
                    source.relative_leaf.as_posix(),
                    source.heritage_key,
                    source.heritage_parent,
                    "/".join(source.heritage_parts),
                ]
            )
    return system


def collect_leaf_dataset(
    root: str | Path,
    output: str | Path,
    *,
    engine: str,
    heritage_depth: int = 2,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 20260730,
    stride: int = 1,
    include_virial: bool = False,
    type_map: Sequence[str] = (),
    split_mode: str = "heritage",
    frames_per_leaf: int | None = None,
    reference_provenance: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Collect terminal VASP calculations into MACE or DeePMD training data."""
    engine = engine.lower()
    split_mode = split_mode.lower()
    if engine not in {"mace", "deepmd"}:
        raise ValueError("engine must be 'mace' or 'deepmd'")
    if split_mode not in {"heritage", "random-frame"}:
        raise ValueError("split_mode must be 'heritage' or 'random-frame'")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if frames_per_leaf is not None and frames_per_leaf < 1:
        raise ValueError("frames_per_leaf must be positive")

    source_root = Path(root).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    sources = discover_leaf_outcars(source_root, heritage_depth=heritage_depth)
    if not sources:
        raise SafetyError(f"No leaf-directory OUTCARs found below {source_root}")
    assignments = (
        assign_heritage_groups(sources, ratios, seed=seed)
        if split_mode == "heritage"
        else {}
    )

    preview_rows = [
        {
            "relative_leaf": source.relative_leaf.as_posix(),
            "outcar": str(source.outcar),
            "run_id": source.run_id,
            "heritage_key": source.heritage_key,
            "heritage_parent": source.heritage_parent,
            "heritage_context": "/".join(source.heritage_parts),
            "split": (
                assignments[source.outcar]
                if split_mode == "heritage"
                else "randomized within leaf"
            ),
        }
        for source in sources
    ]
    if dry_run:
        return {
            "engine": engine,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "heritage_depth": heritage_depth,
            "split_mode": split_mode,
            "ratios": list(ratios),
            "leaves": len(sources),
            "heritage_groups": len({source.heritage_key for source in sources}),
            "dry_run": True,
            "sources": preview_rows,
        }

    _prepare_output(output_root, force=force)
    split_roots = {name: output_root / name for name in SPLITS}
    if engine == "deepmd":
        for path in split_roots.values():
            path.mkdir(parents=True, exist_ok=True)
        resolved_type_map = _discover_type_map(sources, type_map)
    else:
        resolved_type_map = []

    manifest_rows: list[dict[str, Any]] = []
    frame_counts = {name: 0 for name in SPLITS}
    failed_sources: set[Path] = set()

    for source in sources:
        base_row = {
            "relative_leaf": source.relative_leaf.as_posix(),
            "outcar": str(source.outcar),
            "run_id": source.run_id,
            "heritage_key": source.heritage_key,
            "heritage_parent": source.heritage_parent,
            "heritage_context": "/".join(source.heritage_parts),
            "available_frames": "",
            "sampled_frames": "",
        }
        try:
            frames = list(
                iter_leaf_frames(source, stride=stride, include_virial=include_virial)
            )
            if not frames:
                raise SafetyError("no readable labelled frames")
            available_frames = len(frames)
            frames = balance_leaf_frames(
                frames,
                frames_per_leaf,
                seed=seed,
                leaf_key=source.relative_leaf.as_posix(),
            )
            base_row["available_frames"] = available_frames
            base_row["sampled_frames"] = len(frames)
            if split_mode == "heritage":
                partitions = {
                    assignments[source.outcar]: frames,
                }
            else:
                partitions = assign_random_frame_splits(
                    frames,
                    ratios,
                    seed=seed,
                    leaf_key=source.relative_leaf.as_posix(),
                )
        except Exception as exc:
            failed_sources.add(source.outcar)
            manifest_rows.append(
                {
                    **base_row,
                    "split": assignments.get(source.outcar, "unassigned"),
                    "frames": 0,
                    "frame_digest": "",
                    "status": "FAILED",
                    "output": "",
                    "detail": str(exc),
                }
            )
            continue

        for split in SPLITS:
            selected = partitions.get(split, [])
            if not selected:
                continue
            row = {
                **base_row,
                "split": split,
                "frames": len(selected),
                "frame_digest": _frame_digest(selected),
                "status": "OK",
                "output": "",
                "detail": "",
            }
            try:
                if engine == "mace":
                    target = output_root / f"{split}.extxyz"
                    _write_mace_frames(target, source, split, selected)
                    row["output"] = str(target)
                else:
                    system = _write_deepmd_system(
                        split_roots[split],
                        source,
                        split,
                        selected,
                        resolved_type_map,
                    )
                    row["output"] = str(system)
                frame_counts[split] += len(selected)
            except Exception as exc:
                failed_sources.add(source.outcar)
                row["status"] = "FAILED"
                row["detail"] = str(exc)
            manifest_rows.append(row)

    manifest_csv = output_root / "leaf_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    group_splits: dict[str, str] = {}
    if split_mode == "heritage":
        for source in sources:
            split = assignments[source.outcar]
            previous = group_splits.setdefault(source.heritage_key, split)
            if previous != split:
                raise AssertionError(
                    f"heritage leakage detected internally: "
                    f"{source.heritage_key} -> {previous}, {split}"
                )

    empty_splits = [name for name in SPLITS if frame_counts[name] == 0]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "leaf-heritage-collector",
        "engine": engine,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "heritage_depth": heritage_depth,
        "split_mode": split_mode,
        "split_rule": (
            "full source-root-relative parent lineage is indivisible across splits"
            if split_mode == "heritage"
            else "frames are deterministically randomized within every leaf"
        ),
        "ratios": list(ratios),
        "seed": seed,
        "stride": stride,
        "frames_per_leaf": frames_per_leaf,
        "include_virial": include_virial,
        "preserve_raw_forces": True,
        "label_convention": {
            "energy": "ASE VASP reader atoms.get_potential_energy() in eV",
            "forces": "raw unconstrained VASP forces in eV/angstrom",
            "virial": (
                "-volume*ASE stress tensor in eV" if include_virial else "not exported"
            ),
            "positions": "angstrom",
        },
        "leaves_discovered": len(sources),
        "heritage_groups": len(
            {source.heritage_key for source in sources}
        ),
        "heritage_group_splits": group_splits,
        "frame_counts": frame_counts,
        "empty_splits": empty_splits,
        "failed_leaves": len(failed_sources),
        "manifest_csv": str(manifest_csv),
    }
    if reference_provenance is not None:
        provenance_path = Path(reference_provenance).expanduser().resolve()
        if not provenance_path.is_file():
            raise SafetyError(f"Missing VASP reference provenance: {provenance_path}")
        payload["reference_provenance"] = {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        }
    if engine == "mace":
        payload["extxyz"] = {
            name: (
                str(output_root / f"{name}.extxyz")
                if (output_root / f"{name}.extxyz").is_file()
                else None
            )
            for name in SPLITS
        }
    else:
        payload["type_map"] = resolved_type_map
        payload["deepmd"] = {name: str(split_roots[name]) for name in SPLITS}
        payload["context_preservation"] = (
            "split/<full source-root-relative leaf>/set.000 plus "
            "heritage.json and frame_map.csv"
        )

    manifest_json = output_root / "leaf_manifest.json"
    manifest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["manifest_json"] = str(manifest_json)
    return payload


def build_parser(default_engine: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect VASP OUTCARs from terminal directories while keeping directory heritage "
            "intact across train/valid/test splits."
        )
    )
    if default_engine is None:
        parser.add_argument("engine", choices=("mace", "deepmd"))
    parser.add_argument("--root", default=".", help="Calculation tree root (default: current directory)")
    parser.add_argument("--output", required=True, help="New dataset output directory")
    parser.add_argument(
        "--heritage-depth",
        type=int,
        default=2,
        help=(
            "Number of immediate ancestor labels repeated in provenance metadata (default: 2); "
            "the full parent lineage is always the split grouping key"
        ),
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALID", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--split-mode",
        choices=("heritage", "random-frame"),
        default="heritage",
        help=(
            "heritage keeps complete parent lineages indivisible; random-frame "
            "deterministically stratifies every leaf across active splits"
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--frames-per-leaf",
        type=int,
        help="Deterministically subsample every leaf to this many retained frames",
    )
    parser.add_argument(
        "--reference-provenance",
        help="Reference-provenance JSON to hash into the dataset manifest",
    )
    parser.add_argument("--include-virial", action="store_true")
    parser.add_argument(
        "--type-map",
        nargs="*",
        default=(),
        help="Explicit DeePMD type map; otherwise inferred in first-seen order",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, default_engine: str | None = None) -> int:
    parser = build_parser(default_engine)
    args = parser.parse_args(argv)
    engine = default_engine or args.engine
    payload = collect_leaf_dataset(
        args.root,
        args.output,
        engine=engine,
        heritage_depth=args.heritage_depth,
        ratios=args.ratios,
        seed=args.seed,
        stride=args.stride,
        include_virial=args.include_virial,
        type_map=args.type_map,
        split_mode=args.split_mode,
        frames_per_leaf=args.frames_per_leaf,
        reference_provenance=args.reference_provenance,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2))
    return 1 if payload.get("failed_leaves", 0) else 0


def main_mace(argv: Sequence[str] | None = None) -> int:
    """Installed entry point for the MACE leaf collector."""
    return main(argv, default_engine="mace")


def main_deepmd(argv: Sequence[str] | None = None) -> int:
    """Installed entry point for the DeePMD leaf collector."""
    return main(argv, default_engine="deepmd")


if __name__ == "__main__":
    raise SystemExit(main())
