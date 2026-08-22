"""Leaf-directory VASP collectors for MACE and DeePMD.

Terminal VASP folders are the data sources. One or more immediate ancestor
folders define a heritage group. Heritage groups, not individual leaves, are
the indivisible unit for train/validation/test splitting so sibling leaves from
the same physical/systematic campaign cannot leak across splits.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import DependencyError, SafetyError

SPLITS = ("train", "valid", "test")
DEFAULT_EXCLUDES = (
    "*/.git/*",
    "*/.interfaceforge/*",
    "*/archive/*",
    "*/restart_archive_*/*",
    "*/refit_archive_*/*",
    "*/stability_archive_*/*",
    "*/backup/*",
    "*/X*/*",
)


@dataclass(frozen=True)
class LeafSource:
    outcar: Path
    leaf: Path
    relative_leaf: Path
    run_id: str
    heritage_parts: tuple[str, ...]
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


def _is_excluded(relative: str, patterns: Sequence[str]) -> bool:
    wrapped = f"/{relative}"
    return any(
        fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(wrapped, pattern)
        for pattern in patterns
    )


def _has_child_directories(directory: Path) -> bool:
    try:
        return any(child.is_dir() for child in directory.iterdir())
    except OSError:
        return True


def _heritage_for(relative_leaf: Path, depth: int) -> tuple[str, ...]:
    """Use the immediate ancestors above the leaf as its grouping context.

    Example: material/termination/400K/run_03 with depth=2 becomes
    (termination, 400K). The leaf name itself is not part of the group.
    """
    if depth < 1:
        raise ValueError("heritage_depth must be at least 1")
    ancestors = relative_leaf.parts[:-1]
    if ancestors:
        return tuple(ancestors[-depth:])
    return (relative_leaf.name,)


def discover_leaf_outcars(
    root: str | Path,
    *,
    heritage_depth: int = 2,
    outcar_name: str = "OUTCAR",
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[LeafSource]:
    """Discover OUTCARs only in terminal/deepest directories."""
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise SafetyError(f"Leaf source root is not a directory: {source_root}")

    discovered: list[LeafSource] = []
    seen_ids: dict[str, int] = {}
    for outcar in sorted(source_root.rglob(outcar_name)):
        if not outcar.is_file():
            continue
        relative_outcar = outcar.relative_to(source_root).as_posix()
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
        heritage_parts = _heritage_for(relative_leaf, heritage_depth)
        discovered.append(
            LeafSource(
                outcar=outcar,
                leaf=leaf,
                relative_leaf=relative_leaf,
                run_id=run_id,
                heritage_parts=heritage_parts,
                heritage_key=safe_name("__".join(heritage_parts)),
            )
        )
    return discovered


def assign_heritage_groups(
    sources: Sequence[LeafSource],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 20260730,
) -> dict[Path, str]:
    """Assign whole heritage groups, never individual sibling leaves, to splits."""
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

    # Populate requested splits when enough independent heritage groups exist,
    # moving a whole group at a time.
    if len(groups) >= len(active):
        for empty in [name for name in active if totals[name] == 0]:
            donor = max(active, key=lambda name: totals[name] - target[name])
            donor_groups = [
                (key, items)
                for key, items in groups
                if group_split[key] == donor
            ]
            if len(donor_groups) <= 1:
                continue
            key, items = min(donor_groups, key=lambda item: len(item[1]))
            group_split[key] = empty
            totals[donor] -= len(items)
            totals[empty] += len(items)

    return {source.outcar: group_split[source.heritage_key] for source in sources}


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
            # Incomplete trailing VASP steps are common. Keep readable labelled
            # frames and ignore an incomplete final step.
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
    heritage_path = "/".join(source.heritage_parts)
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
                "IF_heritage_path": heritage_path,
            }
        )
        atoms.arrays["REF_forces"] = frame.forces.copy()
        atoms.arrays["move_mask"] = frame.move_mask.copy()
        if frame.virial is not None:
            atoms.info["REF_virial"] = frame.virial.reshape(-1)
        write(str(path), atoms, format="extxyz", append=path.exists())


def _safe_heritage_path(parts: Sequence[str]) -> Path:
    return Path(*(safe_name(part) for part in parts))


def _write_deepmd_system(
    split_root: Path,
    source: LeafSource,
    split: str,
    frames: Sequence[LeafFrame],
    type_map: Sequence[str],
) -> Path:
    """Write one leaf as one DeePMD system beneath its heritage hierarchy."""
    if not frames:
        raise ValueError("Cannot write an empty DeePMD system")

    # Preserve context physically: split/<heritage...>/<leaf>/set.000.
    system = split_root / _safe_heritage_path(source.heritage_parts) / safe_name(source.leaf.name)
    if system.exists():
        system = split_root / _safe_heritage_path(source.heritage_parts) / source.run_id
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
        "heritage_parts": list(source.heritage_parts),
        "heritage_path": "/".join(source.heritage_parts),
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
                "heritage_path",
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
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Collect terminal VASP calculations into MACE or DeePMD training data."""
    engine = engine.lower()
    if engine not in {"mace", "deepmd"}:
        raise ValueError("engine must be 'mace' or 'deepmd'")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    source_root = Path(root).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    sources = discover_leaf_outcars(source_root, heritage_depth=heritage_depth)
    if not sources:
        raise SafetyError(f"No leaf-directory OUTCARs found below {source_root}")
    assignments = assign_heritage_groups(sources, ratios, seed=seed)

    preview_rows = [
        {
            "relative_leaf": source.relative_leaf.as_posix(),
            "outcar": str(source.outcar),
            "run_id": source.run_id,
            "heritage_key": source.heritage_key,
            "heritage_path": "/".join(source.heritage_parts),
            "split": assignments[source.outcar],
        }
        for source in sources
    ]
    if dry_run:
        return {
            "engine": engine,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "heritage_depth": heritage_depth,
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
    failures = 0

    for source in sources:
        split = assignments[source.outcar]
        row = {
            "relative_leaf": source.relative_leaf.as_posix(),
            "outcar": str(source.outcar),
            "run_id": source.run_id,
            "heritage_key": source.heritage_key,
            "heritage_path": "/".join(source.heritage_parts),
            "split": split,
            "frames": 0,
            "status": "OK",
            "output": "",
            "detail": "",
        }
        try:
            frames = list(iter_leaf_frames(source, stride=stride, include_virial=include_virial))
            if not frames:
                raise SafetyError("no readable labelled frames")
            row["frames"] = len(frames)
            frame_counts[split] += len(frames)
            if engine == "mace":
                target = output_root / f"{split}.extxyz"
                _write_mace_frames(target, source, split, frames)
                row["output"] = str(target)
            else:
                system = _write_deepmd_system(
                    split_roots[split], source, split, frames, resolved_type_map
                )
                row["output"] = str(system)
        except Exception as exc:
            failures += 1
            row["status"] = "FAILED"
            row["detail"] = str(exc)
        manifest_rows.append(row)

    manifest_csv = output_root / "leaf_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    group_splits: dict[str, str] = {}
    for source in sources:
        split = assignments[source.outcar]
        previous = group_splits.setdefault(source.heritage_key, split)
        if previous != split:
            raise AssertionError(
                f"heritage leakage detected internally: {source.heritage_key} -> {previous}, {split}"
            )

    empty_splits = [name for name in SPLITS if frame_counts[name] == 0]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "leaf-heritage-collector",
        "engine": engine,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "heritage_depth": heritage_depth,
        "heritage_rule": "immediate ancestors excluding leaf; groups are indivisible across splits",
        "ratios": list(ratios),
        "seed": seed,
        "stride": stride,
        "include_virial": include_virial,
        "preserve_raw_forces": True,
        "leaves_discovered": len(sources),
        "heritage_groups": len(group_splits),
        "heritage_group_splits": group_splits,
        "frame_counts": frame_counts,
        "empty_splits": empty_splits,
        "failed_leaves": failures,
        "manifest_csv": str(manifest_csv),
    }
    if engine == "mace":
        payload["extxyz"] = {
            name: str(output_root / f"{name}.extxyz")
            if (output_root / f"{name}.extxyz").is_file()
            else None
            for name in SPLITS
        }
    else:
        payload["type_map"] = resolved_type_map
        payload["deepmd"] = {name: str(split_roots[name]) for name in SPLITS}
        payload["context_preservation"] = (
            "split/<heritage...>/<leaf>/set.000 plus heritage.json and frame_map.csv"
        )

    manifest_json = output_root / "leaf_manifest.json"
    manifest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["manifest_json"] = str(manifest_json)
    return payload


def build_parser(default_engine: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect VASP OUTCARs from terminal directories while keeping parent/grandparent "
            "heritage groups intact across train/valid/test splits."
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
        help="Immediate ancestors above each leaf used as one indivisible group (default: 2)",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALID", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--stride", type=int, default=1)
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
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2))
    return 1 if payload.get("failed_leaves", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
