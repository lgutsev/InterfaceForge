"""Archive trained committees and canonical datasets for reuse and Hugging Face upload.

Three artifacts are produced here:

* ``pack_dataset_archive`` writes one checksummed ``.zip`` of a canonical dataset
  (extxyz + DeePMD NPY + manifests, from ``iface collect`` or ``iface-mapped-collect``)
  for cold storage. ``data/`` inside the archive is a byte-for-byte dataset
  directory, so restore is ``unzip``.
* ``pack_huggingface`` turns a verified committee bundle (from
  ``iface committee collect``, MACE or DeePMD) into an upload-ready Hugging Face
  model repository: a generated model card with YAML frontmatter, ``.gitattributes``
  for Git LFS, a provenance manifest, checksums, and the exact ``hf upload``
  command.
* ``pack_campaign`` runs collect + package for every committee a campaign has
  (by the same directory conventions ``iface mlip-progress`` / ``iface train``
  use) plus the dataset archive, in one call.

InterfaceForge never contacts the Hugging Face Hub. It stops at a ready-to-push
directory; the user runs ``hf upload`` themselves.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import __version__
from .committee import collect_committee, verify_committee_bundle
from .errors import ConfigurationError, DependencyError, SafetyError
from .state import sha256_file, utc_now

if TYPE_CHECKING:
    from .config import Campaign

_COMPRESSION = {"deflated": zipfile.ZIP_DEFLATED, "stored": zipfile.ZIP_STORED}
_LFS_PATTERNS = ("*.model", "*.pth", "*.pt", "*.pt2", "*.pb", "*.npy", "*.extxyz", "*.xyz")
_DATASET_TOP_FILES = ("manifest.json", "manifest.csv", "frames.csv")
_LEAF_DATASET_TOP_FILES = ("leaf_manifest.json", "leaf_manifest.csv")
_DATASET_EXTXYZ = ("train.extxyz", "valid.extxyz", "test.extxyz")

_MACE_TAGS = ("mace",)
_DEEPMD_TAGS = ("deepmd", "deepmd-kit")
_COMMON_TAGS = (
    "interatomic-potential",
    "machine-learning-potential",
    "molecular-dynamics",
    "chemistry",
    "materials-science",
    "computational-chemistry",
    "interface",
)

_MATURITY_NOTE = (
    "This committee has automated file-level integrity checks only. It is **not** "
    "scientifically validated here: no held-out DFT property comparison, stability "
    "campaign, or transferability study is attested. Committee spread is an "
    "uncalibrated heuristic, not a guaranteed error bar. Compare against your own "
    "held-out DFT data in the intended regime before relying on predictions."
)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _zip_output(requested: str | Path) -> Path:
    value = _resolved(requested)
    return value if value.suffix.lower() == ".zip" else Path(f"{value}.zip")


def _temp_sibling(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _sha_lines(entries: Sequence[tuple[str, str]]) -> str:
    """Render ``sha256sum -c`` compatible lines from (relative_name, digest) pairs."""

    return "".join(f"{digest}  {name}\n" for name, digest in entries)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _write_dir_zip(source_dir: Path, archive: Path, top_level: str) -> None:
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    temporary = _temp_sibling(archive)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            prefix = pathlib.PurePosixPath(top_level)
            for path in files:
                relative = path.relative_to(source_dir).as_posix()
                handle.write(path, (prefix / relative).as_posix())
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _detect_dataset_flavor(source: Path) -> tuple[str, dict[str, Any]]:
    """Identify which InterfaceForge dataset collector wrote ``source``.

    ``iface_collect``: the unified ``manifest.json`` written by ``iface collect``.
    ``mapped_leaf``: the ``iface-mapped-collect`` / leaf-heritage layout, which
    writes ``leaf_manifest.json`` at the MACE-side root and a second one under
    ``deepmd/`` -- e.g. the periodic SiN/TiN/TiO example. Only ``iface_collect``
    datasets carry the ``move_mask.npy`` / ``system_meta.json`` sidecars
    ``dedupe`` needs; ``mapped_leaf`` datasets can still be mirror-archived.
    """

    manifest_path = source / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid dataset manifest JSON: {manifest_path}") from exc
        if "frame_counts" not in manifest and "deepmd" not in manifest:
            raise ConfigurationError(f"{manifest_path} does not look like an 'iface collect' manifest")
        return "iface_collect", manifest

    leaf_manifest_path = source / "leaf_manifest.json"
    if leaf_manifest_path.is_file():
        try:
            manifest = json.loads(leaf_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid leaf manifest JSON: {leaf_manifest_path}") from exc
        if manifest.get("method") != "leaf-heritage-collector":
            raise ConfigurationError(f"{leaf_manifest_path} is not an InterfaceForge leaf manifest")
        deepmd_manifest_path = source / "deepmd" / "leaf_manifest.json"
        if deepmd_manifest_path.is_file():
            try:
                deepmd_manifest = json.loads(deepmd_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                deepmd_manifest = {}
            manifest = {
                **manifest,
                "type_map": deepmd_manifest.get("type_map"),
                "deepmd_leaf_manifest": deepmd_manifest,
            }
        return "mapped_leaf", manifest

    raise ConfigurationError(
        f"Not an InterfaceForge canonical dataset (no manifest.json or leaf_manifest.json): {source}"
    )


# --------------------------------------------------------------------------- #
# dataset archive (task: back up training data for reuse)
# --------------------------------------------------------------------------- #
def pack_dataset_archive(
    dataset_root: str | Path,
    output: str | Path,
    *,
    include_extxyz: bool = True,
    dedupe: bool = False,
    compression: str = "deflated",
    label: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one checksummed ZIP of an ``iface collect`` canonical dataset.

    By default this mirrors ``dataset_root`` exactly (extxyz *and* DeePMD NPY),
    so restore is a plain ``unzip``. ``dedupe=True`` stores the numeric frame
    data only once: the DeePMD NPY tree (full float64) is kept and the
    ``*.extxyz`` files are dropped, since they are a lossy re-encoding of the
    same coordinates/forces (ASE's extxyz writer rounds to ~8 decimal places).
    A deduped archive requires every DeePMD system to already carry the
    ``move_mask.npy`` / ``system_meta.json`` sidecars ``iface collect`` writes
    (needed to regenerate extxyz losslessly) and is restored with
    ``iface package materialize`` before handing it to MACE training.
    """

    source = _resolved(dataset_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")
    flavor, dataset_manifest = _detect_dataset_flavor(source)

    compression_type = _COMPRESSION.get(compression)
    if compression_type is None:
        raise ConfigurationError("compression must be 'deflated' or 'stored'")

    archive = _zip_output(output)
    if archive.exists() and not force:
        raise SafetyError(
            f"Archive already exists: {archive}. Choose a new versioned name or pass force."
        )
    if _inside(archive, source):
        raise SafetyError("The archive must be written outside the dataset directory")

    deepmd_root = source / "deepmd"
    if dedupe:
        if flavor != "iface_collect":
            raise ConfigurationError(
                "dedupe currently requires a dataset written by 'iface collect' -- this "
                f"one was written by {flavor!r} (no move_mask.npy/system_meta.json sidecars "
                "to regenerate extxyz from). Archive it without dedupe instead."
            )
        include_extxyz = False
        _require_materializable(deepmd_root)

    top_file_names = _DATASET_TOP_FILES if flavor == "iface_collect" else _LEAF_DATASET_TOP_FILES
    payload_files: list[tuple[str, Path]] = []
    for name in top_file_names:
        candidate = source / name
        if candidate.is_file():
            payload_files.append((name, candidate))
    if include_extxyz:
        for name in _DATASET_EXTXYZ:
            candidate = source / name
            if candidate.is_file():
                payload_files.append((name, candidate))
    if deepmd_root.is_dir():
        for path in sorted(deepmd_root.rglob("*")):
            if path.is_file():
                payload_files.append((path.relative_to(source).as_posix(), path))
    if not payload_files:
        raise SafetyError(f"No dataset files found under {source}")

    top = archive.stem
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for relative, path in payload_files:
        digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "stored_file": f"data/{relative}",
                "source_file": str(path),
                "sha256": digest,
                "size_bytes": size,
            }
        )

    if_manifest = {
        "schema_version": 1,
        "artifact_type": "mlip_dataset_archive",
        "label": label or top,
        "created_at": utc_now(),
        "interfaceforge_version": __version__,
        "source_root": str(source),
        "dataset_flavor": flavor,
        "include_extxyz": include_extxyz,
        "dedupe": dedupe,
        "materializable_formats": ["extxyz"] if dedupe else [],
        "file_count": len(records),
        "total_bytes": total_bytes,
        "dataset_manifest": dataset_manifest,
        "files": records,
    }
    checksum_body = _sha_lines([(record["stored_file"], record["sha256"]) for record in records])
    readme = _dataset_archive_readme(if_manifest)

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temp_sibling(archive)
    try:
        with zipfile.ZipFile(temporary, "w", compression=compression_type, compresslevel=6) as handle:
            prefix = pathlib.PurePosixPath(top)
            handle.writestr(
                (prefix / "interfaceforge_manifest.json").as_posix(),
                json.dumps(if_manifest, indent=2, sort_keys=True) + "\n",
            )
            handle.writestr((prefix / "checksums.sha256").as_posix(), checksum_body)
            handle.writestr((prefix / "README.md").as_posix(), readme)
            for record in records:
                handle.write(record["source_file"], (prefix / record["stored_file"]).as_posix())
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)

    return {
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "artifact_type": "mlip_dataset_archive",
        "label": if_manifest["label"],
        "file_count": len(records),
        "total_bytes": total_bytes,
        "include_extxyz": include_extxyz,
        "dedupe": dedupe,
    }


def _require_materializable(deepmd_root: Path) -> None:
    """Refuse a deduped archive unless every DeePMD system can regenerate extxyz."""

    if not deepmd_root.is_dir():
        raise ConfigurationError("dedupe requires a deepmd/ tree to materialize extxyz from")
    systems = sorted({path.parent for path in deepmd_root.rglob("type.raw")})
    if not systems:
        raise SafetyError(f"No DeePMD systems found under {deepmd_root}")
    missing = [
        system
        for system in systems
        if not (system / "move_mask.npy").is_file() or not (system / "system_meta.json").is_file()
    ]
    if missing:
        names = ", ".join(str(system.relative_to(deepmd_root)) for system in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise SafetyError(
            "dedupe requires move_mask.npy and system_meta.json in every DeePMD system; "
            f"missing for: {names}{more}. Re-run 'iface collect' with this InterfaceForge "
            "version, or archive with dedupe=False."
        )


def _dataset_archive_readme(manifest: dict[str, Any]) -> str:
    dataset = manifest["dataset_manifest"]
    counts = dataset.get("frame_counts", {})
    type_map = ", ".join(dataset.get("type_map", []) or []) or "not recorded"
    strategy = dataset.get("strategy") or dataset.get("split_mode", "?")
    ratios = dataset.get("ratios", "?")
    stride = dataset.get("stride", "?")
    frames = (
        f"train {counts.get('train', '?')}, valid {counts.get('valid', '?')}, "
        f"test {counts.get('test', '?')}"
    )
    size_mb = manifest["total_bytes"] / 1e6
    dedupe = manifest.get("dedupe", False)
    mode_note = (
        "**Deduped**: only the DeePMD NPY tree is stored (full float64 precision); "
        "`*.extxyz` is not included because it would be a second, lower-precision copy "
        "of the same positions/forces. Regenerate it with `iface package materialize` "
        "below before MACE training -- DeePMD training can use `data/deepmd/` as-is."
        if dedupe
        else "Mirrors `iface collect`'s output exactly: both `*.extxyz` and the DeePMD "
        "NPY tree are stored (the same frames, twice, once per format)."
    )
    restore_extra = (
        """

## Materialize MACE extxyz (deduped archives only)

```bash
iface package materialize <extracted-top-dir>
```

Regenerates `train.extxyz` / `valid.extxyz` / `test.extxyz` from the DeePMD
NPY + `move_mask.npy` / `system_meta.json` sidecars, using a full-precision
writer (not ASE's ~8-decimal default), and verifies every regenerated frame
against the source arrays before writing anything -- a mismatch aborts rather
than publishing a silently wrong file."""
        if dedupe
        else ""
    )
    return f"""# {manifest['label']}

InterfaceForge canonical training dataset, archived {manifest['created_at']} with
InterfaceForge {manifest['interfaceforge_version']} for cold storage and reuse.

{mode_note}

- Source: `{manifest['source_root']}`
- Split strategy: `{strategy}` (ratios `{ratios}`, stride `{stride}`)
- Frames: {frames}
- Type map: {type_map}
- Files: {manifest['file_count']} ({size_mb:.1f} MB uncompressed)

## Restore

```bash
unzip <this-archive>.zip
```

`data/` is a byte-for-byte canonical dataset directory. Repoint the campaign's
MACE `train_file` / `valid_file` / `test_file` and DeePMD `dataset_root` at the
restored `data/` (and `data/deepmd/`), then:

```bash
iface train mace   -c campaign.yaml
iface train deepmd -c campaign.yaml
```
{restore_extra}

## Verify

```bash
iface package verify <this-archive>.zip
# or, from the extracted top directory:
sha256sum -c checksums.sha256
```

Reference forces are raw DFT labels; constrained-atom mobility is stored
separately as `move_mask`. This archive contains no POTCAR or OUTCAR files.
"""


# --------------------------------------------------------------------------- #
# materialize: regenerate MACE extxyz from a DeePMD-only dataset
# --------------------------------------------------------------------------- #
def _format_float(value: Any) -> str:
    # repr() is the shortest decimal string that round-trips to the exact same
    # float64, unlike ASE's default extxyz writer (~8 decimal places).
    return repr(float(value))


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _extxyz_frame_text(
    *,
    symbols: Sequence[str],
    positions: np.ndarray,
    forces: np.ndarray,
    move_mask: np.ndarray,
    cell: np.ndarray,
    energy: float,
    virial: np.ndarray | None,
    info: dict[str, Any],
) -> str:
    """Render one extxyz frame with exact float64 round-trip precision."""

    natoms = len(symbols)
    lattice = " ".join(_format_float(v) for v in np.asarray(cell, dtype=np.float64).reshape(-1))
    header = [
        f'Lattice="{lattice}"',
        "Properties=species:S:1:pos:R:3:REF_forces:R:3:move_mask:I:1",
        f"REF_energy={_format_float(energy)}",
    ]
    for key, value in info.items():
        if value is None:
            continue
        if isinstance(value, bool):
            header.append(f"{key}={'T' if value else 'F'}")
        elif isinstance(value, (int, np.integer)):
            header.append(f"{key}={int(value)}")
        elif isinstance(value, (float, np.floating)):
            header.append(f"{key}={_format_float(value)}")
        else:
            escaped = str(value).replace('"', "'")
            header.append(f'{key}="{escaped}"')
    if virial is not None:
        vvals = " ".join(_format_float(v) for v in np.asarray(virial, dtype=np.float64).reshape(-1))
        header.append(f'REF_virial="{vvals}"')
    header.append('pbc="T T T"')

    lines = [str(natoms), " ".join(header)]
    for i in range(natoms):
        x, y, z = (float(v) for v in positions[i])
        fx, fy, fz = (float(v) for v in forces[i])
        lines.append(
            f"{symbols[i]} {_format_float(x)} {_format_float(y)} {_format_float(z)} "
            f"{_format_float(fx)} {_format_float(fy)} {_format_float(fz)} {int(move_mask[i])}"
        )
    return "\n".join(lines) + "\n"


def _deepmd_systems(split_dir: Path) -> list[Path]:
    return sorted({path.parent for path in split_dir.rglob("type.raw")})


def _read_deepmd_system(system: Path) -> dict[str, Any]:
    type_map = (system / "type_map.raw").read_text(encoding="utf-8").split()
    atom_types = [int(v) for v in (system / "type.raw").read_text(encoding="utf-8").split()]
    symbols = [type_map[i] for i in atom_types]
    set_dir = system / "set.000"
    coord = np.load(set_dir / "coord.npy")
    box = np.load(set_dir / "box.npy")
    energy = np.load(set_dir / "energy.npy").reshape(-1)
    force = np.load(set_dir / "force.npy")
    nframes, natoms = coord.shape[0], len(symbols)
    virial_path = set_dir / "virial.npy"
    virial = np.load(virial_path) if virial_path.is_file() else None

    warnings: list[str] = []
    mask_path = system / "move_mask.npy"
    if mask_path.is_file():
        move_mask = np.load(mask_path)
    else:
        move_mask = np.ones((nframes, natoms), dtype=np.int8)
        warnings.append(f"{system}: no move_mask.npy; assumed every atom mobile")

    meta_path = system / "system_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            warnings.append(f"{system}: invalid system_meta.json; using defaults")
    else:
        warnings.append(f"{system}: no system_meta.json; using defaults")

    frame_rows: list[dict[str, Any]] = []
    frame_map_path = system / "frame_map.csv"
    if frame_map_path.is_file():
        with frame_map_path.open(newline="", encoding="utf-8") as handle:
            frame_rows = list(csv.DictReader(handle))
    if len(frame_rows) != nframes:
        frame_rows = [{} for _ in range(nframes)]
        warnings.append(f"{system}: frame_map.csv missing or frame-count mismatch; provenance unavailable")

    return {
        "run_id": system.name,
        "symbols": symbols,
        "coord": coord.reshape(nframes, natoms, 3),
        "box": box.reshape(nframes, 3, 3),
        "energy": energy,
        "force": force.reshape(nframes, natoms, 3),
        "virial": virial.reshape(nframes, 3, 3) if virial is not None else None,
        "move_mask": move_mask,
        "meta": meta,
        "frame_rows": frame_rows,
        "warnings": warnings,
    }


def _read_back_move_mask(atoms: Any) -> np.ndarray:
    """Reconstruct the mobility mask ASE moves into ``atoms.constraints`` on read.

    ``move_mask`` is a name ASE's extxyz reader special-cases: it is popped out
    of ``atoms.arrays`` and converted into a real ``FixAtoms``/``FixCartesian``
    constraint (never left as a plain per-atom array), the same convention
    ``data.py``'s own ``_constraint_mask`` reads back.
    """

    mask = np.ones(len(atoms), dtype=np.int8)
    for constraint in atoms.constraints:
        try:
            indices = np.asarray(constraint.get_indices(), dtype=int)
        except (AttributeError, TypeError, ValueError):
            continue
        mask[indices] = 0
    return mask


def _verify_materialized_extxyz(path: Path, expected: Sequence[dict[str, Any]], iread: Any) -> None:
    # format= is explicit because the temp file's name (a .tmp-<pid> suffix, so a
    # publish is atomic) doesn't end in .extxyz/.xyz for ASE to infer from.
    frames = list(iread(str(path), index=":", format="extxyz"))
    if len(frames) != len(expected):
        raise SafetyError(
            f"Materialized {path} has {len(frames)} frames but expected {len(expected)}"
        )
    for index, (atoms, item) in enumerate(zip(frames, expected, strict=True)):
        if not np.array_equal(np.asarray(atoms.positions), item["positions"]):
            raise SafetyError(f"Materialized {path} frame {index}: position round-trip mismatch")
        if not np.array_equal(np.asarray(atoms.cell.array), item["cell"]):
            raise SafetyError(f"Materialized {path} frame {index}: cell round-trip mismatch")
        forces = atoms.arrays.get("REF_forces")
        if forces is None or not np.array_equal(np.asarray(forces), item["forces"]):
            raise SafetyError(f"Materialized {path} frame {index}: force round-trip mismatch")
        mask = _read_back_move_mask(atoms)
        if not np.array_equal(mask, item["move_mask"]):
            raise SafetyError(f"Materialized {path} frame {index}: move_mask round-trip mismatch")
        if float(atoms.info.get("REF_energy", float("nan"))) != item["energy"]:
            raise SafetyError(f"Materialized {path} frame {index}: energy round-trip mismatch")
        if item["virial"] is not None:
            virial = atoms.info.get("REF_virial")
            if virial is None or not np.array_equal(
                np.asarray(virial, dtype=np.float64).reshape(3, 3), item["virial"]
            ):
                raise SafetyError(f"Materialized {path} frame {index}: virial round-trip mismatch")


def materialize_dataset(
    source: str | Path,
    output: str | Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Regenerate MACE ``{split}.extxyz`` files from a DeePMD-only canonical dataset.

    Reads the full-float64 ``coord``/``box``/``energy``/``force``(``/virial``)
    NPY arrays plus the ``move_mask.npy`` / ``system_meta.json`` /
    ``frame_map.csv`` sidecars ``iface collect`` writes, and produces extxyz
    with exact float round-trip precision (Python ``repr``, not ASE's default
    ~8-decimal writer). Every regenerated split is read back with ASE and
    checked frame-by-frame against the source arrays before being published;
    a mismatch raises instead of silently writing a wrong file. A system
    missing its ``move_mask.npy`` / ``system_meta.json`` sidecars degrades to
    the same "unknown -> mobile / unclassified" defaults ``iface collect``
    itself uses, and is reported in ``warnings``.
    """

    dataset = _resolved(source)
    deepmd_root = dataset / "deepmd"
    if not deepmd_root.is_dir():
        raise ConfigurationError(f"No deepmd/ tree under {dataset}; nothing to materialize from")
    destination = _resolved(output) if output is not None else dataset
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from .data import _ase_io
    except ImportError as exc:  # pragma: no cover - defensive
        raise DependencyError("ASE is required to materialize extxyz") from exc
    iread, _ = _ase_io()

    results: dict[str, Any] = {
        "dataset_root": str(dataset),
        "output_root": str(destination),
        "materialized": {},
        "warnings": [],
    }
    for split in ("train", "valid", "test"):
        split_dir = deepmd_root / split
        if not split_dir.is_dir():
            continue
        systems = _deepmd_systems(split_dir)
        if not systems:
            continue
        target = destination / f"{split}.extxyz"
        if target.exists() and not force:
            raise SafetyError(f"Refusing to overwrite existing {target}; pass force=True")

        expected: list[dict[str, Any]] = []
        temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for system in systems:
                    data = _read_deepmd_system(system)
                    results["warnings"].extend(data["warnings"])
                    meta = data["meta"]
                    nframes = len(data["energy"])
                    for index in range(nframes):
                        row = data["frame_rows"][index] if index < len(data["frame_rows"]) else {}
                        info = {
                            "source_run": data["run_id"],
                            "source_path": row.get("source_path"),
                            "source_frame": _maybe_int(row.get("source_frame")),
                            "split": split,
                            "IF_kind": meta.get("kind"),
                            "IF_high_temperature": bool(meta.get("high_temperature", False)),
                            "IF_tebeg_k": meta.get("tebeg_k"),
                            "IF_min_coordination_number": _maybe_int(row.get("min_coordination_number")),
                            "IF_mean_coordination_number": _maybe_float(row.get("mean_coordination_number")),
                        }
                        virial = data["virial"][index] if data["virial"] is not None else None
                        positions = data["coord"][index]
                        forces = data["force"][index]
                        move_mask = data["move_mask"][index]
                        cell = data["box"][index]
                        energy = float(data["energy"][index])
                        handle.write(
                            _extxyz_frame_text(
                                symbols=data["symbols"],
                                positions=positions,
                                forces=forces,
                                move_mask=move_mask,
                                cell=cell,
                                energy=energy,
                                virial=virial,
                                info=info,
                            )
                        )
                        expected.append(
                            {
                                "positions": positions,
                                "forces": forces,
                                "move_mask": move_mask,
                                "cell": cell,
                                "energy": energy,
                                "virial": virial,
                            }
                        )

            _verify_materialized_extxyz(temporary, expected, iread)
            if target.exists():
                target.unlink()
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

        results["materialized"][split] = {
            "path": str(target),
            "frames": len(expected),
            "systems": len(systems),
        }

    if not results["materialized"]:
        raise SafetyError(f"No DeePMD systems found under {deepmd_root}")
    return results


# --------------------------------------------------------------------------- #
# Hugging Face model package
# --------------------------------------------------------------------------- #
def pack_huggingface(
    bundle_root: str | Path,
    output: str | Path,
    *,
    repo_id: str | None = None,
    license_id: str = "mit",
    base_model: str | None = None,
    extra_tags: Sequence[str] = (),
    dataset_repo_id: str | None = None,
    metrics_path: str | Path | None = None,
    make_zip: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build an upload-ready Hugging Face model repo from a committee bundle."""

    bundle = _resolved(bundle_root)
    if bundle.suffix.lower() == ".zip":
        raise ConfigurationError(
            "Point 'iface package huggingface' at an extracted committee bundle "
            "directory, not a .zip archive. Extract it first."
        )
    verification = verify_committee_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    engine = str(manifest.get("engine", "")).lower()
    if engine not in ("mace", "deepmd"):
        raise ConfigurationError(f"Unsupported committee engine for HF packaging: {engine!r}")
    members = manifest.get("members") or []
    if not members:
        raise SafetyError("Committee manifest has no members")

    out = _resolved(output)
    if out.exists() and any(out.iterdir()) and not force:
        raise SafetyError(
            f"Hugging Face output directory is not empty: {out}. Choose a new name."
        )
    archive = _zip_output(out) if make_zip else None
    if archive is not None and archive.exists() and not force:
        raise SafetyError(f"Hugging Face archive already exists: {archive}")
    if _inside(bundle, out) or _inside(out, bundle):
        raise SafetyError("The Hugging Face output must not overlap the source bundle")

    ft_checkpoint = manifest.get("base_checkpoint")
    metrics = _load_metrics(_resolved(metrics_path), engine=engine) if metrics_path else None
    if metrics is None and engine == "deepmd":
        metrics = _autoscan_deepmd_metrics(manifest)

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    published = False
    try:
        (temporary / "models").mkdir()
        model_lines: list[str] = []
        for member in members:
            stored_rel = str(member["stored_model"])
            source_model = bundle / stored_rel
            if not source_model.is_file():
                raise SafetyError(f"Committee bundle is missing {stored_rel}")
            destination = temporary / stored_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_model, destination)
            if sha256_file(destination) != member.get("sha256"):
                raise SafetyError(f"Checksum drift while copying {stored_rel}")
            model_lines.append(stored_rel)
            member_input = member.get("input")
            index = member.get("model_index")
            if member_input and index is not None:
                (temporary / "models" / f"model_{int(index):03d}.input.json").write_text(
                    json.dumps(member_input, indent=2) + "\n", encoding="utf-8"
                )

        (temporary / "committee-models.txt").write_text("\n".join(model_lines) + "\n", encoding="utf-8")
        (temporary / ".gitattributes").write_text(_gitattributes(), encoding="utf-8")
        (temporary / "README.md").write_text(
            _render_model_card(
                engine=engine,
                manifest=manifest,
                verification=verification,
                repo_id=repo_id,
                license_id=license_id,
                base_model=base_model,
                extra_tags=tuple(extra_tags),
                dataset_repo_id=dataset_repo_id,
                metrics=metrics,
                ft_checkpoint=ft_checkpoint,
            ),
            encoding="utf-8",
        )
        (temporary / "interfaceforge_manifest.json").write_text(
            json.dumps(
                _hf_provenance(
                    engine=engine,
                    manifest=manifest,
                    verification=verification,
                    repo_id=repo_id,
                    license_id=license_id,
                    base_model=base_model,
                    dataset_repo_id=dataset_repo_id,
                    metrics=metrics,
                    ft_checkpoint=ft_checkpoint,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "UPLOAD.md").write_text(
            _upload_instructions(repo_id), encoding="utf-8"
        )

        checks: list[tuple[str, str]] = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                checks.append((path.relative_to(temporary).as_posix(), sha256_file(path)))
        (temporary / "checksums.sha256").write_text(_sha_lines(checks), encoding="utf-8")

        if out.exists():
            shutil.rmtree(out)
        temporary.rename(out)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)

    result: dict[str, Any] = {
        "output": str(out),
        "artifact_type": "huggingface_model",
        "engine": engine,
        "architecture": manifest.get("architecture"),
        "model_count": len(members),
        "repo_id": repo_id,
        "has_metrics": metrics is not None,
        "readme": str(out / "README.md"),
        "upload_instructions": str(out / "UPLOAD.md"),
    }
    if archive is not None:
        _write_dir_zip(out, archive, out.name)
        result["archive"] = str(archive)
        result["archive_sha256"] = sha256_file(archive)
    return result


def _gitattributes() -> str:
    return "\n".join(f"{pattern} filter=lfs diff=lfs merge=lfs -text" for pattern in _LFS_PATTERNS) + "\n"


def _upload_instructions(repo_id: str | None) -> str:
    rid = repo_id or "<your-org>/<model-name>"
    return f"""# Uploading to the Hugging Face Hub

InterfaceForge does not contact the Hub. This directory is ready to push as-is.

1. Install the client and sign in (once):

```bash
pip install -U huggingface_hub
hf auth login          # older clients: huggingface-cli login
```

2. Create the model repo (once) and upload this directory:

```bash
hf repo create {rid} --repo-type model
hf upload {rid} . --repo-type model     # older clients: huggingface-cli upload {rid} .
```

`.gitattributes` already routes weight files (`*.model`, `*.pth`, `*.pb`, ...)
through Git LFS. Review `README.md` (the model card) and replace anything marked
`TODO` before uploading.
"""


def _load_metrics(path: Path, *, engine: str | None = None) -> dict[str, Any] | None:
    """Read committee energy/force RMSE from an mlip-compare or deepmd-audit report."""

    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid metrics JSON: {path}") from exc
        rows = [
            row
            for row in payload.get("headline", [])
            if row.get("energy_rmse_mev_per_atom") is not None
            or row.get("force_rmse_mev_per_angstrom") is not None
        ]
        if engine == "mace":
            rows = [row for row in rows if "mace" in str(row.get("engine", "")).lower()] or rows
        elif engine == "deepmd":
            rows = [row for row in rows if "mace" not in str(row.get("engine", "")).lower()] or rows
        if rows:
            row = rows[0]
            return {
                "energy_rmse_mev_per_atom": row.get("energy_rmse_mev_per_atom"),
                "force_rmse_mev_per_angstrom": row.get("force_rmse_mev_per_angstrom"),
                "source": str(path),
                "engine": row.get("engine"),
            }
        return None

    import csv

    energies: list[float] = []
    forces: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("energy_rmse_mev_per_atom", "energy_rmse_meV_per_atom"):
                if row.get(key):
                    energies.append(float(row[key]))
                    break
            for key in ("force_rmse_mev_per_angstrom", "force_rmse_meV_per_angstrom"):
                if row.get(key):
                    forces.append(float(row[key]))
                    break
    if not energies and not forces:
        return None
    return {
        "energy_rmse_mev_per_atom": sum(energies) / len(energies) if energies else None,
        "force_rmse_mev_per_angstrom": sum(forces) / len(forces) if forces else None,
        "source": str(path),
        "committee_members": max(len(energies), len(forces)),
    }


def _autoscan_deepmd_metrics(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort: the newest ``evaluation/<arch>/job_*/rmse_overall.csv``."""

    source_root = manifest.get("source_root")
    architecture = manifest.get("architecture")
    if not source_root or not architecture:
        return None
    eval_root = Path(str(source_root)).parent / "evaluation" / str(architecture)
    if not eval_root.is_dir():
        return None
    jobs = sorted(
        (path for path in eval_root.glob("job_*") if (path / "rmse_overall.csv").is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not jobs:
        return None
    try:
        return _load_metrics(jobs[-1] / "rmse_overall.csv")
    except (ConfigurationError, ValueError):
        return None


def _model_index_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    energy = metrics.get("energy_rmse_mev_per_atom")
    force = metrics.get("force_rmse_mev_per_angstrom")
    if energy is not None:
        rows.append({"type": "rmse", "value": round(float(energy), 4), "name": "Energy RMSE (meV/atom)"})
    if force is not None:
        rows.append({"type": "rmse", "value": round(float(force), 4), "name": "Force RMSE (meV/Angstrom)"})
    return rows


def _hf_provenance(
    *,
    engine: str,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    repo_id: str | None,
    license_id: str,
    base_model: str | None,
    dataset_repo_id: str | None,
    metrics: dict[str, Any] | None,
    ft_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "huggingface_model",
        "created_at": utc_now(),
        "interfaceforge_version": __version__,
        "engine": engine,
        "architecture": manifest.get("architecture"),
        "backend": manifest.get("backend"),
        "repo_id": repo_id,
        "license": license_id,
        "base_model": base_model,
        "base_checkpoint": ft_checkpoint,
        "dataset_repo_id": dataset_repo_id,
        "type_map": manifest.get("type_map"),
        "campaign": manifest.get("campaign"),
        "label": manifest.get("label"),
        "source_bundle": {
            "path": str(verification.get("bundle")),
            "bundle_sha256": manifest.get("bundle_sha256"),
            "source_root": manifest.get("source_root"),
            "model_pattern": manifest.get("model_pattern"),
        },
        "members": [
            {
                "seed": member.get("seed"),
                "model_index": member.get("model_index"),
                "stored_model": member.get("stored_model"),
                "sha256": member.get("sha256"),
                "size_bytes": member.get("size_bytes"),
            }
            for member in manifest.get("members", [])
        ],
        "training_data": manifest.get("training_data", []),
        "metrics": metrics,
    }


def _render_model_card(
    *,
    engine: str,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    repo_id: str | None,
    license_id: str,
    base_model: str | None,
    extra_tags: Sequence[str],
    dataset_repo_id: str | None,
    metrics: dict[str, Any] | None,
    ft_checkpoint: str | None,
) -> str:
    import yaml

    label = manifest.get("label") or repo_id or "InterfaceForge committee"
    architecture = manifest.get("architecture")

    tags = list(_MACE_TAGS if engine == "mace" else _DEEPMD_TAGS)
    if architecture:
        tags.append(str(architecture).replace("_", "-"))
    if architecture and str(architecture).endswith("_ft"):
        tags.append("fine-tuned")
    tags.extend(_COMMON_TAGS)
    tags.extend(extra_tags)
    tags = _dedupe(tags)

    frontmatter: dict[str, Any] = {
        "library_name": "mace" if engine == "mace" else "deepmd-kit",
        "tags": tags,
        "license": license_id,
    }
    if base_model:
        frontmatter["base_model"] = base_model
    if dataset_repo_id:
        frontmatter["datasets"] = [dataset_repo_id]
    if metrics and _model_index_metrics(metrics):
        frontmatter["model-index"] = [
            {
                "name": repo_id or label,
                "results": [
                    {
                        "task": {"type": "force-field", "name": "Interatomic potential energy/forces"},
                        "metrics": _model_index_metrics(metrics),
                    }
                ],
            }
        ]

    rendered = yaml.safe_dump(frontmatter, sort_keys=False, default_flow_style=False).strip()
    body = _card_body(
        engine=engine,
        manifest=manifest,
        verification=verification,
        metrics=metrics,
        ft_checkpoint=ft_checkpoint,
        label=label,
        repo_id=repo_id,
        dataset_repo_id=dataset_repo_id,
    )
    return f"---\n{rendered}\n---\n\n{body}"


def _member_rows(manifest: dict[str, Any]) -> str:
    lines = ["| Member | Seed | File | Size | SHA-256 (first 12) |", "|---|---|---|---|---|"]
    for position, member in enumerate(manifest.get("members", [])):
        index = member.get("model_index")
        if index is None:
            index = member.get("index", position)
        stored = str(member.get("stored_model", "")).split("/")[-1]
        size = member.get("size_bytes") or 0
        digest = str(member.get("sha256", ""))[:12]
        lines.append(
            f"| {index} | {member.get('seed', '')} | `{stored}` "
            f"| {size / 1e6:.1f} MB | `{digest}` |"
        )
    return "\n".join(lines)


def _card_body(
    *,
    engine: str,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    metrics: dict[str, Any] | None,
    ft_checkpoint: str | None,
    label: str,
    repo_id: str | None,
    dataset_repo_id: str | None,
) -> str:
    engine_name = "MACE" if engine == "mace" else "DeePMD"
    architecture = manifest.get("architecture")
    model_count = len(manifest.get("members", []))
    type_map = manifest.get("type_map") or []
    sections: list[str] = []

    heading = f"# {label}"
    sections.append(heading)

    descriptor = f"{engine_name}"
    if architecture:
        descriptor += f" ({architecture})"
    sections.append(
        f"A {model_count}-member {descriptor} committee interatomic potential produced with "
        f"[InterfaceForge](https://github.com/lgutsev/InterfaceForge) {__version__}. "
        "Committee members differ only by random seed; evaluate all of them and use the "
        "mean prediction, with the spread as a rough epistemic-uncertainty signal."
    )

    if ft_checkpoint:
        sections.append(
            f"**Fine-tuned** from foundation checkpoint `{ft_checkpoint}` "
            "(`--use-pretrain-script`: descriptor and fitting-net shapes come from the checkpoint)."
        )

    sections.append("## Committee members\n\n" + _member_rows(manifest))

    protocol: list[str] = ["## Training"]
    protocol.append(f"- Engine: {engine_name}" + (f" / architecture `{architecture}`" if architecture else ""))
    if manifest.get("backend"):
        protocol.append(f"- Backend: `{manifest['backend']}`")
    if manifest.get("numb_steps"):
        protocol.append(f"- Training steps: {manifest['numb_steps']}")
    seeds = [member.get("seed") for member in manifest.get("members", []) if member.get("seed") is not None]
    if seeds:
        protocol.append(f"- Seeds: {', '.join(str(seed) for seed in seeds)}")
    protocol.append(
        "- Reference labels: raw DFT energies and forces (constraints stored as `move_mask`, "
        "not applied to force labels)."
    )
    protocol.append(
        "- Splits assign whole trajectories, so nearby MD frames do not leak across train/valid/test."
    )
    sections.append("\n".join(protocol))

    data: list[str] = ["## Data and provenance"]
    if type_map:
        data.append(f"- Elements (`type_map`): {', '.join(type_map)}")
    if dataset_repo_id:
        data.append(f"- Dataset: [`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id})")
    if manifest.get("campaign"):
        data.append(f"- Campaign: `{manifest['campaign']}`")
    if manifest.get("source_root"):
        data.append(f"- Source training runs: `{manifest['source_root']}`")
    if manifest.get("training_data"):
        names = ", ".join(Path(str(item.get("path", ""))).name for item in manifest["training_data"])
        data.append(f"- Training-data provenance files: {names}")
    data.append(f"- Bundle digest: `{manifest.get('bundle_sha256', '')}`")
    sections.append("\n".join(data))

    sections.append(_load_snippet(engine, manifest))

    if metrics:
        rows = ["## Evaluation", "", "| Metric | Value |", "|---|---:|"]
        if metrics.get("energy_rmse_mev_per_atom") is not None:
            rows.append(f"| Energy RMSE | {float(metrics['energy_rmse_mev_per_atom']):.3f} meV/atom |")
        if metrics.get("force_rmse_mev_per_angstrom") is not None:
            rows.append(f"| Force RMSE | {float(metrics['force_rmse_mev_per_angstrom']):.3f} meV/Å |")
        rows.append("")
        rows.append(f"Source: `{metrics.get('source', 'supplied metrics file')}`. ")
        rows.append(
            "These are in-distribution errors on the campaign's held-out test frames, "
            "not a transferability result."
        )
        sections.append("\n".join(rows))
    else:
        sections.append(
            "## Evaluation\n\nNo held-out metrics were attached at packaging time. "
            "Pass `--metrics <comparison.json|rmse_overall.csv>` to `iface package huggingface` "
            "to embed them, or add them here manually (`TODO`)."
        )

    sections.append(f"## Scientific maturity\n\n{_MATURITY_NOTE}")

    citation = (
        "## Citation\n\n"
        "If this potential contributes to published work, cite the underlying method "
        f"({'MACE' if engine == 'mace' else 'DeePMD-kit / DPA'}), your DFT reference dataset, "
        "and InterfaceForge:\n\n"
        "```bibtex\n"
        "@software{interfaceforge,\n"
        "  title = {InterfaceForge},\n"
        "  author = {Gutsev, Lavrenty G.},\n"
        "  url = {https://github.com/lgutsev/InterfaceForge}\n"
        "}\n"
        "```"
    )
    sections.append(citation)

    sections.append(
        "## Provenance\n\n"
        f"- InterfaceForge {__version__}\n"
        f"- Packaged: {utc_now()}\n"
        f"- Committee bundle verified: {verification.get('valid', False)} "
        f"({verification.get('model_count', model_count)} members)\n"
        + (f"- Target repo: `{repo_id}`\n" if repo_id else "")
    )

    return "\n\n".join(sections) + "\n"


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def _sha256_zip_member(handle: zipfile.ZipFile, name: str, chunk: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with handle.open(name, "r") as member:
        while block := member.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _parse_checksums(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not name:
            digest, _, name = line.partition(" ")
        if digest and name:
            entries.append((name.strip(), digest.strip()))
    return entries


def verify_package(path: str | Path) -> dict[str, Any]:
    """Verify any InterfaceForge archive or package by ``artifact_type`` + checksums.

    Recognises ``mlip_dataset_archive`` and ``huggingface_model``; delegates
    committee bundles / training-data archives to
    :func:`interfaceforge.committee.verify_committee_bundle`.
    """

    target = _resolved(path)
    if target.suffix.lower() == ".zip":
        return _verify_zip(target)
    if not target.is_dir():
        raise FileNotFoundError(f"Package not found: {target}")

    if_manifest = target / "interfaceforge_manifest.json"
    if not if_manifest.is_file():
        return verify_committee_bundle(target)
    try:
        manifest = json.loads(if_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SafetyError(f"Invalid package manifest JSON: {if_manifest}") from exc

    artifact_type = manifest.get("artifact_type")
    if artifact_type not in {"mlip_dataset_archive", "huggingface_model"}:
        return verify_committee_bundle(target)

    checksum_file = target / "checksums.sha256"
    if not checksum_file.is_file():
        raise SafetyError(f"Package has no checksums.sha256: {target}")
    entries = _parse_checksums(checksum_file.read_text(encoding="utf-8"))
    if not entries:
        raise SafetyError(f"checksums.sha256 is empty: {target}")

    for name, expected in entries:
        member = (target / name).resolve()
        if not _inside(member, target) or not member.is_file():
            raise SafetyError(f"Missing or unsafe file listed in checksums: {name}")
        if sha256_file(member) != expected:
            raise SafetyError(f"Checksum mismatch: {name}")

    if artifact_type == "mlip_dataset_archive" and manifest.get("file_count") != len(
        [name for name, _ in entries if name.startswith("data/")]
    ):
        raise SafetyError("Dataset archive file_count does not match its checksums list")
    if artifact_type == "huggingface_model":
        readme = target / "README.md"
        if not readme.is_file() or not readme.read_text(encoding="utf-8").startswith("---\n"):
            raise SafetyError("Hugging Face package README.md has no YAML frontmatter")
        if not (target / ".gitattributes").is_file():
            raise SafetyError("Hugging Face package is missing .gitattributes")

    return {
        "package": str(target),
        "kind": "directory",
        "artifact_type": artifact_type,
        "valid": True,
        "file_count": len(entries),
        "label": manifest.get("label"),
    }


def _verify_zip(archive: Path) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            names = [name for name in handle.namelist() if not name.endswith("/")]
            paths = [pathlib.PurePosixPath(name) for name in names]
            if any(path.is_absolute() or ".." in path.parts for path in paths):
                raise SafetyError(f"Unsafe path in archive: {archive}")
            manifest_names = [
                path
                for path in paths
                if len(path.parts) == 2 and path.name == "interfaceforge_manifest.json"
            ]
            if len(manifest_names) != 1:
                # not one of ours; committee.py handles mlip_committee / training_data zips
                return verify_committee_bundle(archive)
            top = manifest_names[0].parts[0]
            manifest = json.loads(handle.read(manifest_names[0].as_posix()))
            artifact_type = manifest.get("artifact_type")
            if artifact_type not in {"mlip_dataset_archive", "huggingface_model"}:
                return verify_committee_bundle(archive)

            checksum_name = (pathlib.PurePosixPath(top) / "checksums.sha256").as_posix()
            if checksum_name not in names:
                raise SafetyError(f"Archive has no checksums.sha256: {archive}")
            entries = _parse_checksums(handle.read(checksum_name).decode("utf-8"))
            if not entries:
                raise SafetyError(f"checksums.sha256 is empty in {archive}")
            known = set(names)
            for name, expected in entries:
                member = (pathlib.PurePosixPath(top) / name).as_posix()
                if member not in known:
                    raise SafetyError(f"Missing file listed in checksums: {name}")
                if _sha256_zip_member(handle, member) != expected:
                    raise SafetyError(f"Checksum mismatch in archive: {name}")
    except zipfile.BadZipFile as exc:
        raise SafetyError(f"Invalid archive: {archive}") from exc
    except json.JSONDecodeError as exc:
        raise SafetyError(f"Invalid package manifest JSON in {archive}") from exc

    return {
        "package": str(archive),
        "kind": "zip",
        "artifact_type": artifact_type,
        "valid": True,
        "file_count": len(entries),
        "label": manifest.get("label"),
        "archive_sha256": sha256_file(archive),
    }


def _load_snippet(engine: str, manifest: dict[str, Any]) -> str:
    members = manifest.get("members", [])
    stored = [str(member.get("stored_model", "")) for member in members]
    if engine == "mace":
        listed = ", ".join(f'"{name}"' for name in stored) or '"models/seed_0.model"'
        return (
            "## How to load\n\n"
            "```python\n"
            "from mace.calculators import MACECalculator\n\n"
            f"calc = MACECalculator(model_paths=[{listed}], device=\"cuda\")\n"
            "atoms.calc = calc\n"
            "energy = atoms.get_potential_energy()   # committee mean\n"
            "```\n\n"
            "`MACECalculator` with several `model_paths` returns the committee mean and "
            "exposes the per-model spread."
        )
    first = stored[0] if stored else "models/model_000.pth"
    model_args = " ".join(stored) or "models/model_000.pth"
    return (
        "## How to load\n\n"
        "```python\n"
        "from deepmd.infer import DeepPot\n\n"
        f'dp = DeepPot("{first}")\n'
        "e, f, v = dp.eval(coords, cells, atom_types)\n"
        "```\n\n"
        "Committee deviation across all members:\n\n"
        "```bash\n"
        f"dp model-devi -m {model_args} -s <deepmd-system> -o model_devi.out\n"
        "```"
    )


# --------------------------------------------------------------------------- #
# campaign-wide: everything a campaign has, in one call
# --------------------------------------------------------------------------- #
def pack_campaign(
    campaign: Campaign,
    *,
    output_root: str | Path | None = None,
    mace_committee_root: str | Path | None = None,
    deepmd_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    repo_prefix: str | None = None,
    license_id: str = "mit",
    expected_members: int = 4,
    dedupe: bool = False,
    include_huggingface: bool = True,
    include_dataset_archive: bool = True,
    tag: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Collect + package everything a campaign has, in the layout the rest of
    InterfaceForge already writes to -- one call instead of one
    ``committee collect`` / ``package huggingface`` pair per committee plus a
    separate ``dataset-archive``.

    Looks for (each optional; missing pieces are reported in ``skipped``, not
    treated as failures):

    * the canonical dataset at ``dataset_root`` (default
      ``<campaign>/datasets/canonical``, the ``iface collect`` /
      ``iface-mapped-collect`` default);
    * a MACE base and/or fine-tune committee under ``mace_committee_root``
      (default ``<campaign>/models/mace_committee_520eV``, the same default
      ``iface mlip-progress`` uses) as ``mace_committee`` /
      ``mace_finetune_committee``;
    * a DeePMD committee under every architecture directory that actually has
      one found directly under ``deepmd_root`` (default
      ``<campaign>/models/deepmd``, the ``iface train deepmd`` layout) --
      any subdirectory containing at least one ``model_NNN/`` run. This does
      not require ``models.deepmd.enabled`` or an ``architectures`` list in
      ``campaign.yaml``: a trained committee on disk is packaged whether or
      not the campaign file that generated it is still around or current,
      the same way the MACE side is found by directory, not by config.

    One committee failing (a wrong ``--expected-members``, or a bundle name
    that already exists -- committee bundles are immutable, so re-running with
    the same ``--tag`` does not silently skip) is recorded in ``errors`` and
    does not stop the rest: the point of one call is not losing everything
    else to one mistake. Returns a payload with
    ``dataset_archive``, ``committees`` (each with its collected bundle and,
    unless ``include_huggingface=False``, its packaged Hugging Face repo),
    ``skipped``, and ``errors``.
    """

    out_root = _resolved(output_root) if output_root is not None else campaign.root / "packaged"
    mace_root = (
        _resolved(mace_committee_root)
        if mace_committee_root is not None
        else campaign.root / "models" / "mace_committee_520eV"
    )
    deepmd_models_root = (
        _resolved(deepmd_root) if deepmd_root is not None else campaign.root / "models" / "deepmd"
    )
    dataset_dir = (
        _resolved(dataset_root) if dataset_root is not None else campaign.root / "datasets" / "canonical"
    )
    suffix = f"_{tag}" if tag else ""

    result: dict[str, Any] = {
        "campaign": campaign.name,
        "output_root": str(out_root),
        "dataset_archive": None,
        "committees": [],
        "skipped": [],
        "errors": [],
    }

    def _skip(step: str, reason: str) -> None:
        result["skipped"].append({"step": step, "reason": reason})

    def _fail(step: str, exc: Exception) -> None:
        result["errors"].append({"step": step, "detail": str(exc)})

    def _repo_id(component: str) -> str | None:
        return f"{repo_prefix}-{component}" if repo_prefix else None

    def _collect_and_package(step: str, source: Path, *, engine: str, component: str) -> None:
        if not source.is_dir():
            _skip(step, f"not found: {source}")
            return
        bundle_out = out_root / "stored_models" / f"{campaign.name}_{component}{suffix}"
        try:
            # collect_committee has no force= -- bundles are deliberately immutable;
            # an existing bundle name surfaces as an error here, not a silent skip.
            collected = collect_committee(
                source,
                bundle_out,
                engine=engine,
                expected_members=expected_members,
                label=f"{campaign.name} {component}",
            )
        except Exception as exc:  # noqa: BLE001 - one bad committee must not abort the sweep
            _fail(step, exc)
            return
        entry: dict[str, Any] = {"engine": engine, "component": component, "bundle": collected}
        if include_huggingface:
            try:
                entry["huggingface"] = pack_huggingface(
                    collected["bundle"],
                    out_root / "hf" / f"{campaign.name}_{component}{suffix}",
                    repo_id=_repo_id(component),
                    license_id=license_id,
                    force=force,
                )
            except Exception as exc:  # noqa: BLE001
                _fail(f"{step}_huggingface", exc)
        result["committees"].append(entry)

    if include_dataset_archive:
        if dataset_dir.is_dir():
            try:
                result["dataset_archive"] = pack_dataset_archive(
                    dataset_dir,
                    out_root / "backups" / f"{campaign.name}_dataset{suffix}.zip",
                    dedupe=dedupe,
                    force=force,
                )
            except Exception as exc:  # noqa: BLE001
                _fail("dataset_archive", exc)
        else:
            _skip("dataset_archive", f"no dataset at {dataset_dir}")

    _collect_and_package("mace_committee", mace_root / "mace_committee", engine="mace", component="mace")
    _collect_and_package(
        "mace_finetune_committee",
        mace_root / "mace_finetune_committee",
        engine="mace",
        component="mace-ft",
    )

    architectures = _discover_deepmd_architectures(deepmd_models_root)
    if not architectures:
        _skip(
            "deepmd",
            f"no model_NNN/ committees found directly under {deepmd_models_root}",
        )
    for architecture in architectures:
        _collect_and_package(
            f"deepmd_{architecture}",
            deepmd_models_root / architecture,
            engine="deepmd",
            component=architecture,
        )

    return result


def _discover_deepmd_architectures(deepmd_root: Path) -> list[str]:
    """Every immediate subdirectory of ``deepmd_root`` with a ``model_NNN/``
    run directly under it -- i.e. a trained committee, found by what is
    actually on disk rather than ``campaign.yaml``'s ``models.deepmd``
    config. Excludes ``evaluation/`` and ``smoke/``, which hold per-job
    subtrees rather than ``model_NNN/`` runs directly.
    """

    if not deepmd_root.is_dir():
        return []

    def _has_model_run(directory: Path) -> bool:
        return any(child.is_dir() and child.name.startswith("model_") for child in directory.iterdir())

    return sorted(
        child.name for child in deepmd_root.iterdir() if child.is_dir() and _has_model_run(child)
    )
