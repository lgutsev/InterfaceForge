"""Collect completed MLIP committee members into immutable deployment bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ConfigurationError, SafetyError
from .state import sha256_file, utc_now


_SEED_NAME = re.compile(r"^seed[_-](?P<seed>-?\d+)$")
_RUN_ARTIFACTS = ("results", "mace_model", "checkpoints", "logs")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _output_paths(requested: str | Path) -> tuple[Path, Path]:
    value = _resolved(requested)
    if value.suffix.lower() == ".zip":
        bundle = value.with_suffix("")
        archive = value
    else:
        bundle = value
        archive = Path(f"{value}.zip")
    if not bundle.name:
        raise ConfigurationError("Committee output must have a non-empty bundle name")
    return bundle, archive


def _zip_path(requested: str | Path) -> Path:
    value = _resolved(requested)
    return value if value.suffix.lower() == ".zip" else Path(f"{value}.zip")


def _temporary_archive(parent: Path, name: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.tmp-", dir=parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _discover_mace_models(source: Path, pattern: str) -> list[tuple[int, str, Path]]:
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ConfigurationError("The committee model pattern must stay inside the source directory")

    discovered: list[tuple[int, str, Path]] = []
    seen_runs: set[str] = set()
    seen_seeds: set[int] = set()
    for candidate in source.glob(pattern):
        model = candidate.resolve()
        if not model.is_file() or not _inside(model, source):
            continue
        relative = model.relative_to(source)
        if not relative.parts:
            continue
        run_name = relative.parts[0]
        match = _SEED_NAME.fullmatch(run_name)
        if match is None:
            continue
        seed = int(match.group("seed"))
        if seed < 0:
            raise SafetyError(f"Committee seed must be non-negative: {run_name}")
        if seed in seen_seeds:
            raise SafetyError(f"Committee seed appears more than once: {seed}")
        if run_name in seen_runs:
            raise SafetyError(
                f"More than one final model matches run {run_name!r}; use a narrower --model-pattern"
            )
        if model.stat().st_size == 0:
            raise SafetyError(f"Committee model is empty: {model}")
        seen_runs.add(run_name)
        seen_seeds.add(seed)
        discovered.append((seed, run_name, model))

    return sorted(discovered, key=lambda item: (item[0], item[1]))


def _bundle_digest(engine: str, members: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(engine.encode("utf-8"))
    for member in members:
        digest.update(b"\0")
        digest.update(str(member["seed"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(member["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _readme(label: str, manifest: dict[str, Any]) -> str:
    models = "\n".join(f"- `{member['stored_model']}`" for member in manifest["members"])
    return f"""# {label}

Immutable InterfaceForge deployment bundle for a {manifest['engine'].upper()} committee.
The training checkpoints and logs remain in their original run directories; this
bundle contains only the final deployable models and their provenance.

## Models

{models}

Verify the bundle before use:

```bash
iface committee verify .
```

`committee-models.txt` lists the model paths relative to this directory.
`checksums.sha256` can also be checked with `sha256sum -c checksums.sha256`.
The complete machine-readable record is `manifest.json`.
"""


def _write_zip(bundle: Path, archive: Path, top_level: str) -> None:
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in files:
            relative = path.relative_to(bundle)
            handle.write(path, (PurePosixPath(top_level) / relative.as_posix()).as_posix())


def _write_training_data_zip(
    records: Sequence[dict[str, Any]],
    archive: Path,
    *,
    top_level: str,
    label: str,
    committee_sha256: str,
    compression: str,
) -> None:
    compression_type = {
        "deflated": zipfile.ZIP_DEFLATED,
        "stored": zipfile.ZIP_STORED,
    }.get(compression)
    if compression_type is None:
        raise ConfigurationError("training_data_compression must be 'deflated' or 'stored'")

    names: set[str] = set()
    files: list[dict[str, Any]] = []
    checksum_lines: list[str] = []
    for record in records:
        source = Path(str(record["path"]))
        stored = (PurePosixPath("data") / source.name).as_posix()
        if stored in names:
            raise SafetyError(
                f"Training-data archive has duplicate basenames: {source.name}; rename the files first"
            )
        names.add(stored)
        files.append(
            {
                "source_file": str(source),
                "stored_file": stored,
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
        checksum_lines.append(f"{record['sha256']}  {stored}")

    manifest = {
        "schema_version": 1,
        "artifact_type": "mlip_training_data",
        "label": label,
        "created_at": utc_now(),
        "committee_bundle_sha256": committee_sha256,
        "file_count": len(files),
        "files": files,
    }
    prefix = PurePosixPath(top_level)
    readme = f"""# {label}

Separate training-data archive associated with committee bundle
`{committee_sha256}`. The committee model ZIP does not contain these files.

Verify without extracting:

```bash
iface committee verify {Path(archive).name}
```
"""
    with zipfile.ZipFile(
        archive, "w", compression=compression_type, compresslevel=6
    ) as handle:
        handle.writestr(
            (prefix / "manifest.json").as_posix(),
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        handle.writestr(
            (prefix / "checksums.sha256").as_posix(), "\n".join(checksum_lines) + "\n"
        )
        handle.writestr((prefix / "README.md").as_posix(), readme)
        for record in files:
            handle.write(
                record["source_file"], (prefix / str(record["stored_file"])).as_posix()
            )


def _sha256_zip_member(
    handle: zipfile.ZipFile, name: str, chunk_size: int = 1024 * 1024
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with handle.open(name, "r") as member:
        while chunk := member.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def collect_committee(
    source_root: str | Path,
    output_root: str | Path,
    *,
    engine: str = "mace",
    expected_members: int = 4,
    model_pattern: str = "seed_*/mace_model/*_stagetwo.model",
    training_data: Sequence[str | Path] = (),
    training_data_output: str | Path | None = None,
    training_data_compression: str = "deflated",
    label: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Copy final committee models into a compact, checksummed bundle.

    The source training runs are read-only. Existing output directories are never
    replaced; use a new output name to preserve the immutability of stored models.
    """

    engine = engine.strip().lower()
    if engine != "mace":
        raise ConfigurationError("Committee collection currently supports engine='mace'")
    if expected_members < 1:
        raise ConfigurationError("expected_members must be positive")

    source = _resolved(source_root)
    output, archive = _output_paths(output_root)
    data_archive = _zip_path(training_data_output) if training_data_output is not None else None
    if not source.is_dir():
        raise FileNotFoundError(f"Committee source directory not found: {source}")
    if data_archive is not None and not training_data:
        raise ConfigurationError("--training-data-output requires at least one --training-data file")
    if data_archive is not None and data_archive == archive:
        raise ConfigurationError("Committee and training-data ZIP outputs must be different files")
    planned_outputs = [output, archive]
    if data_archive is not None:
        planned_outputs.append(data_archive)
    existing = [path for path in planned_outputs if path.exists()]
    if existing:
        raise SafetyError(
            "Committee bundle output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a new versioned output name."
        )
    if any(path == source or _inside(path, source) for path in planned_outputs):
        raise SafetyError("Committee bundle output must be outside the source run directory")

    discovered = _discover_mace_models(source, model_pattern)
    if len(discovered) != expected_members:
        raise SafetyError(
            f"Expected {expected_members} completed committee members but found {len(discovered)} "
            f"with pattern {model_pattern!r} under {source}"
        )

    source_records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for seed, run_name, model in discovered:
        checksum = sha256_file(model)
        if checksum in hashes:
            raise SafetyError(
                f"Duplicate committee model content detected at {model}; every member must be distinct"
            )
        hashes.add(checksum)
        source_records.append(
            {
                "seed": seed,
                "run_name": run_name,
                "model": model,
                "sha256": checksum,
                "size_bytes": model.stat().st_size,
            }
        )

    data_records: list[dict[str, Any]] = []
    for value in training_data:
        path = _resolved(value)
        if not path.is_file():
            raise FileNotFoundError(f"Training-data provenance file not found: {path}")
        data_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    temporary_archive = _temporary_archive(archive.parent, archive.name)
    temporary_data_archive: Path | None = None
    if data_archive is not None:
        data_archive.parent.mkdir(parents=True, exist_ok=True)
        temporary_data_archive = _temporary_archive(data_archive.parent, data_archive.name)
    published = False
    archive_published = False
    data_archive_published = False
    try:
        model_directory = temporary / "models"
        model_directory.mkdir()
        members: list[dict[str, Any]] = []
        checksum_lines: list[str] = []
        model_lines: list[str] = []
        for index, record in enumerate(source_records):
            stored = Path("models") / f"seed_{record['seed']}.model"
            target = temporary / stored
            shutil.copy2(record["model"], target)
            copied_checksum = sha256_file(target)
            if copied_checksum != record["sha256"]:
                raise SafetyError(f"Checksum changed while copying {record['model']}")
            run_root = source / str(record["run_name"])
            member = {
                "index": index,
                "seed": record["seed"],
                "run_name": record["run_name"],
                "source_model": str(record["model"]),
                "original_filename": Path(record["model"]).name,
                "stored_model": stored.as_posix(),
                "sha256": copied_checksum,
                "size_bytes": target.stat().st_size,
                "source_run_artifacts": {
                    name: (run_root / name).is_dir() for name in _RUN_ARTIFACTS
                },
            }
            members.append(member)
            checksum_lines.append(f"{copied_checksum}  {stored.as_posix()}")
            model_lines.append(stored.as_posix())

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "mlip_committee",
            "engine": engine,
            "label": label or output.name,
            "created_at": utc_now(),
            "source_root": str(source),
            "expected_members": expected_members,
            "model_count": len(members),
            "model_pattern": model_pattern,
            "members": members,
            "training_data": data_records,
            "training_data_archive": str(data_archive) if data_archive is not None else None,
            "notes": notes,
        }
        manifest["bundle_sha256"] = _bundle_digest(engine, members)

        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "committee-models.txt").write_text(
            "\n".join(model_lines) + "\n", encoding="utf-8"
        )
        (temporary / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            _readme(str(manifest["label"]), manifest), encoding="utf-8"
        )
        _write_zip(temporary, temporary_archive, output.name)
        if data_archive is not None and temporary_data_archive is not None:
            _write_training_data_zip(
                data_records,
                temporary_data_archive,
                top_level=data_archive.stem,
                label=f"{manifest['label']} training data",
                committee_sha256=str(manifest["bundle_sha256"]),
                compression=training_data_compression,
            )
        temporary.rename(output)
        published = True
        os.link(temporary_archive, archive)
        archive_published = True
        temporary_archive.unlink()
        if data_archive is not None and temporary_data_archive is not None:
            os.link(temporary_data_archive, data_archive)
            data_archive_published = True
            temporary_data_archive.unlink()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        temporary_archive.unlink(missing_ok=True)
        if temporary_data_archive is not None:
            temporary_data_archive.unlink(missing_ok=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        if archive_published:
            archive.unlink(missing_ok=True)
        if data_archive_published and data_archive is not None:
            data_archive.unlink(missing_ok=True)
        raise

    result = {
        "bundle": str(output),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "manifest": str(output / "manifest.json"),
        "bundle_sha256": manifest["bundle_sha256"],
        "model_count": len(manifest["members"]),
        "models": [str(output / member["stored_model"]) for member in manifest["members"]],
    }
    if data_archive is not None:
        result["training_data_archive"] = str(data_archive)
        result["training_data_archive_sha256"] = sha256_file(data_archive)
    return result


def verify_committee_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Verify all models and metadata in a collected committee bundle."""

    root = _resolved(bundle_root)
    if root.suffix.lower() == ".zip":
        return _verify_committee_zip(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Committee manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SafetyError(f"Invalid committee manifest JSON: {manifest_path}") from exc
    if manifest.get("artifact_type") != "mlip_committee":
        raise SafetyError(f"Not an InterfaceForge MLIP committee bundle: {root}")

    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise SafetyError("Committee manifest does not contain any members")
    if manifest.get("model_count") != len(members):
        raise SafetyError("Committee manifest model_count does not match its member list")

    observed_hashes: set[str] = set()
    for member in members:
        stored = Path(str(member.get("stored_model", "")))
        model = (root / stored).resolve()
        if stored.is_absolute() or not _inside(model, root) or not model.is_file():
            raise SafetyError(f"Missing or unsafe stored committee model path: {stored}")
        checksum = sha256_file(model)
        if checksum != member.get("sha256"):
            raise SafetyError(f"Committee model checksum mismatch: {model}")
        if model.stat().st_size != member.get("size_bytes"):
            raise SafetyError(f"Committee model size mismatch: {model}")
        if checksum in observed_hashes:
            raise SafetyError(f"Duplicate committee model content in bundle: {model}")
        observed_hashes.add(checksum)

    expected_digest = _bundle_digest(str(manifest.get("engine", "")), members)
    if expected_digest != manifest.get("bundle_sha256"):
        raise SafetyError("Committee bundle digest does not match its manifest")

    return {
        "bundle": str(root),
        "kind": "directory",
        "valid": True,
        "engine": manifest.get("engine"),
        "label": manifest.get("label"),
        "model_count": len(members),
        "bundle_sha256": expected_digest,
    }


def _verify_committee_zip(archive: Path) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(f"Committee ZIP archive not found: {archive}")
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            file_names = [name for name in handle.namelist() if not name.endswith("/")]
            paths = [PurePosixPath(name) for name in file_names]
            if any(path.is_absolute() or ".." in path.parts for path in paths):
                raise SafetyError(f"Unsafe path in committee ZIP archive: {archive}")
            manifest_names = [
                path for path in paths if len(path.parts) == 2 and path.name == "manifest.json"
            ]
            if len(manifest_names) != 1:
                raise SafetyError("Committee ZIP must contain one top-level bundle manifest")
            top_level = manifest_names[0].parts[0]
            if any(not path.parts or path.parts[0] != top_level for path in paths):
                raise SafetyError("Committee ZIP entries must share one top-level bundle directory")
            try:
                manifest = json.loads(handle.read(manifest_names[0].as_posix()))
            except json.JSONDecodeError as exc:
                raise SafetyError(f"Invalid committee manifest JSON in {archive}") from exc
            artifact_type = manifest.get("artifact_type")
            if artifact_type == "mlip_training_data":
                return _verify_training_data_zip(
                    handle, archive, manifest, top_level, set(file_names)
                )
            if artifact_type != "mlip_committee":
                raise SafetyError(f"Not an InterfaceForge MLIP committee archive: {archive}")

            members = manifest.get("members")
            if not isinstance(members, list) or not members:
                raise SafetyError("Committee manifest does not contain any members")
            if manifest.get("model_count") != len(members):
                raise SafetyError("Committee manifest model_count does not match its member list")

            known_names = set(file_names)
            observed_hashes: set[str] = set()
            for member in members:
                stored = PurePosixPath(str(member.get("stored_model", "")))
                if stored.is_absolute() or ".." in stored.parts:
                    raise SafetyError(f"Unsafe stored committee model path in ZIP: {stored}")
                member_name = (PurePosixPath(top_level) / stored).as_posix()
                if member_name not in known_names:
                    raise SafetyError(f"Missing committee model in ZIP: {stored}")
                checksum, size = _sha256_zip_member(handle, member_name)
                if checksum != member.get("sha256"):
                    raise SafetyError(f"Committee model checksum mismatch in ZIP: {stored}")
                if size != member.get("size_bytes"):
                    raise SafetyError(f"Committee model size mismatch in ZIP: {stored}")
                if checksum in observed_hashes:
                    raise SafetyError(f"Duplicate committee model content in ZIP: {stored}")
                observed_hashes.add(checksum)

            expected_digest = _bundle_digest(str(manifest.get("engine", "")), members)
            if expected_digest != manifest.get("bundle_sha256"):
                raise SafetyError("Committee bundle digest does not match its ZIP manifest")
    except zipfile.BadZipFile as exc:
        raise SafetyError(f"Invalid committee ZIP archive: {archive}") from exc

    return {
        "bundle": str(archive),
        "kind": "zip",
        "valid": True,
        "engine": manifest.get("engine"),
        "label": manifest.get("label"),
        "model_count": len(members),
        "bundle_sha256": expected_digest,
        "archive_sha256": sha256_file(archive),
    }


def _verify_training_data_zip(
    handle: zipfile.ZipFile,
    archive: Path,
    manifest: dict[str, Any],
    top_level: str,
    known_names: set[str],
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SafetyError("Training-data ZIP manifest does not contain any files")
    if manifest.get("file_count") != len(files):
        raise SafetyError("Training-data ZIP file_count does not match its manifest")

    observed_hashes: set[str] = set()
    for record in files:
        stored = PurePosixPath(str(record.get("stored_file", "")))
        if stored.is_absolute() or ".." in stored.parts:
            raise SafetyError(f"Unsafe stored training-data path in ZIP: {stored}")
        member_name = (PurePosixPath(top_level) / stored).as_posix()
        if member_name not in known_names:
            raise SafetyError(f"Missing training-data file in ZIP: {stored}")
        checksum, size = _sha256_zip_member(handle, member_name)
        if checksum != record.get("sha256"):
            raise SafetyError(f"Training-data checksum mismatch in ZIP: {stored}")
        if size != record.get("size_bytes"):
            raise SafetyError(f"Training-data size mismatch in ZIP: {stored}")
        if checksum in observed_hashes:
            raise SafetyError(f"Duplicate training-data content in ZIP: {stored}")
        observed_hashes.add(checksum)

    return {
        "bundle": str(archive),
        "kind": "training_data_zip",
        "valid": True,
        "label": manifest.get("label"),
        "file_count": len(files),
        "committee_bundle_sha256": manifest.get("committee_bundle_sha256"),
        "archive_sha256": sha256_file(archive),
    }
