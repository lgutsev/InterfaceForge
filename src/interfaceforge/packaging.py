"""Archive trained committees and canonical datasets for reuse and Hugging Face upload.

Two artifacts are produced here:

* ``pack_dataset_archive`` writes one checksummed ``.zip`` of an ``iface collect``
  canonical dataset (extxyz + DeePMD NPY + manifests) for cold storage. ``data/``
  inside the archive is a byte-for-byte dataset directory, so restore is ``unzip``.
* ``pack_huggingface`` turns a verified committee bundle (from
  ``iface committee collect``, MACE or DeePMD) into an upload-ready Hugging Face
  model repository: a generated model card with YAML frontmatter, ``.gitattributes``
  for Git LFS, a provenance manifest, checksums, and the exact ``hf upload``
  command.

InterfaceForge never contacts the Hugging Face Hub. It stops at a ready-to-push
directory; the user runs ``hf upload`` themselves.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .committee import verify_committee_bundle
from .errors import ConfigurationError, SafetyError
from .state import sha256_file, utc_now

_COMPRESSION = {"deflated": zipfile.ZIP_DEFLATED, "stored": zipfile.ZIP_STORED}
_LFS_PATTERNS = ("*.model", "*.pth", "*.pt", "*.pt2", "*.pb", "*.npy", "*.extxyz", "*.xyz")
_DATASET_TOP_FILES = ("manifest.json", "manifest.csv", "frames.csv")
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


# --------------------------------------------------------------------------- #
# dataset archive (task: back up training data for reuse)
# --------------------------------------------------------------------------- #
def pack_dataset_archive(
    dataset_root: str | Path,
    output: str | Path,
    *,
    include_extxyz: bool = True,
    compression: str = "deflated",
    label: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one checksummed ZIP of an ``iface collect`` canonical dataset."""

    source = _resolved(dataset_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(
            f"Not an 'iface collect' dataset (no manifest.json): {source}"
        )
    try:
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid dataset manifest JSON: {manifest_path}") from exc
    if "frame_counts" not in dataset_manifest and "deepmd" not in dataset_manifest:
        raise ConfigurationError(
            f"{manifest_path} does not look like an 'iface collect' manifest"
        )

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

    payload_files: list[tuple[str, Path]] = []
    for name in _DATASET_TOP_FILES:
        candidate = source / name
        if candidate.is_file():
            payload_files.append((name, candidate))
    if include_extxyz:
        for name in _DATASET_EXTXYZ:
            candidate = source / name
            if candidate.is_file():
                payload_files.append((name, candidate))
    deepmd_root = source / "deepmd"
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
        "include_extxyz": include_extxyz,
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
    }


def _dataset_archive_readme(manifest: dict[str, Any]) -> str:
    dataset = manifest["dataset_manifest"]
    counts = dataset.get("frame_counts", {})
    type_map = ", ".join(dataset.get("type_map", []) or []) or "not recorded"
    strategy = dataset.get("strategy", "?")
    ratios = dataset.get("ratios", "?")
    stride = dataset.get("stride", "?")
    frames = (
        f"train {counts.get('train', '?')}, valid {counts.get('valid', '?')}, "
        f"test {counts.get('test', '?')}"
    )
    size_mb = manifest["total_bytes"] / 1e6
    return f"""# {manifest['label']}

InterfaceForge canonical training dataset, archived {manifest['created_at']} with
InterfaceForge {manifest['interfaceforge_version']} for cold storage and reuse.

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
