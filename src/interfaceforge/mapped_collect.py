"""Stage mapped VASP trees and build synchronized MACE/DeePMD leaf datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError, SafetyError
from .leaf_audit import audit_leaf_manifests, write_leaf_audit
from .leaf_collect import collect_leaf_dataset
from .vasp_provenance import (
    DEFAULT_REQUIRED_INCAR_TAGS,
    audit_vasp_reference_records,
    build_vasp_reference_record,
    write_vasp_reference_provenance,
)

DEFAULT_FILES = ("OUTCAR", "INCAR", "POSCAR", "CONTCAR", "KPOINTS")


@dataclass(frozen=True)
class Mapping:
    source: Path
    target: Path
    required: bool = True


def _expand(value: str, *, root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    unresolved = re.findall(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", expanded)
    if unresolved:
        raise ConfigurationError(f"Unresolved environment variable in path {value!r}: {unresolved}")
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_mapped_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ConfigurationError("Mapped collection config requires schema_version: 1")
    if not raw.get("campaign_root"):
        raise ConfigurationError("campaign_root is required")
    campaign_root = _expand(str(raw["campaign_root"]), root=config_path.parent)
    staging_value = str(raw.get("staging_root", "reference_runs"))
    staging_root = _expand(staging_value, root=campaign_root)
    mappings_raw = raw.get("sources")
    if not isinstance(mappings_raw, list) or not mappings_raw:
        raise ConfigurationError("sources must be a non-empty list")
    mappings: list[Mapping] = []
    disabled_sources: list[dict[str, str]] = []
    role_manifests: dict[Path, Path] = {}
    seen_targets: set[Path] = set()
    for index, item in enumerate(mappings_raw):
        if not isinstance(item, dict) or not item.get("source") or not item.get("target"):
            raise ConfigurationError(f"sources[{index}] requires source and target")
        source = _expand(str(item["source"]), root=config_path.parent)
        if item.get("enabled", True) is False:
            if not item.get("reason"):
                raise ConfigurationError("Disabled sources require a quarantine reason")
            disabled_sources.append({"source": str(source), "reason": str(item["reason"])})
            continue
        if item.get("train_role_manifest"):
            role_manifests[source] = _expand(str(item["train_role_manifest"]), root=config_path.parent)
        target = Path(str(item["target"]))
        if target.is_absolute() or ".." in target.parts:
            raise SafetyError(f"sources[{index}].target must stay below staging_root: {target}")
        if target in seen_targets:
            raise ConfigurationError(f"Duplicate mapped target: {target}")
        seen_targets.add(target)
        mappings.append(Mapping(source=source, target=target, required=bool(item.get("required", True))))
    collection = dict(raw.get("collection") or {})
    ratios = list(collection.get("ratios", [0.8, 0.1, 0.1]))
    split_mode = str(collection.get("split_mode", "heritage")).lower()
    if split_mode not in {"heritage", "random-frame"}:
        raise ConfigurationError("collection.split_mode must be 'heritage' or 'random-frame'")
    if len(ratios) != 3:
        raise ConfigurationError("collection.ratios requires train, valid, test")
    include_files = tuple(str(name) for name in raw.get("include_files", DEFAULT_FILES))
    if "OUTCAR" not in include_files:
        raise ConfigurationError("include_files must contain OUTCAR")
    provenance = dict(raw.get("provenance") or {})
    required_incar_tags = [
        str(tag).upper() for tag in provenance.get("required_incar_tags", DEFAULT_REQUIRED_INCAR_TAGS)
    ]
    consistent_incar_tags = [str(tag).upper() for tag in provenance.get("consistent_incar_tags", required_incar_tags)]
    if not required_incar_tags:
        raise ConfigurationError("provenance.required_incar_tags cannot be empty")
    hash_files = list(dict.fromkeys(str(name) for name in provenance.get("hash_files", [*include_files, "POTCAR"])))
    return {
        "config_path": config_path,
        "campaign_root": campaign_root,
        "staging_root": staging_root,
        "initialize_campaign": bool(raw.get("initialize_campaign", True)),
        "mappings": mappings,
        "disabled_sources": disabled_sources,
        "role_manifests": role_manifests,
        "source_audit": (_expand(str(raw["source_audit"]), root=campaign_root) if raw.get("source_audit") else None),
        "include_files": include_files,
        "provenance": {
            "required_incar_tags": required_incar_tags,
            "consistent_incar_tags": consistent_incar_tags,
            "hash_files": hash_files,
        },
        "collection": {
            "ratios": ratios,
            "split_mode": split_mode,
            "seed": int(collection.get("seed", 20260730)),
            "stride": int(collection.get("stride", 1)),
            "heritage_depth": int(collection.get("heritage_depth", 2)),
            "include_virial": bool(collection.get("include_virial", False)),
            "balance_frames_per_leaf": bool(collection.get("balance_frames_per_leaf", True)),
            "type_map": [str(value) for value in collection.get("type_map", [])],
            "mace_output": str(collection.get("mace_output", "datasets/canonical")),
            "deepmd_output": str(collection.get("deepmd_output", "datasets/canonical/deepmd")),
        },
        "audit_output": str((raw.get("audit") or {}).get("output", "audit/leaf_datasets")),
    }


def _excluded(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if (
            "backup" in lowered
            or part.startswith("X")
            or lowered in {".interfaceforge", "archive"}
            or lowered.startswith(("restart_archive_", "refit_archive_", "stability_archive_"))
        ):
            return True
    return False


def discover_mapped_leaves(config: dict[str, Any]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    destinations: set[Path] = set()
    errors: list[str] = []
    for mapping in config["mappings"]:
        roles_path = config.get("role_manifests", {}).get(mapping.source)
        roles = None
        if roles_path is not None:
            roles = json.loads(roles_path.read_text(encoding="utf-8"))
            if roles.get("schema_version") != 1 or not isinstance(roles.get("sources"), dict):
                raise ConfigurationError(f"Invalid A2 train role manifest: {roles_path}")
        if not mapping.source.is_dir():
            if mapping.required:
                errors.append(f"Missing required source directory: {mapping.source}")
            continue
        found = 0
        for outcar in sorted(mapping.source.rglob("OUTCAR")):
            if not outcar.is_file() or outcar.stat().st_size == 0:
                continue
            relative_leaf = outcar.parent.relative_to(mapping.source)
            if _excluded(relative_leaf):
                continue
            if roles is not None:
                record = roles["sources"].get(relative_leaf.as_posix(), {})
                if record.get("role") != "train" or record.get("frames") != 1:
                    errors.append(f"A2 source lacks a single-frame train role: {outcar}")
                    continue
                from .source_audit import scan_outcar
                from .vasp_provenance import sha256_file

                if scan_outcar(outcar)["frames"] != 1:
                    errors.append(f"A2 source must contain exactly one ionic frame: {outcar}")
                    continue
                if record.get("outcar_sha256") != sha256_file(outcar):
                    errors.append(f"A2 role manifest hash mismatch: {outcar}")
                    continue
            destination = config["staging_root"] / mapping.target / relative_leaf
            if destination in destinations:
                errors.append(f"Mapped leaf collision: {destination}")
                continue
            destinations.add(destination)
            found += 1
            leaves.append(
                {
                    "source_leaf": outcar.parent,
                    "source_root": mapping.source,
                    "source_outcar": outcar,
                    "mapped_prefix": mapping.target,
                    "relative_leaf": relative_leaf,
                    "destination_leaf": destination,
                }
            )
        if mapping.required and found == 0:
            errors.append(f"No nonempty OUTCAR found below required source: {mapping.source}")
    if errors:
        raise SafetyError("\n".join(errors))
    if not leaves:
        raise SafetyError("No mapped VASP leaves were discovered")
    return leaves


def _resolve_source_file(leaf: dict[str, Any], name: str) -> Path:
    """Resolve shared VASP inputs without escaping the configured source root."""
    direct = leaf["source_leaf"] / name
    if direct.is_file() or name not in {"INCAR", "KPOINTS", "POTCAR"}:
        return direct
    source_root: Path = leaf["source_root"]
    current: Path = leaf["source_leaf"]
    while current != source_root:
        current = current.parent
        candidate = current / name
        if candidate.is_file():
            return candidate
    return direct


def _initialize_campaign(root: Path) -> list[str]:
    written: list[str] = []
    templates = {
        "campaign.yaml": root / "campaign.yaml",
        "profile_loni.yaml": root / "profiles" / "loni.yaml",
        "profile_local.yaml": root / "profiles" / "local.yaml",
        "potcar_pbe_54.yaml": root / "profiles" / "potcar_pbe_54.yaml",
    }
    for resource_name, destination in templates.items():
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        template = resources.files("interfaceforge").joinpath(f"templates/{resource_name}")
        destination.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(str(destination))
    for directory in (root / "inputs", root / "structures"):
        directory.mkdir(parents=True, exist_ok=True)
    return written


def stage_mapped_leaves(config: dict[str, Any], leaves: list[dict[str, Any]]) -> dict[str, Any]:
    staging_root: Path = config["staging_root"]
    rows: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    linked = skipped = 0
    for leaf in leaves:
        destination: Path = leaf["destination_leaf"]
        destination.mkdir(parents=True, exist_ok=True)
        resolved_files = {
            name: _resolve_source_file(leaf, name)
            for name in dict.fromkeys([*config["include_files"], *config["provenance"]["hash_files"]])
        }
        files = 0
        for name in config["include_files"]:
            source = resolved_files[name]
            if not source.is_file():
                continue
            target = destination / name
            if target.exists():
                try:
                    same = os.path.samefile(source, target)
                except OSError:
                    same = False
                if not same:
                    raise SafetyError(
                        f"Refusing to replace nonmatching staged file: {target}. "
                        "Move the staging tree or use a new campaign root."
                    )
                skipped += 1
            else:
                try:
                    os.link(source, target)
                except OSError as exc:
                    raise SafetyError(
                        f"Could not hard-link {source} to {target}: {exc}. Keep source and campaign on one filesystem."
                    ) from exc
                linked += 1
            files += 1
        rows.append(
            {
                "source_outcar": str(leaf["source_outcar"]),
                "staged_leaf": str(destination.relative_to(staging_root)),
                "files_present": files,
            }
        )
        provenance_records.append(
            build_vasp_reference_record(
                source_leaf=leaf["source_leaf"],
                source_outcar=leaf["source_outcar"],
                staged_leaf=str(destination.relative_to(staging_root)),
                included_files=config["provenance"]["hash_files"],
                file_paths=resolved_files,
                required_incar_tags=config["provenance"]["required_incar_tags"],
            )
        )
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest = staging_root / "mapped_sources.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance_audit = audit_vasp_reference_records(
        provenance_records,
        consistent_incar_tags=config["provenance"]["consistent_incar_tags"],
    )
    provenance_outputs = write_vasp_reference_provenance(provenance_records, provenance_audit, staging_root)
    if provenance_audit["status"] != "OK":
        details = "; ".join(issue for problem in provenance_audit["problems"] for issue in problem["issues"])
        raise SafetyError(f"VASP reference provenance audit failed: {details}. See {provenance_outputs['audit']}")
    stride = config["collection"]["stride"]
    available_after_stride = [
        (int(record["ionic_frames_detected"]) + stride - 1) // stride for record in provenance_records
    ]
    balanced_frames = min(available_after_stride)
    return {
        "staging_root": str(staging_root),
        "leaves": len(leaves),
        "files_linked": linked,
        "existing_hardlinks": skipped,
        "manifest": str(manifest),
        "provenance": provenance_audit,
        "provenance_outputs": provenance_outputs,
        "available_frames_after_stride": {
            "minimum": min(available_after_stride),
            "maximum": max(available_after_stride),
            "unique": sorted(set(available_after_stride)),
        },
        "balanced_frames_per_leaf": (balanced_frames if config["collection"]["balance_frames_per_leaf"] else None),
    }


def run_mapped_collection(
    config_path: str | Path,
    *,
    execute: bool = False,
    collect: bool = False,
    force_datasets: bool = False,
    audit_only: bool = False,
) -> dict[str, Any]:
    config = load_mapped_config(config_path)
    campaign_root: Path = config["campaign_root"]
    collection = config["collection"]
    mace_output = _expand(collection["mace_output"], root=campaign_root)
    deepmd_output = _expand(collection["deepmd_output"], root=campaign_root)
    audit_output = _expand(config["audit_output"], root=campaign_root)

    payload: dict[str, Any] = {
        "mode": "execute" if execute else "dry-run",
        "config": str(config["config_path"]),
        "campaign_root": str(campaign_root),
        "staging_root": str(config["staging_root"]),
        "mace_output": str(mace_output),
        "deepmd_output": str(deepmd_output),
        "audit_output": str(audit_output),
        "disabled_sources": config["disabled_sources"],
        "source_audit": str(config["source_audit"]) if config["source_audit"] else None,
    }
    if audit_only:
        if not execute:
            raise SafetyError("--audit-only writes reports and therefore requires --execute")
        report = audit_leaf_manifests(
            mace_output / "leaf_manifest.csv",
            deepmd_output / "leaf_manifest.csv",
            reference_audit=config["staging_root"] / "reference_provenance_audit.json",
            require_balanced_frames=collection["balance_frames_per_leaf"],
        )
        report["outputs"] = write_leaf_audit(report, audit_output)
        payload["audit"] = report
        return payload

    leaves = discover_mapped_leaves(config)
    payload["leaves"] = [
        {
            "source_outcar": str(leaf["source_outcar"]),
            "staged_leaf": str(leaf["destination_leaf"].relative_to(config["staging_root"])),
        }
        for leaf in leaves
    ]
    if not execute:
        payload["would_initialize_campaign"] = (
            config["initialize_campaign"] and not (campaign_root / "campaign.yaml").exists()
        )
        payload["would_collect"] = collect
        return payload

    if config["source_audit"] is not None:
        from .source_audit import require_source_admission

        payload["source_admission"] = require_source_admission(
            config["source_audit"], [leaf["source_outcar"] for leaf in leaves]
        )
    if config["initialize_campaign"]:
        payload["initialized_files"] = _initialize_campaign(campaign_root)
    payload["staging"] = stage_mapped_leaves(config, leaves)
    if not collect:
        return payload

    # A previous collection must not resurrect a source removed from the map.
    expected_outcars = {leaf["destination_leaf"] / "OUTCAR" for leaf in leaves}
    staged_outcars = set(config["staging_root"].rglob("OUTCAR"))
    if staged_outcars != expected_outcars:
        raise SafetyError("Staging contains missing or unmapped OUTCARs; use a fresh campaign root")

    common = {
        "heritage_depth": collection["heritage_depth"],
        "ratios": collection["ratios"],
        "split_mode": collection["split_mode"],
        "seed": collection["seed"],
        "stride": collection["stride"],
        "include_virial": collection["include_virial"],
        "force": force_datasets,
        "frames_per_leaf": payload["staging"]["balanced_frames_per_leaf"],
        "reference_provenance": payload["staging"]["provenance_outputs"]["records"],
    }
    payload["mace"] = collect_leaf_dataset(config["staging_root"], mace_output, engine="mace", **common)
    payload["deepmd"] = collect_leaf_dataset(
        config["staging_root"],
        deepmd_output,
        engine="deepmd",
        type_map=collection["type_map"],
        **common,
    )
    if payload.get("source_admission"):
        final_admission = require_source_admission(
            config["source_audit"], [leaf["source_outcar"] for leaf in leaves]
        )
        if final_admission != payload["source_admission"]:
            raise SafetyError("Source admission report changed during collection; discard this export")
        for engine in ("mace", "deepmd"):
            manifest = Path(payload[engine]["manifest_json"])
            content = json.loads(manifest.read_text())
            content["source_admission"] = payload["source_admission"]
            manifest.write_text(json.dumps(content, indent=2) + "\n")
    report = audit_leaf_manifests(
        mace_output / "leaf_manifest.csv",
        deepmd_output / "leaf_manifest.csv",
        reference_audit=payload["staging"]["provenance"],
        require_balanced_frames=collection["balance_frames_per_leaf"],
    )
    report["outputs"] = write_leaf_audit(report, audit_output)
    payload["audit"] = report
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML source-mapping configuration")
    parser.add_argument("--execute", action="store_true", help="Perform staging writes")
    parser.add_argument("--collect", action="store_true", help="Run both collectors and audit")
    parser.add_argument("--force-datasets", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.collect or args.force_datasets) and not args.execute:
        parser.error("--collect/--force-datasets require --execute")
    if args.force_datasets and not args.collect:
        parser.error("--force-datasets requires --collect")
    try:
        payload = run_mapped_collection(
            args.config,
            execute=args.execute,
            collect=args.collect,
            force_datasets=args.force_datasets,
            audit_only=args.audit_only,
        )
    except (ConfigurationError, FileNotFoundError, OSError, SafetyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2))
    audit = payload.get("audit", {})
    return 1 if audit and audit.get("status") != "OK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
