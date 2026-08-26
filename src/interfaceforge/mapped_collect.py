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

DEFAULT_FILES = ("OUTCAR", "INCAR", "POSCAR", "CONTCAR")


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
    seen_targets: set[Path] = set()
    for index, item in enumerate(mappings_raw):
        if not isinstance(item, dict) or not item.get("source") or not item.get("target"):
            raise ConfigurationError(f"sources[{index}] requires source and target")
        source = _expand(str(item["source"]), root=config_path.parent)
        target = Path(str(item["target"]))
        if target.is_absolute() or ".." in target.parts:
            raise SafetyError(f"sources[{index}].target must stay below staging_root: {target}")
        if target in seen_targets:
            raise ConfigurationError(f"Duplicate mapped target: {target}")
        seen_targets.add(target)
        mappings.append(Mapping(source=source, target=target, required=bool(item.get("required", True))))
    collection = dict(raw.get("collection") or {})
    ratios = list(collection.get("ratios", [0.8, 0.1, 0.1]))
    if len(ratios) != 3:
        raise ConfigurationError("collection.ratios requires train, valid, test")
    include_files = tuple(str(name) for name in raw.get("include_files", DEFAULT_FILES))
    if "OUTCAR" not in include_files:
        raise ConfigurationError("include_files must contain OUTCAR")
    return {
        "config_path": config_path,
        "campaign_root": campaign_root,
        "staging_root": staging_root,
        "initialize_campaign": bool(raw.get("initialize_campaign", True)),
        "mappings": mappings,
        "include_files": include_files,
        "collection": {
            "ratios": ratios,
            "seed": int(collection.get("seed", 20260730)),
            "stride": int(collection.get("stride", 1)),
            "heritage_depth": int(collection.get("heritage_depth", 2)),
            "include_virial": bool(collection.get("include_virial", False)),
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
            destination = config["staging_root"] / mapping.target / relative_leaf
            if destination in destinations:
                errors.append(f"Mapped leaf collision: {destination}")
                continue
            destinations.add(destination)
            found += 1
            leaves.append(
                {
                    "source_leaf": outcar.parent,
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
    linked = skipped = 0
    for leaf in leaves:
        destination: Path = leaf["destination_leaf"]
        destination.mkdir(parents=True, exist_ok=True)
        files = 0
        for name in config["include_files"]:
            source = leaf["source_leaf"] / name
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
                        f"Could not hard-link {source} to {target}: {exc}. "
                        "Keep source and campaign on one filesystem."
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
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest = staging_root / "mapped_sources.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "staging_root": str(staging_root),
        "leaves": len(leaves),
        "files_linked": linked,
        "existing_hardlinks": skipped,
        "manifest": str(manifest),
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
    }
    if audit_only:
        if not execute:
            raise SafetyError("--audit-only writes reports and therefore requires --execute")
        report = audit_leaf_manifests(
            mace_output / "leaf_manifest.csv", deepmd_output / "leaf_manifest.csv"
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
        payload["would_initialize_campaign"] = config["initialize_campaign"] and not (
            campaign_root / "campaign.yaml"
        ).exists()
        payload["would_collect"] = collect
        return payload

    if config["initialize_campaign"]:
        payload["initialized_files"] = _initialize_campaign(campaign_root)
    payload["staging"] = stage_mapped_leaves(config, leaves)
    if not collect:
        return payload

    common = {
        "heritage_depth": collection["heritage_depth"],
        "ratios": collection["ratios"],
        "seed": collection["seed"],
        "stride": collection["stride"],
        "include_virial": collection["include_virial"],
        "force": force_datasets,
    }
    payload["mace"] = collect_leaf_dataset(
        config["staging_root"], mace_output, engine="mace", **common
    )
    payload["deepmd"] = collect_leaf_dataset(
        config["staging_root"],
        deepmd_output,
        engine="deepmd",
        type_map=collection["type_map"],
        **common,
    )
    report = audit_leaf_manifests(
        mace_output / "leaf_manifest.csv", deepmd_output / "leaf_manifest.csv"
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
