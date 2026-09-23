# ruff: noqa: E501
"""Pre-GPU readiness audit for the canonical NiO MLIP dataset.

Answers, from the files on disk and without training anything:

1. which trajectories are discoverable (by stage / temperature / status);
2. how many usable frames exist per system and temperature;
3. which runs are incomplete or problematic, and why;
4. how many frames land in train / valid / test;
5. whether any trajectory lineage leaks across splits;
6. whether MACE, DeePMD/DPA and NequIP would consume the same dataset and split.

Writes ``readiness.json`` (machine-readable) and ``readiness.md`` (for review).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Campaign
from .errors import InterfaceForgeError
from .nio_dataset import (
    SPLITS,
    ExportConfig,
    _plan_summary,
    _trajectory_row,
    load_dataset_manifest,
    plan_export,
    verify_dataset,
)
from .provenance import interfaceforge_commit
from .state import sha256_file, utc_now


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def backend_consumption(campaign: Campaign | None, dataset: str | Path | None) -> dict[str, Any]:
    """Would the three backends read the same canonical files?

    Compares each backend's configured input against the canonical manifest's
    file hashes. Without a campaign, reports what each backend *would* read from
    the canonical layout. Cutoffs are listed for information only.
    """

    if dataset is None:
        return {"status": "not-checked", "reason": "no exported dataset given (--dataset)"}
    root, manifest = load_dataset_manifest(dataset)
    hashes = manifest.get("file_hashes", {})
    canonical = {name: (root / name).resolve() for name in ("train.extxyz", "valid.extxyz", "test.extxyz")}

    def _check_file(path: Path, expected_name: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        entry["is_canonical_file"] = path == canonical[expected_name]
        if path.is_file() and hashes.get(expected_name):
            entry["matches_manifest_hash"] = sha256_file(path) == hashes[expected_name]
        else:
            entry["matches_manifest_hash"] = False
        return entry

    backends: dict[str, Any] = {}
    cutoffs: dict[str, Any] = {}
    models = campaign.models if campaign is not None else {}
    base = campaign.root if campaign is not None else Path.cwd()

    mace = dict(models.get("mace", {}))
    if campaign is None or mace:
        files = {
            "train.extxyz": _resolve(base, mace.get("train_file", canonical["train.extxyz"])),
            "valid.extxyz": _resolve(base, mace.get("valid_file", canonical["valid.extxyz"])),
            "test.extxyz": _resolve(base, mace.get("test_file", canonical["test.extxyz"])),
        }
        checks = {name: _check_file(path, name) for name, path in files.items()}
        backends["mace"] = {
            "enabled": bool(mace.get("enabled", campaign is None)),
            "roi_enabled": bool(dict(mace.get("roi", {})).get("enabled", False)),
            "files": checks,
            "same_dataset": all(item["matches_manifest_hash"] for item in checks.values()),
        }
        cutoffs["mace_r_max"] = float(mace.get("r_max", 6.0)) if mace else None
        if backends["mace"]["roi_enabled"]:
            backends["mace"]["note"] = "MACE-ROI trains on a derived copy; verify its manifest source hashes"

    deepmd = dict(models.get("deepmd", {}))
    if campaign is None or deepmd:
        deepmd_root = _resolve(base, deepmd.get("dataset_root", root / "deepmd"))
        entry: dict[str, Any] = {
            "enabled": bool(deepmd.get("enabled", campaign is None)),
            "dataset_root": str(deepmd_root),
        }
        entry["is_canonical_root"] = deepmd_root == (root / "deepmd").resolve()
        entry["same_dataset"] = bool(entry["is_canonical_root"]) and all(
            sha256_file(root / name) == digest
            for name, digest in hashes.items()
            if name.startswith("deepmd/") and (root / name).is_file()
        )
        backends["deepmd"] = entry
        cutoffs["deepmd_architectures"] = deepmd.get("architectures")

    nequip = dict(models.get("nequip", {}))
    if campaign is None or nequip:
        if nequip.get("dataset"):
            nequip_root = _resolve(base, nequip["dataset"])
            files = {name: nequip_root / name for name in canonical}
        else:
            files = {
                "train.extxyz": _resolve(base, nequip.get("train_file", canonical["train.extxyz"])),
                "valid.extxyz": _resolve(base, nequip.get("valid_file", canonical["valid.extxyz"])),
                "test.extxyz": _resolve(base, nequip.get("test_file", canonical["test.extxyz"])),
            }
        checks = {name: _check_file(path, name) for name, path in files.items()}
        backends["nequip"] = {
            "enabled": bool(nequip.get("enabled", campaign is None)),
            "files": checks,
            "same_dataset": all(item["matches_manifest_hash"] for item in checks.values()),
        }
        cutoffs["nequip_r_max"] = nequip.get("r_max")

    enabled = {name: item for name, item in backends.items() if item.get("enabled")}
    consistent = bool(enabled) and all(item["same_dataset"] for item in enabled.values())
    return {
        "status": "checked",
        "dataset_root": str(root),
        "dataset_hash": manifest.get("dataset_hash"),
        "split_hash": manifest.get("split_hash"),
        "backends": backends,
        "all_enabled_backends_share_dataset_and_split": consistent,
        "cutoffs_for_information": cutoffs,
        "note": (
            "Identical files guarantee identical frames, labels and split for every backend; "
            "cutoffs and hyperparameters are backend choices and are only listed."
        ),
    }


def readiness_audit(
    roots: Sequence[str | Path],
    config: ExportConfig,
    *,
    output: str | Path,
    campaign: Campaign | None = None,
    dataset: str | Path | None = None,
) -> dict[str, Any]:
    plan = plan_export(roots, config)
    summary = _plan_summary(plan)
    rows = [_trajectory_row(analysis, plan) for analysis in plan.analyses]
    usable = [row for row in rows if row["status"] in {"ok", "ok_incomplete"}]

    per_system_temperature: dict[str, dict[str, int]] = defaultdict(dict)
    for row in usable:
        per_system_temperature[row["case"]][f"{row['stage']}@{row['temperature_k']}"] = int(row["frames_selected"])
    discoverable = Counter((row["stage"], row["temperature_k"], row["status"]) for row in rows)

    verification = None
    if dataset is not None:
        try:
            verification = verify_dataset(dataset)
        except InterfaceForgeError as exc:
            verification = {"valid": False, "problems": [str(exc)]}
    consumption = backend_consumption(campaign, dataset)

    blocking: list[str] = []
    if summary["leakage"]["leakage_detected"]:
        blocking.append("split leakage detected")
    for split, count in summary["frames_per_split"].items():
        if config.ratios[SPLITS.index(split)] > 0 and count == 0:
            blocking.append(f"{split} split is empty")
    if verification is not None and not verification.get("valid"):
        blocking.append("exported dataset failed verification")
    if consumption.get("status") == "checked" and not consumption.get("all_enabled_backends_share_dataset_and_split"):
        blocking.append("enabled backends do not all read the canonical dataset")
    if dataset is None:
        blocking.append(
            "dataset not exported/verified yet (run `iface dataset export`, then re-run readiness with --dataset)"
        )

    payload = {
        "schema": "interfaceforge-nio-readiness",
        "schema_version": 1,
        "created_at": utc_now(),
        "interfaceforge_commit": interfaceforge_commit(),
        "roots": [str(root) for root in plan.roots],
        "config": config.to_dict(),
        "answers": {
            "discoverable_trajectories": {
                "total": len(rows),
                "by_stage_temperature_status": [
                    {"stage": stage, "temperature_k": temperature, "status": status, "trajectories": count}
                    for (stage, temperature, status), count in sorted(
                        discoverable.items(), key=lambda item: tuple(str(v) for v in item[0])
                    )
                ],
            },
            "usable_frames_per_system_temperature": dict(sorted(per_system_temperature.items())),
            "frames_by_stage_temperature": summary["frames_by_stage_temperature"],
            "problem_runs": summary["problem_trajectories"],
            "rejection_reasons": summary["rejection_reasons"],
            "frames_per_split": summary["frames_per_split"],
            "trajectories_per_split": summary["trajectories_per_split"],
            "groups_per_split": summary["groups_per_split"],
            "achieved_frame_fractions": summary["achieved_frame_fractions"],
            "leakage": summary["leakage"],
            "backend_consumption": consumption,
            "dataset_verification": verification,
        },
        "summary": summary,
        "ready_for_gpu_smoke_tests": not blocking,
        "blocking_items": blocking,
        "scope_note": (
            "Readiness means the data plumbing is consistent. It does not validate the DFT "
            "labels, the MLIPs, or their transferability to phosphonate chemistry or long MD."
        ),
        "trajectories": rows,
    }
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "readiness.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "readiness.md").write_text(render_markdown(payload), encoding="utf-8")
    payload["outputs"] = {"json": str(out / "readiness.json"), "markdown": str(out / "readiness.md")}
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    answers = payload["answers"]
    summary = payload["summary"]
    lines = [
        "# NiO MLIP dataset readiness audit",
        "",
        f"Generated {payload['created_at']} from {', '.join(f'`{root}`' for root in payload['roots'])}.",
        "",
        f"**Ready for GPU smoke tests:** {'yes' if payload['ready_for_gpu_smoke_tests'] else 'no'}",
        "",
    ]
    if payload["blocking_items"]:
        lines += ["Blocking items:", ""] + [f"- {item}" for item in payload["blocking_items"]] + [""]
    lines += [
        f"> {payload['scope_note']}",
        "",
        "## 1. Discoverable trajectories",
        "",
        f"{answers['discoverable_trajectories']['total']} trajectories found.",
        "",
        "| Stage | T (K) | Status | Trajectories |",
        "|---|---:|---|---:|",
    ]
    for row in answers["discoverable_trajectories"]["by_stage_temperature_status"]:
        lines.append(f"| {row['stage']} | {_fmt(row['temperature_k'])} | {row['status']} | {row['trajectories']} |")
    lines += ["", "## 2. Usable frames per system and temperature", ""]
    columns = sorted({key for values in answers["usable_frames_per_system_temperature"].values() for key in values})
    if columns:
        lines.append("| System | " + " | ".join(columns) + " |")
        lines.append("|---|" + "---:|" * len(columns))
        for system, values in answers["usable_frames_per_system_temperature"].items():
            lines.append(f"| `{system}` | " + " | ".join(str(values.get(column, 0)) for column in columns) + " |")
    else:
        lines.append("No usable frames.")
    lines += ["", "## 3. Incomplete or problematic runs", ""]
    if answers["problem_runs"]:
        lines += ["| Trajectory | Status | Reasons | Warnings |", "|---|---|---|---|"]
        for row in answers["problem_runs"]:
            lines.append(
                f"| `{row['trajectory_id']}` | {row['status']} | {row['reasons'] or '–'} | {row['warnings'] or '–'} |"
            )
    else:
        lines.append("None.")
    if answers["rejection_reasons"]:
        lines += ["", "Rejected frames by reason:", ""]
        lines += [f"- {reason}: {count}" for reason, count in answers["rejection_reasons"].items()]
    lines += [
        "",
        "## 4. Split",
        "",
        f"Method `{payload['config']['split_method']}`, group_by `{payload['config']['group_by']}`, "
        f"stratify_by `{', '.join(payload['config']['stratify_by'])}`, seed {payload['config']['seed']}.",
        "",
        "| Split | Frames | Fraction (target) | Trajectories | Groups |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        fraction = answers["achieved_frame_fractions"][split]
        lines.append(
            f"| {split} | {answers['frames_per_split'][split]} | {_fmt(fraction)} ({summary['target_fractions'][split]:.2f}) | "
            f"{answers['trajectories_per_split'][split]} | {answers['groups_per_split'][split]} |"
        )
    leakage = answers["leakage"]
    lines += [
        "",
        "## 5. Leakage",
        "",
        f"Leakage detected: **{'yes' if leakage['leakage_detected'] else 'no'}** ({leakage['groups_checked']} groups checked: "
        "group membership, identical starting structures, CONTCAR→POSCAR lineage).",
        "",
        f"Informational: {leakage['informational']['surface_families_shared_across_splits']} surface families "
        "(same hydroxylated slab, different ligand/anchor) appear in more than one split under group_by=case.",
    ]
    for problem in leakage["problems"][:20]:
        lines.append(f"- {problem}")
    consumption = answers["backend_consumption"]
    lines += ["", "## 6. Backend consumption", ""]
    if consumption.get("status") != "checked":
        lines.append(f"Not checked: {consumption.get('reason')}.")
    else:
        lines.append(
            f"All enabled backends share dataset and split: **{'yes' if consumption['all_enabled_backends_share_dataset_and_split'] else 'no'}** "
            f"(dataset `{consumption['dataset_hash'][:12]}`, split `{consumption['split_hash'][:12]}`)."
        )
        lines += ["", "| Backend | Enabled | Same dataset |", "|---|---|---|"]
        for name, item in consumption["backends"].items():
            lines.append(f"| {name} | {item.get('enabled')} | {item.get('same_dataset')} |")
    verification = answers.get("dataset_verification")
    if verification is not None:
        lines += ["", f"Dataset verification: **{'valid' if verification.get('valid') else 'INVALID'}**."]
        lines += [f"- {problem}" for problem in verification.get("problems", [])[:20]]
    lines += ["", "Machine-readable detail: `readiness.json` (every trajectory with QC counts and reasons).", ""]
    return "\n".join(lines)
