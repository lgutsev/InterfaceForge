"""Prepare a rescue inventory from an audit snapshot without reading remote sources.

This is an offline planning tool, not a source admission or dataset exporter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .errors import SafetyError


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SafetyError(f"Invalid {description}: {value!r}")
    return value


def _mapped_target(source_id: str) -> str:
    return source_id.removeprefix("mapped/").removesuffix("/.")


def _group(source_id: str) -> str:
    parts = _mapped_target(source_id).split("/")
    if parts[0] == "interface" and len(parts) >= 5:
        return "/".join(parts[:4])
    if parts[0] == "bulk" and len(parts) == 2:
        return "/".join(parts)
    raise SafetyError(f"Unsupported probe source: {source_id}")


def _check_membership(path: Path, rows: dict[str, dict], active: dict[str, set[int]],
                      stride: int, report_hash: str) -> dict:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1 or ledger.get("report_sha256") != report_hash:
        raise SafetyError("Historical membership ledger must bind this audit report hash")
    if not isinstance(ledger.get("frames"), list):
        raise SafetyError("Historical membership ledger requires a frames list")
    seen: set[tuple[str, int]] = set()
    counts: Counter = Counter()
    excluded = 0
    for entry in ledger["frames"]:
        source_id = entry.get("source_id")
        row = rows.get(source_id)
        if row is None or row["role"] != "train" or not source_id.startswith("mapped/"):
            raise SafetyError(f"Historical split contains an unknown or non-training source: {source_id}")
        index = _integer(entry.get("source_frame"), "historical source_frame")
        frames = row.get("outcar", {}).get("frames", 0)
        if index >= frames or index % stride:
            raise SafetyError(f"Historical frame outside retained source indices: {source_id}:{index}")
        if entry.get("outcar_sha256") != row["files"]["OUTCAR"]["sha256"]:
            raise SafetyError(f"Historical OUTCAR hash mismatch: {source_id}")
        split = entry.get("split")
        if split not in {"train", "valid", "test"}:
            raise SafetyError(f"Invalid historical split: {split}")
        key = (source_id, index)
        if key in seen:
            raise SafetyError(f"Duplicate historical frame or split leakage: {source_id}:{index}")
        seen.add(key)
        if source_id in active:
            counts[split] += 1
        else:
            excluded += 1
    missing = sum((source_id, index) not in seen for source_id, indices in active.items() for index in indices)
    if missing:
        raise SafetyError(f"Historical membership is missing {missing} active frames; never re-split survivors")
    return {"status": "COMPLETE", "sha256": _digest(path), "active_split_counts": dict(sorted(counts.items())),
            "excluded_historical_frames": excluded, "source": "supplied historical ledger; not reconstructed"}


def prepare_rescue(report_path: Path, map_path: Path, policy_path: Path,
                   membership_path: Path | None = None) -> dict:
    """Bind a candidate inventory and probe indices to the supplied audit snapshot."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_hash = _digest(report_path)
    if report.get("schema_version") != 1 or report.get("labels_scanned") is not True:
        raise SafetyError("A full schema-1 source audit with scanned labels is required")
    if report.get("root_errors"):
        raise SafetyError("Audit has root errors; inventory is incomplete")
    if _digest(map_path) != report.get("mapped_config", {}).get("sha256"):
        raise SafetyError("Mapped YAML differs from this audit snapshot")
    if _digest(policy_path) != report.get("policy_sha256"):
        raise SafetyError("Qualification policy differs from this audit snapshot")
    config = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    stride = _integer(config.get("collection", {}).get("stride", 1), "stride", minimum=1)
    if config.get("collection", {}).get("balance_frames_per_leaf", True):
        raise SafetyError("Balanced-leaf selection requires its own explicit retained-frame policy")
    if config.get("collection", {}).get("include_virial") is not True:
        raise SafetyError("Rescue planning requires the full virial channel")
    mappings = config["sources"]
    inventory: list[dict] = []
    active: dict[str, set[int]] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    rows: dict[str, dict] = {}
    evidence_hashes = {}
    probe_holds = []
    for row in report["rows"]:
        source_id = row["source_id"]
        if source_id in rows:
            raise SafetyError(f"Duplicate audit source ID: {source_id}")
        rows[source_id] = row
        target = _mapped_target(source_id)
        matches = [m for m in mappings if target == m["target"] or target.startswith(m["target"] + "/")]
        if source_id.startswith("mapped/") and len(matches) != 1:
            raise SafetyError(f"Mapped source must match exactly one map entry: {source_id}")
        mapping = matches[0] if source_id.startswith("mapped/") else None
        enabled = mapping is not None and mapping.get("enabled", True) is True
        if enabled and row["status"] == "QUARANTINED":
            raise SafetyError(f"Quarantined source is enabled in the map: {source_id}")
        candidate = enabled and row["role"] == "train"
        frames = _integer(row.get("outcar", {}).get("frames", 0), "source frame count")
        indices = set(range(0, frames, stride)) if candidate else set()
        inventory.append({"source_id": source_id, "role": row["role"], "status": row["status"],
                          "candidate": candidate, "source_frames": frames, "candidate_frames": len(indices),
                          "fingerprint": row["fingerprint"], "outcar_sha256": row.get("files", {}).get(
                              "OUTCAR", {}).get("sha256"), "run": row["run"],
                          "reason": row.get("reason", ""),
                          "issues": sorted({issue["code"] for issue in row["issues"]})})
        if not candidate:
            continue
        if frames == 0 or row.get("labels", {}).get("frames") != frames:
            raise SafetyError(f"Candidate source lacks a complete label scan: {source_id}")
        active[source_id] = indices
        evidence = report_path.parent / "frames" / (sha256(source_id.encode()).hexdigest()[:16] + ".scf.json")
        if not evidence.is_file():
            raise SafetyError(f"Missing frame-level SCF evidence: {evidence.name}")
        history = json.loads(evidence.read_text(encoding="utf-8"))
        if (any(type(f.get("source_frame")) is not int for f in history)
                or [f.get("source_frame") for f in history] != list(range(frames))):
            raise SafetyError(f"SCF evidence omits, duplicates or reorders source frames: {source_id}")
        evidence_hashes[source_id] = _digest(evidence)
        for frame in history:
            if frame["source_frame"] not in indices:
                continue
            residual = frame.get("final_rms_c")
            kind = "final_rms_c"
            if residual is None:
                residual = frame.get("last_rms_c")
                kind = "last_printed_rms_c"  # VASP often omits charge residual on its final row.
            if (isinstance(residual, bool) or not isinstance(residual, (float, int))
                    or not math.isfinite(residual) or residual < 0):
                probe_holds.append(f"Missing retained charge residual: {source_id}:{frame['source_frame']}")
                continue
            groups[_group(source_id)].append({"source_id": source_id, "source_frame": frame["source_frame"],
                                              "ionic_step": frame.get("ionic_step"), "charge_residual": residual,
                                              "residual_kind": kind, "scf_steps": frame.get("scf_steps"),
                                              "fingerprint": row["fingerprint"],
                                              "outcar_sha256": row["files"]["OUTCAR"]["sha256"],
                                              "scf_evidence_sha256": evidence_hashes[source_id]})
    if not active:
        raise SafetyError("No active mapped original training sources in this audit")
    probes = []
    if not probe_holds:
        for group, candidates in sorted(groups.items()):
            ordered = sorted(candidates, key=lambda f: (f["charge_residual"], f["source_id"], f["source_frame"]))
            median = statistics.median(f["charge_residual"] for f in ordered)
            typical = min(ordered, key=lambda f: abs(f["charge_residual"] - median))
            worst = next((f for f in reversed(ordered) if f is not typical), None)
            if worst is None:
                raise SafetyError(f"Two distinct retained probe geometries required: {group}")
            probes.extend([{**typical, "group": group, "kind": "typical"}, {**worst, "group": group, "kind": "worst"}])
    membership = (_check_membership(membership_path, rows, active, stride, report_hash) if membership_path else
                  {"status": "MISSING", "reason": "Recover source/frame/split membership from historical exports"})
    return {"schema_version": 1, "mode": "offline-preparation", "training_ready": False,
            "report_sha256": report_hash, "policy_sha256": _digest(policy_path), "map_sha256": _digest(map_path),
            "report_status": report["status"], "raw_sources_rechecked": False, "stride": stride,
            "counts": {"candidate_sources": len(active), "candidate_frames": sum(map(len, active.values())),
                       "quarantined_records": sum(r["status"] == "QUARANTINED" for r in inventory),
                       "roles_and_statuses": {f"{role}/{status}": n for (role, status), n in sorted(
                           Counter((r["role"], r["status"]) for r in inventory).items())}},
            "historical_membership": membership, "inventory": inventory, "probes": probes,
            "probe_holds": probe_holds, "scf_evidence_hashes": evidence_hashes,
            "duplicate_output_groups": report.get("duplicate_outputs", []), "copies_merged": False,
            "disabled_mappings": [m for m in mappings if m.get("enabled", True) is False],
            "gates": ["Review package-23 full E/F/stress against predeclared tolerances",
                      "Qualify every retained source and staged replacement; triage is not full-source acceptance",
                      "Freeze historical splits and implement validated partial replacement export",
                      "Finish and role-curate the A2 44 train / 16 holdout plan if included",
                      "Run a fresh LONI admission audit and synchronized MACE/DeePMD export checks"]}


def write_preparation(plan: dict, output: Path) -> None:
    """Create a fresh preparation directory; never overwrite an earlier snapshot."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "rebuild_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, records in (("source_inventory.csv", plan["inventory"]), ("probe_selection.csv", plan["probes"])):
        if records:
            with (output / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
    counts = plan["counts"]
    text = (f"# Rescue preparation: training remains blocked\n\n"
            f"Snapshot report SHA256: `{plan['report_sha256']}`\n\n"
            f"Candidate original sources: {counts['candidate_sources']}; candidate frames at stride {plan['stride']}: "
            f"{counts['candidate_frames']}. These are inventory counts, not accepted labels.\n\n"
            f"Quarantined records: {counts['quarantined_records']}. Probe geometries: {len(plan['probes'])}.\n\n"
            f"Historical split membership: {plan['historical_membership']['status']}.\n\n"
            "This offline preparation did not read live raw sources, qualify labels, merge copies, "
            "stage a dataset or submit jobs. Frame indices are zero-based. Probe selection uses the "
            "median and worst available printed charge residual among stride-retained frames. "
            "A last printed residual is not necessarily the residual at the final iteration. "
            "Verify geometry/input hashes before generating probes; "
            "reconcile existing controls to avoid duplicates.\n\n"
            "Remaining gates:\n\n" + "\n".join(f"- {gate}" for gate in plan["gates"]) + "\n")
    (output / "SUMMARY.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("mapped_config", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--historical-membership", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = prepare_rescue(args.report, args.mapped_config, args.policy, args.historical_membership)
        write_preparation(plan, args.output)
    except (SafetyError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Rescue preparation failed: {error}\n")
    print(json.dumps({"output": str(args.output), "counts": plan["counts"], "training_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
