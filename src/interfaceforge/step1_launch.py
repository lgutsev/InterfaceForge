"""Submit prepared and repaired Step1 preheat runs.

Like ``step2-launch`` / ``opt-launch``, this is the *only* place a Step1 run
reaches ``sbatch``.  The default is a non-mutating launch plan; ``execute=True``
is the sole path that submits, and every root is fully preflighted before the
first job is sent.  A run is launchable when it was written by
``step1-prepare`` (hashes still match ``step1_manifest.json``) or by
``step1-repair`` (``step1_repair.json`` is ``PREPARED``), it carries no runtime
outputs, and it is not already recorded as submitted in ``step1_launch.json``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .errors import SafetyError
from .step1_repair import _discover_runs
from .vasp import _sha256_file, resolve_launcher, submit_run

_STARTED_MARKERS = ("OUTCAR", "OSZICAR", "vasprun.xml")
_REQUIRED_INPUTS = ("INCAR", "POSCAR", "KPOINTS")


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _already_submitted(root: Path) -> set[str]:
    payload = _json_or_empty(root / "step1_launch.json")
    return {
        str(row.get("relative_path"))
        for row in payload.get("runs", [])
        if row.get("status") == "SUBMITTED"
    }


def _manifest_rows(root: Path) -> dict[str, dict[str, Any]]:
    payload = _json_or_empty(root / "step1_manifest.json")
    return {str(row.get("relative_path")): row for row in payload.get("runs", [])}


def _preflight_run(
    run: Path,
    tree_root: Path,
    *,
    manifest_rows: dict[str, dict[str, Any]],
    submitted: set[str],
    launcher: str | None,
    only_repaired: bool,
    emit: Callable[[str], None],
) -> tuple[dict[str, Any] | None, str]:
    """Return ``(plan, skip_reason)``; exactly one is truthy."""

    relative = run.relative_to(tree_root).as_posix() if run != tree_root else "."

    started = [name for name in _STARTED_MARKERS if _nonempty(run / name)]
    if started:
        return None, f"already started ({', '.join(started)})"
    if relative in submitted:
        return None, "already recorded as submitted in step1_launch.json"

    repair = _json_or_empty(run / "step1_repair.json")
    if repair:
        if repair.get("status") != "PREPARED":
            return None, f"step1_repair.json status is {repair.get('status')!r}, not PREPARED"
        kind = "repair-prepared"
    elif only_repaired:
        return None, "not a repaired run (--only-repaired)"
    elif relative in manifest_rows:
        row = manifest_rows[relative]
        for name, key in (("INCAR", "step1_incar_sha256"), ("POSCAR", "step1_poscar_sha256")):
            path = run / name
            expected = row.get(key)
            if not path.is_file() or not expected or _sha256_file(path) != expected:
                raise SafetyError(
                    f"{path} changed since step1-prepare; re-run "
                    "'iface vasp step1-prepare --audit-only' and inspect before launching"
                )
        kind = "prepared"
    else:
        return None, "not written by step1-prepare or step1-repair"

    for name in _REQUIRED_INPUTS:
        if not _nonempty(run / name):
            raise SafetyError(f"{run} is missing {name}")
    script = resolve_launcher(run, launcher)

    notes: list[str] = []
    if not _nonempty(run / "POTCAR"):
        notes.append("POTCAR absent; the launcher must generate it")
    emit(f"[{tree_root.name}] preflight OK: {relative} ({kind}, launcher={script.name})")
    return {
        "root": str(tree_root),
        "relative_path": relative,
        "directory": str(run),
        "launcher": script.name,
        "kind": kind,
        "notes": "; ".join(notes),
    }, ""


def launch_step1_runs(
    roots: Iterable[str | Path],
    *,
    execute: bool = False,
    launcher: str | None = None,
    only_repaired: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Plan (default) or submit prepared/repaired Step1 runs."""

    emit = progress or (lambda _message: None)
    resolved = [Path(value).expanduser().resolve() for value in roots]
    if not resolved:
        raise SafetyError("At least one Step1 root (or run directory) is required")

    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for tree_root in resolved:
        if not tree_root.is_dir():
            raise FileNotFoundError(tree_root)
        submitted = _already_submitted(tree_root)
        manifest_rows = _manifest_rows(tree_root)
        runs = _discover_runs(tree_root)
        if not runs:
            raise SafetyError(f"No Step1 run directories found under {tree_root}")
        emit(f"[{tree_root.name}] preflighting {len(runs)} run directory(ies)")
        for run in runs:
            plan, reason = _preflight_run(
                run,
                tree_root,
                manifest_rows=manifest_rows,
                submitted=submitted,
                launcher=launcher,
                only_repaired=only_repaired,
                emit=emit,
            )
            if plan is not None:
                planned.append(plan)
            else:
                relative = run.relative_to(tree_root).as_posix() if run != tree_root else "."
                skipped.append({"relative_path": relative, "root": str(tree_root), "reason": reason})
                emit(f"[{tree_root.name}] skip {relative}: {reason}")

    if not planned:
        raise SafetyError(
            "No launchable Step1 runs (nothing prepared/repaired and idle). "
            + (f"{len(skipped)} directory(ies) skipped." if skipped else "")
        )
    emit(f"Preflight PASS: {len(planned)} run(s) ready to submit, {len(skipped)} skipped")

    if not execute:
        emit("Dry run only; no jobs submitted. Re-run with --execute to submit.")
        return {
            "format": "interfaceforge-step1-launch-plan",
            "mode": "dry-run",
            "roots": [str(root) for root in resolved],
            "runs": len(planned),
            "skipped": len(skipped),
            "preflight": "PASS",
            "submission": "not performed; pass --execute after review",
            "planned": planned,
            "skipped_runs": skipped,
        }

    rows: list[dict[str, Any]] = []
    failure: str | None = None
    for index, item in enumerate(planned, start=1):
        emit(f"[{index}/{len(planned)}] sbatch {item['relative_path']} in {item['directory']}")
        try:
            job_id = submit_run(item["directory"], item["launcher"])
            rows.append({**item, "status": "SUBMITTED", "job_id": job_id, "detail": ""})
            emit(f"    submitted job {job_id}")
        except Exception as exc:  # noqa: BLE001 - recorded and re-raised below
            failure = f"{item['directory']}: {exc}"
            rows.append({**item, "status": "FAILED", "job_id": "", "detail": str(exc)})
            emit(f"    FAILED: {exc}")
            break

    reports: list[str] = []
    for tree_root in resolved:
        root_rows = [row for row in rows if row["root"] == str(tree_root)]
        if not root_rows:
            continue
        if any(row["status"] == "FAILED" for row in root_rows):
            root_status = "FAILED"
        else:
            root_status = "SUBMITTED"
        payload = {
            "format": "interfaceforge-step1-launch",
            "schema_version": 1,
            "status": root_status,
            "root": str(tree_root),
            "preflight": "PASS",
            "runs": root_rows,
        }
        json_path = tree_root / "step1_launch.json"
        tsv_path = tree_root / "step1_launch.tsv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("status", "job_id", "kind", "relative_path", "directory", "launcher", "detail"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in writer.fieldnames} for row in root_rows)
        reports.extend((str(json_path), str(tsv_path)))
        emit(f"[{tree_root.name}] {root_status}; wrote {json_path.name}, {tsv_path.name}")

    if failure is not None:
        raise SafetyError(
            f"Step1 launch stopped after a submission failure ({failure}). "
            f"Review partial launch records: {', '.join(reports)}"
        )
    emit(f"Done: {len(rows)} job(s) submitted across {len(resolved)} root(s)")
    return {
        "format": "interfaceforge-step1-launch",
        "mode": "submitted",
        "roots": [str(root) for root in resolved],
        "runs": len(rows),
        "skipped": len(skipped),
        "preflight": "PASS",
        "submitted": len(rows),
        "reports": reports,
        "jobs": rows,
    }
