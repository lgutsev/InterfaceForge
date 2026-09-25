# ruff: noqa: E501
"""Runtime status of a prepared Step1 preheat tree.

Read-only. For every run under a ``Step1/`` root (or a single run directory)
this reports what a human checks by hand:

* **frames produced** -- MD steps written so far (from ``OSZICAR``, cross
  checked against ``XDATCAR``), versus the ``NSW`` target;
* **the INCAR** -- the electronic-quality knobs (``ENCUT``, ``PREC``,
  ``EDIFF``, ``ALGO``, ``LREAL``, smearing) and the physics inherited from
  OPT (``ISPIN``, the Hubbard ``U`` values, ``IBRION``/``NSW``/``POTIM``),
  plus ``ENCUT/ENMAX`` when a ``POTCAR`` is present;
* **which job is done** -- ``not-started`` / ``repair-prepared`` /
  ``resume-prepared`` / ``queued`` / ``running`` / ``stalled?`` /
  ``interrupted`` / ``done`` / ``done-early`` / ``error`` / ``unstable``, from
  ``OUTCAR`` completion markers, the last written step and (when ``squeue``
  answers) the Slurm queue;
* **the lineage** -- the current generation (original / repair / resume), the
  accepted prefix before it, the cumulative progress, and whether the current
  generation was already submitted (``step1_launch.json`` ledgers);
* **the recovery category** -- ``done`` / ``resume`` / ``repair`` / ``launch`` /
  ``review`` / ``active`` (:func:`recovery_category`, reused by
  ``step1-recover``).

Thermal state is reported separately from readiness.  ``thermal_tail_ok`` only
says the late-window mean temperature (last <= 50 steps) reached 5/6 of the
target -- a 100->300 K ramp has a whole-run mean near 200 K, so the tail is the
diagnostic.  ``ready_for_step2`` additionally needs the run to be complete and
the trajectory free of hard instability; ``thermal_ready`` is the legacy
"tail ok AND stable" flag, kept with its old meaning.

Never touches a running job; only reads generated files and (optionally) asks
``squeue``.  A failing scheduler query is reported, never raised.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aimd import _first_float, _first_int, preheat_ps
from .audit import parse_oszicar, read_tail
from .step1_lineage import (
    ACTIVITY_FILES,
    MANIFEST,
    REPAIR_RECORD,
    RESUME_RECORD,
    Generation,
    accepted_ps,
    current_generation,
    incar_schedule,
    interrupted_archive,
    ledger_paths_for,
    read_json,
    rows_for_run,
    schedule_temperature,
    submission_state,
)
from .step1_lineage import _int as _finite_int
from .step1_repair import diagnose_step1_run, parse_step1_oszicar
from .step1_scheduler import (
    SCHEDULER_MODES,
    SchedulerGuard,
    SchedulerSnapshot,
    as_guard,
    resolve_stale_hours,
)
from .vasp import _sha256_file, parse_incar

_EXCLUDED = ("archive", "backup", ".interfaceforge", "precondition")
_TIMING_MARKER = "General timing and accounting informations"
_XDATCAR_FRAME = re.compile(r"^\s*Direct configuration=", re.MULTILINE)
_ENMAX = re.compile(r"ENMAX\s*=\s*([0-9.]+)")
_ERROR_MARKERS = (
    "VERY BAD NEWS",
    "ZBRENT: fatal error",
    "internal error",
    "Error EDDDAV",
    "The distance between some ions is very small",
    "WARNING: DENTET",
    "PRICEL",
)
_STALE_HOURS_DEFAULT = 6.0

# Files whose modification time says a run is still being written; the same
# set step1-resume uses for its ACTIVE_OR_RECENT guard, so both agree.
_ACTIVITY_FILES = ACTIVITY_FILES
# squeue states of a job that has not started running yet.
_QUEUED_STATES = frozenset({"PENDING", "CONFIGURING"})
_NOT_STARTED_STATES = frozenset({"not-started", "repair-prepared", "resume-prepared"})
_MAX_MANIFEST_ANCESTORS = 8
# Render/tally order of the recovery categories (same order as step1-recover).
_CATEGORY_ORDER = ("active", "done", "review", "launch", "resume", "repair")
_THERMAL_TAIL_STEPS = 50
# What step1-launch needs before it will submit a run (its _REQUIRED_INPUTS,
# default launchers and extra "already started" marker), mirrored read-only so
# the ``launch`` category never names a run that launch would refuse.
_LAUNCH_REQUIRED_INPUTS = ("INCAR", "POSCAR", "KPOINTS")
_DEFAULT_LAUNCHERS = ("runvasp.sh", "run.slurm")
_LAUNCH_STARTED_EXTRA = ("vasprun.xml",)


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="seconds") if moment else None


def _age_hours(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    return (datetime.now(tz=timezone.utc) - moment).total_seconds() / 3600.0


def _discover_runs(root: Path) -> list[Path]:
    if (root / "INCAR").is_file():
        return [root]
    runs: list[Path] = []
    for incar in sorted(root.rglob("INCAR")):
        parts = {part.lower() for part in incar.parent.relative_to(root).parts}
        if parts & set(_EXCLUDED) or any(p.startswith("x") for p in parts):
            continue
        runs.append(incar.parent)
    return runs


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _xdatcar_frames(path: Path) -> int | None:
    if not _nonempty(path):
        return None
    return len(_XDATCAR_FRAME.findall(read_tail(path)))


def _potcar_enmax(path: Path) -> float | None:
    if not _nonempty(path):
        return None
    values = [float(v) for v in _ENMAX.findall(read_tail(path))]
    return max(values) if values else None


def _incar_summary(incar: dict[str, str], potcar_enmax: float | None) -> dict[str, Any]:
    encut = _first_float(incar.get("ENCUT"))
    ldauu = incar.get("LDAUU")
    summary: dict[str, Any] = {
        "istart": _first_int(incar.get("ISTART")),
        "encut_ev": encut,
        "prec": incar.get("PREC"),
        "ediff": incar.get("EDIFF"),
        "algo": incar.get("ALGO"),
        "lreal": incar.get("LREAL"),
        "ismear": _first_int(incar.get("ISMEAR")),
        "sigma": _first_float(incar.get("SIGMA")),
        "ispin": _first_int(incar.get("ISPIN")),
        "ldau": incar.get("LDAU"),
        "ldauu": ldauu,
        "lmaxmix": _first_int(incar.get("LMAXMIX")),
        "ibrion": _first_int(incar.get("IBRION")),
        "nsw": _first_int(incar.get("NSW")),
        "potim_fs": _first_float(incar.get("POTIM")),
        "smass": incar.get("SMASS"),
        "nblock": _first_int(incar.get("NBLOCK")),
        "tebeg": _first_float(incar.get("TEBEG")),
        "teend": _first_float(incar.get("TEEND")),
        "ivdw": _first_int(incar.get("IVDW")),
        "encut_over_enmax": None,
    }
    if encut and potcar_enmax:
        summary["encut_over_enmax"] = round(encut / potcar_enmax, 3)
    return summary


def _classify(
    *,
    has_incar: bool,
    outcar_tail: str,
    started: bool,
    last_step: int | None,
    nsw: int | None,
    updated: datetime | None,
    stale_hours: float,
) -> tuple[str, bool]:
    """Return ``(state, stale)`` from the files alone (no scheduler, no lineage)."""

    if not has_incar:
        return "no-incar", False
    if not started:
        return "not-started", False
    finished = _TIMING_MARKER in outcar_tail
    has_error = any(marker in outcar_tail for marker in _ERROR_MARKERS)
    if finished:
        if nsw and last_step is not None and last_step >= nsw:
            return "done", False
        return ("error" if has_error else "done-early"), False
    if has_error:
        return "error", False
    stale = False
    if updated is not None:
        age_h = (datetime.now(tz=timezone.utc) - updated).total_seconds() / 3600.0
        stale = age_h > stale_hours
    return ("stalled?" if stale else "running"), stale


# --------------------------------------------------------------------------- #
# Scheduler (read-only, non-fatal)
# --------------------------------------------------------------------------- #


def _printable(text: str) -> str:
    """``text`` with lone surrogates (non-UTF-8 WorkDir bytes) spelled out, so it always prints."""

    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _scheduler_payload(guard: SchedulerGuard, snapshot: SchedulerSnapshot | None, error: str | None) -> dict[str, Any]:
    if snapshot is not None:
        return snapshot.to_dict()
    return {
        "requested": guard.mode,
        "mode": "unknown",
        "verified": False,
        "reason": error or "scheduler not queried",
        "taken_at": None,
        "active_jobs": None,
        "error": error,
    }


def _row_scheduler(run: Path, snapshot: SchedulerSnapshot | None, error: str | None) -> dict[str, Any]:
    """``{"verified", "active", "jobs"}`` for one run.

    ``active`` is True when the snapshot lists a job whose WorkDir is (inside)
    the run, False when a *verified* snapshot does not, and None when nothing
    can be said (scheduler unverified or the query failed).  A job listed by an
    unverified snapshot still counts as active: evidence of activity is never
    ignored.
    """

    if snapshot is None:
        return {"verified": False, "active": None, "jobs": [], "error": error}
    jobs = [
        {
            "job_id": _printable(str(job.get("job_id"))),
            "state": _printable(str(job.get("state"))),
            "workdir": _printable(str(job.get("workdir") or "")),
        }
        for job in snapshot.active_jobs_for(run)
    ]
    active: bool | None = True if jobs else (False if snapshot.verified else None)
    return {"verified": bool(snapshot.verified), "active": active, "jobs": jobs}


def _resolve_stale(
    stale_hours: float | None, snapshot: SchedulerSnapshot | None, error: str | None, requested: str
) -> tuple[float, str]:
    """``resolve_stale_hours`` that also works when the scheduler query failed (treated as unverified)."""

    if snapshot is None:
        snapshot = SchedulerSnapshot(
            requested=requested,
            mode="none",
            verified=False,
            reason=f"scheduler query failed: {error}" if error else "scheduler not queried",
            taken_at="",
            jobs=[],
        )
    return resolve_stale_hours(stale_hours, snapshot)


# --------------------------------------------------------------------------- #
# Lineage + launchability
# --------------------------------------------------------------------------- #


def _completed_steps(rows: list[dict[str, Any]]) -> int:
    """Completed ionic steps of the current segment: MD rows numbered 1, 2, 3, ... without a gap.

    The same rule step1-resume uses for ``n_osz`` (rows from
    ``parse_step1_oszicar``, which already drops a torn final line).
    """

    count = 0
    for row in rows:
        if row.get("step") != count + 1:
            break
        count += 1
    return count


def _ps(steps: int | None, potim_fs: float | None) -> float | None:
    if steps is None or potim_fs is None:
        return None
    return round(float(steps) * float(potim_fs) / 1000.0, 9)


def _public_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """A ledger/record submission row without the ``_``-prefixed matching annotations."""

    if not isinstance(row, dict):
        return None
    public = {key: value for key, value in row.items() if not str(key).startswith("_")}
    if row.get("_ledger") and "ledger" not in public:
        public["ledger"] = row["_ledger"]
    return public


def _lineage(
    run: Path, generation: Generation, *, schedule: dict[str, Any], segment_steps: int, root: Path
) -> dict[str, Any]:
    """The lineage block: generation, accepted prefix + current segment, ps, schedule, submissions."""

    prefix = int(generation.accepted_prefix_steps)
    cumulative = prefix + segment_steps
    original = generation.original_nsw
    ledger = [dict(row) for row in generation.ledger]
    # Accepted prefix time from the ledger; unknown when the ledger does not
    # add up to the prefix (hand edit, broken legacy chain) or lacks a POTIM.
    ledger_steps = sum(_finite_int(row.get("steps"), 0) or 0 for row in ledger)
    prefix_ps = accepted_ps(ledger) if ledger_steps == prefix else None
    # The INCAR is what VASP actually integrates with; the record is the fallback.
    potim = schedule["potim_fs"] if schedule["potim_fs"] is not None else generation.segment_potim_fs
    segment_ps = _ps(segment_steps, potim)
    total_ps = round(prefix_ps + segment_ps, 9) if prefix_ps is not None and segment_ps is not None else None
    ramp_nsw = schedule["nsw"] if schedule["nsw"] else generation.segment_nsw
    tebeg, teend = schedule["tebeg_k"], schedule["teend_k"]

    rows = rows_for_run(run, ledger_paths_for(run, [root]))
    submissions = submission_state(run, generation, rows)
    current_submissions = submissions.get("current_submissions") or []
    record_name = generation.record_path.name if generation.record_path is not None else None

    warnings: list[str] = []
    if generation.conflict:
        warnings.append(generation.conflict)
    if generation.status == "UNREADABLE" and record_name:
        warnings.append(f"{record_name} exists but cannot be parsed")
    if len(current_submissions) > 1:
        jobs = ", ".join(str(item.get("job_id") or "?") for item in current_submissions)
        warnings.append(
            f"current generation recorded as submitted {len(current_submissions)} times (jobs {jobs}); "
            "check for duplicate jobs"
        )
    for path in submissions.get("unreadable_ledgers") or []:
        warnings.append(
            f"launch ledger {path} is unreadable; the current generation counts as submitted until it is "
            "repaired or moved aside"
        )
    if record_name and generation.status not in ("UNREADABLE",):
        if generation.segment_nsw is not None and schedule["nsw"] is not None and generation.segment_nsw != schedule["nsw"]:
            warnings.append(f"INCAR NSW {schedule['nsw']} differs from {record_name} segment NSW {generation.segment_nsw}")
        record_potim = generation.segment_potim_fs
        if (
            record_potim is not None
            and schedule["potim_fs"] is not None
            and not math.isclose(record_potim, schedule["potim_fs"], rel_tol=1e-9, abs_tol=1e-12)
        ):
            warnings.append(f"INCAR POTIM {schedule['potim_fs']:g} differs from {record_name} segment POTIM {record_potim:g}")

    archive = interrupted_archive(run)
    return {
        "generation": generation.generation,
        "generation_id": generation.generation_id,
        "segment_kind": generation.kind,
        "legacy_record": bool(generation.legacy and generation.record_path is not None),
        "record": record_name,
        "record_status": generation.status,
        "conflict": generation.conflict,
        "prepared_at": generation.prepared_at,
        "original_nsw": original,
        "accepted_prefix_steps": prefix,
        "current_segment_steps": segment_steps,
        "cumulative_steps": cumulative,
        "remaining_steps": max(0, original - cumulative) if original is not None else None,
        "segment_nsw": generation.segment_nsw,
        "segment_potim_fs": generation.segment_potim_fs,
        "accepted_prefix_ps": prefix_ps,
        "current_segment_ps": segment_ps,
        "accepted_total_ps": total_ps,
        "segments": ledger,
        "ledger_exact": bool(generation.ledger_exact),
        "temperature": {
            "segment_tebeg_k": tebeg,
            "segment_teend_k": teend,
            "segment_nsw": ramp_nsw,
            "ramp": bool(schedule["ramp"]),
            "current_target_k": schedule_temperature(tebeg, teend, ramp_nsw, segment_steps),
            "thermostat": schedule["thermostat"],
        },
        "submitted": bool(submissions.get("current_submitted")),
        "submission": _public_row(submissions.get("current_submission")),
        "submission_count": len(current_submissions),
        "historical_submissions": [_public_row(row) for row in submissions.get("historical_submissions") or []],
        "submission_rule": submissions.get("match_rule"),
        "unreadable_ledgers": list(submissions.get("unreadable_ledgers") or []),
        "interrupted_mutation": str(archive) if archive is not None else None,
        "warnings": warnings,
    }


def _manifest_entry(run: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """``(nearest step1_manifest.json at or above run, its row for run or None)``."""

    current = run
    for _ in range(_MAX_MANIFEST_ANCESTORS + 1):
        manifest = current / MANIFEST
        if manifest.is_file():
            relative = run.relative_to(current).as_posix() if run != current else "."
            rows = read_json(manifest).get("runs")
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict) and str(row.get("relative_path")) == relative:
                    return manifest, row
            return manifest, None
        if current.parent == current:
            break
        current = current.parent
    return None, None


def _prepared_kind(generation: Generation) -> str:
    """``repair-prepared`` / ``resume-prepared`` for a segment record (the rule step1-launch uses).

    The record's ``segment_kind`` decides; an unexpected kind falls back to the
    record's file name.
    """

    if generation.kind in ("repair", "resume"):
        return f"{generation.kind}-prepared"
    name = generation.record_path.name if generation.record_path is not None else ""
    return "resume-prepared" if name == RESUME_RECORD else "repair-prepared"


def _default_launcher(run: Path) -> str | None:
    """The launcher ``step1-launch`` picks without ``--launcher`` (``resolve_launcher``), read-only."""

    for name in _DEFAULT_LAUNCHERS:
        if (run / name).is_file():
            return name
    return None


def _launch_info(run: Path, generation: Generation) -> dict[str, Any]:
    """Whether the (not yet started) current generation is launchable, mirroring step1-launch's preflight.

    Gen 0 must be listed in the nearest ``step1_manifest.json`` with matching
    INCAR/POSCAR hashes; a repair/resume generation needs its record to be
    ``PREPARED``.  Either way the inputs launch requires and a default launcher
    must exist, else launch would refuse the run (the category is then
    ``review``, never ``launch``).  The duplicate-submission and scheduler checks
    live in :func:`recovery_category`.
    """

    launcher = _default_launcher(run)
    if generation.record_path is not None:
        name = generation.record_path.name
        kind = _prepared_kind(generation)
        if generation.status != "PREPARED":
            reason = f"{name} status is {generation.status!r}, not PREPARED"
            if generation.conflict:
                reason += f" ({generation.conflict})"
            return {"launchable": False, "kind": kind, "reason": reason, "launcher": launcher}
        basis = f"{name} is PREPARED"
    else:
        kind = "prepared"
        manifest, entry = _manifest_entry(run)
        if manifest is None:
            return {
                "launchable": False,
                "kind": None,
                "reason": "not written by step1-prepare (no step1_manifest.json)",
                "launcher": launcher,
            }
        if entry is None:
            return {"launchable": False, "kind": None, "reason": f"not listed in {manifest}", "launcher": launcher}
        for name, key in (("INCAR", "step1_incar_sha256"), ("POSCAR", "step1_poscar_sha256")):
            path = run / name
            expected = entry.get(key)
            try:
                matches = bool(expected) and path.is_file() and _sha256_file(path) == expected
            except OSError:
                matches = False
            if not matches:
                return {
                    "launchable": False,
                    "kind": kind,
                    "reason": f"{name} changed since step1-prepare (hash differs from {manifest.name}); inspect before launching",
                    "launcher": launcher,
                }
        basis = f"listed in {manifest.name} with matching INCAR/POSCAR hashes"

    started = [name for name in _LAUNCH_STARTED_EXTRA if _nonempty(run / name)]
    if started:
        return {"launchable": False, "kind": kind, "reason": f"already started ({', '.join(started)})", "launcher": launcher}
    missing = [name for name in _LAUNCH_REQUIRED_INPUTS if not _nonempty(run / name)]
    if missing:
        return {"launchable": False, "kind": kind, "reason": f"missing {', '.join(missing)}", "launcher": launcher}
    if launcher is None:
        return {
            "launchable": False,
            "kind": kind,
            "reason": f"no VASP launcher ({' or '.join(_DEFAULT_LAUNCHERS)})",
            "launcher": None,
        }
    return {"launchable": True, "kind": kind, "reason": basis, "launcher": launcher}


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def _run_status(
    run: Path,
    *,
    root: Path,
    stale_hours: float,
    snapshot: SchedulerSnapshot | None = None,
    scheduler_error: str | None = None,
) -> dict[str, Any]:
    incar_path = run / "INCAR"
    incar = parse_incar(incar_path)
    potcar_enmax = _potcar_enmax(run / "POTCAR")
    summary = _incar_summary(incar, potcar_enmax)
    nsw = summary["nsw"]
    potim = summary["potim_fs"]

    oszicar = parse_oszicar(run / "OSZICAR")
    frames_segment = oszicar["md_steps"] or 0
    last_step = oszicar["last_oszicar_step"]
    frames_xdatcar = _xdatcar_frames(run / "XDATCAR")

    # The current generation (gen 0, a new record, or a reconstructed legacy
    # repair) supplies the accepted prefix and the whole-run target.
    generation = current_generation(run, incar)
    repair = read_json(run / REPAIR_RECORD)
    prefix_steps = int(generation.accepted_prefix_steps)
    original_target = generation.original_nsw or nsw
    frames_oszicar = prefix_steps + frames_segment

    outcar = run / "OUTCAR"
    outcar_tail = read_tail(outcar) if _nonempty(outcar) else ""
    started = _nonempty(outcar) or _nonempty(run / "OSZICAR")
    # One activity moment for the state, the stale flag, "updated" and the
    # recovery category: the newest of OSZICAR/OUTCAR/CONTCAR/XDATCAR.
    updated = max((moment for moment in (_mtime(run / name) for name in _ACTIVITY_FILES) if moment), default=None)

    state, stale = _classify(
        has_incar=incar_path.is_file(),
        outcar_tail=outcar_tail,
        started=started,
        last_step=last_step,
        nsw=nsw,
        updated=updated,
        stale_hours=stale_hours,
    )
    age_hours = _age_hours(updated)
    if state == "not-started" and generation.record_path is not None:
        # A segment record with nothing started yet: prepared by repair/resume,
        # even when the accepted prefix is 0 (rewind to the segment start).
        state = _prepared_kind(generation)
    stability = diagnose_step1_run(run)
    if stability["unstable"] and state not in {"no-incar"} | _NOT_STARTED_STATES:
        # A VASP timing footer only means the executable exited normally.  It
        # does not make a physically/numerically runaway trajectory usable.
        state = "unstable"

    # Scheduler state beats file age: a listed job is queued/running however
    # old its files are; a verified absence turns running/stalled? into
    # interrupted.  Unverified -> the file-based state stands.
    scheduler = _row_scheduler(run, snapshot, scheduler_error)
    if scheduler["active"]:
        job_states = {str(job["state"]).upper() for job in scheduler["jobs"]}
        state = "queued" if job_states and job_states <= _QUEUED_STATES else "running"
    elif scheduler["active"] is False and state in {"running", "stalled?"}:
        state = "interrupted"

    thermal_steps = parse_step1_oszicar(run / "OSZICAR", nelm=stability["scf_nelm"])["steps"]
    segment_steps = _completed_steps(thermal_steps)
    valid_temperatures = [row["temperature_k"] for row in thermal_steps if row["temperature_k"] is not None]
    tail_count = min(_THERMAL_TAIL_STEPS, len(valid_temperatures))
    tail_temperatures = valid_temperatures[-tail_count:] if tail_count else []
    thermal_target_k = summary["teend"] or summary["tebeg"]
    thermal_tail_mean_k = sum(tail_temperatures) / len(tail_temperatures) if tail_temperatures else None
    thermal_tail_min_k = min(tail_temperatures) if tail_temperatures else None
    thermal_tail_max_k = max(tail_temperatures) if tail_temperatures else None
    thermal_ready_threshold_k = (5.0 / 6.0) * thermal_target_k if thermal_target_k is not None else None
    # Thermal state alone (independent of stability); None when it cannot be judged.
    thermal_tail_ok: bool | None = None
    if thermal_tail_mean_k is not None and thermal_ready_threshold_k is not None:
        thermal_tail_ok = thermal_tail_mean_k >= thermal_ready_threshold_k
    trajectory_stable = not stability["unstable"]
    # Legacy flag, unchanged meaning: tail ok AND trajectory stable.
    thermal_ready = thermal_tail_ok is True and trajectory_stable

    lineage = _lineage(run, generation, schedule=incar_schedule(incar), segment_steps=segment_steps, root=root)
    cumulative = lineage["cumulative_steps"]
    whole_target = lineage["original_nsw"]
    complete = bool(
        whole_target is not None
        and cumulative >= whole_target
        and scheduler["active"] is not True
        and state not in {"running", "queued"}
    )
    ready_for_step2 = bool(complete and trajectory_stable and thermal_tail_ok is True)
    severity = stability.get("severity") or ("unstable" if stability["unstable"] else "ok")

    # Accepted time from the ledger; a ledger that cannot price the prefix
    # (broken legacy chain) falls back to the record's original POTIM.
    prefix_ps = lineage["accepted_prefix_ps"]
    if prefix_ps is None:
        prefix_ps = preheat_ps(prefix_steps, _first_float(repair.get("original_potim_fs"), potim)) or 0.0
    segment_potim = potim if potim is not None else generation.segment_potim_fs
    produced_ps = prefix_ps + (_ps(segment_steps, segment_potim) or 0.0)
    target_ps = None
    if original_target:
        segment_target = generation.segment_nsw if generation.segment_nsw is not None else nsw
        target_ps = prefix_ps + (preheat_ps(segment_target, segment_potim) or 0.0)

    activity_moments = [moment for moment in (_mtime(run / name) for name in _ACTIVITY_FILES) if moment is not None]
    newest = max(activity_moments) if activity_moments else None
    activity_age = _age_hours(newest)
    activity = {
        "updated": _iso(newest),
        "age_hours": activity_age,
        "window_hours": stale_hours,
        "recent": activity_age is not None and activity_age < stale_hours,
    }

    row: dict[str, Any] = {
        "run": run.name,
        "path": str(run),
        "relative_path": run.relative_to(root).as_posix() if run != root else ".",
        "state": state,
        "started": started,
        "stale": stale,
        "age_hours": age_hours,
        "frames_oszicar": frames_oszicar,
        "frames_oszicar_segment": frames_segment,
        "accepted_prefix_steps": prefix_steps,
        "frames_xdatcar": frames_xdatcar,
        "last_step": last_step,
        "nsw_target": original_target,
        "nsw_segment_target": nsw,
        "percent_complete": (
            round(100.0 * frames_oszicar / original_target, 1) if original_target and frames_oszicar is not None else None
        ),
        "produced_ps": produced_ps if cumulative else None,
        "target_ps": target_ps,
        "temperature_mean_k": oszicar["temperature_mean_k"],
        "temperature_std_k": oszicar["temperature_std_k"],
        "temperature_last_k": oszicar["temperature_last_k"],
        "thermal_target_k": thermal_target_k,
        "thermal_tail_window_steps": tail_count,
        "thermal_tail_mean_k": thermal_tail_mean_k,
        "thermal_tail_min_k": thermal_tail_min_k,
        "thermal_tail_max_k": thermal_tail_max_k,
        "thermal_ready_threshold_k": thermal_ready_threshold_k,
        "thermal_ready": thermal_ready,
        "thermal_tail_ok": thermal_tail_ok,
        "trajectory_stable": trajectory_stable,
        "complete": complete,
        "ready_for_step2": ready_for_step2,
        "review_required": severity == "warning",
        "severity": severity,
        "wavecar_present": _nonempty(run / "WAVECAR"),
        "contcar_present": _nonempty(run / "CONTCAR"),
        "potcar_present": _nonempty(run / "POTCAR"),
        "potcar_enmax_ev": potcar_enmax,
        "updated": _iso(updated),
        "activity": activity,
        "incar": summary,
        "stability": stability,
        "repair": repair or None,
        "scheduler": scheduler,
        "lineage": lineage,
        "launch": None if started else _launch_info(run, generation),
    }
    row["recovery"] = recovery_category(row)
    return row


# --------------------------------------------------------------------------- #
# Recovery category
# --------------------------------------------------------------------------- #


def _first_warning(stability: dict[str, Any]) -> str:
    warnings = stability.get("warnings") or []
    return str(warnings[0]) if warnings else "review-level warning"


# Short labels for review reasons and the frames line; the full warning text is
# printed once, on the stability line (and kept in --json).
_WARNING_LABELS: dict[str, tuple[str, str | None]] = {
    "startup_energy_excursion": ("startup excursion", "startup_excursion_steps"),
    "isolated_energy_spike": ("isolated energy spike", "isolated_spike_steps"),
    "scf_elevated": ("elevated SCF ceiling use", None),
    "temperature_elevated": ("elevated temperature", None),
}


def _warning_label(stability: dict[str, Any]) -> str:
    """``benign startup transient (step 1)`` / ``startup excursion (step 1) + isolated energy spike (step 250)``."""

    classes = stability.get("warning_classes") or []
    if not classes:
        return _first_warning(stability)
    parts: list[str] = []
    for name in classes:
        label, steps_key = _WARNING_LABELS.get(str(name), (str(name).replace("_", " "), None))
        steps = stability.get(steps_key) if steps_key else None
        if steps:
            label += f" (step {steps[0]})" if len(steps) == 1 else f" (steps {steps[0]}-{steps[-1]})"
        parts.append(label)
    text = " + ".join(parts)
    if stability.get("benign_warnings_only"):
        text = text.replace("startup excursion", "benign startup transient", 1)
    return text


def recovery_category(row: dict[str, Any]) -> dict[str, str]:
    """``{"category", "reason"}`` for one status row; the first matching rule wins.

    1. scheduler lists a job -> ``active``;
    2. an archive left ``IN_PROGRESS`` (interrupted recovery mutation), or an
       unreadable / conflicting segment record -> ``review``;
    3. no INCAR -> ``review``;
    4. not started: launchable and never submitted -> ``launch``; submitted ->
       ``review`` when Slurm is verified (not queued, no output), else
       ``active``; otherwise ``review``;
    5. files modified within the activity window -> ``active``;
    6. hard-unstable -> ``repair``;
    7. complete: Step2-ready and severity ok -> ``done``; ready with a warning
       (or with a VASP error marker and no timing footer) -> ``review``; else
       ``review`` (thermal tail below threshold);
    8. incomplete: VASP error, a non-benign warning or no completed ionic step
       -> ``review``; else ``resume``.

    Only ``resume``, ``repair`` and ``launch`` are ever acted on automatically
    (by step1-recover); everything else is left untouched.
    """

    scheduler = row.get("scheduler") or {}
    lineage = row.get("lineage") or {}
    stability = row.get("stability") or {}
    state = row.get("state")

    def verdict(category: str, reason: str) -> dict[str, str]:
        return {"category": category, "reason": reason}

    # 1. A listed Slurm job beats every file-based rule.
    if scheduler.get("active"):
        jobs = scheduler.get("jobs") or []
        listing = ", ".join(f"{job.get('job_id')} {job.get('state')}" for job in jobs) or "listed"
        return verdict("active", f"Slurm job {listing}")

    # 2. Half-done recovery or records that cannot be trusted: never mutate.
    if lineage.get("interrupted_mutation"):
        return verdict("review", f"interrupted recovery mutation; inspect {lineage['interrupted_mutation']}")
    if lineage.get("record_status") in ("UNREADABLE", "CONFLICT"):
        detail = lineage.get("conflict") or f"{lineage.get('record') or 'segment record'} cannot be parsed"
        return verdict("review", f"segment record {lineage['record_status']}: {detail}; inspect before any recovery")

    # 3.
    if state == "no-incar":
        return verdict("review", "no INCAR")

    # 4. Nothing has run in the current generation yet.
    started = row.get("started")
    if started is None:
        started = state not in _NOT_STARTED_STATES
    if not started:
        submission = lineage.get("submission") or {}
        if lineage.get("submitted"):
            if submission.get("unreadable_ledger"):
                return verdict(
                    "review",
                    f"launch ledger {submission.get('ledger') or '?'} is unreadable; repair or move it aside before launching",
                )
            job = submission.get("job_id") or "?"
            if scheduler.get("verified"):
                return verdict("review", f"submitted (job {job}) but not queued and no output; inspect slurm log")
            return verdict("active", f"submitted (job {job}); scheduler not verified")
        launch = row.get("launch") or {}
        if launch.get("launchable"):
            return verdict("launch", f"{launch.get('kind')}, never submitted ({launch.get('reason')})")
        return verdict("review", f"not started and not launchable: {launch.get('reason') or 'unknown'}")

    # 5. Recently written files: treat as still running.
    activity = row.get("activity") or {}
    if activity.get("recent"):
        age = activity.get("age_hours")
        minutes = max(0, round(float(age) * 60.0)) if age is not None else 0
        window = activity.get("window_hours")
        suffix = f" (activity window {float(window):g} h)" if window is not None else ""
        return verdict("active", f"updated {minutes} min ago{suffix}")

    # 6. Hard instability -> conservative repair.
    if stability.get("unstable") or row.get("trajectory_stable") is False:
        reasons = stability.get("hard_reasons") or stability.get("first_bad_reasons") or []
        reason = f"hard-unstable: {reasons[0]}" if reasons else "hard-unstable trajectory"
        if stability.get("first_bad_step") is not None:
            reason += f" (first unsafe step {stability['first_bad_step']})"
        return verdict("repair", reason)

    # 7. Complete runs.
    if row.get("complete"):
        if row.get("ready_for_step2"):
            if state == "error":
                # Every step is there, but VASP stopped on an error marker without
                # its timing footer: a human confirms before Step2 uses it.
                return verdict(
                    "review", "complete and Step2-ready by hard criteria, but OUTCAR has a VASP error marker — confirm before Step2"
                )
            if row.get("severity") == "ok":
                return verdict("done", "complete, stable and thermally ready for Step2")
            return verdict(
                "review", f"complete and Step2-ready by hard criteria; {_warning_label(stability)} — confirm before Step2"
            )
        tail = row.get("thermal_tail_mean_k")
        threshold = row.get("thermal_ready_threshold_k")
        if row.get("thermal_tail_ok") is False and tail is not None and threshold is not None:
            window = row.get("thermal_tail_window_steps")
            return verdict(
                "review", f"complete but Ttail below threshold (Ttail{window}={tail:.0f} K < {threshold:.0f} K)"
            )
        return verdict("review", "complete but the thermal tail could not be evaluated")

    # 8. Incomplete runs.
    if state == "error":
        return verdict("review", "VASP error marker in OUTCAR; inspect before resuming")
    if row.get("severity") == "warning" and not stability.get("benign_warnings_only"):
        return verdict("review", f"{_warning_label(stability)}; review before resuming")
    if not (lineage.get("current_segment_steps") or 0):
        return verdict("review", "no completed ionic step in the current segment; inspect the Slurm log")
    target = lineage.get("original_nsw")
    progress = f"{lineage.get('cumulative_steps')}/{target if target is not None else '?'}"
    reason = f"incomplete ({progress}), trajectory healthy"
    if stability.get("benign_warnings_only"):
        reason += "; benign startup transient"
    return verdict("resume", reason)


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #


def step1_status(
    root: str | Path,
    *,
    stale_hours: float | None = None,
    scheduler: str | SchedulerGuard = "auto",
) -> dict[str, Any]:
    """Read-only status of every Step1 run under ``root``.

    ``scheduler`` is a mode (``auto``/``slurm``/``none``) or a shared
    ``SchedulerGuard``; it is queried ONCE (``try_snapshot``) and a failure is
    recorded in the payload instead of raised.  ``stale_hours`` is the file-age
    window for ``stalled?`` and the "recently modified -> active" rule; None
    resolves it from the scheduler (0.1 h when Slurm is verified, else 6 h).
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    guard = as_guard(scheduler)
    snapshot, scheduler_error = guard.try_snapshot()
    hours, stale_reason = _resolve_stale(stale_hours, snapshot, scheduler_error, guard.mode)

    manifest: dict[str, Any] | None = None
    manifest_path = root_path / "step1_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
    if not isinstance(manifest, dict):
        manifest = None

    runs = [
        _run_status(run, root=root_path, stale_hours=hours, snapshot=snapshot, scheduler_error=scheduler_error)
        for run in _discover_runs(root_path)
    ]
    tally: dict[str, int] = {}
    actions: dict[str, int] = {}
    for row in runs:
        tally[row["state"]] = tally.get(row["state"], 0) + 1
        category = row["recovery"]["category"]
        actions[category] = actions.get(category, 0) + 1

    return {
        "schema_version": 1,
        "root": str(root_path),
        "stale_hours": hours,
        "stale_hours_reason": stale_reason,
        "protocol": (manifest or {}).get("protocol"),
        "manifest_temperature_k": (manifest or {}).get("temperature_k"),
        "manifest_nsw": (manifest or {}).get("nsw"),
        "scheduler": _scheduler_payload(guard, snapshot, scheduler_error),
        "state_tally": tally,
        "action_tally": actions,
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# Human output
# --------------------------------------------------------------------------- #


def _fmt(value: Any, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def _readiness_text(row: dict[str, Any]) -> str:
    if row.get("trajectory_stable") is False:
        return "UNSTABLE"
    if not row.get("complete"):
        return "incomplete"
    if row.get("ready_for_step2"):
        text = "ready for Step2"
        if row.get("review_required"):
            text += " (review)"  # the reason is on the action and stability lines
        return text
    return "complete; not ready"


def _thermal_text(row: dict[str, Any]) -> str:
    """``  Ttail50=300 K thermal-ok; incomplete`` -- never ``(<250 K)`` for a tail that passed."""

    tail_mean = row.get("thermal_tail_mean_k")
    tail_n = row.get("thermal_tail_window_steps") or 0
    if tail_mean is None or not tail_n:
        return ""
    text = f"  Ttail{tail_n}={tail_mean:.0f} K"
    threshold = row.get("thermal_ready_threshold_k")
    if row.get("thermal_tail_ok"):
        text += " thermal-ok"
    elif threshold is not None:
        text += f" (<{threshold:.0f} K)"
    return text + "; " + _readiness_text(row)


def _lineage_text(lineage: dict[str, Any]) -> str | None:
    """The ``lineage:`` line, for generation >= 1 or when a submission is known."""

    if not lineage:
        return None
    generation = lineage.get("generation") or 0
    if generation < 1 and not (lineage.get("submitted") or lineage.get("historical_submissions")):
        return None
    head = f"{lineage.get('segment_kind')} g{generation} ({lineage.get('generation_id')}"
    if lineage.get("legacy_record"):
        head += "; legacy record"
    if lineage.get("record_status"):
        head += f"; record {lineage['record_status']}"
    parts = [head + ")"]
    if lineage.get("original_nsw") is not None:
        parts.append(f"target {lineage['original_nsw']}")
    parts.append(
        f"accepted {lineage.get('accepted_prefix_steps', 0)} + segment {lineage.get('current_segment_steps', 0)}"
        f" = {lineage.get('cumulative_steps', 0)}"
    )
    parts.append(f"segment NSW {_fmt(lineage.get('segment_nsw'))} @ POTIM {_fmt(lineage.get('segment_potim_fs'), 'g')} fs")
    total_ps = lineage.get("accepted_total_ps")
    parts.append(f"accepted {total_ps:.3f} ps" if total_ps is not None else "accepted ps unknown")
    if not lineage.get("ledger_exact", True):
        parts.append("ledger inexact")
    temperature = lineage.get("temperature") or {}
    tebeg = temperature.get("segment_tebeg_k")
    if tebeg is not None:
        if temperature.get("ramp"):
            parts.append(
                f"T {tebeg:g}→{_fmt(temperature.get('segment_teend_k'), 'g')} K ramp"
                f" (now {_fmt(temperature.get('current_target_k'), '.0f')} K)"
            )
        else:
            parts.append(f"T {tebeg:g} K")
    if lineage.get("submitted"):
        submission = lineage.get("submission") or {}
        parts.append(f"current generation submitted (job {submission.get('job_id') or '?'})")
    else:
        parts.append("current generation not submitted")
    older = len(lineage.get("historical_submissions") or [])
    if older:
        parts.append(f"{older} older submission{'s' if older != 1 else ''}")
    return "lineage: " + " · ".join(parts)


def _stability_line(stability: dict[str, Any]) -> str | None:
    if stability.get("unstable"):
        detail: list[str] = []
        if stability.get("first_bad_step") is not None:
            detail.append(f"first unsafe ionic step {stability['first_bad_step']}")
        detail.extend(stability.get("hard_reasons") or stability.get("first_bad_reasons") or [])
        fraction = stability.get("scf_ceiling_fraction")
        if fraction is not None and not stability.get("scf_unreliable"):
            detail.append(
                f"NELM ceiling {stability['scf_ceiling_steps']}/"
                f"{stability['scf_window_steps']} steps ({100.0*fraction:.0f}%)"
            )
        return "stability: UNSTABLE — " + "; ".join(detail)
    if stability.get("severity") == "warning":
        text = "; ".join(stability.get("warnings") or ["review-level warning"])
        if stability.get("benign_warnings_only"):
            text += " (benign startup transient)"
        return "stability: WARNING — " + text
    return None


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = [f"Step1 status: {payload['root']}"]
    header_bits = []
    if payload.get("protocol"):
        header_bits.append(f"protocol {payload['protocol']}")
    if payload.get("manifest_nsw"):
        header_bits.append(f"NSW target {payload['manifest_nsw']}")
    if payload.get("manifest_temperature_k"):
        header_bits.append(f"{payload['manifest_temperature_k']:g} K")
    if header_bits:
        lines.append("  " + "  ·  ".join(header_bits))
    scheduler = payload.get("scheduler")
    if scheduler:
        verified = "verified" if scheduler.get("verified") else "NOT verified"
        lines.append(f"  scheduler: {verified} — {_printable(str(scheduler.get('reason')))}")
    if payload.get("stale_hours") is not None:
        reason = payload.get("stale_hours_reason")
        lines.append(f"  activity window: {payload['stale_hours']:g} h" + (f" ({reason})" if reason else ""))
    lines.append("")

    if not payload["runs"]:
        lines.append("  (no runs with an INCAR found)")
        return "\n".join(lines)

    for row in payload["runs"]:
        inc = row["incar"]
        frames = row["frames_oszicar"]
        target = row["nsw_target"]
        pct = row["percent_complete"]
        count = f"{frames}/{target}" if target else f"{frames}/?"
        pct_txt = f"{pct:>5.1f}%" if pct is not None else "   -  "
        xdat = row["frames_xdatcar"]
        prefix = row.get("accepted_prefix_steps", 0)
        if prefix:
            segment = (row.get("lineage") or {}).get("segment_kind") or "repair"
            xdat_txt = f" (accepted prefix {prefix}; {segment} XDATCAR {xdat or 0})"
        else:
            xdat_txt = f" (XDATCAR {xdat})" if xdat is not None and xdat != frames else ""
        ps_txt = ""
        if row["produced_ps"] is not None and row["target_ps"] is not None:
            ps_txt = f"  {row['produced_ps']:.2f}/{row['target_ps']:.2f} ps"
        temp_txt = ""
        if row["temperature_mean_k"] is not None:
            mean_k = row["temperature_mean_k"]
            std = row["temperature_std_k"]
            if std is None:
                temp_txt = f"  Tmean={mean_k:.0f} K"
            else:
                temp_txt = f"  Tmean={mean_k:.0f}+/-{std:.0f} K"
            temp_txt += _thermal_text(row)

        first = f"  [{row['state']:<11}] {row['run']}"
        recovery = row.get("recovery")
        if recovery:
            first += f"  → {recovery['category']}: {recovery['reason']}"
        lines.append(first)
        lines.append(
            f"      frames {count}{xdat_txt}  {pct_txt}{ps_txt}{temp_txt}  updated {row['updated'] or '-'}"
        )
        ratio = inc["encut_over_enmax"]
        ratio_txt = f" ({ratio:g}x ENMAX)" if ratio is not None else ""
        lines.append(
            "      INCAR: "
            + f"ISTART={_fmt(inc['istart'])}  "
            + f"ENCUT={_fmt(inc['encut_ev'], '.0f')}{ratio_txt}  "
            + f"PREC={_fmt(inc['prec'])}  EDIFF={_fmt(inc['ediff'])}  "
            + f"ALGO={_fmt(inc['algo'])}  LREAL={_fmt(inc['lreal'])}  "
            + f"ISMEAR={_fmt(inc['ismear'])}/{_fmt(inc['sigma'], '.2g')}"
        )
        lines.append(
            "             "
            + f"ISPIN={_fmt(inc['ispin'])}  U(eV)={_fmt(inc['ldauu'])}  LMAXMIX={_fmt(inc['lmaxmix'])}  "
            + f"IBRION={_fmt(inc['ibrion'])}  NSW={_fmt(inc['nsw'])}  POTIM={_fmt(inc['potim_fs'])}  "
            + f"SMASS={_fmt(inc['smass'])}  TEBEG={_fmt(inc['tebeg'], '.0f')}"
        )
        lineage = row.get("lineage") or {}
        lineage_line = _lineage_text(lineage)
        if lineage_line:
            lines.append("      " + lineage_line)
        for warning in lineage.get("warnings") or []:
            lines.append("      lineage warning: " + warning)
        stability_line = _stability_line(row.get("stability") or {})
        if stability_line:
            lines.append("      " + stability_line)

    summary = "  ".join(f"{state}: {n}" for state, n in sorted(payload["state_tally"].items()))
    lines += ["", f"{len(payload['runs'])} runs  ({summary})"]
    actions = payload.get("action_tally") or {}
    if actions:
        ordered = [name for name in _CATEGORY_ORDER if name in actions] + sorted(set(actions) - set(_CATEGORY_ORDER))
        lines.append("actions  (" + "  ".join(f"{name}: {actions[name]}" for name in ordered) + ")")
    return "\n".join(lines)


# Plain-ASCII stand-ins for the typographic characters in the table, used only
# when the output stream cannot encode them (e.g. stdout redirected to a file
# under a cp1252 locale on Windows).
_ASCII_FALLBACK = str.maketrans({"→": "->", "·": "|", "—": "-"})


def _emit(text: str) -> None:
    """Print ``text``; a stream that cannot encode ``→``/``·``/``—`` gets ASCII stand-ins instead of a crash."""

    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.translate(_ASCII_FALLBACK).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", help="Step1 tree root or a single run directory")
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=None,
        help=(
            "Files modified within this window count as active; a started, unfinished run older than it is "
            "'stalled?'. Default: 0.1 h when squeue answered (Slurm verified), 6 h when Slurm is not verified"
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_MODES,
        default="auto",
        help="Ask squeue which runs are queued/running (auto: only when squeue is on PATH; never fatal here)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the raw payload instead of a table")
    args = parser.parse_args(argv)
    payload = step1_status(args.root, stale_hours=args.stale_hours, scheduler=args.scheduler)
    _emit(json.dumps(payload, indent=2, default=str) if args.json else render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
