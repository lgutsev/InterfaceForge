"""Submit prepared, repaired and resumed Step1 preheat runs (generation-aware).

Like ``step2-launch`` / ``opt-launch``, this is the *only* place a Step1 run
reaches ``sbatch``.  The default is a non-mutating launch plan; ``execute=True``
is the sole path that submits, and every root is fully preflighted before the
first job is sent.

A run is launchable when its *current generation* (see ``step1_lineage``) is

* generation 0 written by ``step1-prepare`` -- listed in ``step1_manifest.json``
  with INCAR/POSCAR hashes still matching (kind ``prepared``);
* a ``step1-repair`` segment whose ``step1_repair.json`` is ``PREPARED``
  (kind ``repair-prepared``); or
* a ``step1-resume`` segment whose ``step1_resume.json`` is ``PREPARED``
  (kind ``resume-prepared``),

it carries no runtime outputs, no Slurm job uses it as WorkDir, no recovery
mutation of it was interrupted (an ``IN_PROGRESS`` archive manifest), and the
current generation is not already recorded as submitted.  The duplicate guard reads
every launch ledger that may mention the run -- the run itself, each root, and
its ancestors up to the Step1 root -- so a leaf launched as its own root and
later launched from the Step1 root (or the reverse) is still seen, while a
historical ``SUBMITTED`` row of an *older* generation never blocks the current
one (no more renaming ``step1_launch.json`` by hand after a repair).

Execution submits one batch and records each job the moment ``sbatch``
returns: a row in the invoked root's ``step1_launch.json`` (schema 2,
cumulative history, mirrored by ``step1_launch.tsv``) and a submission entry on
the run's current record.  A crash mid-batch therefore still leaves every
submitted job on record.  Immediately before each ``sbatch`` the scheduler is
re-queried and the run's fingerprint, generation and ledgers are re-checked;
the first failure is recorded as a ``FAILED`` row and stops the batch.  A
schema-1 ledger is upgraded (its rows imported verbatim as legacy rows) only by
an executing launch that appends to it; a dry run never writes anything.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .errors import SafetyError
from .step1_lineage import (
    LAUNCH_LEDGER,
    LAUNCH_TSV,
    MANIFEST,
    RESUME_RECORD,
    Generation,
    append_launch_rows,
    current_generation,
    interrupted_archive,
    ledger_paths_for,
    mark_record_submitted,
    read_json,
    rows_for_run,
    run_fingerprint,
    submission_state,
    utc_now_iso,
    utc_stamp,
)
from .step1_repair import _discover_runs
from .step1_scheduler import SchedulerGuard, SchedulerSnapshot, as_guard
from .vasp import _sha256_file, resolve_launcher, submit_run

_STARTED_MARKERS = ("OUTCAR", "OSZICAR", "vasprun.xml")
_REQUIRED_INPUTS = ("INCAR", "POSCAR", "KPOINTS")

KIND_PREPARED = "prepared"
KIND_REPAIR = "repair-prepared"
KIND_RESUME = "resume-prepared"
_KIND_BY_SEGMENT = {"repair": KIND_REPAIR, "resume": KIND_RESUME}

# How many skip reasons the "nothing launchable" error spells out.
_SKIP_REASONS_IN_ERROR = 10


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _dir_key(path: str | Path) -> str:
    """Comparison key for directories: realpath, case-folded where the OS is case-insensitive."""

    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _within(path: Path, root: Path) -> bool:
    path_key, root_key = _dir_key(path), _dir_key(root)
    if path_key == root_key:
        return True
    prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
    return path_key.startswith(prefix)


def _relative(run: Path, tree_root: Path) -> str:
    return run.relative_to(tree_root).as_posix() if run != tree_root else "."


def _started(run: Path) -> list[str]:
    return [name for name in _STARTED_MARKERS if _nonempty(run / name)]


def _paths(values: Iterable[str | Path] | str | Path) -> list[Path]:
    """Resolved paths; a lone ``str``/``Path`` is one path, never iterated per character."""

    if isinstance(values, (str, os.PathLike)):
        values = [values]
    return [Path(value).expanduser().resolve() for value in values]


def _manifest_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(root / MANIFEST).get("runs")
    if not isinstance(rows, list):
        return {}
    return {str(row.get("relative_path")): row for row in rows if isinstance(row, dict)}


def _segment_kind(generation: Generation) -> str:
    """``"original"`` for gen 0, else ``"repair"`` / ``"resume"`` (by record kind, then file name)."""

    if generation.record_path is None:
        return "original"
    if generation.kind in _KIND_BY_SEGMENT:
        return generation.kind
    return "resume" if generation.record_path.name == RESUME_RECORD else "repair"


def _kind_filter_reason(kind: str, *, only_repaired: bool, only_resumed: bool) -> str | None:
    """Why ``kind`` is excluded by ``--only-repaired`` / ``--only-resumed`` (both = union), else None."""

    if not (only_repaired or only_resumed):
        return None
    allowed = {KIND_REPAIR} if only_repaired else set()
    if only_resumed:
        allowed.add(KIND_RESUME)
    if kind in allowed:
        return None
    if only_repaired and only_resumed:
        return "not a repaired or resumed run (--only-repaired --only-resumed)"
    if only_repaired:
        return "not a repaired run (--only-repaired)"
    return "not a resumed run (--only-resumed)"


def _active_reason(jobs: list[dict[str, Any]]) -> str:
    listing = ", ".join(f"job {job.get('job_id')} {job.get('state')}" for job in jobs)
    return f"active in Slurm ({listing})"


def _interrupted_reason(run: Path) -> str | None:
    """Skip reason when a repair/resume of ``run`` stopped mid-mutation (or is running now), else None.

    ``archive_step1_state`` leaves its manifest ``IN_PROGRESS`` until the
    mutation finished; such a run may be half rewritten, so it is never
    submitted -- the operator inspects the archive first (read-only check).
    """

    archive = interrupted_archive(run)
    if archive is None:
        return None
    return f"interrupted recovery mutation (archive {archive} is IN_PROGRESS); inspect it before launching"


def _submitted_reason(generation: Generation, state: dict[str, Any]) -> str:
    """Skip reason for a current generation that is already on record as submitted."""

    current = state.get("current_submission") or {}
    job = current.get("job_id") or "unknown"
    reason = f"current generation {generation.generation_id} already submitted (job {job})"
    if current.get("unreadable_ledger"):
        reason += (
            f": launch ledger {current.get('_ledger')} is unreadable, so a submission cannot be ruled out; "
            "repair it or move it aside after inspection"
        )
        return reason
    count = len(state.get("current_submissions") or [])
    if count > 1:
        reason += f"; {count} submissions of this generation are recorded"
    return reason


def _submission_state(run: Path, generation: Generation, roots: list[Path]) -> dict[str, Any]:
    return submission_state(run, generation, rows_for_run(run, ledger_paths_for(run, roots)))


def _public(plan: dict[str, Any]) -> dict[str, Any]:
    """``plan`` without the private (``_``-prefixed) bookkeeping keys."""

    return {key: value for key, value in plan.items() if not key.startswith("_")}


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def _preflight_run(
    run: Path,
    tree_root: Path,
    *,
    roots: list[Path],
    snapshot: SchedulerSnapshot,
    manifest_rows: dict[str, dict[str, Any]],
    launcher: str | None,
    only_repaired: bool,
    only_resumed: bool,
    emit: Callable[[str], None],
) -> tuple[dict[str, Any] | None, str]:
    """Return ``(plan, skip_reason)``; exactly one is truthy.

    Read-only.  Inconsistent inputs (a prepared run whose INCAR/POSCAR changed
    since ``step1-prepare``, a missing input, no launcher) raise instead of
    skipping, exactly as before generations existed.
    """

    relative = _relative(run, tree_root)
    fingerprint = run_fingerprint(run)  # captured first: execute refuses if anything moves after this

    # Scheduler state beats file state: a queued job has no outputs yet.
    jobs = snapshot.active_jobs_for(run)
    if jobs:
        return None, _active_reason(jobs)
    started = _started(run)
    if started:
        return None, f"already started ({', '.join(started)})"
    interrupted = _interrupted_reason(run)
    if interrupted:
        return None, interrupted

    generation = current_generation(run)
    segment = _segment_kind(generation)
    kind = KIND_PREPARED if segment == "original" else _KIND_BY_SEGMENT[segment]
    excluded = _kind_filter_reason(kind, only_repaired=only_repaired, only_resumed=only_resumed)
    if excluded:
        return None, excluded

    # Generation-aware duplicate guard: only a submission of THIS generation blocks.
    state = _submission_state(run, generation, roots)
    if state["current_submitted"]:
        return None, _submitted_reason(generation, state)

    if generation.record_path is not None:
        if generation.status != "PREPARED":
            reason = f"{generation.record_path.name} status is {generation.status!r}, not PREPARED"
            if generation.conflict:
                reason += f" ({generation.conflict})"
            return None, reason
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
    else:
        return None, "not written by step1-prepare, step1-repair or step1-resume"

    for name in _REQUIRED_INPUTS:
        if not _nonempty(run / name):
            raise SafetyError(f"{run} is missing {name}")
    script = resolve_launcher(run, launcher)

    notes: list[str] = []
    if not _nonempty(run / "POTCAR"):
        notes.append("POTCAR absent; the launcher must generate it")
    historical = len(state["historical_submissions"])
    if historical:
        notes.append(f"{historical} older submission(s) of earlier generations on record")
    emit(f"[{tree_root.name}] preflight OK: {relative} ({kind}, {generation.generation_id}, launcher={script.name})")
    return {
        "root": str(tree_root),
        "relative_path": relative,
        "directory": str(run),
        "launcher": script.name,
        "kind": kind,
        "notes": "; ".join(notes),
        "generation": generation.generation,
        "generation_id": generation.generation_id,
        "record": generation.record_path.name if generation.record_path is not None else None,
        "historical_submissions": historical,
        "_fingerprint": fingerprint,
    }, ""


def _selected_runs(runs: Iterable[str | Path] | str | Path | None, roots: list[Path]) -> dict[str, Path] | None:
    """``{dir key: run}`` for an explicit ``runs=`` restriction (None = every discovered run)."""

    if runs is None:
        return None
    selected: dict[str, Path] = {}
    for path in _paths(runs):
        if not any(_within(path, root) for root in roots):
            raise SafetyError(
                f"{path} is not under any of the given Step1 roots ({', '.join(str(root) for root in roots)})"
            )
        selected.setdefault(_dir_key(path), path)
    if not selected:
        raise SafetyError("No Step1 runs selected (runs= is empty); nothing to launch")
    return selected


def _nothing_launchable(skipped: list[dict[str, Any]]) -> SafetyError:
    shown = [
        f"{row['relative_path'] if row['relative_path'] != '.' else row['directory']}: {row['reason']}"
        for row in skipped[:_SKIP_REASONS_IN_ERROR]
    ]
    if len(skipped) > _SKIP_REASONS_IN_ERROR:
        shown.append(f"... {len(skipped) - _SKIP_REASONS_IN_ERROR} more")
    return SafetyError(
        "No launchable Step1 runs (nothing prepared/repaired/resumed and idle). "
        + (f"{len(skipped)} directory(ies) skipped: {'; '.join(shown)}" if skipped else "")
    )


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #


def _recheck_before_submit(plan: dict[str, Any], *, roots: list[Path]) -> None:
    """Raise ``SafetyError`` if ``plan``'s run changed in any way since preflight (read-only)."""

    run = Path(plan["directory"])
    if run_fingerprint(run) != plan["_fingerprint"]:
        raise SafetyError(f"{run} changed since planning; nothing was submitted for it. Re-run step1-launch to re-plan")
    started = _started(run)
    if started:
        raise SafetyError(f"{run} started since planning ({', '.join(started)}); nothing was submitted for it")
    interrupted = _interrupted_reason(run)
    if interrupted:
        raise SafetyError(f"{run}: {interrupted}; nothing was submitted for it")
    generation = current_generation(run)
    if generation.generation_id != plan["generation_id"]:
        raise SafetyError(
            f"{run}: current generation is now {generation.generation_id}, planned {plan['generation_id']}; "
            "nothing was submitted for it. Re-run step1-launch to re-plan"
        )
    if generation.record_path is not None and generation.status != "PREPARED":
        raise SafetyError(
            f"{run}: {generation.record_path.name} status is now {generation.status!r}, not PREPARED; "
            "nothing was submitted for it"
        )
    state = _submission_state(run, generation, roots)
    if state["current_submitted"]:
        raise SafetyError(f"{run}: {_submitted_reason(generation, state)} (recorded since planning)")


def _launch_row(
    plan: dict[str, Any], *, status: str, job_id: str, detail: str, submitted_at: str, batch_id: str
) -> dict[str, Any]:
    return {
        "status": status,
        "job_id": job_id,
        "kind": plan["kind"],
        "root": plan["root"],
        "relative_path": plan["relative_path"],
        "directory": plan["directory"],
        "launcher": plan["launcher"],
        "notes": plan["notes"],
        "detail": detail,
        "generation": plan["generation"],
        "generation_id": plan["generation_id"],
        "submitted_at": submitted_at,
        "batch_id": batch_id,
    }


def _new_batch_id(ledger_dirs: Iterable[Path]) -> str:
    """``b-<UTC stamp>``, suffixed ``-2``, ``-3``, ... when a target ledger already knows that id.

    Stamps have one-second resolution and ``append_launch_rows`` merges rows
    into an existing batch entry of the same id, so two launches within one
    second (e.g. ``step1-recover`` submitting run after run) must not share
    one.  Read-only.
    """

    taken: set[str] = set()
    for directory in ledger_dirs:
        payload = read_json(directory / LAUNCH_LEDGER)
        for key in ("batches", "runs"):
            items = payload.get(key)
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and item.get("batch_id"):
                    taken.add(str(item["batch_id"]))
    base = f"b-{utc_stamp()}"
    candidate, suffix = base, 1
    while candidate in taken:
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def _note_report(reports: list[str], ledger_json: Path) -> None:
    for path in (ledger_json, ledger_json.with_name(LAUNCH_TSV)):
        if str(path) not in reports:
            reports.append(str(path))


def _failure_message(
    plan: dict[str, Any],
    exc: BaseException,
    *,
    batch_id: str,
    submitted: list[dict[str, Any]],
    reports: list[str],
    ledger_note: str,
) -> str:
    done = ", ".join(f"{row['relative_path']} (job {row['job_id']})" for row in submitted)
    message = (
        f"Step1 launch stopped at {plan['relative_path']} ({plan['directory']}): {exc}. "
        f"Batch {batch_id}: {len(submitted)} job(s) submitted before the failure" + (f": {done}" if done else "")
    )
    message += f". {ledger_note}" if ledger_note else ""
    if reports:
        message += f". Review partial launch records: {', '.join(reports)}"
    return message


def _submit_planned(
    planned: list[dict[str, Any]],
    *,
    guard: SchedulerGuard,
    roots: list[Path],
    batch_id: str,
    emit: Callable[[str], None],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Submit ``planned`` in order, recording each job immediately; returns ``(rows, reports)``.

    Per run: ``guard.assert_inactive`` (sbatch counts as a mutation), the
    read-only planning re-check, ``submit_run``, then at once the ledger row in
    the plan's root and the submission on the run record.  When the run IS its
    own root (a leaf launched directly) the ledger is written into the run after
    its ``sbatch`` -- launch's own record of the job it just queued; a run whose
    Slurm state refused it gets no FAILED row written into it.
    """

    started_at = utc_now_iso()
    planned_per_ledger = Counter(_dir_key(plan["root"]) for plan in planned)
    rows: list[dict[str, Any]] = []
    reports: list[str] = []
    for index, plan in enumerate(planned, start=1):
        run = Path(plan["directory"])
        tree_root = Path(plan["root"])
        ledger_json = tree_root / LAUNCH_LEDGER
        # "planned" counts the runs recorded in THIS ledger (a batch may span several roots).
        batch = {"batch_id": batch_id, "started_at": started_at, "planned": planned_per_ledger[_dir_key(tree_root)]}
        emit(f"[{index}/{len(planned)}] sbatch {plan['relative_path']} ({plan['generation_id']}) in {run}")

        scheduler_refused = False
        try:
            try:
                guard.assert_inactive(run)  # sbatch is a mutation: never submit into an active WorkDir
            except SafetyError:
                scheduler_refused = True
                raise
            _recheck_before_submit(plan, roots=roots)
            job_id = submit_run(run, plan["launcher"])
        except Exception as exc:  # noqa: BLE001 - recorded as a FAILED row and re-raised as SafetyError
            emit(f"    FAILED: {exc}")
            ledger_note = ""
            if scheduler_refused and _dir_key(tree_root) == _dir_key(run):
                # The ledger lives in the run itself (a leaf launched as its own
                # root) and that directory may be an active WorkDir: leave it alone.
                ledger_note = f"Not recorded in {ledger_json}: that directory is the run whose Slurm state blocked it"
            else:
                failed = _launch_row(
                    plan,
                    status="FAILED",
                    job_id="",
                    detail=str(exc),
                    submitted_at=utc_now_iso(),
                    batch_id=batch_id,
                )
                try:
                    _note_report(reports, append_launch_rows(tree_root, [failed], batch=batch))
                    ledger_note = f"FAILED row recorded in {ledger_json}"
                except Exception as write_exc:  # noqa: BLE001 - reported in the SafetyError below
                    ledger_note = f"recording the failure in {ledger_json} also failed: {write_exc}"
            raise SafetyError(
                _failure_message(plan, exc, batch_id=batch_id, submitted=rows, reports=reports, ledger_note=ledger_note)
            ) from exc

        submitted_at = utc_now_iso()
        row = _launch_row(
            plan, status="SUBMITTED", job_id=str(job_id), detail="", submitted_at=submitted_at, batch_id=batch_id
        )
        rows.append(row)
        emit(f"    submitted job {job_id}")
        submission = {
            "job_id": str(job_id),
            "submitted_at": submitted_at,
            "batch_id": batch_id,
            "launcher": plan["launcher"],
            "ledger": str(ledger_json),
        }
        # Record the job before anything else can fail, so a crash mid-batch
        # never leaves a submitted job unrecorded.
        try:
            _note_report(reports, append_launch_rows(tree_root, [row], batch=batch))
        except Exception as exc:  # noqa: BLE001 - the job exists; say so loudly
            record_note = ""
            try:
                mark_record_submitted(run, plan["generation_id"], submission)
                if plan["record"]:
                    record_note = f" It was recorded on {run / plan['record']}."
            except Exception as record_exc:  # noqa: BLE001
                record_note = f" Marking the run record also failed: {record_exc}."
            raise SafetyError(
                f"Job {job_id} for {run} WAS submitted (batch {batch_id}) but could not be recorded in "
                f"{ledger_json}: {exc}.{record_note} Record it by hand before any relaunch; "
                f"jobs submitted earlier in this batch: {', '.join(r['job_id'] for r in rows[:-1]) or 'none'}"
            ) from exc
        try:
            mark_record_submitted(run, plan["generation_id"], submission)
        except Exception as exc:  # noqa: BLE001 - the ledger row already guards against a duplicate
            raise SafetyError(
                f"Job {job_id} for {run} was submitted and recorded in {ledger_json}, but marking "
                f"{run / (plan['record'] or '?')} SUBMITTED failed: {exc}. The ledger row still blocks a duplicate "
                f"launch; review partial launch records: {', '.join(reports)}"
            ) from exc
    return rows, reports


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def launch_step1_runs(
    roots: Iterable[str | Path],
    *,
    execute: bool = False,
    launcher: str | None = None,
    only_repaired: bool = False,
    only_resumed: bool = False,
    progress: Callable[[str], None] | None = None,
    scheduler: str | SchedulerGuard = "auto",
    runs: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Plan (default) or submit prepared / repaired / resumed Step1 runs.

    ``only_repaired`` / ``only_resumed`` restrict the launchable kinds (both =
    union).  ``runs`` restricts preflight to those run directories, each of
    which must lie under one of ``roots``.  ``scheduler`` is a
    ``--scheduler`` mode or a shared ``SchedulerGuard``; a Slurm query that
    fails raises ``SafetyError`` rather than guessing.  Raises
    ``SafetyError`` when nothing is launchable.

    The dry run (default) only reads files and queries the scheduler.  The
    payload keeps the schema-1 keys (``planned``/``skipped_runs`` for a dry
    run; ``submitted``, ``reports``, ``jobs`` after ``execute``) and adds
    ``scheduler`` (planning snapshot) and ``batch_id`` (None for a dry run).
    """

    emit = progress or (lambda _message: None)
    resolved: list[Path] = []
    for root in _paths(roots):
        if all(_dir_key(root) != _dir_key(seen) for seen in resolved):
            resolved.append(root)
    if not resolved:
        raise SafetyError("At least one Step1 root (or run directory) is required")
    for tree_root in resolved:
        if not tree_root.is_dir():
            raise FileNotFoundError(tree_root)
    selected = _selected_runs(runs, resolved)

    guard = as_guard(scheduler)
    snapshot = guard.snapshot  # raises SafetyError when Slurm was requested but cannot be queried
    emit(f"scheduler: {snapshot.reason}" + ("" if snapshot.verified else " (not verified)"))

    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    found: set[str] = set()
    preflighted: dict[str, Path] = {}
    for tree_root in resolved:
        manifest_rows = _manifest_rows(tree_root)
        discovered = _discover_runs(tree_root)
        if not discovered:
            raise SafetyError(f"No Step1 run directories found under {tree_root}")
        if selected is not None:
            discovered = [run for run in discovered if _dir_key(run) in selected]
            found.update(_dir_key(run) for run in discovered)
            if not discovered:
                continue
        emit(f"[{tree_root.name}] preflighting {len(discovered)} run directory(ies)")
        for run in discovered:
            relative = _relative(run, tree_root)
            earlier = preflighted.get(_dir_key(run))
            if earlier is not None:
                reason = f"already preflighted under root {earlier}"
                plan = None
            else:
                preflighted[_dir_key(run)] = tree_root
                plan, reason = _preflight_run(
                    run,
                    tree_root,
                    roots=resolved,
                    snapshot=snapshot,
                    manifest_rows=manifest_rows,
                    launcher=launcher,
                    only_repaired=only_repaired,
                    only_resumed=only_resumed,
                    emit=emit,
                )
            if plan is not None:
                planned.append(plan)
            else:
                skipped.append(
                    {"relative_path": relative, "root": str(tree_root), "directory": str(run), "reason": reason}
                )
                emit(f"[{tree_root.name}] skip {relative}: {reason}")
    if selected is not None:
        missing = [str(path) for key, path in selected.items() if key not in found]
        if missing:
            raise SafetyError(f"Not a Step1 run directory under the given roots: {', '.join(missing)}")

    if not planned:
        raise _nothing_launchable(skipped)
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
            "planned": [_public(plan) for plan in planned],
            "skipped_runs": skipped,
            "scheduler": snapshot.to_dict(),
            "batch_id": None,
        }

    batch_id = _new_batch_id(Path(root) for root in dict.fromkeys(plan["root"] for plan in planned))
    rows, reports = _submit_planned(planned, guard=guard, roots=resolved, batch_id=batch_id, emit=emit)
    emit(f"Done: {len(rows)} job(s) submitted in batch {batch_id}; launch records: {', '.join(reports)}")
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
        "skipped_runs": skipped,
        "scheduler": snapshot.to_dict(),
        "batch_id": batch_id,
    }
