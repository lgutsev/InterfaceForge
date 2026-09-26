"""State-aware recovery of a whole Step1 preheat tree (``iface vasp step1-recover``).

``step1-recover`` is the one command an operator runs after a batch of Step1
jobs has left the queue.  Every run under the root is classified by
:func:`interfaceforge.step1_status.recovery_category` (the same rules
``step1-status`` prints) into one of :data:`RECOVER_CATEGORIES`:

``done``
    complete, stable and thermally ready -- left alone (eligible for Step2);
``resume``
    healthy but interrupted -- ``step1_resume`` continues it from its latest
    trusted state (CONTCAR, else an XDATCAR frame) with every INCAR tag kept;
``repair``
    hard-unstable -- ``step1_repair`` rewinds it to a safe frame and prepares a
    conservative segment (recover's defaults are the NiO rescue:
    ``POTIM=0.5`` fs, ``ALGO=Normal``, a preconditioning static SCF and a
    100 K -> target ramp);
``launch``
    prepared (generation 0 in ``step1_manifest.json``, or a PREPARED repair /
    resume record) but never submitted -- ``step1_launch`` submits it;
``review``
    anything a human must look at: review-level warnings (e.g. a startup
    transient on a completed run), an interrupted recovery mutation, an
    unreadable or conflicting segment record, an unreadable launch ledger, a
    launcher that cannot be preconditioned (or an explicit ``--launcher`` that
    would bypass the preconditioning wrapper), or *any* disagreement between
    the classifier and the planner that would act on the run;
``active``
    a Slurm job uses the run as its WorkDir (or its files are fresh).

Only :data:`AUTO_CATEGORIES` (resume, repair, launch) are ever acted on;
``done``, ``review`` and ``active`` runs are never touched.

Planning (:func:`plan_step1_recovery`) is read-only: it reads files and asks
the scheduler once.  For every resume/repair entry the dry
``plan_resume_run`` / ``plan_repair_run`` result is attached, for every launch
entry the launch preflight; an entry moves to ``review`` whenever that planner
does not agree (status other than READY, no repair needed under the requested
thresholds, preflight refused), so nothing is mutated on a disagreement.

Execution (:func:`execute_step1_recovery`) plans afresh with ONE
:class:`~interfaceforge.step1_scheduler.SchedulerGuard`, then processes the
selected entries -- repairs and resumes first, then launches, each group in
relative-path order.  Per entry: ``guard.assert_inactive(run)`` and a
fingerprint comparison against the planning pass, then the repair/resume
executor (which archives first and re-checks both again), then -- unless
``submit=False`` -- ``launch_step1_runs([root], execute=True, runs=[run],
scheduler=guard)`` for exactly that run.  After every step the execution is
rewritten atomically into the journal ``<root>/step1_recover.json``
(``executions`` list, one entry per ``--execute``), so the journal always
says which runs were changed and which were submitted (execution ``status``
RUNNING -> COMPLETED or FAILED).  The first failure stops the execution: the
remaining entries are marked ``not attempted`` and a
:class:`~interfaceforge.errors.SafetyError` names the journal.  A dry run --
and an execution with nothing selected -- writes nothing at all (no journal).

When the root is itself a run directory, the journal lives in that run: it is
written only after the run passed its Slurm check, and after a successful
``sbatch`` it records that submission (the same rule ``step1-launch`` follows
for a leaf launched as its own root).
"""

from __future__ import annotations

import os
import re
import shlex
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import SafetyError
from .step1_launch import launch_step1_runs
from .step1_lineage import (
    GEN0_ID,
    RECOVER_JOURNAL,
    _file_lock,
    atomic_write_json,
    format_temperature,
    interrupted_archive,
    ledger_paths_for,
    read_json,
    run_fingerprint,
    unreadable_ledgers,
    utc_now_iso,
    utc_stamp,
)
from .step1_repair import (
    DEFAULT_CATASTROPHIC_ENERGY_EV,
    DEFAULT_ENERGY_JUMP_EV,
    DEFAULT_REFERENCE_WINDOW_STEPS,
    DEFAULT_STARTUP_GRACE_STEPS,
    _diagnostic_settings,
    _precondition_launcher,
    _validate_repair_options,
    diagnose_step1_run,
    execute_repair_plan,
    plan_repair_run,
    precondition_blocker,
)
from .step1_resume import (
    DEFAULT_CONTCAR_TOLERANCE_ANGSTROM,
    _validate_resume_options,
    execute_resume_plan,
    plan_resume_run,
)
from .step1_scheduler import SchedulerGuard, as_guard
from .step1_status import _discover_runs, _manifest_entry, step1_status
from .vasp import resolve_launcher

RECOVER_CATEGORIES = ("done", "resume", "repair", "launch", "review", "active")
AUTO_CATEGORIES = ("resume", "repair", "launch")
# Human output order (the same order step1-status tallies its actions in).
RENDER_ORDER = ("active", "done", "review", "launch", "resume", "repair")

PLAN_FORMAT = "interfaceforge-step1-recover-plan"
JOURNAL_FORMAT = "interfaceforge-step1-recover"
EXECUTION_FORMAT = "interfaceforge-step1-recover-execution"
PLAN_SCHEMA_VERSION = 1
JOURNAL_SCHEMA_VERSION = 1

# Recover's repair defaults = the conservative NiO rescue (overridable per call).
RECOVER_REPAIR_DEFAULTS: dict[str, Any] = {
    "potim_fs": 0.5,
    "algo": "Normal",
    "safety_steps": 8,
    "langevin_gamma": None,
    "ramp_from": 100.0,
    "precondition": True,
}
# A ramp restart needs room: step1-status judges thermal readiness on the last
# 50 rows of the segment (tail mean >= 5/6 of the target), which a 100 -> 300 K
# linear ramp only reaches with >= ~98 steps.  Shorter repairs continue at the
# schedule temperature of the rewind point instead of quenching and reheating.
RECOVER_MIN_RAMP_STEPS = 100
RECOVER_RESUME_DEFAULTS: dict[str, Any] = {
    "contcar_tolerance_angstrom": DEFAULT_CONTCAR_TOLERANCE_ANGSTROM,
    "precondition": False,
    "fresh_start": False,
}
DIAGNOSTIC_OPTION_KEYS = (
    "energy_jump_ev",
    "max_temperature_k",
    "startup_grace_steps",
    "catastrophic_energy_ev",
    "reference_window_steps",
)
# Repair options for which None means "no such feature" (not "default").
_NULLABLE_REPAIR_OPTIONS = ("langevin_gamma", "ramp_from")

# Journal row outcomes.
OUTCOME_PENDING = "pending"
OUTCOME_RUNNING = "running"
OUTCOME_PREPARED = "prepared"
OUTCOME_SUBMITTED = "submitted"
OUTCOME_FAILED = "failed"
OUTCOME_NOT_ATTEMPTED = "not attempted"

# The typographic characters step1-status reasons use, spelled in ASCII so the
# rendered plan prints on any console (e.g. a redirected cp1252 stdout).
_ASCII = str.maketrans({"→": "->", "·": "|", "—": "-", "–": "-", "…": "...", "≥": ">=", "≤": "<="})
_SHELL_SAFE = re.compile(r"^[\w@%+=:,./\\-]+$")


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


def _reject_unknown(kind: str, options: Mapping[str, Any], allowed: Iterable[str]) -> None:
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ValueError(f"unknown step1-recover {kind} option(s): {', '.join(unknown)}")


def _repair_settings(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recover's repair defaults overlaid with ``options`` (validated).

    ``None`` selects the default, except for ``ramp_from`` / ``langevin_gamma``
    where ``None`` means "no ramp" / "no Langevin thermostat".
    """

    given = dict(options or {})
    _reject_unknown("repair", given, RECOVER_REPAIR_DEFAULTS)
    settings = dict(RECOVER_REPAIR_DEFAULTS)
    for key, value in given.items():
        if value is None and key not in _NULLABLE_REPAIR_OPTIONS:
            continue
        settings[key] = value
    settings["potim_fs"] = float(settings["potim_fs"])
    settings["algo"] = str(settings["algo"])
    settings["safety_steps"] = int(settings["safety_steps"])
    settings["precondition"] = bool(settings["precondition"])
    for key in _NULLABLE_REPAIR_OPTIONS:
        settings[key] = None if settings[key] is None else float(settings[key])
    _validate_repair_options(
        settings["potim_fs"], settings["safety_steps"], settings["langevin_gamma"], settings["ramp_from"]
    )
    return settings


def _resume_settings(options: Mapping[str, Any] | None) -> dict[str, Any]:
    given = dict(options or {})
    _reject_unknown("resume", given, RECOVER_RESUME_DEFAULTS)
    settings = dict(RECOVER_RESUME_DEFAULTS)
    settings.update({key: value for key, value in given.items() if value is not None})
    settings["precondition"] = bool(settings["precondition"])
    settings["fresh_start"] = bool(settings["fresh_start"])
    # ValueError for a negative tolerance or precondition + fresh_start together.
    settings["contcar_tolerance_angstrom"] = _validate_resume_options(
        settings["contcar_tolerance_angstrom"], settings["precondition"], settings["fresh_start"]
    )
    return settings


def _diagnostic_settings_for(options: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """``(options passed to the planners, resolved values, whether they equal the defaults)``."""

    given = {key: value for key, value in dict(options or {}).items() if value is not None}
    _reject_unknown("diagnostic", given, DIAGNOSTIC_OPTION_KEYS)
    jump, limit, grace, catastrophic, window = _diagnostic_settings(
        given.get("energy_jump_ev"),
        given.get("max_temperature_k"),
        given.get("startup_grace_steps"),
        given.get("catastrophic_energy_ev"),
        given.get("reference_window_steps"),
    )
    resolved = {
        "energy_jump_ev": jump,
        "max_temperature_k": limit,
        "startup_grace_steps": grace,
        "catastrophic_energy_ev": catastrophic,
        "reference_window_steps": window,
    }
    defaults = (
        DEFAULT_ENERGY_JUMP_EV,
        None,
        DEFAULT_STARTUP_GRACE_STEPS,
        DEFAULT_CATASTROPHIC_ENERGY_EV,
        DEFAULT_REFERENCE_WINDOW_STEPS,
    )
    return given, resolved, (jump, limit, grace, catastrophic, window) == defaults


def _selected_categories(only: Iterable[str] | str) -> tuple[str, ...]:
    names = [only] if isinstance(only, str) else list(only)
    unknown = [name for name in names if name not in AUTO_CATEGORIES]
    if unknown:
        raise ValueError(
            f"step1-recover acts only on {', '.join(AUTO_CATEGORIES)}; cannot select {', '.join(map(str, unknown))}"
        )
    if not names:
        raise ValueError(f"select at least one of {', '.join(AUTO_CATEGORIES)}")
    return tuple(name for name in AUTO_CATEGORIES if name in names)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _key(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _progress(lineage: Mapping[str, Any]) -> str:
    target = lineage.get("original_nsw")
    return f"{lineage.get('cumulative_steps', 0)}/{target if target is not None else '?'}"


def _temperature(value: Any) -> str:
    try:
        return format_temperature(float(value))
    except (TypeError, ValueError):
        return "?"


def _quoted(path: str) -> str:
    return path if _SHELL_SAFE.match(path) else shlex.quote(path)


def execute_command(root: str | Path, selected: Iterable[str] = AUTO_CATEGORIES, *, submit: bool = True) -> str:
    """The ``iface vasp step1-recover ... --execute`` command line for ``root`` and a selection."""

    command = f"iface vasp step1-recover {_quoted(str(root))} --execute"
    chosen = tuple(selected)
    if set(chosen) != set(AUTO_CATEGORIES):
        command += " --only " + " ".join(chosen)
    if not submit:
        command += " --no-submit"
    return command


def _resume_summary(plan: Mapping[str, Any]) -> str:
    source = str(plan.get("restart_source"))
    if source == "XDATCAR":
        source = f"{plan.get('restart_file')} frame {plan.get('restart_frame')}"
    continuation = plan.get("temperature_continuation") or {}
    before, after = continuation.get("previous_tebeg_k"), continuation.get("resumed_tebeg_k")
    if before is not None and after is not None and abs(float(before) - float(after)) > 1e-9:
        tebeg = f"TEBEG {_temperature(before)} -> {_temperature(after)} K"
    else:
        tebeg = f"TEBEG {_temperature(after)} K"
    potim = plan.get("segment_potim_fs")
    electronic = (plan.get("electronic_start") or {}).get("mode") or "?"
    return (
        f"resume g{plan.get('generation')} from {source} (+{plan.get('segment_accepted_steps')} accepted -> "
        f"{plan.get('accepted_prefix_steps')}/{plan.get('original_nsw')}), NSW={plan.get('resume_nsw')}"
        + (f" @ {float(potim):g} fs" if potim is not None else "")
        + f", {tebeg}, TEEND {_temperature(continuation.get('teend_k'))} K, electronic start {electronic}"
    )


def _repair_summary(plan: Mapping[str, Any]) -> str:
    text = (
        f"repair g{plan.get('generation')}: rewind to segment step {plan.get('safe_segment_steps')} "
        f"(cumulative {plan.get('safe_prefix_steps')}/{plan.get('original_nsw')}), NSW={plan.get('repair_nsw')} "
        f"@ {float(plan.get('repair_potim_fs') or 0.0):g} fs, ALGO={plan.get('repair_algo')}"
    )
    if plan.get("repair_precondition"):
        text += ", precondition"
    if plan.get("repair_langevin_gamma") is not None:
        text += f", Langevin {float(plan['repair_langevin_gamma']):g}/ps"
    schedule = plan.get("segment_schedule") or {}
    if schedule.get("ramp"):
        text += f", ramp {_temperature(schedule.get('tebeg_k'))}->{_temperature(schedule.get('teend_k'))} K"
    else:
        text += f", T {_temperature(schedule.get('tebeg_k'))} K"
    if plan.get("ramp_skipped"):
        text += f" ({plan['ramp_skipped']})"
    return text


def _launch_summary(preflight: Mapping[str, Any]) -> str:
    text = f"launch {preflight.get('kind')} {preflight.get('generation_id')} with {preflight.get('launcher')}"
    if preflight.get("notes"):
        text += f" ({preflight['notes']})"
    return text


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _launch_preflights(
    root: Path, runs: list[Path], guard: SchedulerGuard, launcher: str | None
) -> dict[str, tuple[dict[str, Any] | None, str]]:
    """``{run key: (planned preflight row | None, refusal reason)}`` from step1-launch's dry run (read-only).

    One dry launch for every candidate; if it refuses as a whole (nothing
    launchable, or an inconsistency such as a changed INCAR hash) each run is
    preflighted on its own so one bad run cannot hide the others.
    """

    if not runs:
        return {}
    results: dict[str, tuple[dict[str, Any] | None, str]] = {}
    try:
        payload = launch_step1_runs([root], execute=False, launcher=launcher, scheduler=guard, runs=runs)
    except (SafetyError, OSError, ValueError) as exc:
        if len(runs) == 1:
            return {_key(runs[0]): (None, str(exc))}
        for run in runs:
            results.update(_launch_preflights(root, [run], guard, launcher))
        return results
    for row in payload.get("planned") or []:
        results[_key(row["directory"])] = (dict(row), "")
    for row in payload.get("skipped_runs") or []:
        results[_key(row["directory"])] = (None, str(row.get("reason") or "skipped by step1-launch"))
    return results


def _new_entry(row: Mapping[str, Any], fingerprint: dict[str, Any]) -> dict[str, Any]:
    lineage = row.get("lineage") or {}
    recovery = row.get("recovery") or {}
    category = str(recovery.get("category") or "review")
    reason = str(recovery.get("reason") or "")
    if category not in RECOVER_CATEGORIES:
        category, reason = "review", f"unknown recovery category {category!r}: {reason}"
    return {
        "run": str(row["path"]),
        "relative_path": str(row["relative_path"]),
        "category": category,
        "reason": reason,
        "status_category": category,
        "status_reason": reason,
        "state": row.get("state"),
        "progress": _progress(lineage),
        "generation": lineage.get("generation"),
        "generation_id": lineage.get("generation_id"),
        "action_kind": None,
        "action": None,
        "action_summary": None,
        "fingerprint": fingerprint,
    }


def _to_review(entry: dict[str, Any], reason: str) -> None:
    """Move an entry to ``review`` (the attached dry plan stays for inspection, never executed)."""

    if entry["category"] != "review":
        entry["category"] = "review"
        entry["reason"] = reason


def _check_auto_entry(entry: dict[str, Any], run: Path, root: Path, launcher: str | None) -> None:
    """Refusals shared by every acting category: evaluated read-only before anything is attached."""

    unreadable = unreadable_ledgers(ledger_paths_for(run, [root]))
    if unreadable:
        _to_review(
            entry,
            f"launch ledger {unreadable[0]} is unreadable; a submission cannot be ruled out -- repair it or move it "
            "aside (after inspection) before any recovery",
        )
        return
    if entry["category"] in ("resume", "repair"):
        try:
            resolve_launcher(run, launcher)
        except FileNotFoundError as exc:
            # Preparing a segment that step1-launch would then refuse helps nobody.
            _to_review(entry, f"cannot submit a prepared segment: {exc}")


def _launcher_bypasses_precondition(run: Path, launcher: str | None, wrapped: str | None) -> str | None:
    """Why an explicit ``--launcher`` would bypass the preconditioning wrapper (read-only), else None.

    The preconditioner is wrapped into ``runvasp.sh`` (else ``run.slurm``); submitting
    another launcher would run the MD without it, against what the plan says.
    """

    if not launcher or not wrapped:
        return None
    try:
        chosen = resolve_launcher(run, launcher)
    except FileNotFoundError:
        return None  # already routed to review by _check_auto_entry
    if _key(chosen) == _key(run / wrapped):
        return None
    return (
        f"--launcher {launcher} would be submitted, but the preconditioning static SCF is wrapped into {wrapped}; "
        f"submit {wrapped} instead"
    )


def _attach_resume(
    entry: dict[str, Any],
    run: Path,
    *,
    guard: SchedulerGuard,
    hours: float,
    resume: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    launcher: str | None,
) -> None:
    plan = plan_resume_run(
        run,
        snapshot=guard.snapshot,
        stale_hours=hours,
        contcar_tolerance_angstrom=resume["contcar_tolerance_angstrom"],
        precondition=resume["precondition"],
        fresh_start=resume["fresh_start"],
        accept_warnings=False,
        diagnostic_options=diagnostic,
    )
    entry.update({"action_kind": "resume", "action": plan, "action_summary": _resume_summary(plan)})
    electronic = plan.get("electronic_start") or {}
    wrapped = electronic.get("launcher") if electronic.get("mode") == "precondition" else None
    if plan["status"] != "READY":
        _to_review(entry, f"resume planner disagrees ({plan['status']}): {plan.get('skip_reason')}")
    elif resume["precondition"] and (blocker := precondition_blocker(run)) is not None:
        _to_review(entry, f"cannot precondition: {blocker}")
    elif (bypass := _launcher_bypasses_precondition(run, launcher, wrapped)) is not None:
        _to_review(entry, bypass)
    elif plan.get("fingerprint") != entry["fingerprint"]:
        _to_review(entry, "files changed while the plan was being made; plan again")


def _attach_repair(
    entry: dict[str, Any],
    run: Path,
    *,
    guard: SchedulerGuard,
    hours: float,
    repair: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    launcher: str | None,
) -> None:
    def plan_with(ramp_from: float | None) -> dict[str, Any] | None:
        return plan_repair_run(
            run,
            snapshot=guard.snapshot,
            stale_hours=hours,
            potim_fs=repair["potim_fs"],
            algo=repair["algo"],
            safety_steps=repair["safety_steps"],
            diagnostic_options=diagnostic,
            langevin_gamma=repair["langevin_gamma"],
            ramp_from=ramp_from,
            precondition=repair["precondition"],
        )

    plan = plan_with(repair["ramp_from"])
    remaining = plan.get("repair_nsw") if plan is not None else None
    if repair["ramp_from"] is not None and isinstance(remaining, int) and 0 < remaining < RECOVER_MIN_RAMP_STEPS:
        plan = plan_with(None)
        if plan is not None:
            plan["ramp_skipped"] = (
                f"ramp from {float(repair['ramp_from']):g} K skipped: only {remaining} steps remain "
                f"(< {RECOVER_MIN_RAMP_STEPS}), too few to reheat to the Step2 thermal threshold"
            )
    if plan is None:
        entry["action_kind"] = "repair"
        _to_review(entry, "repair planner disagrees: the run is not hard-unstable under the requested thresholds")
        return
    entry.update({"action_kind": "repair", "action": plan, "action_summary": _repair_summary(plan)})
    wrapped = _precondition_launcher(run) if repair["precondition"] else None
    if plan["status"] != "READY":
        _to_review(entry, f"repair planner disagrees ({plan['status']}): {plan.get('skip_reason')}")
    elif repair["precondition"] and (blocker := precondition_blocker(run)) is not None:
        _to_review(entry, f"cannot precondition: {blocker}")
    elif (bypass := _launcher_bypasses_precondition(run, launcher, wrapped)) is not None:
        _to_review(entry, bypass)
    elif plan.get("fingerprint") != entry["fingerprint"]:
        _to_review(entry, "files changed while the plan was being made; plan again")


def _manifest_hint(run: Path, root: Path) -> str:
    """For a generation-0 run whose step1_manifest.json lies below ``root``: where to launch it from."""

    manifest, row = _manifest_entry(run)
    if manifest is None or row is None or _key(manifest.parent) == _key(root):
        return ""
    # step1-launch reads generation-0 hashes from the manifest of the root it is invoked on.
    return f"; it is listed in {manifest}, so run step1-recover (or step1-launch) on {manifest.parent} to launch it"


def _attach_launch(
    entry: dict[str, Any], preflight: tuple[dict[str, Any] | None, str] | None, *, run: Path, root: Path
) -> None:
    entry["action_kind"] = "launch"
    planned, refusal = preflight if preflight is not None else (None, "not preflighted")
    if planned is None:
        gen0_hint_needed = entry["generation_id"] == GEN0_ID and "launch it from" not in refusal
        hint = _manifest_hint(run, root) if gen0_hint_needed else ""
        _to_review(entry, f"launch preflight refused: {refusal}{hint}")
        return
    entry.update({"action": planned, "action_summary": _launch_summary(planned)})
    if planned.get("generation_id") != entry["generation_id"]:
        _to_review(
            entry,
            f"launch preflight sees generation {planned.get('generation_id')}, status saw {entry['generation_id']}; "
            "plan again",
        )


def plan_step1_recovery(
    root: str | Path,
    *,
    stale_hours: float | None = None,
    scheduler: str | SchedulerGuard = "auto",
    repair_options: Mapping[str, Any] | None = None,
    resume_options: Mapping[str, Any] | None = None,
    diagnostic_options: Mapping[str, Any] | None = None,
    launcher: str | None = None,
) -> dict[str, Any]:
    """Read-only recovery plan for every Step1 run under ``root`` (see the module docstring).

    ``scheduler`` is a ``--scheduler`` mode or a shared ``SchedulerGuard``;
    it is queried once, and ``--scheduler slurm`` refuses when squeue cannot
    be asked.  ``stale_hours=None`` resolves to 0.1 h when Slurm is verified
    and 6 h otherwise.  ``repair_options`` overlay :data:`RECOVER_REPAIR_DEFAULTS`
    (keys of ``plan_repair_run``: ``potim_fs``, ``algo``, ``safety_steps``,
    ``langevin_gamma``, ``ramp_from``, ``precondition``), ``resume_options``
    overlay :data:`RECOVER_RESUME_DEFAULTS` (``contcar_tolerance_angstrom``,
    ``precondition``, ``fresh_start``; review-level warnings are never
    accepted automatically), ``diagnostic_options`` are
    ``diagnose_step1_run`` thresholds (``None`` values select the defaults).
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    repair = _repair_settings(repair_options)
    resume = _resume_settings(resume_options)
    diagnostic, diagnostic_resolved, diagnostic_default = _diagnostic_settings_for(diagnostic_options)

    guard = as_guard(scheduler)
    snapshot = guard.snapshot  # fails closed when Slurm was requested but cannot be queried
    # Fingerprints come first: anything that changes after this point makes
    # the entry refuse at execution time.
    fingerprints = {_key(run): run_fingerprint(run) for run in _discover_runs(root_path)}
    status = step1_status(root_path, stale_hours=stale_hours, scheduler=guard)
    hours = float(status["stale_hours"])

    entries: list[dict[str, Any]] = []
    launch_candidates: list[Path] = []
    for row in status["runs"]:
        run = Path(row["path"])
        entry = _new_entry(row, fingerprints.get(_key(run)) or run_fingerprint(run))
        entries.append(entry)
        if entry["category"] == "done" and not diagnostic_default:
            # step1-status judged the run with the default thresholds.
            verdict = diagnose_step1_run(run, **diagnostic)
            if verdict.get("severity") != "ok":
                detail = (verdict.get("hard_reasons") or verdict.get("warnings") or ["see diagnostic"])[0]
                _to_review(entry, f"the requested diagnostic thresholds report {verdict.get('severity')}: {detail}")
        if entry["category"] not in AUTO_CATEGORIES:
            continue
        _check_auto_entry(entry, run, root_path, launcher)
        if entry["category"] == "resume":
            _attach_resume(
                entry, run, guard=guard, hours=hours, resume=resume, diagnostic=diagnostic, launcher=launcher
            )
        elif entry["category"] == "repair":
            _attach_repair(
                entry, run, guard=guard, hours=hours, repair=repair, diagnostic=diagnostic, launcher=launcher
            )
        elif entry["category"] == "launch":
            launch_candidates.append(run)

    preflights = _launch_preflights(root_path, launch_candidates, guard, launcher)
    for entry in entries:
        if entry["category"] == "launch":
            _attach_launch(entry, preflights.get(_key(entry["run"])), run=Path(entry["run"]), root=root_path)

    categories: dict[str, list[dict[str, Any]]] = {name: [] for name in RECOVER_CATEGORIES}
    for entry in sorted(entries, key=lambda item: item["relative_path"]):
        categories[entry["category"]].append(entry)
    return {
        "format": PLAN_FORMAT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "mode": "dry-run",
        "root": str(root_path),
        "planned_at": utc_now_iso(),
        "scheduler": snapshot.to_dict(),
        "settings": {
            "stale_hours": hours,
            "stale_hours_requested": stale_hours,
            "stale_hours_reason": status.get("stale_hours_reason"),
            "repair": repair,
            "resume": dict(resume, accept_warnings=False),
            "diagnostic": diagnostic_resolved,
            "launcher": launcher,
        },
        "categories": categories,
        "counts": {name: len(categories[name]) for name in RECOVER_CATEGORIES},
        "auto_categories": list(AUTO_CATEGORIES),
        "execute_command": execute_command(root_path),
    }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _work_list(plan: Mapping[str, Any], selected: tuple[str, ...], submit: bool) -> list[dict[str, Any]]:
    """Selected entries in execution order: repairs + resumes (by relative path), then launches."""

    categories = plan["categories"]
    prepare = [entry for name in ("repair", "resume") if name in selected for entry in categories[name]]
    work = sorted(prepare, key=lambda entry: entry["relative_path"])
    if submit and "launch" in selected:
        work += sorted(categories["launch"], key=lambda entry: entry["relative_path"])
    return work


def _journal_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    prepares = entry["category"] in ("repair", "resume")
    return {
        "relative_path": entry["relative_path"],
        "directory": entry["run"],
        "category": entry["category"],
        "action": entry.get("action_summary"),
        "outcome": OUTCOME_PENDING,
        "parent_generation_id": entry.get("generation_id") if prepares else None,
        "prepared": False,
        "archive": None,
        "generation_id": None if prepares else entry.get("generation_id"),
        "submitted": False,
        "job_id": None,
        "batch_id": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }


def _empty_journal(root: Path) -> dict[str, Any]:
    return {"format": JOURNAL_FORMAT, "schema_version": JOURNAL_SCHEMA_VERSION, "root": str(root), "executions": []}


def _load_journal(path: Path, root: Path) -> dict[str, Any]:
    """The existing journal (or a new one); refuses one that exists but cannot be read (it is provenance)."""

    if not path.exists():
        return _empty_journal(root)
    payload = read_json(path)
    if payload.get("format") != JOURNAL_FORMAT or not isinstance(payload.get("executions"), list):
        raise SafetyError(
            f"{path} exists but is not a readable step1-recover journal; it is provenance, so move it aside "
            "(after inspection) before executing again"
        )
    return payload


def _write_journal(path: Path, root: Path, execution: dict[str, Any]) -> None:
    """Insert/replace ``execution`` in the journal (read-modify-write under the journal lock, atomic write)."""

    with _file_lock(path):
        journal = _load_journal(path, root)
        executions = [
            item
            for item in journal["executions"]
            if not (isinstance(item, dict) and item.get("execution_id") == execution["execution_id"])
        ]
        executions.append(execution)
        journal["executions"] = executions
        journal["updated_at"] = utc_now_iso()
        atomic_write_json(path, journal)


def _labels(rows: Iterable[Mapping[str, Any]]) -> str:
    return ", ".join(str(row["relative_path"]) for row in rows) or "none"


def _submitted_labels(rows: Iterable[Mapping[str, Any]]) -> str:
    return ", ".join(f"{row['relative_path']} (job {row['job_id']})" for row in rows) or "none"


def execute_step1_recovery(
    root: str | Path,
    *,
    only: Iterable[str] | str = AUTO_CATEGORIES,
    submit: bool = True,
    stale_hours: float | None = None,
    scheduler: str | SchedulerGuard = "auto",
    repair_options: Mapping[str, Any] | None = None,
    resume_options: Mapping[str, Any] | None = None,
    diagnostic_options: Mapping[str, Any] | None = None,
    launcher: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Plan afresh, then prepare (and, unless ``submit=False``, submit) the selected entries.

    ``only`` selects among :data:`AUTO_CATEGORIES`; with ``submit=False`` the
    repairs/resumes are prepared without ``sbatch`` and launch entries are
    left alone (a later plan lists the prepared runs under ``launch``).  The
    options are those of :func:`plan_step1_recovery`.  Returns the execution
    payload (also journalled in ``<root>/step1_recover.json``); raises
    :class:`SafetyError` at the first failure, after journalling which runs
    were changed, which were submitted and which were not attempted.  When no
    entry is selected nothing is written at all.
    """

    selected = _selected_categories(only)
    emit = progress or (lambda _message: None)
    guard = as_guard(scheduler)
    plan = plan_step1_recovery(
        root,
        stale_hours=stale_hours,
        scheduler=guard,
        repair_options=repair_options,
        resume_options=resume_options,
        diagnostic_options=diagnostic_options,
        launcher=launcher,
    )
    root_path = Path(plan["root"])
    journal_path = root_path / RECOVER_JOURNAL
    work = _work_list(plan, selected, bool(submit))
    skipped_launch: list[str] = []
    if not submit and "launch" in selected:
        # --no-submit: a launch entry IS a submission, so it is left alone.
        skipped_launch = [entry["relative_path"] for entry in plan["categories"]["launch"]]
    execution: dict[str, Any] = {
        "execution_id": f"x-{utc_stamp()}-{uuid.uuid4().hex[:6]}",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "RUNNING",
        "selected_categories": list(selected),
        "submit": bool(submit),
        "scheduler": plan["scheduler"],
        "settings": plan["settings"],
        "plan_counts": plan["counts"],
        "skipped_launch_entries": skipped_launch,
        "error": None,
        "runs": [_journal_row(entry) for entry in work],
    }
    result: dict[str, Any] = {
        "format": EXECUTION_FORMAT,
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "mode": "executed",
        "root": str(root_path),
        "journal": None,
        "status": None,
        "changed": [],
        "submitted": [],
        "execution": execution,
        "plan": plan,
    }
    if skipped_launch:
        emit(f"--no-submit: leaving {len(skipped_launch)} launch entr(y/ies) alone: {', '.join(skipped_launch)}")
    if not work:
        execution.update({"status": "NOTHING_TO_DO", "finished_at": utc_now_iso()})
        result["status"] = "NOTHING_TO_DO"
        emit("step1-recover: nothing selected to resume, repair or launch; nothing was changed")
        return result

    if any(_key(entry["run"]) == _key(root_path) for entry in work):
        # The root is itself the run: its journal lives inside it, so never
        # write it into a directory that has become an active WorkDir.
        guard.assert_inactive(root_path)
    _write_journal(journal_path, root_path, execution)
    result["journal"] = str(journal_path)
    emit(f"step1-recover: executing {len(work)} entr(y/ies); journal {journal_path}")

    changed: list[dict[str, Any]] = []
    submitted: list[dict[str, Any]] = []
    rows = execution["runs"]
    for index, (entry, row) in enumerate(zip(work, rows, strict=True)):
        run = Path(entry["run"])
        label = entry["relative_path"]
        category = entry["category"]
        row.update({"outcome": OUTCOME_RUNNING, "started_at": utc_now_iso()})
        emit(f"[{index + 1}/{len(work)}] {category} {label}: {entry.get('action_summary') or ''}".rstrip(": "))
        try:
            guard.assert_inactive(run)
            if run_fingerprint(run) != entry["fingerprint"]:
                raise SafetyError(
                    f"Refusing to mutate {run}: it changed since planning (file fingerprint differs); plan again"
                )
            if category in ("repair", "resume"):
                executor = execute_repair_plan if category == "repair" else execute_resume_plan
                prepared = executor(entry["action"], guard=guard, ledger_roots=(root_path,))
                row.update(
                    {
                        "outcome": OUTCOME_PREPARED,
                        "prepared": True,
                        "prepared_at": prepared.get("prepared_at"),
                        "archive": prepared.get("archive"),
                        "generation_id": prepared.get("generation_id"),
                    }
                )
                changed.append({"relative_path": label, "generation_id": row["generation_id"]})
                emit(f"    CHANGED {label}: prepared {row['generation_id']} (archive {row['archive']})")
                _write_journal(journal_path, root_path, execution)
            if submit:
                launched = launch_step1_runs(
                    [root_path], execute=True, launcher=launcher, scheduler=guard, runs=[run], progress=emit
                )
                job = (launched.get("jobs") or [{}])[0]
                row.update(
                    {
                        "outcome": OUTCOME_SUBMITTED,
                        "submitted": True,
                        "job_id": job.get("job_id"),
                        "batch_id": launched.get("batch_id"),
                        "launch_kind": job.get("kind"),
                        "generation_id": row["generation_id"] or job.get("generation_id"),
                    }
                )
                submitted.append(
                    {"relative_path": label, "job_id": row["job_id"], "generation_id": row["generation_id"]}
                )
                emit(f"    SUBMITTED {label}: job {row['job_id']} ({row['generation_id']})")
            row["finished_at"] = utc_now_iso()
            _write_journal(journal_path, root_path, execution)
        except BaseException as exc:  # noqa: BLE001 - journalled, then re-raised
            now = utc_now_iso()
            row.update({"outcome": OUTCOME_FAILED, "error": str(exc) or type(exc).__name__, "finished_at": now})
            interrupted = interrupted_archive(run)
            if interrupted is not None:
                row["interrupted_archive"] = str(interrupted)
            for later in rows[index + 1 :]:
                later["outcome"] = OUTCOME_NOT_ATTEMPTED
            # KeyboardInterrupt / SystemExit: journalled as FAILED too (flagged), then re-raised as is.
            interrupted_run = not isinstance(exc, Exception)
            execution.update({"status": "FAILED", "finished_at": now, "error": row["error"]})
            if interrupted_run:
                execution["interrupted"] = True
            try:
                _write_journal(journal_path, root_path, execution)
                journal_note = f"Journal: {journal_path}"
            except Exception as journal_exc:  # noqa: BLE001 - reported below
                journal_note = f"Writing the journal {journal_path} ALSO failed ({journal_exc}); record this by hand"
            if interrupted_run:
                raise
            if row["submitted"]:
                done = "prepared and submitted" if row["prepared"] else "submitted"
                state = f"it WAS {done} (job {row['job_id']}); only journalling that failed"
            elif row["prepared"]:
                state = "it WAS prepared (changed) but not submitted"
            elif interrupted is not None:
                state = f"its interrupted mutation is archived at {interrupted} (ARCHIVE_MANIFEST status IN_PROGRESS)"
            else:
                state = "it was not modified"
            result.update({"status": "FAILED", "changed": changed, "submitted": submitted})
            raise SafetyError(
                f"step1-recover stopped at {label} ({category}): {exc}; {state}. "
                f"Changed (prepared): {_labels(changed)}. Submitted: {_submitted_labels(submitted)}. "
                f"Not attempted: {_labels(rows[index + 1 :])}. {journal_note}"
            ) from exc

    execution.update({"status": "COMPLETED", "finished_at": utc_now_iso()})
    try:
        _write_journal(journal_path, root_path, execution)
    except Exception as exc:  # noqa: BLE001 - every run is done; say exactly what happened
        raise SafetyError(
            f"step1-recover finished every entry, but marking {journal_path} COMPLETED failed ({exc}); its per-run "
            f"rows are current. Changed (prepared): {_labels(changed)}. Submitted: {_submitted_labels(submitted)}"
        ) from exc
    result.update({"status": "COMPLETED", "changed": changed, "submitted": submitted})
    emit(
        f"step1-recover: COMPLETED; changed {len(changed)} run(s): {_labels(changed)}; "
        f"submitted {len(submitted)} job(s): {_submitted_labels(submitted)}"
    )
    return result


# --------------------------------------------------------------------------- #
# Human output
# --------------------------------------------------------------------------- #


def _settings_lines(settings: Mapping[str, Any]) -> list[str]:
    repair = settings.get("repair") or {}
    resume = settings.get("resume") or {}
    diagnostic = settings.get("diagnostic") or {}
    repair_bits = [f"POTIM {float(repair.get('potim_fs') or 0.0):g} fs", f"ALGO {repair.get('algo')}"]
    repair_bits.append("precondition" if repair.get("precondition") else "no precondition")
    ramp = repair.get("ramp_from")
    repair_bits.append(f"ramp from {float(ramp):g} K" if ramp is not None else "no ramp")
    if repair.get("langevin_gamma") is not None:
        repair_bits.append(f"Langevin {float(repair['langevin_gamma']):g}/ps")
    repair_bits.append(f"safety {repair.get('safety_steps')} steps")
    resume_bits = [f"CONTCAR tolerance {float(resume.get('contcar_tolerance_angstrom') or 0.0):g} A"]
    if resume.get("precondition"):
        resume_bits.append("precondition")
    if resume.get("fresh_start"):
        resume_bits.append("fresh electronic start")
    limit = diagnostic.get("max_temperature_k")
    diagnostic_bits = [
        f"energy jump {diagnostic.get('energy_jump_ev'):g} eV",
        f"catastrophic {diagnostic.get('catastrophic_energy_ev'):g} eV",
        f"startup grace {diagnostic.get('startup_grace_steps')} steps",
        f"T limit {float(limit):g} K" if limit else "T limit max(1200 K, 4*T_target)",
    ]
    lines = [
        "  repair: " + ", ".join(repair_bits),
        "  resume: " + ", ".join(resume_bits),
        "  diagnostics: " + ", ".join(diagnostic_bits),
    ]
    if settings.get("launcher"):
        lines.append(f"  launcher: {settings['launcher']}")
    return lines


def _plan_body(plan: Mapping[str, Any]) -> list[str]:
    scheduler = plan.get("scheduler") or {}
    settings = plan.get("settings") or {}
    verified = "verified" if scheduler.get("verified") else "NOT verified"
    lines = [
        f"Step1 recovery plan: {plan.get('root')}",
        f"  scheduler: {verified} - {scheduler.get('reason')}",
    ]
    if settings.get("stale_hours") is not None:
        reason = settings.get("stale_hours_reason")
        lines.append(f"  activity window: {float(settings['stale_hours']):g} h" + (f" ({reason})" if reason else ""))
    lines += _settings_lines(settings)
    counts = plan.get("counts") or {}
    lines.append("  counts: " + ", ".join(f"{name} {counts.get(name, 0)}" for name in RENDER_ORDER))

    categories = plan.get("categories") or {}
    for name in RENDER_ORDER:
        entries = categories.get(name) or []
        suffix = " -- acted on by --execute" if name in AUTO_CATEGORIES else " -- left untouched"
        lines += ["", f"{name} ({len(entries)}){suffix}"]
        width = max((len(str(entry["relative_path"])) for entry in entries), default=0)
        for entry in entries:
            lines.append(f"  {str(entry['relative_path']):<{width}}  {entry.get('progress', ''):>9}  {entry['reason']}")
            if name in AUTO_CATEGORIES and entry.get("action_summary"):
                lines.append(f"      {entry['action_summary']}")
            elif name == "review" and entry.get("status_category") not in (None, "review"):
                lines.append(f"      (step1-status classified it '{entry['status_category']}'; not acted on)")
    return lines


def render_recovery_plan(payload: Mapping[str, Any]) -> str:
    """Human-readable plan: header, one block per category (active, done, review, launch, resume, repair), footer.

    The text is ASCII apart from run names/paths (the typographic characters of
    step1-status reasons are spelled in ASCII).  ``payload`` may carry the
    optional keys ``selected_categories`` / ``submit`` (the CLI sets them from
    ``--only`` / ``--no-submit``) to tailor the execute command in the footer.
    """

    lines = _plan_body(payload)
    selected = tuple(payload.get("selected_categories") or AUTO_CATEGORIES)
    submit = bool(payload.get("submit", True))
    counts = payload.get("counts") or {}
    actionable = sum(counts.get(name, 0) for name in selected if submit or name != "launch")
    lines.append("")
    if actionable:
        verbs = "+".join(name for name in AUTO_CATEGORIES if name in selected and (submit or name != "launch"))
        lines.append(
            f"Dry run: nothing changed. Execute {verbs} with: "
            f"{execute_command(payload.get('root', '.'), selected, submit=submit)}"
        )
    else:
        lines.append("Dry run: nothing changed. Nothing to resume, repair or launch automatically.")
    return "\n".join(lines).translate(_ASCII)


def render_recovery_execution(payload: Mapping[str, Any]) -> str:
    """Human-readable summary of an :func:`execute_step1_recovery` payload (the plan it acted on, then outcomes)."""

    plan = payload.get("plan") or {}
    execution = payload.get("execution") or {}
    lines = _plan_body(plan) if plan else [f"Step1 recovery: {payload.get('root')}"]
    lines += [
        "",
        f"Execution {execution.get('execution_id')}: {execution.get('status')} "
        f"(selected {', '.join(execution.get('selected_categories') or [])}; "
        f"submit {'on' if execution.get('submit') else 'off'})",
    ]
    rows = execution.get("runs") or []
    width = max((len(str(row["relative_path"])) for row in rows), default=0)
    for row in rows:
        detail = []
        if row.get("generation_id"):
            detail.append(str(row["generation_id"]))
        if row.get("job_id"):
            detail.append(f"job {row['job_id']}")
        if row.get("error"):
            detail.append(f"error: {row['error']}")
        label = f"{str(row['relative_path']):<{width}}"
        lines.append(f"  [{row.get('outcome')}] {label}  {row.get('category')}  {'; '.join(detail)}".rstrip())
    if not rows:
        lines.append("  nothing selected; nothing was changed or submitted")
    for name in execution.get("skipped_launch_entries") or []:
        lines.append(f"  [left alone: --no-submit] {name}  launch")
    if payload.get("journal"):
        lines.append(f"Journal: {payload['journal']}")
    return "\n".join(lines).translate(_ASCII)
