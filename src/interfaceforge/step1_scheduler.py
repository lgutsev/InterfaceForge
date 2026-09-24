"""Slurm activity guard for the mutating Step1 recovery commands.

Every Step1 command that rewrites a run directory (repair, resume, recover,
launch) must first prove that no Slurm job is using that directory as its
WorkDir.  File age alone is a poor proxy: a job can sit in the queue for days
without touching its files, and a healthy run can go quiet during a long SCF
cycle.  This module asks ``squeue`` instead and turns the answer into a
:class:`SchedulerSnapshot`; :class:`SchedulerGuard` re-checks it immediately
before each mutation so a job submitted between planning and execution is
still caught.

When Slurm cannot be queried the behaviour depends on what was requested:
``--scheduler auto`` on a machine without ``squeue`` falls back to the
file-age guard (unverified), while ``--scheduler slurm`` refuses outright
rather than guessing.  Read-only callers (status, plan display) use
:meth:`SchedulerGuard.try_snapshot`, which reports the failure instead of
raising and remembers it so a hung ``squeue`` is not re-run for every row.
"""

from __future__ import annotations

import getpass
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from .errors import SafetyError

SCHEDULER_MODES = ("auto", "slurm", "none")
DEFAULT_RECHECK_SECONDS = 15.0
DEFAULT_STALE_HOURS = 6.0  # file-age guard when Slurm is NOT verified
SLURM_VERIFIED_SETTLE_HOURS = 0.1  # 6 min settle window when Slurm IS verified

# ``%Z`` is the job WorkDir; squeue prints these placeholders when it has none.
_NO_WORKDIR = {"", "N/A", "(null)"}
_SQUEUE_FORMAT = "%i|%T|%Z"
_SQUEUE_TIMEOUT_SECONDS = 60


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _printable(text: str) -> str:
    """``text`` with lone surrogates (from ``surrogateescape`` decoding) spelled as ``\\udcXX``.

    Keeps error messages and job ids encodable as UTF-8 whatever bytes squeue emitted.
    """

    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _check_mode(mode: str) -> str:
    if mode not in SCHEDULER_MODES:
        raise ValueError(f"unknown scheduler mode {mode!r}; expected one of {', '.join(SCHEDULER_MODES)}")
    return mode


def _same_or_inside(path_real: str, run_real: str) -> bool:
    """True when ``path_real`` is ``run_real`` or lies below it (both already realpath'd).

    ``normcase`` makes the comparison case-insensitive on Windows and is the
    identity on POSIX.  The separator suffix stops ``leaf1`` matching
    ``leaf10``.
    """

    path_norm = os.path.normcase(path_real)
    run_norm = os.path.normcase(run_real)
    if path_norm == run_norm:
        return True
    prefix = run_norm if run_norm.endswith(os.sep) else run_norm + os.sep
    return path_norm.startswith(prefix)


@dataclass
class SchedulerSnapshot:
    """One answer from the scheduler (or the reason there is none)."""

    requested: str  # "auto" | "slurm" | "none"
    mode: str  # resolved: "slurm" | "none"
    verified: bool  # True only if squeue answered successfully
    reason: str  # human-readable, e.g. "squeue -u lg: 3 active job(s)"
    taken_at: str  # UTC ISO-8601, seconds
    jobs: list[dict[str, Any]]  # [{"job_id","state","workdir","workdir_real"}]
    monotonic: float = 0.0  # time.monotonic() when taken (not serialised)

    def active_jobs_for(self, run: str | os.PathLike[str]) -> list[dict[str, Any]]:
        """Jobs whose realpath(workdir) == realpath(run) or lies INSIDE run
        (e.g. run/precondition). Ancestor WorkDirs do NOT count (a recovery job
        submitted from the Step1 root must not block every leaf)."""

        run_real = os.path.realpath(os.fspath(run))
        matches = []
        for job in self.jobs:
            workdir_real = job.get("workdir_real")
            if not workdir_real:
                # Hand-built job rows may omit workdir_real; never realpath("")
                # (that would be the cwd).
                workdir = job.get("workdir")
                if workdir is None or str(workdir).strip() in _NO_WORKDIR:
                    continue
                workdir_real = os.path.realpath(os.fspath(workdir))
            if _same_or_inside(workdir_real, run_real):
                matches.append(job)
        return matches

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "mode": self.mode,
            "verified": self.verified,
            "reason": self.reason,
            "taken_at": self.taken_at,
            "active_jobs": len(self.jobs),
        }


def _own_job_ids() -> frozenset[str]:
    """The squeue ``%i`` ids under which the current (recovery) job itself may be listed.

    ``SLURM_JOB_ID`` for a plain job.  squeue prints an array task as
    ``<SLURM_ARRAY_JOB_ID>_<SLURM_ARRAY_TASK_ID>`` rather than by its own
    ``SLURM_JOB_ID``, so that form is added when both variables are set.
    """

    ids = set()
    job_id = (os.environ.get("SLURM_JOB_ID") or "").strip()
    if job_id:
        ids.add(job_id)
    array_job_id = (os.environ.get("SLURM_ARRAY_JOB_ID") or "").strip()
    array_task_id = (os.environ.get("SLURM_ARRAY_TASK_ID") or "").strip()
    if array_job_id and array_task_id:
        ids.add(f"{array_job_id}_{array_task_id}")
    return frozenset(ids)


def _parse_squeue(stdout: str, *, own_job_ids: Collection[str] = ()) -> tuple[list[dict[str, Any]], list[str]]:
    """Jobs with a usable WorkDir from ``squeue -o %i|%T|%Z`` output.

    Returns ``(jobs, dropped_own_ids)``.  Lines that do not split into three
    fields are ignored; ``split("|", 2)`` keeps any ``|`` inside the WorkDir.
    """

    jobs: list[dict[str, Any]] = []
    dropped_own: list[str] = []
    for line in stdout.splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) != 3:
            continue
        job_id, state, workdir = (part.strip() for part in parts)
        if workdir in _NO_WORKDIR:
            continue
        if job_id and job_id in own_job_ids:
            # The recovery job itself (submitted from the Step1 root) is not a
            # conflicting writer.
            dropped_own.append(_printable(job_id))
            continue
        jobs.append(
            {
                # A line with a WorkDir but no id/state is still a listed job:
                # keep it (blocking is the safe failure) with a placeholder.
                "job_id": _printable(job_id) or "?",
                "state": _printable(state) or "UNKNOWN",
                # Kept verbatim (it may carry surrogate escapes) so it maps back to
                # the real directory bytes through os.fsencode on POSIX.
                "workdir": workdir,
                "workdir_real": os.path.realpath(workdir),
            }
        )
    return jobs, dropped_own


def _default_user() -> str:
    user = os.environ.get("USER")
    if user:
        return user
    try:
        return getpass.getuser()
    except (ImportError, KeyError, OSError) as exc:
        # getpass raises OSError (3.13+), KeyError or ImportError (no pwd) when
        # it cannot name the user; squeue -u without a user is meaningless.
        raise SafetyError(
            f"could not query Slurm (cannot determine the user: {exc}); refusing rather than guessing job state"
        ) from exc


def _query_failed(exc: BaseException) -> SafetyError:
    detail = f"{type(exc).__name__}: {exc}"
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if isinstance(stderr, str) and stderr.strip():
        detail += f"; stderr: {stderr.strip()[:300]}"
    return SafetyError(f"could not query Slurm ({_printable(detail)}); refusing rather than guessing job state")


def take_snapshot(mode: str = "auto", *, user: str | None = None) -> SchedulerSnapshot:
    """Query the scheduler once.  ``"slurm"`` raises :class:`SafetyError` if squeue fails."""

    requested = _check_mode(mode)
    if requested == "none":
        return SchedulerSnapshot(
            requested=requested,
            mode="none",
            verified=False,
            reason="scheduler check disabled (--scheduler none)",
            taken_at=_utc_now_iso(),
            jobs=[],
            monotonic=time.monotonic(),
        )
    if requested == "auto" and shutil.which("squeue") is None:
        return SchedulerSnapshot(
            requested=requested,
            mode="none",
            verified=False,
            reason="no squeue on PATH; relying on file-age guard",
            taken_at=_utc_now_iso(),
            jobs=[],
            monotonic=time.monotonic(),
        )

    squeue_user = user or _default_user()
    # -a (--all): also list the user's jobs in hidden partitions.
    command = ["squeue", "-h", "-a", "-u", squeue_user, "-o", _SQUEUE_FORMAT]
    try:
        # errors="surrogateescape": decoding can never fail -- not even inside the
        # Windows reader threads, where a decode error would silently leave stdout
        # None -- and a non-UTF-8 WorkDir round-trips through os.fsencode on POSIX.
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            errors="surrogateescape",
            timeout=_SQUEUE_TIMEOUT_SECONDS,
        )
        if not isinstance(result.stdout, str):
            # capture_output always yields a str ("" when there are no jobs); None
            # means the output was lost, which must never read as "no jobs".
            raise ValueError(f"squeue output could not be read (stdout is {type(result.stdout).__name__})")
        jobs, dropped_own = _parse_squeue(result.stdout, own_job_ids=_own_job_ids())
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        # OSError covers FileNotFoundError (no squeue); SubprocessError covers
        # CalledProcessError (squeue exit != 0) and TimeoutExpired; UnicodeError /
        # ValueError cover undecodable or unusable output (e.g. realpath of a
        # WorkDir with an embedded NUL).
        raise _query_failed(exc) from exc

    reason = f"squeue -u {squeue_user}: {len(jobs)} active job(s)"
    if dropped_own:
        reason += f" (ignoring this job {', '.join(dropped_own)})"
    return SchedulerSnapshot(
        requested=requested,
        mode="slurm",
        verified=True,
        reason=reason,
        taken_at=_utc_now_iso(),
        jobs=jobs,
        monotonic=time.monotonic(),
    )


class SchedulerGuard:
    """Holds the planning snapshot and re-checks it right before each mutation."""

    def __init__(
        self,
        mode: str = "auto",
        *,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        snapshot_factory: Callable[[], SchedulerSnapshot] | None = None,
    ) -> None:
        self.mode = _check_mode(mode)
        self.recheck_seconds = recheck_seconds  # validated by the property setter
        self._factory: Callable[[], SchedulerSnapshot] = snapshot_factory or (lambda: take_snapshot(self.mode))
        self._snapshot: SchedulerSnapshot | None = None
        self._failure: str | None = None  # last refresh error, remembered until a refresh succeeds

    @property
    def recheck_seconds(self) -> float:
        """Maximum age of a snapshot that :meth:`assert_inactive` reuses; ``<= 0`` re-checks every time."""

        return self._recheck_seconds

    @recheck_seconds.setter
    def recheck_seconds(self, value: float) -> None:
        seconds = float(value)
        if math.isnan(seconds) or seconds == math.inf:
            # Either would silently switch off the pre-mutation re-check.
            raise ValueError(f"recheck_seconds must be a finite number of seconds (<= 0 = always), got {value!r}")
        self._recheck_seconds = seconds

    @classmethod
    def fixed(cls, jobs: Iterable[Mapping[str, Any]] = (), *, verified: bool = True) -> SchedulerGuard:
        """Test/analysis helper: every snapshot returns these jobs. Each job dict needs
        job_id, state, workdir (workdir_real is computed)."""

        rows = []
        for job in jobs:
            missing = [key for key in ("job_id", "state", "workdir") if key not in job]
            if missing:
                raise ValueError(f"fixed scheduler job is missing {', '.join(missing)}: {dict(job)!r}")
            if job["workdir"] is None or str(job["workdir"]).strip() in _NO_WORKDIR:
                # Same rule as squeue parsing: a job without a WorkDir cannot be
                # matched to a run (and realpath("") would be the cwd).
                continue
            row = dict(job)
            row["job_id"] = str(job["job_id"])
            row["state"] = str(job["state"])
            row["workdir"] = os.fspath(job["workdir"])
            row["workdir_real"] = os.path.realpath(row["workdir"])
            rows.append(row)
        mode = "slurm" if verified else "none"
        reason = f"fixed scheduler snapshot: {len(rows)} active job(s)" + ("" if verified else " (unverified)")

        def factory() -> SchedulerSnapshot:
            return SchedulerSnapshot(
                requested=mode,
                mode=mode,
                verified=verified,
                reason=reason,
                taken_at=_utc_now_iso(),
                jobs=[dict(row) for row in rows],
                monotonic=time.monotonic(),
            )

        return cls(mode, snapshot_factory=factory)

    @property
    def snapshot(self) -> SchedulerSnapshot:
        """The current snapshot; the first access takes it.

        While no snapshot is held (first access, or after a failed refresh) every
        access queries the scheduler and raises :class:`SafetyError` if that fails.
        Read-only callers should use :meth:`try_snapshot` instead.
        """

        if self._snapshot is None:
            return self.refresh()
        return self._snapshot

    def refresh(self) -> SchedulerSnapshot:
        """Take a new snapshot.

        On failure the previous snapshot is dropped (nothing keeps using an answer
        the scheduler could not confirm), the error is remembered for
        :meth:`try_snapshot`, and the exception propagates.
        """

        self._snapshot = None
        try:
            snapshot = self._factory()
        except SafetyError as exc:
            self._failure = str(exc)
            raise
        if snapshot.monotonic == 0.0:
            # Factories that do not stamp the snapshot (the dataclass default): age
            # it from now.  Only exactly 0.0 means "unstamped" -- a real stamp can be
            # <= 0 (monotonic() shortly after boot minus a backdated age).
            snapshot = replace(snapshot, monotonic=time.monotonic())
        self._failure = None
        self._snapshot = snapshot
        return snapshot

    def try_snapshot(self) -> tuple[SchedulerSnapshot | None, str | None]:
        """READ-ONLY callers (status, planning display): (snapshot, None) or (None, error text).
        A failure is remembered for this guard until refresh() succeeds, so a hung squeue is
        not re-run per row. Mutation paths must still use assert_inactive (fails closed)."""

        if self._failure is not None:
            return None, self._failure
        if self._snapshot is not None:
            return self._snapshot, None
        try:
            return self.refresh(), None
        except SafetyError as exc:
            return None, str(exc)

    def active_jobs_for(self, run: str | os.PathLike[str]) -> list[dict[str, Any]]:
        """Active jobs for ``run`` in the current snapshot (an existing snapshot is never refreshed)."""

        return self.snapshot.active_jobs_for(run)

    def _stale(self, snapshot: SchedulerSnapshot) -> bool:
        recheck = self.recheck_seconds
        if not recheck > 0.0:
            return True
        age = time.monotonic() - snapshot.monotonic
        # Fail safe: reuse only a snapshot of known, non-negative age younger than
        # recheck_seconds.  A negative age (clock went backwards) or a NaN stamp
        # cannot be trusted, so re-check.
        return not (0.0 <= age < recheck)

    def assert_inactive(self, run: str | os.PathLike[str]) -> SchedulerSnapshot:
        """Refresh if snapshot older than recheck_seconds (always when recheck_seconds <= 0),
        then raise SafetyError("Refusing to mutate <run>: active Slurm job(s) <id> (<STATE>)
        use it as WorkDir") if any. Returns the snapshot used."""

        snapshot = self._snapshot
        if snapshot is None or self._stale(snapshot):
            snapshot = self.refresh()
        jobs = snapshot.active_jobs_for(run)
        if jobs:
            listing = ", ".join(f"{job.get('job_id')} ({job.get('state')})" for job in jobs)
            raise SafetyError(f"Refusing to mutate {os.fspath(run)}: active Slurm job(s) {listing} use it as WorkDir")
        return snapshot


def as_guard(scheduler: str | SchedulerGuard) -> SchedulerGuard:
    """Accept a ``--scheduler`` mode string or an existing guard (shared across commands)."""

    if isinstance(scheduler, SchedulerGuard):
        return scheduler
    if isinstance(scheduler, str):
        return SchedulerGuard(scheduler)
    raise TypeError(f"scheduler must be a mode string or SchedulerGuard, not {type(scheduler).__name__}")


def resolve_stale_hours(stale_hours: float | None, snapshot: SchedulerSnapshot) -> tuple[float, str]:
    """Explicit value wins. Else SLURM_VERIFIED_SETTLE_HOURS if snapshot.verified, else
    DEFAULT_STALE_HOURS. Returns (hours, human reason)."""

    if stale_hours is not None:
        hours = float(stale_hours)
        if math.isnan(hours) or hours < 0.0:
            raise ValueError(f"stale_hours must be >= 0, got {stale_hours!r}")
        return hours, f"explicit --stale-hours {hours:g} h"
    if snapshot.verified:
        minutes = SLURM_VERIFIED_SETTLE_HOURS * 60.0
        return (
            SLURM_VERIFIED_SETTLE_HOURS,
            f"Slurm verified ({snapshot.reason}): squeue decides activity; "
            f"{minutes:g} min settle window for files written after a job left the queue",
        )
    return (
        DEFAULT_STALE_HOURS,
        f"Slurm not verified ({snapshot.reason}): files modified within {DEFAULT_STALE_HOURS:g} h count as active",
    )
