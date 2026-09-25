"""Generations, launch ledgers, archives and temperature schedules for Step1 recovery.

A Step1 run evolves through *generations*.  Generation 0 is the run exactly as
``step1-prepare`` wrote it (no record).  Every repair or resume creates
generation N+1 and writes exactly one current segment record at the run top
level (``step1_repair.json`` or ``step1_resume.json``); the previous record is
archived and retired first, so at most one of the two is current.

This module is the single place that reads that lineage -- including schema-1
repair records written before generations existed, which are reconstructed
through their archive chain -- and the generation-aware ``step1_launch.json``
ledgers used to decide whether the *current* generation was already submitted.
A historical ``SUBMITTED`` row for an older generation therefore never blocks
the launch of a newer one.

Nothing here decides dry-run versus execute or talks to Slurm: every function
is either read-only or an explicit, atomic write that callers invoke only after
their own scheduler and fingerprint checks.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import socket
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aimd import _first_float
from .errors import SafetyError
from .vasp import archive_run, parse_incar

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPAIR_RECORD = "step1_repair.json"
RESUME_RECORD = "step1_resume.json"
LAUNCH_LEDGER = "step1_launch.json"
LAUNCH_TSV = "step1_launch.tsv"
MANIFEST = "step1_manifest.json"
RECOVER_JOURNAL = "step1_recover.json"
ARCHIVE_MANIFEST = "ARCHIVE_MANIFEST.json"
GEN0_ID = "g0-prepare"

REPAIR_FORMAT = "interfaceforge-step1-repair"
RESUME_FORMAT = "interfaceforge-step1-resume"
LAUNCH_FORMAT = "interfaceforge-step1-launch"
ARCHIVE_FORMAT = "interfaceforge-step1-archive"
REPAIR_SCHEMA_VERSION = 2
RESUME_SCHEMA_VERSION = 1
LAUNCH_SCHEMA_VERSION = 2

STEP1_EXTRA_ARCHIVE_FILES = (
    "step1_resume.json",
    "step1_launch.json",
    "step1_launch.tsv",
    "step1_recover.json",
    "INCAR.precondition",
    "vasp_md.dat",
    "vasp_md_FINAL.dat",
    ".vasp_md.dat",
    "MD_TempPlot.png",
    "PCDAT",
    "EIGENVAL",
    "IBZKPT",
)
NOT_ARCHIVED = ("WAVECAR", "CHG", "CHGCAR")
PRECONDITION_ARCHIVE_FILES = ("INCAR", "OSZICAR", "OUTCAR", "vasprun.xml")
FINGERPRINT_FILES = (
    "INCAR",
    "POSCAR",
    "CONTCAR",
    "OSZICAR",
    "OUTCAR",
    "XDATCAR",
    "step1_repair.json",
    "step1_resume.json",
)

# Every key a schema-1 step1_repair.json carried beyond the common segment
# keys; schema-2 repair records keep all of them with identical meaning so an
# older InterfaceForge (and any operator script) still reads them correctly.
LEGACY_REPAIR_KEYS = (
    "safe_prefix_steps",
    "safe_segment_steps",
    "previous_safe_prefix_steps",
    "rewind_frame",
    "repair_nsw",
    "original_potim_fs",
    "repair_potim_fs",
    "repair_algo",
    "repair_electronic",
    "repair_langevin_gamma",
    "repair_ramp_from_k",
    "repair_precondition",
    "source",
    "diagnostic",
    "age_hours",
)
RESUME_KEYS = (
    "resume_nsw",
    "restart_source",
    "restart_frame",
    "segment_accepted_steps",
    "contcar_check",
    "electronic_start",
    "temperature_continuation",
    "diagnostic",
    "age_hours",
)
SCHEDULE_KEYS = ("tebeg_k", "teend_k", "nsw", "thermostat", "ramp")

LAUNCH_ROW_KEYS = (
    "status",
    "job_id",
    "kind",
    "root",
    "relative_path",
    "directory",
    "launcher",
    "notes",
    "detail",
    "generation",
    "generation_id",
    "submitted_at",
    "batch_id",
)
LAUNCH_TSV_COLUMNS = (
    "batch_id",
    "submitted_at",
    "status",
    "job_id",
    "kind",
    "generation",
    "generation_id",
    "relative_path",
    "directory",
    "launcher",
    "detail",
)

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_ARCHIVE_STAMP = re.compile(r"_(\d{8}T\d{6}Z)$")
_MAX_CHAIN_DEPTH = 32
_MAX_LEDGER_ANCESTORS = 8

# Ledger read-modify-write is serialised with an O_EXCL lock file next to the
# ledger (portable, works on NFS/Lustre where flock may be unavailable).  The
# lock is held for milliseconds, so one older than the stale limit belongs to a
# dead process and is broken; the wait outlasts the stale limit so a stale lock
# never makes a post-sbatch append fail.
_LOCK_STALE_SECONDS = 120.0
_LOCK_TIMEOUT_SECONDS = 180.0
_LOCK_POLL_SECONDS = 0.05
_REPLACE_ATTEMPTS = 8

UNREADABLE_LEDGER_JOB_ID = "unknown"


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def _float(value: Any, default: float | None = None) -> float | None:
    """``aimd._first_float`` restricted to finite values.

    ``json.loads`` accepts ``NaN``/``Infinity``; a hand-edited record carrying
    them must read as "unknown", never crash a read-only status path.
    """

    number = _first_float(value)
    if number is None or not math.isfinite(number):
        return default
    return number


def _int(value: Any, default: int | None = None) -> int | None:
    """``aimd._first_int`` restricted to finite values (see ``_float``)."""

    number = _float(value)
    return int(number) if number is not None else default


# --------------------------------------------------------------------------- #
# Time + JSON I/O
# --------------------------------------------------------------------------- #


def utc_now_iso() -> str:
    """Current UTC time, ISO-8601 to the second (``2026-09-23T10:15:00+00:00``)."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_stamp() -> str:
    """Current UTC time as a compact archive/batch stamp (``20260923T101500Z``)."""

    return datetime.now(timezone.utc).strftime(_STAMP_FORMAT)


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(microsecond=0).isoformat()


def _stamp_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(_STAMP_FORMAT)


def _epoch_from_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):  # fromisoformat only learned "Z" in Python 3.11
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _epoch_from_stamp(stamp: str) -> float | None:
    try:
        return datetime.strptime(stamp, _STAMP_FORMAT).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def read_json(path: str | Path) -> dict[str, Any]:
    """Parsed JSON object at ``path``; ``{}`` when missing, unreadable, invalid or not an object."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _replace(source: Path, target: Path) -> None:
    """``os.replace``, retried briefly on Windows where a concurrently open target refuses it."""

    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_LOCK_POLL_SECONDS * (attempt + 1))


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` via a unique ``<name>.<token>.tmp`` in the same directory, then ``os.replace``.

    A reader never sees a half-written file; on failure the ``.tmp`` is removed
    and the previous content is left untouched.  The temporary name is unique
    per call (created with ``O_EXCL``, honouring the umask like a plain open),
    so two concurrent writers can never truncate or interleave each other's
    temporary file.
    """

    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """``atomic_write_text`` for raw bytes.

    Replacing the directory entry also means a hard-linked or symlinked target
    is never written through: the file outside the run keeps its content.
    """

    target = Path(path)
    temporary = target.with_name(f"{target.name}.{os.getpid()}-{uuid.uuid4().hex[:12]}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomic JSON write (``indent=2``, ``sort_keys=True``, trailing newline)."""

    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


@contextmanager
def _file_lock(target: Path) -> Iterator[Path]:
    """Hold ``<target>.lock`` (created with ``O_EXCL``) around a read-modify-write of ``target``.

    Serialises concurrent InterfaceForge writers of one ledger (e.g. recovery
    running as a Slurm array job).  A lock older than ``_LOCK_STALE_SECONDS``
    is left over from a dead process and is removed; waiting longer than
    ``_LOCK_TIMEOUT_SECONDS`` raises ``SafetyError``.
    """

    lock = target.with_name(target.name + ".lock")
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        try:
            descriptor = os.open(lock, flags, 0o666)
            break
        except FileExistsError:
            pass
        except PermissionError:
            # Windows reports a lock file that is being deleted as access denied.
            if os.name != "nt":
                raise
        modified = _mtime(lock)
        if modified is not None and time.time() - modified > _LOCK_STALE_SECONDS:
            try:
                lock.unlink()
            except OSError:
                pass
            continue
        if time.monotonic() >= deadline:
            raise SafetyError(
                f"Timed out after {_LOCK_TIMEOUT_SECONDS:g} s waiting for {lock}: another InterfaceForge "
                f"process is writing {target.name}. Remove the lock only if no such process is running."
            )
        time.sleep(_LOCK_POLL_SECONDS)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid {os.getpid()} at {utc_now_iso()}\n")
        yield lock
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _real(path: str | Path) -> str:
    return os.path.realpath(str(path))


def _path_key(path: str | Path) -> str:
    """Comparison key for directories: realpath, case-folded where the OS is case-insensitive."""

    return os.path.normcase(_real(path))


# --------------------------------------------------------------------------- #
# Temperature schedule
# --------------------------------------------------------------------------- #


def schedule_temperature(tebeg: float, teend: float | None, nsw: int | None, step: int) -> float:
    """Thermostat target after ``step`` ionic steps of a TEBEG->TEEND segment.

    VASP ramps the target linearly in ionic step over NSW:
    ``tebeg + (teend - tebeg) * clamp(step, 0, nsw) / nsw``.  ``teend`` None
    means a constant-temperature segment (``tebeg``); a falsy or non-positive
    ``nsw`` has no ramp length, so the endpoint ``teend`` is returned.
    """

    start = float(tebeg)
    if teend is None:
        return start
    end = float(teend)
    if not nsw or nsw <= 0:
        return end
    fraction = min(max(int(step), 0), int(nsw)) / int(nsw)
    return start + (end - start) * fraction


def _thermostat(incar: dict[str, str]) -> str:
    if _int(incar.get("MDALGO")) == 3:
        return "langevin (MDALGO=3)"
    smass = _float(incar.get("SMASS"))
    if smass is None:
        return "unknown"
    if smass == -1.0:
        return "velocity-rescale (SMASS=-1)"
    if smass >= 0.0:
        return f"nose (SMASS={smass:g})"
    return "unknown"


def incar_schedule(incar: dict[str, str]) -> dict[str, Any]:
    """Temperature/timestep schedule of one segment from its parsed INCAR.

    ``tebeg_k`` defaults to 300 K when TEBEG is absent (the same default the
    Step1 diagnostics use); ``teend_k`` defaults to TEBEG when TEEND is absent
    (``teend_explicit`` records which).  ``potim_fs``/``nsw`` are None when the
    tag is missing; ``nblock`` defaults to 1.
    """

    tebeg = _float(incar.get("TEBEG"), 300.0)
    tebeg = 300.0 if tebeg is None else tebeg
    teend_value = _float(incar.get("TEEND"))
    teend = tebeg if teend_value is None else teend_value
    nblock = _int(incar.get("NBLOCK"), 1) or 1
    return {
        "tebeg_k": tebeg,
        "teend_k": teend,
        "teend_explicit": teend_value is not None,
        "nsw": _int(incar.get("NSW")),
        "potim_fs": _float(incar.get("POTIM")),
        "nblock": max(1, nblock),
        "thermostat": _thermostat(incar),
        "ramp": tebeg != teend,
    }


def format_temperature(value: float) -> str:
    """INCAR-ready temperature text: rounded to 0.01 K, no trailing zeros (``116.5``, ``300``)."""

    return f"{round(float(value), 2):g}"


# --------------------------------------------------------------------------- #
# Generations (current segment)
# --------------------------------------------------------------------------- #


@dataclass
class Generation:
    """The current segment of a Step1 run and the accepted history before it."""

    generation: int
    generation_id: str
    kind: str  # "original" | "repair" | "resume"
    legacy: bool  # True for gen 0 without record, or a schema-1 repair record
    record_path: Path | None
    record: dict[str, Any]
    prepared_at: str | None  # ISO; None for gen 0
    prepared_epoch: float | None  # POSIX seconds used for legacy launch matching; None for gen 0
    original_nsw: int | None  # whole-run target ionic steps
    accepted_prefix_steps: int  # cumulative accepted BEFORE the current segment
    segment_nsw: int | None  # NSW of the current segment (record value, else INCAR NSW)
    segment_potim_fs: float | None  # POTIM of the current segment (record, else INCAR)
    ledger: list[dict[str, Any]] = field(default_factory=list)  # accepted history BEFORE current segment
    ledger_exact: bool = True
    status: str | None = None  # record "status"; None for gen 0
    submissions: list[dict[str, Any]] = field(default_factory=list)
    # Set when both step1_repair.json and step1_resume.json exist and disagree
    # (mixed versions, time order contradicting generation order, unreadable
    # loser); ``status`` is then "CONFLICT" (or "UNREADABLE") so nothing launches.
    conflict: str | None = None


@dataclass
class _ChainLink:
    record: dict[str, Any]
    record_path: Path
    archive: Path | None  # resolved archive dir the record's repair created


def new_generation_id(kind: str, generation: int, stamp: str | None = None) -> str:
    """Identifier of a newly created generation: ``g{generation}-{kind}-{stamp}``."""

    return f"g{generation}-{kind}-{stamp or utc_stamp()}"


def _segment_ps(steps: Any, potim_fs: Any) -> float | None:
    if steps is None or potim_fs is None:
        return None
    try:
        return round(float(steps) * float(potim_fs) / 1000.0, 9)
    except (TypeError, ValueError):
        return None


def accepted_ps(ledger: list[dict[str, Any]]) -> float | None:
    """Accepted simulated time in ps (sum of steps * POTIM / 1000); None if any POTIM is unknown."""

    total = 0.0
    for row in ledger:
        value = _segment_ps(_int(row.get("steps")), _float(row.get("potim_fs")))
        if value is None:
            return None
        total += value
    return round(total, 9)


def _first_present_int(record: dict[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        value = _int(record.get(key))
        if value is not None:
            return value
    return None


def _first_present_float(record: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _float(record.get(key))
        if value is not None:
            return value
    return None


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _generation_zero(incar: dict[str, str]) -> Generation:
    schedule = incar_schedule(incar)
    return Generation(
        generation=0,
        generation_id=GEN0_ID,
        kind="original",
        legacy=True,
        record_path=None,
        record={},
        prepared_at=None,
        prepared_epoch=None,
        original_nsw=schedule["nsw"],
        accepted_prefix_steps=0,
        segment_nsw=schedule["nsw"],
        segment_potim_fs=schedule["potim_fs"],
        ledger=[],
        ledger_exact=True,
        status=None,
        submissions=[],
    )


def _unreadable_generation(path: Path, kind: str, incar: dict[str, str]) -> Generation:
    """A record file exists but cannot be parsed: expose it, never pretend it is gen 0.

    ``status="UNREADABLE"`` keeps launch from treating the run as launchable
    and makes status/recover route it to review.
    """

    schedule = incar_schedule(incar)
    mtime = _mtime(path)
    stamp = _stamp_from_epoch(mtime) if mtime is not None else utc_stamp()
    return Generation(
        generation=1,
        generation_id=f"unreadable-{kind}-{stamp}",
        kind=kind,
        legacy=True,
        record_path=path,
        record={},
        prepared_at=_iso_from_epoch(mtime) if mtime is not None else None,
        prepared_epoch=mtime,
        original_nsw=None,
        accepted_prefix_steps=0,
        segment_nsw=schedule["nsw"],
        segment_potim_fs=schedule["potim_fs"],
        ledger=[],
        ledger_exact=False,
        status="UNREADABLE",
        submissions=[],
    )


def _new_record_generation(
    path: Path, kind: str, record: dict[str, Any], incar: dict[str, str], *, legacy: bool = False
) -> Generation:
    schedule = incar_schedule(incar)
    generation = _int(record.get("generation"))
    generation = generation if generation is not None and generation >= 1 else 1
    prefix = _first_present_int(record, ("accepted_prefix_steps", "safe_prefix_steps")) or 0
    segment_nsw = _first_present_int(record, ("segment_nsw", "repair_nsw", "resume_nsw"))
    if segment_nsw is None:
        segment_nsw = schedule["nsw"]
    segment_potim = _first_present_float(record, ("segment_potim_fs", "repair_potim_fs"))
    if segment_potim is None:
        segment_potim = schedule["potim_fs"]
    original_nsw = _int(record.get("original_nsw"))
    if original_nsw is None and segment_nsw is not None:
        original_nsw = prefix + segment_nsw
    has_id = bool(record.get("generation_id"))
    prepared_at = record.get("prepared_at") if isinstance(record.get("prepared_at"), str) else None
    prepared_epoch = _epoch_from_iso(prepared_at)
    if prepared_epoch is None and not has_id:
        # mark_record_submitted pins a derived identity; the rewrite moved the mtime.
        pinned_at = record.get("legacy_prepared_at")
        prepared_epoch = _epoch_from_iso(pinned_at)
        prepared_at = pinned_at if prepared_epoch is not None else None
    if prepared_epoch is None:
        # Whole seconds, so the value survives being pinned as ISO seconds.
        mtime = _mtime(path)
        prepared_epoch = float(int(mtime)) if mtime is not None else None
        prepared_at = _iso_from_epoch(prepared_epoch) if prepared_epoch is not None else None
    generation_id = record.get("generation_id")
    if not generation_id:
        pinned_id = record.get("legacy_generation_id")
        if isinstance(pinned_id, str) and pinned_id:
            generation_id = pinned_id
        else:
            stamp = _stamp_from_epoch(prepared_epoch) if prepared_epoch is not None else utc_stamp()
            generation_id = f"legacy-{kind}-g{generation}-{stamp}"
    status = record.get("status")
    return Generation(
        generation=generation,
        generation_id=str(generation_id),
        kind=str(record.get("segment_kind") or kind),
        legacy=legacy,
        record_path=path,
        record=record,
        prepared_at=prepared_at,
        prepared_epoch=prepared_epoch,
        original_nsw=original_nsw,
        accepted_prefix_steps=prefix,
        segment_nsw=segment_nsw,
        segment_potim_fs=segment_potim,
        ledger=_dict_list(record.get("accepted_segments")),
        # A new record always states it; a missing flag is treated as inexact.
        ledger_exact=bool(record.get("ledger_exact", False)),
        status=str(status) if status is not None else None,
        submissions=_dict_list(record.get("submissions")),
    )


def _resolve_archive(run: Path, value: Any) -> Path | None:
    """Archive dir named by a record; falls back to ``<run>/.interfaceforge/archive/<name>``.

    The fallback finds the archive after the tree was moved or copied (the
    record stores the absolute path of the machine that wrote it).
    """

    if not value or not isinstance(value, str):
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = run / candidate
    if candidate.is_dir():
        return candidate
    name = re.split(r"[\\/]", value.rstrip("\\/"))[-1]
    if name:
        local = run / ".interfaceforge" / "archive" / name
        if local.is_dir():
            return local
    return None


def _archive_stamp(value: Any) -> str | None:
    if not value or not isinstance(value, str):
        return None
    name = re.split(r"[\\/]", value.rstrip("\\/"))[-1]
    match = _ARCHIVE_STAMP.search(name)
    return match.group(1) if match else None


def _legacy_stamp(record: dict[str, Any], record_path: Path) -> str:
    stamp = _archive_stamp(record.get("archive"))
    if stamp:
        return stamp
    mtime = _mtime(record_path)
    return _stamp_from_epoch(mtime) if mtime is not None else utc_stamp()


def _legacy_generation_id(generation: int, record: dict[str, Any], record_path: Path) -> str:
    pinned = record.get("legacy_generation_id")
    if isinstance(pinned, str) and pinned:
        return pinned
    return f"legacy-repair-g{generation}-{_legacy_stamp(record, record_path)}"


def _legacy_prepared_epoch(record: dict[str, Any], record_path: Path) -> float | None:
    pinned = _epoch_from_iso(record.get("legacy_prepared_at"))
    if pinned is not None:
        return pinned
    stamp = _archive_stamp(record.get("archive"))
    epoch = _epoch_from_stamp(stamp) if stamp else None
    if epoch is not None:
        return epoch
    # Whole seconds, like an archive stamp, so the value is unchanged once
    # mark_record_submitted pins it as legacy_prepared_at (ISO seconds).
    mtime = _mtime(record_path)
    return float(int(mtime)) if mtime is not None else None


def _is_first_repair(link: _ChainLink) -> bool:
    """True when the legacy record ``link`` is R_1 (no earlier repair to follow).

    A record without ``previous_safe_prefix_steps`` predates the cumulative
    fix and is generation 1.  A positive previous prefix always has a
    predecessor.  A previous prefix of 0 is ambiguous: the first repair, or a
    repair of a repair that had rewound to step 0 (early first bad step,
    SCF-only failure).  archive_run copies the top-level ``step1_repair.json``,
    so the predecessor exists exactly when the record's archive holds one.
    """

    record = link.record
    if "previous_safe_prefix_steps" not in record:
        return True
    if (_int(record.get("previous_safe_prefix_steps"), 0) or 0) > 0:
        return False
    return link.archive is None or not (link.archive / REPAIR_RECORD).is_file()


def _walk_legacy_chain(
    run: Path, path: Path, record: dict[str, Any]
) -> tuple[list[_ChainLink], _ChainLink | None, bool, str]:
    """Follow ``record["archive"]`` back through earlier repairs.

    Returns ``(links newest-first, new-format base record or None, complete, note)``.
    Each repair's archive holds the *previous* ``step1_repair.json`` (when the
    run had been repaired before) and the INCAR of the segment it rewound.
    The walk stops at R_1, or (broken) at a missing/unreadable record, a cycle
    or ``_MAX_CHAIN_DEPTH``.
    """

    links = [_ChainLink(record, path, _resolve_archive(run, record.get("archive")))]
    seen: set[str] = set()
    while True:
        current = links[-1]
        if _is_first_repair(current):
            return links, None, True, ""
        if len(links) >= _MAX_CHAIN_DEPTH:
            return links, None, False, f"archive chain deeper than {_MAX_CHAIN_DEPTH} records"
        archive = current.archive
        if archive is None:
            return links, None, False, f"archive {current.record.get('archive')!r} not found"
        key = _path_key(archive)
        if key in seen:
            return links, None, False, f"archive chain cycles at {archive}"
        seen.add(key)
        previous_path = archive / REPAIR_RECORD
        previous = read_json(previous_path)
        if not previous:
            return links, None, False, f"{previous_path} missing or unreadable"
        if previous.get("generation_id"):
            # An older InterfaceForge repaired a run whose previous record was
            # already generation-aware: that record carries its own ledger.
            return links, _ChainLink(previous, previous_path, None), True, ""
        links.append(_ChainLink(previous, previous_path, _resolve_archive(run, previous.get("archive"))))


def _segment_steps(record: dict[str, Any], previous_prefix: int) -> int:
    explicit = _int(record.get("safe_segment_steps"))
    if explicit is not None:
        return explicit
    return max(0, (_int(record.get("safe_prefix_steps"), 0) or 0) - previous_prefix)


def _restart_source(record: dict[str, Any]) -> str | None:
    source = record.get("source")
    frame = record.get("rewind_frame")
    if source and frame is not None:
        return f"{source} frame {frame}"
    return str(source) if source else None


def _legacy_ledger_row(
    link: _ChainLink, *, generation: int | None, generation_id: str | None, kind: str, steps: int
) -> dict[str, Any]:
    """Ledger row for the segment the legacy record ``link`` rewound and partly accepted.

    Legacy records read ``original_potim_fs`` from the INCAR being rewound, so
    it IS the POTIM of the accepted steps.  ``teend_k`` is the schedule
    temperature reached at the last accepted step (from the archived INCAR).
    """

    record = link.record
    archived_incar = parse_incar(link.archive / "INCAR") if link.archive is not None else {}
    potim = _float(record.get("original_potim_fs"))
    if potim is None and archived_incar:
        potim = _float(archived_incar.get("POTIM"))
    row: dict[str, Any] = {
        "generation": generation,
        "generation_id": generation_id,
        "kind": kind,
        "steps": steps,
        "potim_fs": potim,
        "ps": _segment_ps(steps, potim),
        "restart_source": _restart_source(record),
        "legacy": True,
    }
    if archived_incar.get("TEBEG") is not None:
        schedule = incar_schedule(archived_incar)
        row["tebeg_k"] = schedule["tebeg_k"]
        row["teend_k"] = round(
            schedule_temperature(schedule["tebeg_k"], schedule["teend_k"], schedule["nsw"], steps), 2
        )
    return row


def _legacy_repair_generation(run: Path, path: Path, record: dict[str, Any], incar: dict[str, str]) -> Generation:
    schedule = incar_schedule(incar)
    links, base, complete, note = _walk_legacy_chain(run, path, record)
    chronological = list(reversed(links))
    prefix = _int(record.get("safe_prefix_steps"), 0) or 0
    ledger: list[dict[str, Any]] = []

    if complete:
        exact = True
        if base is None:
            base_generation, previous_gid, previous_kind, previous_prefix = 0, GEN0_ID, "original", 0
        else:
            base_record = base.record
            base_generation = max(1, _int(base_record.get("generation"), 1) or 1)
            previous_gid = str(base_record.get("generation_id"))
            previous_kind = str(base_record.get("segment_kind") or "repair")
            previous_prefix = _first_present_int(base_record, ("accepted_prefix_steps", "safe_prefix_steps")) or 0
            ledger.extend(_dict_list(base_record.get("accepted_segments")))
            exact = bool(base_record.get("ledger_exact", False))
        generation = base_generation + len(links)
        previous_generation = base_generation
        for index, link in enumerate(chronological):
            current = link.record
            if index == 0 and base is None:
                steps = _int(current.get("safe_prefix_steps"), 0) or 0
            else:
                steps = _segment_steps(current, previous_prefix)
                stated = _int(current.get("previous_safe_prefix_steps"))
                if stated is not None and stated != previous_prefix:
                    exact = False  # the chain disagrees with itself; keep it but flag it
            ledger.append(
                _legacy_ledger_row(
                    link, generation=previous_generation, generation_id=previous_gid, kind=previous_kind, steps=steps
                )
            )
            this_generation = base_generation + index + 1
            previous_gid = _legacy_generation_id(this_generation, current, link.record_path)
            previous_kind = "repair"
            previous_generation = this_generation
            previous_prefix = _int(current.get("safe_prefix_steps"), previous_prefix + steps) or 0
        if sum(_int(row.get("steps"), 0) or 0 for row in ledger) != prefix:
            exact = False
    else:
        # Broken chain: an earlier prefix is known only as a step count.  At
        # least one earlier generation produced it, so numbering is a lower bound.
        exact = False
        oldest = chronological[0].record
        previous_prefix = _int(oldest.get("previous_safe_prefix_steps"), 0) or 0
        ledger.append(
            {
                "generation": None,
                "generation_id": None,
                "kind": "unknown",
                "steps": previous_prefix,
                "potim_fs": None,
                "ps": None,
                "legacy": True,
                "note": f"legacy archive chain broken: {note}",
            }
        )
        generation = len(links) + 1
        # The oldest walked record rewound a repair segment whose record is lost
        # (generation >= 1, identity unknown).
        segment_generation = 1
        segment_gid: str | None = None
        for index, link in enumerate(chronological):
            current = link.record
            steps = _segment_steps(current, previous_prefix)
            ledger.append(
                _legacy_ledger_row(
                    link, generation=segment_generation, generation_id=segment_gid, kind="repair", steps=steps
                )
            )
            segment_generation = index + 2
            segment_gid = _legacy_generation_id(segment_generation, current, link.record_path)
            previous_prefix = _int(current.get("safe_prefix_steps"), previous_prefix + steps) or 0

    prepared_epoch = _legacy_prepared_epoch(record, path)
    segment_nsw = _int(record.get("repair_nsw"))
    if segment_nsw is None:
        segment_nsw = schedule["nsw"]
    segment_potim = _float(record.get("repair_potim_fs"))
    if segment_potim is None:
        segment_potim = schedule["potim_fs"]
    original_nsw = _int(record.get("original_nsw"))
    if original_nsw is None and segment_nsw is not None:
        original_nsw = prefix + segment_nsw
    status = record.get("status")
    return Generation(
        generation=generation,
        generation_id=_legacy_generation_id(generation, record, path),
        kind="repair",
        legacy=True,
        record_path=path,
        record=record,
        prepared_at=_iso_from_epoch(prepared_epoch) if prepared_epoch is not None else None,
        prepared_epoch=prepared_epoch,
        original_nsw=original_nsw,
        accepted_prefix_steps=prefix,
        segment_nsw=segment_nsw,
        segment_potim_fs=segment_potim,
        ledger=ledger,
        ledger_exact=exact,
        status=str(status) if status is not None else None,
        submissions=_dict_list(record.get("submissions")),
    )


def _record_generation(run: Path, path: Path, kind: str, incar: dict[str, str]) -> Generation:
    record = read_json(path)
    if not record:
        return _unreadable_generation(path, kind, incar)
    if record.get("generation_id"):
        return _new_record_generation(path, kind, record, incar)
    if kind == "repair":
        return _legacy_repair_generation(run, path, record, incar)
    # A resume record without generation_id was never written by InterfaceForge;
    # read it with the new-record fallbacks but keep it legacy and inexact.
    generation = _new_record_generation(path, kind, record, incar, legacy=True)
    generation.ledger_exact = False
    return generation


def _records_conflict(candidates: list[Generation]) -> str | None:
    """Why two co-existing top-level records cannot be ordered safely, else None.

    Normal operation never leaves both (the previous record is retired before
    the new one is written), so both present means a hand edit or an older
    InterfaceForge acting on a newer run -- e.g. an old ``step1-repair`` writing
    a schema-1 ``step1_repair.json`` next to a stale ``step1_resume.json``,
    where the larger generation number is NOT the current segment.
    """

    if len(candidates) < 2:
        return None
    names = " and ".join(item.record_path.name for item in candidates if item.record_path is not None)
    unreadable = [item.record_path.name for item in candidates if item.status == "UNREADABLE" and item.record_path]
    if unreadable:
        return f"both {names} exist and {', '.join(unreadable)} is unreadable"
    if len({item.legacy for item in candidates}) > 1:
        return f"both {names} exist and only one is generation-aware (written by different InterfaceForge versions?)"
    first, second = candidates
    if first.generation != second.generation and first.prepared_epoch is not None and second.prepared_epoch is not None:
        later, earlier = (first, second) if first.generation > second.generation else (second, first)
        if later.prepared_epoch < earlier.prepared_epoch:
            return (
                f"both {names} exist and generation {later.generation} ({later.generation_id}) was prepared "
                f"before generation {earlier.generation} ({earlier.generation_id})"
            )
    return None


def current_generation(run: str | Path, incar: dict[str, str] | None = None) -> Generation:
    """The current generation of ``run`` (gen 0, a new record, or a reconstructed legacy repair).

    If both ``step1_repair.json`` and ``step1_resume.json`` exist (hand edits),
    the one with the larger generation, then the newer ``prepared_at``, then
    the newer mtime wins.  When the two records contradict each other (see
    ``_records_conflict``) the winner is still returned for display, but with
    ``status="CONFLICT"`` (kept ``"UNREADABLE"`` if the winner is unreadable)
    and ``conflict`` set, so launch refuses it and status/recover route the run
    to review; the record's own status stays in ``record["status"]``.
    """

    folder = Path(run).expanduser().resolve()
    parsed = parse_incar(folder / "INCAR") if incar is None else incar
    candidates = [
        _record_generation(folder, folder / name, kind, parsed)
        for name, kind in ((REPAIR_RECORD, "repair"), (RESUME_RECORD, "resume"))
        if (folder / name).is_file()
    ]
    if not candidates:
        return _generation_zero(parsed)
    winner = max(
        candidates,
        key=lambda item: (item.generation, item.prepared_epoch or 0.0, _mtime(item.record_path) or 0.0),
    )
    conflict = _records_conflict(candidates)
    if conflict is not None:
        winner.conflict = conflict
        if winner.status != "UNREADABLE":
            winner.status = "CONFLICT"
    return winner


# --------------------------------------------------------------------------- #
# Segment records
# --------------------------------------------------------------------------- #


def build_segment_record(
    kind: str,
    *,
    run: Path,
    generation: int,
    generation_id: str,
    parent: Generation,
    prepared_at: str,
    original_nsw: int,
    accepted_prefix_steps: int,
    accepted_segments: list[dict[str, Any]],
    ledger_exact: bool,
    segment_nsw: int,
    segment_potim_fs: float,
    segment_schedule: dict[str, Any],
    archive: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """The top-level record of a new repair (schema 2) or resume (schema 1) generation.

    Common keys always win over ``extra`` (so a plan dict carrying e.g.
    ``status: READY`` or ``archive: None`` cannot leak into the record).  Repair
    records additionally keep every schema-1 key with its original meaning:
    ``safe_prefix_steps`` == ``accepted_prefix_steps``, ``repair_nsw`` ==
    ``segment_nsw``, ``repair_potim_fs`` == ``segment_potim_fs``, and
    ``previous_safe_prefix_steps`` / ``safe_segment_steps`` /
    ``original_potim_fs`` default to what the parent generation implies.
    """

    if kind not in ("repair", "resume"):
        raise ValueError(f"segment kind must be 'repair' or 'resume', not {kind!r}")
    segments = [dict(row) for row in accepted_segments]
    for row in segments:
        if "ps" not in row:
            row["ps"] = _segment_ps(_int(row.get("steps")), _float(row.get("potim_fs")))
    common: dict[str, Any] = {
        "format": REPAIR_FORMAT if kind == "repair" else RESUME_FORMAT,
        "schema_version": REPAIR_SCHEMA_VERSION if kind == "repair" else RESUME_SCHEMA_VERSION,
        "status": "PREPARED",
        "run": str(run),
        "generation": int(generation),
        "generation_id": generation_id,
        "parent_generation_id": parent.generation_id,
        "segment_kind": kind,
        "prepared_at": prepared_at,
        "original_nsw": original_nsw,
        "accepted_prefix_steps": int(accepted_prefix_steps),
        "accepted_segments": segments,
        "accepted_ps": accepted_ps(segments),
        "ledger_exact": bool(ledger_exact),
        "segment_nsw": segment_nsw,
        "segment_potim_fs": segment_potim_fs,
        "segment_schedule": {key: segment_schedule.get(key) for key in SCHEDULE_KEYS},
        "archive": str(archive),
        "submissions": [],
    }
    segment_steps = int(accepted_prefix_steps) - int(parent.accepted_prefix_steps)
    record: dict[str, Any] = dict(extra)
    if kind == "repair":
        record.setdefault("previous_safe_prefix_steps", parent.accepted_prefix_steps)
        record.setdefault("safe_segment_steps", segment_steps)
        record.setdefault("original_potim_fs", parent.segment_potim_fs)
        record["safe_prefix_steps"] = int(accepted_prefix_steps)
        record["repair_nsw"] = segment_nsw
        record["repair_potim_fs"] = segment_potim_fs
        for key in LEGACY_REPAIR_KEYS:
            record.setdefault(key, None)
    else:
        record.setdefault("segment_accepted_steps", segment_steps)
        record["resume_nsw"] = segment_nsw
        for key in RESUME_KEYS:
            record.setdefault(key, None)
    record.update(common)
    return record


def mark_record_submitted(run: str | Path, generation_id: str, submission: dict[str, Any]) -> None:
    """Record a submission on the current top-level record if it is ``generation_id``.

    Appends ``submission`` (``{job_id, submitted_at, batch_id, launcher,
    ledger}``, plus ``generation_id``) to ``submissions`` and sets status
    ``SUBMITTED`` (atomic write).  No-op for gen 0, an unreadable record, or a
    different generation.  A schema-1 record is marked in place without gaining
    a ``generation_id`` (which would change how it is read); its derived
    identity is pinned as ``legacy_generation_id`` / ``legacy_prepared_at``.
    """

    folder = Path(run).expanduser().resolve()
    generation = current_generation(folder)
    if generation.record_path is None or generation.generation_id != generation_id:
        return
    record = read_json(generation.record_path)
    if not record:
        return
    entry = dict(submission)
    entry.setdefault("generation_id", generation_id)
    entry.setdefault("submitted_at", utc_now_iso())
    if not record.get("generation_id"):
        record.setdefault("legacy_generation_id", generation.generation_id)
        if generation.prepared_at is not None:
            record.setdefault("legacy_prepared_at", generation.prepared_at)
    record["submissions"] = [*_dict_list(record.get("submissions")), entry]
    record["status"] = "SUBMITTED"
    atomic_write_json(generation.record_path, record)


def retire_current_records(run: str | Path, archive: Path) -> list[str]:
    """Delete the top-level segment records once ``archive`` holds identical copies.

    Every present record is checked before anything is deleted: a missing or
    differing archived copy raises ``SafetyError`` and nothing is removed.
    """

    folder = Path(run).expanduser().resolve()
    archive_dir = Path(archive)
    present = [name for name in (REPAIR_RECORD, RESUME_RECORD) if (folder / name).is_file()]
    for name in present:
        copy = archive_dir / name
        if not copy.is_file():
            raise SafetyError(f"Refusing to retire {folder / name}: no archived copy in {archive_dir}")
        if copy.read_bytes() != (folder / name).read_bytes():
            raise SafetyError(f"Refusing to retire {folder / name}: it changed after it was archived to {archive_dir}")
    for name in present:
        (folder / name).unlink()
    return present


# --------------------------------------------------------------------------- #
# Archive + fingerprint
# --------------------------------------------------------------------------- #


def archive_step1_state(run: str | Path, operation: str) -> Path:
    """Archive a Step1 run before mutating it; returns the archive dir.

    ``archive_run`` copies the standard run state (incl. ``step1_repair.json``,
    launchers, Slurm logs); then the Step1 extras and ``precondition/`` outputs
    are copied and ``ARCHIVE_MANIFEST.json`` is written with status
    ``IN_PROGRESS``.  The caller calls ``finalize_archive`` once its mutation
    finished, so an interrupted mutation stays discoverable via
    ``interrupted_archive``.  Regenerable WAVECAR/CHG/CHGCAR are not copied.
    """

    folder = Path(run).expanduser().resolve()
    try:
        archive = archive_run(folder, operation)
    except FileExistsError:
        # archive_run stamps to the second; a second archive of the same
        # operation within that second collides.  Wait for the next stamp once.
        time.sleep(1.1)
        archive = archive_run(folder, operation)
    for name in STEP1_EXTRA_ARCHIVE_FILES:
        source = folder / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    precondition = folder / "precondition"
    if precondition.is_dir():
        for name in PRECONDITION_ARCHIVE_FILES:
            source = precondition / name
            if source.is_file():
                (archive / "precondition").mkdir(exist_ok=True)
                shutil.copy2(source, archive / "precondition" / name)
    files = [
        {"name": path.relative_to(archive).as_posix(), "bytes": path.stat().st_size}
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name != ARCHIVE_MANIFEST
    ]
    manifest = {
        "format": ARCHIVE_FORMAT,
        "schema_version": 1,
        "operation": operation,
        "run": str(folder),
        "created_at": utc_now_iso(),
        "status": "IN_PROGRESS",
        "files": files,
        "not_archived": [name for name in NOT_ARCHIVED if (folder / name).exists()],
        "generation_id": None,
    }
    atomic_write_json(archive / ARCHIVE_MANIFEST, manifest)
    return archive


def finalize_archive(archive: Path, *, generation_id: str) -> None:
    """Mark an archive's mutation complete (status ``COMPLETE``, ``generation_id`` set)."""

    path = Path(archive) / ARCHIVE_MANIFEST
    manifest = read_json(path)
    if not manifest:
        raise SafetyError(f"{path} is missing or unreadable; it was not written by archive_step1_state")
    manifest["status"] = "COMPLETE"
    manifest["generation_id"] = generation_id
    manifest["finalized_at"] = utc_now_iso()
    atomic_write_json(path, manifest)


def abandon_archive(archive: Path, *, reason: str) -> None:
    """Mark an archive ``ABANDONED``: it was taken but the run was left untouched.

    Used when the re-check after archiving refuses the mutation, so the copy is
    kept for reference without reporting an interrupted mutation.
    """

    path = Path(archive) / ARCHIVE_MANIFEST
    manifest = read_json(path)
    if not manifest:
        return
    manifest["status"] = "ABANDONED"
    manifest["abandoned_reason"] = reason
    manifest["finalized_at"] = utc_now_iso()
    atomic_write_json(path, manifest)


def recheck_after_archive(run: Path, archive: Path, *, guard: Any, fingerprint: Any) -> None:
    """Re-run the scheduler and fingerprint checks after a (possibly slow) archive copy.

    Called after ``archive_step1_state`` and before the first write to the run:
    a job that started, or a file that changed, while large outputs were being
    copied refuses the mutation.  On refusal the archive is marked
    ``ABANDONED`` (the run is untouched) and ``SafetyError`` is raised.
    """

    try:
        guard.assert_inactive(run)
        if run_fingerprint(run) != fingerprint:
            raise SafetyError(
                f"Refusing to mutate {run}: it changed while it was being archived (file fingerprint differs); "
                "plan again"
            )
    except SafetyError as exc:
        abandon_archive(archive, reason=str(exc))
        raise


# --------------------------------------------------------------------------- #
# Per-run exclusive lock (recovery mutations and submissions)
# --------------------------------------------------------------------------- #

RUN_LOCK = "step1.lock"


def run_lock_path(run: str | Path) -> Path:
    return Path(run).expanduser() / ".interfaceforge" / RUN_LOCK


@contextmanager
def run_lock(run: str | Path, operation: str) -> Iterator[Path]:
    """Hold ``<run>/.interfaceforge/step1.lock`` (``O_EXCL``) for one recovery action on ``run``.

    Covers the whole check -> archive -> mutate -> record sequence of
    step1-repair / step1-resume and the re-check -> sbatch -> ledger sequence
    of step1-launch, so two concurrent InterfaceForge processes can never both
    act on one run.  It does not wait: a held lock raises ``SafetyError``
    naming the holder.  It is never broken automatically (a mutation may copy
    GB of outputs); a lock left by a killed process must be removed by hand.
    """

    lock = run_lock_path(run)
    created_parent = not lock.parent.exists()
    lock.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(lock, flags, 0o666)
    except (FileExistsError, PermissionError) as exc:
        if isinstance(exc, PermissionError) and os.name != "nt":
            raise
        try:
            holder = lock.read_text(encoding="utf-8", errors="replace").strip() or "unknown holder"
        except OSError:
            holder = "unknown holder"
        raise SafetyError(
            f"Refusing to {operation} {Path(run)}: another InterfaceForge recovery command holds {lock} "
            f"({holder}). Remove the lock only if no such process is running."
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{operation} pid {os.getpid()} on {socket.gethostname()} at {utc_now_iso()}\n")
        yield lock
    finally:
        try:
            lock.unlink()
            if created_parent:
                lock.parent.rmdir()  # leave no trace in a run that had no .interfaceforge/
        except OSError:
            pass


# Files whose modification marks a run as recently active (status, resume and repair share it).
ACTIVITY_FILES = ("OSZICAR", "OUTCAR", "CONTCAR", "XDATCAR")


def newest_activity(folder: str | Path) -> tuple[str | None, float | None]:
    """``(file name, age in hours)`` of the most recently modified ``ACTIVITY_FILES`` entry."""

    newest: tuple[float, str] | None = None
    for name in ACTIVITY_FILES:
        moment = _mtime(Path(folder) / name)
        if moment is not None and (newest is None or moment > newest[0]):
            newest = (moment, name)
    if newest is None:
        return None, None
    return newest[1], (time.time() - newest[0]) / 3600.0


def interrupted_archive(run: str | Path) -> Path | None:
    """Newest archive of ``run`` whose manifest is still ``IN_PROGRESS``, else None."""

    root = Path(run).expanduser().resolve() / ".interfaceforge" / "archive"
    if not root.is_dir():
        return None
    pending: list[tuple[float, float, str, Path]] = []
    for manifest_path in root.glob(f"*/{ARCHIVE_MANIFEST}"):
        manifest = read_json(manifest_path)
        if manifest.get("status") != "IN_PROGRESS":
            continue
        created = _epoch_from_iso(manifest.get("created_at")) or 0.0
        pending.append((created, _mtime(manifest_path) or 0.0, manifest_path.parent.name, manifest_path.parent))
    if not pending:
        return None
    return max(pending, key=lambda item: item[:3])[3]


def run_fingerprint(run: str | Path) -> dict[str, list[int] | None]:
    """``{name: [size, mtime_ns] or None}`` for the files whose change invalidates a plan."""

    folder = Path(run).expanduser()
    fingerprint: dict[str, list[int] | None] = {}
    for name in FINGERPRINT_FILES:
        try:
            stat = (folder / name).stat()
        except OSError:
            fingerprint[name] = None
            continue
        fingerprint[name] = [int(stat.st_size), int(stat.st_mtime_ns)]
    return fingerprint


# --------------------------------------------------------------------------- #
# Launch ledgers (generation-aware)
# --------------------------------------------------------------------------- #


def ledger_paths_for(run: str | Path, roots: Iterable[Path] = ()) -> list[Path]:
    """Existing ``step1_launch.json`` files that may hold rows for ``run``.

    Looks in the run itself, each given root, and every ancestor of the run up
    to and including the nearest one containing ``step1_manifest.json``
    (at most 8 levels).  De-duplicated by realpath, order stable.
    """

    folder = Path(run).expanduser().resolve()
    candidates: list[Path] = [folder / LAUNCH_LEDGER]
    candidates.extend(Path(root).expanduser() / LAUNCH_LEDGER for root in roots)
    current = folder
    for _ in range(_MAX_LEDGER_ANCESTORS):
        parent = current.parent
        if parent == current:
            break
        current = parent
        candidates.append(current / LAUNCH_LEDGER)
        if (current / MANIFEST).is_file():
            break
    found: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        key = _path_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        found.append(Path(_real(candidate)))
    return found


def _row_directory_real(row: dict[str, Any], base: Path) -> str | None:
    directory = row.get("directory")
    if isinstance(directory, str) and directory:
        return _real(directory)
    relative = row.get("relative_path")
    if isinstance(relative, str) and relative:
        return _real(base / relative)
    return None


def _row_directory_keys(row: dict[str, Any], base: Path | None) -> set[str]:
    """Every directory a row may refer to: its absolute ``directory`` and ``<ledger dir>/<relative_path>``.

    Matching on either keeps a moved/copied tree (stale absolute paths) and a
    leaf launched as its own root (``relative_path == "."``) both visible.
    """

    keys: set[str] = set()
    directory = row.get("directory")
    if isinstance(directory, str) and directory:
        keys.add(_path_key(directory))
    relative = row.get("relative_path")
    if base is not None and isinstance(relative, str) and relative:
        keys.add(_path_key(base / relative))
    annotated = row.get("_directory_real")
    if isinstance(annotated, str) and annotated:
        keys.add(os.path.normcase(annotated))
    return keys


def _ledger_payload(path: Path) -> dict[str, Any] | None:
    """The parsed ledger at ``path``; None when the file exists but is not a usable ledger.

    Usable means a JSON object with a ``runs`` list.  Invalid/truncated JSON
    (e.g. the pre-generation ``step1-launch`` used a plain, non-atomic write),
    a non-object, an unreadable file, or an object without a ``runs`` list are
    all unusable.  A missing file is an empty ledger (``{}``).
    """

    if not path.is_file():
        return {}
    payload = read_json(path)
    if not payload or not isinstance(payload.get("runs"), list):
        return None
    return payload


def unreadable_ledgers(ledger_paths: Iterable[str | Path]) -> list[Path]:
    """Existing launch ledgers among ``ledger_paths`` that cannot be read as a ledger.

    Such a ledger may hold a SUBMITTED row for any run below its directory, so
    the duplicate-launch check cannot be evaluated: ``submission_state`` then
    reports the current generation as submitted (fail closed) and launch or
    recovery must refuse / route the run to review until the file is repaired
    or moved aside.  Read-only.
    """

    found: list[Path] = []
    seen: set[str] = set()
    for ledger_path in ledger_paths:
        path = Path(ledger_path)
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        if _ledger_payload(path) is None:
            found.append(path)
    return found


def _unreadable_ledger_row(path: Path) -> dict[str, Any]:
    return {
        "status": "UNREADABLE",
        "unreadable_ledger": True,
        "job_id": UNREADABLE_LEDGER_JOB_ID,
        "generation": None,
        "generation_id": None,
        "detail": f"{path} exists but is not a readable launch ledger; repair it or move it aside after inspection",
        "_ledger": str(path),
        "_directory_real": None,
    }


def load_ledger_rows(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Rows of a schema-1 or schema-2 launch ledger, annotated for matching.

    Schema-1 rows and rows lacking ``generation_id`` get ``legacy=True`` and
    ``legacy_recorded_at`` = file mtime (unless already set).  Every row gets
    ``_ledger`` (path) and ``_directory_real``.  A missing or unreadable ledger
    gives ``[]`` here; ``rows_for_run`` / ``unreadable_ledgers`` expose the
    unreadable case.
    """

    path = Path(ledger_path)
    payload = _ledger_payload(path)
    runs = payload.get("runs") if payload else None
    if not payload or not isinstance(runs, list):
        return []
    schema = _int(payload.get("schema_version"), 1) or 1
    mtime = _mtime(path)
    recorded = _iso_from_epoch(mtime) if mtime is not None else None
    rows: list[dict[str, Any]] = []
    for raw in runs:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if schema < 2 or not row.get("generation_id"):
            row["legacy"] = True
            if not row.get("legacy_recorded_at") and recorded is not None:
                row["legacy_recorded_at"] = recorded
        row["_ledger"] = str(path)
        row["_directory_real"] = _row_directory_real(row, path.parent)
        rows.append(row)
    return rows


def rows_for_run(run: str | Path, ledger_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every ledger row (from ``ledger_paths``) that refers to ``run``'s directory.

    An existing ledger that cannot be read contributes one placeholder row
    (``status: "UNREADABLE"``, ``unreadable_ledger: True``) instead of
    silently contributing nothing, so ``submission_state`` fails closed.
    """

    target = _path_key(Path(run).expanduser())
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ledger_path in ledger_paths:
        key = _path_key(ledger_path)
        if key in seen:  # the same ledger passed twice must not double its rows
            continue
        seen.add(key)
        path = Path(ledger_path)
        if _ledger_payload(path) is None:
            matched.append(_unreadable_ledger_row(path))
            continue
        matched.extend(row for row in load_ledger_rows(path) if target in _row_directory_keys(row, path.parent))
    return matched


def _row_time(row: dict[str, Any]) -> float | None:
    submitted = _epoch_from_iso(row.get("submitted_at"))
    return submitted if submitted is not None else _epoch_from_iso(row.get("legacy_recorded_at"))


def _row_current(row: dict[str, Any], generation: Generation) -> tuple[bool, str]:
    row_gid = row.get("generation_id")
    if row_gid:
        if str(row_gid) == generation.generation_id:
            return True, f"generation_id == {generation.generation_id}"
        return False, f"row generation_id {row_gid} != {generation.generation_id}"
    if not generation.legacy:
        return False, f"legacy row cannot belong to generation-aware {generation.generation_id}"
    if generation.generation == 0:
        return True, "legacy row counts for generation 0"
    row_time = _row_time(row)
    if row_time is None or generation.prepared_epoch is None:
        # Unknown ordering: count it, so a duplicate submission is never risked.
        return True, "legacy row with unknown time relative to the legacy record (conservatively current)"
    recorded = _iso_from_epoch(row_time)
    prepared = _iso_from_epoch(generation.prepared_epoch)
    if row_time >= generation.prepared_epoch:
        return True, f"legacy row recorded {recorded} >= legacy {generation.kind} prepared {prepared}"
    return False, f"legacy row recorded {recorded} < legacy {generation.kind} prepared {prepared}"


def submission_state(run: str | Path, generation: Generation, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Whether the *current* generation of ``run`` was already submitted.

    A ``SUBMITTED`` row is current iff it carries the current
    ``generation_id``, or it is a legacy row (no ``generation_id``) and the
    current generation is legacy and either generation 0 or prepared no later
    than the row was recorded.  A submission on the current record (record
    status ``SUBMITTED``) also counts.  Rows that demonstrably name another
    directory are ignored.

    Fail closed: if ``rows`` carry an unreadable-ledger placeholder (see
    ``rows_for_run``) and nothing else is current, the generation is reported
    as submitted with that placeholder as ``current_submission``; the ledger
    paths are listed in ``unreadable_ledgers`` either way.
    ``current_submissions`` lists every current submission (oldest first, one
    per job id) so a generation submitted twice is visible;
    ``current_submission`` is the newest.
    """

    target = _path_key(Path(run).expanduser())
    current_rows: list[tuple[float, int, dict[str, Any], str]] = []
    historical: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("unreadable_ledger"):
            if all(item.get("_ledger") != row.get("_ledger") for item in unreadable):
                unreadable.append(row)
            continue
        if row.get("status") != "SUBMITTED":
            continue
        ledger = row.get("_ledger")
        keys = _row_directory_keys(row, Path(ledger).parent if isinstance(ledger, str) and ledger else None)
        if keys and target not in keys:
            continue
        is_current, why = _row_current(row, generation)
        if not is_current:
            historical.append(row)
            continue
        row_time = _row_time(row)
        current_rows.append((row_time if row_time is not None else float("-inf"), index, row, why))

    # Oldest first (input order breaks ties); the same job in two ledgers is one submission.
    current_rows.sort(key=lambda item: (item[0], item[1]))
    current_submissions: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    current: dict[str, Any] | None = None
    rule = ""
    for _order, _index, row, why in current_rows:
        job = str(row.get("job_id") or "")
        if job and job in positions:
            current_submissions[positions[job]] = row
        else:
            if job:
                positions[job] = len(current_submissions)
            current_submissions.append(row)
        current, rule = row, why

    record_status = generation.record.get("status") if generation.record else None
    if current is None and "SUBMITTED" in (generation.status, record_status):
        for entry in reversed(generation.submissions):
            entry_gid = entry.get("generation_id")
            if not entry_gid or entry_gid == generation.generation_id:
                current = entry
                current_submissions = [entry]
                rule = f"{generation.record_path.name if generation.record_path else 'record'} submission"
                if entry.get("job_id"):
                    rule += f" (job {entry.get('job_id')})"
                break
    if current is None and unreadable:
        current = unreadable[0]
        current_submissions = [current]
        rule = (
            f"launch ledger {current.get('_ledger')} is unreadable; the current generation is treated as "
            "submitted until that file is repaired or moved aside"
        )
    if current is None:
        rule = f"no SUBMITTED row matches {generation.generation_id}"
        if historical:
            rule += f" ({len(historical)} historical)"
    elif len(current_submissions) > 1:
        rule += f"; {len(current_submissions)} submissions of this generation are recorded"
    return {
        "current_submitted": current is not None,
        "current_submission": current,
        "current_submissions": current_submissions,
        "historical_submissions": historical,
        "match_rule": rule,
        "unreadable_ledgers": [str(row.get("_ledger")) for row in unreadable],
    }


def _legacy_import(rows: Any, recorded: str | None) -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["legacy"] = True
        if not row.get("legacy_recorded_at") and recorded is not None:
            row["legacy_recorded_at"] = recorded
        imported.append(row)
    return imported


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    """De-duplication key: ``(directory, generation_id, job_id, status)``.

    A row without a job id (a FAILED attempt) also keys on ``batch_id`` and
    ``detail``: every failed batch is distinct provenance, and only a verbatim
    re-append of the same attempt is a duplicate.
    """

    key = tuple(str(row.get(name) or "") for name in ("directory", "generation_id", "job_id", "status"))
    if not key[2]:
        key += (str(row.get("batch_id") or ""), str(row.get("detail") or ""))
    return key


def _tsv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LAUNCH_TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in LAUNCH_TSV_COLUMNS})
    return buffer.getvalue()


def append_launch_rows(ledger_dir: str | Path, rows: list[dict[str, Any]], *, batch: dict[str, Any]) -> Path:
    """Append launch rows to ``<ledger_dir>/step1_launch.json`` (schema 2) and rewrite the TSV.

    A schema-1 ledger is upgraded by importing its rows as legacy rows (with
    ``legacy_recorded_at`` = its mtime, so their time survives the rewrite).
    An existing but unreadable ledger is first copied aside
    (``step1_launch.json.unreadable-<stamp>``), never silently overwritten.
    Exact duplicates by ``(directory, generation_id, job_id, status)`` are
    skipped (rows without a job id also key on ``batch_id`` + ``detail``, so
    every failed attempt is kept).  ``runs`` stays cumulative (oldest first) so
    an older InterfaceForge still sees every historical submission.  The batch
    status is ``FAILED`` if any of its rows failed, else ``SUBMITTED`` if any
    was submitted (``EMPTY`` for a batch without rows).  The read-modify-write
    holds ``step1_launch.json.lock`` so concurrent launches never lose rows.
    """

    directory = Path(ledger_dir).expanduser().resolve()
    json_path = directory / LAUNCH_LEDGER
    with _file_lock(json_path):
        return _append_launch_rows_locked(directory, json_path, rows, batch)


def _append_launch_rows_locked(
    directory: Path, json_path: Path, rows: list[dict[str, Any]], batch: dict[str, Any]
) -> Path:
    batch_id = str(batch.get("batch_id") or f"b-{utc_stamp()}")
    now = utc_now_iso()

    existing = _ledger_payload(json_path)
    backup: Path | None = None
    if existing is None:
        # Never overwrite provenance we cannot read: keep a verbatim copy.
        backup = json_path.with_name(f"{LAUNCH_LEDGER}.unreadable-{utc_stamp()}")
        shutil.copy2(json_path, backup)
        existing = {}

    payload: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    schema = LAUNCH_SCHEMA_VERSION
    legacy_note: dict[str, Any] | None = None
    if existing:
        existing_schema = _int(existing.get("schema_version"), 1) or 1
        if existing_schema < LAUNCH_SCHEMA_VERSION:
            mtime = _mtime(json_path)
            recorded = _iso_from_epoch(mtime) if mtime is not None else None
            history = _legacy_import(existing.get("runs"), recorded)
            legacy_note = {
                "schema_version": existing_schema,
                "status": existing.get("status"),
                "rows": len(history),
                "recorded_at": recorded,
                "imported_at": now,
            }
        else:
            payload = dict(existing)
            history = _dict_list(existing.get("runs"))
            schema = existing_schema
            # Rows without generation_id are read as legacy with the file mtime
            # as their time; pin it before the rewrite moves the mtime forward.
            mtime = _mtime(json_path)
            recorded = _iso_from_epoch(mtime) if mtime is not None else None
            for row in history:
                if not row.get("generation_id"):
                    row["legacy"] = True
                    if not row.get("legacy_recorded_at") and recorded is not None:
                        row["legacy_recorded_at"] = recorded
    if backup is not None:
        payload["unreadable_backup"] = str(backup)

    seen = {_row_key(row) for row in history}
    incoming: set[str] = set()
    for raw in rows:
        row = {key: value for key, value in dict(raw).items() if not str(key).startswith("_")}
        for key in LAUNCH_ROW_KEYS:
            row.setdefault(key, None if key in ("generation", "generation_id") else "")
        if not row.get("submitted_at"):
            row["submitted_at"] = now
        if not row.get("batch_id"):
            row["batch_id"] = batch_id
        incoming.add(str(row.get("status") or ""))
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        history.append(row)

    batches = _dict_list(payload.get("batches"))
    entry = next((item for item in batches if item.get("batch_id") == batch_id), None)
    if entry is None:
        entry = {"batch_id": batch_id, "started_at": now}
        batches.append(entry)
    entry.update({key: value for key, value in batch.items() if key not in ("submitted", "failed")})
    entry["batch_id"] = batch_id
    entry.setdefault("started_at", now)
    if not batch.get("finished_at"):
        entry["finished_at"] = now
    batch_rows = [row for row in history if row.get("batch_id") == batch_id]
    entry["submitted"] = sum(row.get("status") == "SUBMITTED" for row in batch_rows)
    entry["failed"] = sum(row.get("status") == "FAILED" for row in batch_rows)
    # Derived from this batch's rows AND the incoming ones: a row skipped as a
    # duplicate of an older batch still tells what this call recorded.
    if entry["failed"] or "FAILED" in incoming:
        entry["status"] = "FAILED"
    elif batch.get("status"):
        entry["status"] = str(batch["status"])
    elif entry["submitted"] or "SUBMITTED" in incoming:
        entry["status"] = "SUBMITTED"
    elif not entry.get("status"):
        entry["status"] = "EMPTY"

    payload.update(
        {
            "format": LAUNCH_FORMAT,
            "schema_version": schema,
            "root": str(directory),
            "status": entry["status"],
            "preflight": payload.get("preflight") or "PASS",
            "latest_batch_id": batch_id,
            "batches": batches,
            "runs": history,
        }
    )
    if legacy_note is not None:
        payload["legacy_import"] = legacy_note
    atomic_write_json(json_path, payload)
    atomic_write_text(directory / LAUNCH_TSV, _tsv_text(history))
    return json_path


def seal_launch_rows(
    run: str | Path,
    ledger_paths: Iterable[str | Path],
    *,
    retired_generation_id: str,
    new_generation_id: str,
) -> list[str]:
    """Annotate rows of a retired generation of ``run`` with ``superseded_by_generation_id``.

    Informational provenance only (matching never relies on it).  Rows for
    ``run`` that are legacy or carry ``retired_generation_id`` and are not yet
    sealed get ``superseded_by_generation_id`` / ``superseded_at``.  Because a
    rewrite changes the file mtime, every legacy row in a rewritten ledger is
    first pinned with ``legacy_recorded_at`` (the pre-rewrite mtime).  Returns
    the ledgers modified.  Missing or unreadable ledgers are left untouched.
    Each read-modify-write holds the ledger's lock (see ``append_launch_rows``).
    Call ONLY during ``--execute`` when a new generation is created.
    """

    target = _path_key(Path(run).expanduser())
    modified: list[str] = []
    seen: set[str] = set()
    now = utc_now_iso()
    for ledger_path in ledger_paths:
        path = Path(ledger_path)
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        if not _ledger_payload(path):  # missing, empty or unreadable: nothing to seal, never rewrite
            continue
        with _file_lock(path):
            if _seal_ledger_locked(path, target, retired_generation_id, new_generation_id, now):
                modified.append(str(path))
    return modified


def _seal_ledger_locked(path: Path, target: str, retired_generation_id: str, new_generation_id: str, now: str) -> bool:
    payload = _ledger_payload(path)
    if not payload:
        return False
    runs = payload["runs"]
    schema = _int(payload.get("schema_version"), 1) or 1
    changed = False
    for row in runs:
        if not isinstance(row, dict) or row.get("superseded_by_generation_id"):
            continue
        if target not in _row_directory_keys(row, path.parent):
            continue
        legacy = schema < 2 or bool(row.get("legacy")) or not row.get("generation_id")
        if legacy or row.get("generation_id") == retired_generation_id:
            row["superseded_by_generation_id"] = new_generation_id
            row["superseded_at"] = now
            changed = True
    if not changed:
        return False
    mtime = _mtime(path)
    recorded = _iso_from_epoch(mtime) if mtime is not None else None
    for row in runs:
        if not isinstance(row, dict):
            continue
        if (schema < 2 or row.get("legacy") or not row.get("generation_id")) and not row.get("legacy_recorded_at"):
            if recorded is not None:
                row["legacy_recorded_at"] = recorded
    atomic_write_json(path, payload)
    return True
