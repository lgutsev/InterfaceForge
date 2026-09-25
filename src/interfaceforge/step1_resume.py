"""Resume healthy but interrupted Step1 AIMD runs from their latest trustworthy state.

A Step1 preheat that stopped on wall time (or any other clean kill) with a
healthy trajectory must not be *repaired*: nothing went wrong physically, so
nothing is rewound to a "safe" frame and no conservative settings are forced.
``step1-resume`` instead continues the run where it stopped:

* **Eligibility** -- the run is not an active Slurm WorkDir, its files are not
  recent (``stale_hours``; 0.1 h when Slurm is verified, 6 h otherwise), it has
  started, its inputs are present, ``diagnose_step1_run`` does not call it
  hard-unstable (that is ``step1-repair``'s job) nor raise a non-benign warning
  (unless ``accept_warnings``), OUTCAR carries no VASP error marker, at least
  one ionic step completed in the current segment, and the accepted total is
  still below the whole-run target.
* **Restart source** -- the CONTCAR when it is *trusted* (same lattice and
  species as the segment-start POSCAR, finite coordinates and velocities, and
  within a displacement tolerance of the latest XDATCAR frame); it is copied
  verbatim, so the velocities and predictor-corrector block continue the
  dynamics.  Otherwise the latest XDATCAR frame (``k * NBLOCK`` steps, written
  without velocities so VASP redraws Maxwell-Boltzmann velocities at the
  resumed TEBEG), else the unchanged segment-start POSCAR.
* **Settings** -- every INCAR tag is kept byte-for-byte except ``NSW`` (exactly
  the remaining steps), ``TEBEG`` (the schedule temperature reached, so an
  interrupted 100->300 K ramp continues instead of restarting at 100 K),
  ``ISTART`` (electronic start mode) and ``ICHARG`` (removed).  ``TEEND`` stays
  explicit so the ramp keeps its endpoint.  A root or generic INCAR is never
  copied in.
* **Generations** -- execution archives the run first
  (``archive_step1_state``), retires the previous segment record, seals the
  retired generation's launch-ledger rows and writes ``step1_resume.json``
  (generation N+1) whose accepted-segment ledger is the parent's plus the
  closed segment.  The prepared run is then launchable as ``resume-prepared``.

Dry run (the default) reads files and asks the scheduler, nothing else.
Execution re-checks Slurm and the planning fingerprint immediately before the
first write of each run and stops at the first failure.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aimd import _first_int
from .audit import read_tail
from .errors import SafetyError
from .step1_lineage import (
    ARCHIVE_MANIFEST,
    NOT_ARCHIVED,
    RESUME_RECORD,
    accepted_ps,
    archive_step1_state,
    atomic_write_json,
    build_segment_record,
    current_generation,
    finalize_archive,
    format_temperature,
    incar_schedule,
    interrupted_archive,
    ledger_paths_for,
    new_generation_id,
    read_json,
    retire_current_records,
    run_fingerprint,
    schedule_temperature,
    seal_launch_rows,
    utc_now_iso,
)
from .step1_repair import (
    DEFAULT_CATASTROPHIC_ENERGY_EV,
    DEFAULT_ENERGY_JUMP_EV,
    DEFAULT_STARTUP_GRACE_STEPS,
    _diagnostic_kwargs,
    _diagnostic_settings,
    _discover_runs,
    _precondition_launcher,
    _write_rewind_poscar,
    _xdatcar_frames,
    apply_precondition,
    diagnose_step1_run,
    parse_step1_oszicar,
    precondition_blocker,
)
from .step1_scheduler import SchedulerGuard, SchedulerSnapshot, as_guard, resolve_stale_hours
from .step1_status import _ACTIVITY_FILES, _ERROR_MARKERS, _completed_steps
from .vasp import _PRECONDITION_MARKER, _poscar_layout_indices, parse_incar, require_files, update_incar

# Plan statuses.  Only READY plans are executed; PREPARED is a READY plan after
# execution.  Every other status is a skip that is never mutated.
RESUME_READY = "READY"
RESUME_PREPARED = "PREPARED"
RESUME_ACTIVE_SLURM = "ACTIVE_SLURM"
RESUME_ACTIVE_OR_RECENT = "ACTIVE_OR_RECENT"
RESUME_UNSTABLE = "UNSTABLE"
RESUME_REVIEW = "REVIEW"
RESUME_COMPLETE = "COMPLETE"
RESUME_NOT_STARTED = "NOT_STARTED"
RESUME_MISSING_INPUTS = "MISSING_INPUTS"
RESUME_NO_PROGRESS = "NO_PROGRESS"
RESUME_SKIP_STATUSES = (
    RESUME_ACTIVE_SLURM,
    RESUME_ACTIVE_OR_RECENT,
    RESUME_UNSTABLE,
    RESUME_REVIEW,
    RESUME_COMPLETE,
    RESUME_NOT_STARTED,
    RESUME_MISSING_INPUTS,
    RESUME_NO_PROGRESS,
)

DEFAULT_CONTCAR_TOLERANCE_ANGSTROM = 1.0
# Allowance for how far an ion may travel per femtosecond between the reference
# frame and the CONTCAR (0.05 A/fs is several times a 300-1200 K thermal speed).
CONTCAR_DRIFT_ANGSTROM_PER_FS = 0.05
_LATTICE_REL_TOL = 1e-6

RESUME_REQUIRED_INPUTS = ("INCAR", "POSCAR", "KPOINTS", "OSZICAR")
# Runtime outputs of the interrupted segment that a resume removes once they are
# archived.  WAVECAR is handled by the electronic-start mode; a current segment
# record is retired separately through ``retire_current_records``.
RESUME_RUNTIME_OUTPUTS = (
    "OSZICAR",
    "OUTCAR",
    "XDATCAR",
    "XDATCAR_FINAL",
    "CONTCAR",
    "vasprun.xml",
    "REPORT",
    "PCDAT",
    "EIGENVAL",
    "DOSCAR",
    "PROCAR",
    "IBZKPT",
    "CHG",
    "CHGCAR",
    "vasp_md.dat",
    "vasp_md_FINAL.dat",
    ".vasp_md.dat",
    "MD_TempPlot.png",
)

# Plan keys that describe the planning pass, not the prepared segment; they are
# kept out of the written step1_resume.json.
_PLAN_ONLY_KEYS = ("fingerprint", "skip_reason", "review_reasons", "active_jobs")
_BLOCKED_LINEAGE_STATUSES = ("UNREADABLE", "CONFLICT")
# Electronic-start fields that must be identical at planning and execution.
_ELECTRONIC_ACTION_KEYS = ("mode", "istart", "remove_wavecar", "apply_precondition")
_ARCHIVE_STAMP = re.compile(r"_(\d{8}T\d{6}Z)$")
_XDATCAR_CARTESIAN = re.compile(r"^\s*Cartesian\s+configuration\s*=", re.I | re.M)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _newest_activity(folder: Path) -> tuple[str | None, float | None]:
    """``(file name, age in hours)`` of the most recently modified activity file (status's set)."""

    newest: tuple[float, str] | None = None
    for name in _ACTIVITY_FILES:
        try:
            mtime = (folder / name).stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, name)
    if newest is None:
        return None, None
    modified = datetime.fromtimestamp(newest[0], tz=timezone.utc)
    return newest[1], (datetime.now(tz=timezone.utc) - modified).total_seconds() / 3600.0


def _active_label(jobs: list[dict[str, Any]]) -> str:
    return ", ".join(f"job {job.get('job_id')} {job.get('state')}" for job in jobs)


def _run_label(root: Path, run: str | Path) -> str:
    try:
        relative = Path(run).relative_to(root).as_posix()
    except ValueError:
        return str(run)
    return relative if relative != "." else Path(run).name


def _validate_resume_options(contcar_tolerance_angstrom: float, precondition: bool, fresh_start: bool) -> float:
    tolerance = float(contcar_tolerance_angstrom)
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError("contcar_tolerance_angstrom must be a finite distance >= 0 in Angstrom")
    if precondition and fresh_start:
        # Opposite electronic starts (preconditioned WAVECAR vs ISTART=0): refuse to guess.
        raise ValueError("precondition and fresh_start are mutually exclusive; choose one electronic start")
    return tolerance


# --------------------------------------------------------------------------- #
# Structures (POSCAR / CONTCAR / XDATCAR frames)
# --------------------------------------------------------------------------- #


class _StructureError(ValueError):
    """A POSCAR/CONTCAR/XDATCAR frame that cannot be trusted as a structure (the message says why)."""


def _leading_floats(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split():
        try:
            values.append(float(token))
        except ValueError:
            break
    return values


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inverse3(m: list[list[float]]) -> list[list[float]]:
    det = _det3(m)
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ]


def _finite_row(text: str, what: str, number: int) -> list[float]:
    row = _leading_floats(text)[:3]
    if len(row) != 3 or not all(math.isfinite(value) for value in row):
        raise _StructureError(f"{what} row {number} is not three finite numbers: {text.strip()!r}")
    return row


def _read_structure(path: Path, *, check_trailing: bool) -> dict[str, Any]:
    """Lattice (scaled, Cartesian A), species/count lines and fractional coordinates of a POSCAR/CONTCAR.

    ``check_trailing`` also validates an MD velocity block after the
    coordinates: a header line (blank or starting with a letter, e.g.
    ``Cartesian``) followed by ``ion_count`` rows of finite numbers.  Anything
    after that block (the predictor-corrector data) is not inspected.  Raises
    ``_StructureError`` with a human-readable reason.
    """

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise _StructureError(f"cannot read {path.name}: {exc}") from exc
    try:
        count_index, ion_count, mode_index = _poscar_layout_indices(lines, path.name)
    except SafetyError as exc:
        raise _StructureError(str(exc)) from exc

    scales = _leading_floats(lines[1])
    vectors = [_finite_row(lines[index], "lattice", index - 1) for index in (2, 3, 4)]
    if not scales or not all(math.isfinite(value) for value in scales[:3]):
        raise _StructureError(f"{path.name} has no finite scale factor: {lines[1].strip()!r}")
    if len(scales) >= 3:
        axis_scale = scales[:3]  # VASP 6: one factor per Cartesian axis
    elif scales[0] > 0.0:
        axis_scale = [scales[0]] * 3
    elif scales[0] < 0.0:
        volume = abs(_det3(vectors))
        if volume <= 0.0:
            raise _StructureError(f"{path.name} lattice vectors are degenerate")
        axis_scale = [(-scales[0] / volume) ** (1.0 / 3.0)] * 3  # a negative factor is the cell volume
    else:
        raise _StructureError(f"{path.name} has a zero scale factor")
    lattice = [[vector[axis] * axis_scale[axis] for axis in range(3)] for vector in vectors]
    if abs(_det3(lattice)) <= 0.0:
        raise _StructureError(f"{path.name} lattice vectors are degenerate")

    species = lines[5].split() if count_index == 6 else None
    counts = [int(token) for token in lines[count_index].split()]
    start = mode_index + 1
    rows = lines[start : start + ion_count]
    if len(rows) != ion_count:
        raise _StructureError(f"{path.name} has {len(rows)} of {ion_count} coordinate rows (truncated)")
    coordinates = [_finite_row(text, f"{path.name} coordinate", index + 1) for index, text in enumerate(rows)]
    if lines[mode_index].strip()[:1].lower() in {"c", "k"}:
        inverse = _inverse3(lattice)
        cartesian = [[row[axis] * axis_scale[axis] for axis in range(3)] for row in coordinates]
        fractional = [[sum(row[i] * inverse[i][j] for i in range(3)) for j in range(3)] for row in cartesian]
    else:
        fractional = coordinates

    velocities = False
    trailing = lines[start + ion_count :]
    if check_trailing and any(line.strip() for line in trailing):
        header = trailing[0].strip()
        if header and not header[0].isalpha():
            raise _StructureError(f"{path.name} has an unrecognised block after its coordinates: {header!r}")
        block = trailing[1 : 1 + ion_count]
        if len(block) != ion_count:
            raise _StructureError(f"{path.name} velocity block has {len(block)} of {ion_count} rows (truncated)")
        for index, text in enumerate(block):
            _finite_row(text, f"{path.name} velocity", index + 1)
        velocities = True
    return {
        "lattice": lattice,
        "species": species,
        "counts": counts,
        "ion_count": ion_count,
        "fractional": fractional,
        "velocities": velocities,
    }


def _frame_fractional(frame: list[str]) -> list[list[float]]:
    return [_finite_row(text, "XDATCAR frame", index + 1) for index, text in enumerate(frame)]


def _same_lattice(first: list[list[float]], second: list[list[float]]) -> bool:
    largest = max(abs(value) for row in (*first, *second) for value in row)
    return all(
        math.isclose(a, b, rel_tol=_LATTICE_REL_TOL, abs_tol=_LATTICE_REL_TOL * largest)
        for row_a, row_b in zip(first, second, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )


def _max_displacement(first: list[list[float]], second: list[list[float]], lattice: list[list[float]]) -> float:
    """Largest minimum-image distance (A) between matching ions of two fractional coordinate sets."""

    worst = 0.0
    for a, b in zip(first, second, strict=True):
        delta = [x - y for x, y in zip(a, b, strict=True)]
        delta = [value - round(value) for value in delta]
        cartesian = [sum(delta[i] * lattice[i][j] for i in range(3)) for j in range(3)]
        worst = max(worst, math.sqrt(sum(value * value for value in cartesian)))
    return worst


def _xdatcar_source(folder: Path) -> Path | None:
    return next(
        (candidate for candidate in (folder / "XDATCAR", folder / "XDATCAR_FINAL") if _nonempty(candidate)),
        None,
    )


def _choose_restart(folder: Path, *, n_osz: int, nblock: int, potim_fs: float, tolerance_base: float) -> dict[str, Any]:
    """The last trustworthy ionic state of the current segment (read-only).

    Returns ``{"source", "file", "frame", "segment_accepted", "contcar_check",
    "velocities", "xdatcar_frames", "k_use", "reviews"}``.  Frame ``k`` of the
    XDATCAR is segment ionic step ``k * NBLOCK`` (the same convention as repair).
    """

    reviews: list[str] = []
    check: dict[str, Any] = {
        "trusted": False,
        "reason": "",
        "max_displacement_angstrom": None,
        "tolerance_angstrom": None,
        "reference": None,
    }
    choice: dict[str, Any] = {
        "source": "POSCAR",
        "file": "POSCAR",
        "frame": None,
        "segment_accepted": 0,
        "contcar_check": check,
        "velocities": False,
        "xdatcar_frames": 0,
        "k_use": 0,
        "reviews": reviews,
    }
    try:
        poscar = _read_structure(folder / "POSCAR", check_trailing=False)
    except _StructureError as exc:
        reviews.append(f"segment-start POSCAR cannot be parsed: {exc}")
        check["reason"] = "segment-start POSCAR cannot be parsed"
        return choice

    frames: list[list[str]] = []
    xdatcar = _xdatcar_source(folder)
    if xdatcar is not None:
        if _XDATCAR_CARTESIAN.search(read_tail(xdatcar)):
            reviews.append(f"{xdatcar.name} stores Cartesian frames, which resume cannot restore")
        else:
            frames = _xdatcar_frames(xdatcar, poscar["ion_count"])
    k_use = min(len(frames), n_osz // nblock)
    choice.update({"xdatcar_frames": len(frames), "k_use": k_use})
    reference: list[list[float]] | None = None
    reference_text = "segment-start POSCAR (segment step 0)"
    if k_use >= 1:
        reference_text = f"{xdatcar.name if xdatcar else 'XDATCAR'} frame {k_use} (segment step {k_use * nblock})"
        try:
            reference = _frame_fractional(frames[k_use - 1])
        except _StructureError as exc:
            reviews.append(f"{reference_text} is not usable: {exc}")
    else:
        reference = poscar["fractional"]
    check["reference"] = reference_text

    # --- CONTCAR trust check -------------------------------------------------
    contcar = folder / "CONTCAR"
    lag = n_osz - k_use * nblock
    tolerance = tolerance_base + CONTCAR_DRIFT_ANGSTROM_PER_FS * (n_osz + 1 - k_use * nblock) * potim_fs
    check["tolerance_angstrom"] = round(tolerance, 6)
    try:
        if not _nonempty(contcar):
            raise _StructureError("CONTCAR is missing or empty")
        structure = _read_structure(contcar, check_trailing=True)
        if structure["species"] != poscar["species"] or structure["counts"] != poscar["counts"]:
            raise _StructureError(
                f"CONTCAR species/counts {structure['species']} {structure['counts']} differ from the "
                f"segment-start POSCAR {poscar['species']} {poscar['counts']}"
            )
        if not _same_lattice(structure["lattice"], poscar["lattice"]):
            raise _StructureError("CONTCAR lattice differs from the segment-start POSCAR (rel. tol 1e-6)")
        if reference is None:
            raise _StructureError(f"no usable reference frame to compare it with ({reference_text})")
        if lag > nblock:
            raise _StructureError(
                f"the reference ({reference_text}) lags the {n_osz} completed step(s) by {lag} > NBLOCK={nblock}"
            )
        displacement = _max_displacement(structure["fractional"], reference, poscar["lattice"])
        check["max_displacement_angstrom"] = round(displacement, 6)
        if not displacement <= tolerance:
            raise _StructureError(
                f"CONTCAR is {displacement:.3f} A from {reference_text}, beyond the {tolerance:.3f} A tolerance"
            )
    except _StructureError as exc:
        check["reason"] = str(exc)
    else:
        check["trusted"] = True
        check["reason"] = f"matches {reference_text} within {tolerance:.3f} A (max displacement {displacement:.3f} A)"
        choice.update(
            {
                "source": "CONTCAR",
                "file": "CONTCAR",
                "segment_accepted": n_osz,
                "velocities": bool(structure["velocities"]),
            }
        )
        return choice

    # --- fallbacks -------------------------------------------------------------
    # (k_use >= 1 with an unusable frame already added a review reason: never resume from it.)
    if k_use >= 1 and reference is not None and xdatcar is not None:
        choice.update({"source": "XDATCAR", "file": xdatcar.name, "frame": k_use, "segment_accepted": k_use * nblock})
    return choice


# --------------------------------------------------------------------------- #
# Electronic start
# --------------------------------------------------------------------------- #


def _electronic_start(
    folder: Path, incar: Mapping[str, str], *, precondition: bool, fresh_start: bool
) -> tuple[dict[str, Any], list[str]]:
    """How the resumed segment starts electronically, and any review reasons (read-only).

    Modes (first match wins): the launcher is already precondition-wrapped ->
    ``precondition`` (WAVECAR removed so the wrapper reconverges the magnetic
    DFT+U state at the resumed geometry, ISTART=1); ``precondition`` requested
    -> ``precondition`` (``apply_precondition`` after the INCAR update);
    ``fresh_start`` -> ``fresh`` (ISTART=0, WAVECAR removed); ISTART >= 1 with a
    nonempty WAVECAR -> ``wavecar`` (kept, ISTART unchanged; a hard-linked
    WAVECAR is replaced by a private copy first); otherwise ``fresh`` (ISTART=0,
    an empty WAVECAR removed).
    """

    launcher = _precondition_launcher(folder)
    launcher_text = ""
    if launcher is not None:
        try:
            launcher_text = (folder / launcher).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            launcher_text = ""
    wrapped = _PRECONDITION_MARKER in launcher_text
    has_precondition_incar = _nonempty(folder / "INCAR.precondition")
    wavecar_size = _size(folder / "WAVECAR")
    istart = _first_int(incar.get("ISTART"))
    reviews: list[str] = []
    notes: list[str] = []
    apply = False
    remove_wavecar = False
    if wrapped and has_precondition_incar:
        mode, new_istart, remove_wavecar = "precondition", 1, True
        reason = f"{launcher} is precondition-wrapped: it reconverges the electronic state at the resumed geometry"
        if fresh_start:
            notes.append("fresh start requested, but the precondition-wrapped launcher already starts the MD afresh")
    elif precondition:
        mode, new_istart, remove_wavecar, apply = "precondition", 1, True, True
        reason = "--precondition: INCAR.precondition written and the launcher wrapped after the INCAR update"
        blocker = precondition_blocker(folder)
        if blocker is not None:
            reviews.append(f"cannot precondition: {blocker}")
    elif fresh_start:
        mode, new_istart, remove_wavecar = "fresh", 0, True
        reason = "--fresh-start: ISTART=0 and WAVECAR removed"
    elif istart is not None and istart >= 1 and wavecar_size:
        mode, new_istart = "wavecar", istart
        reason = f"ISTART={istart} with a nonempty WAVECAR: the resumed MD reads it"
    else:
        mode, new_istart = "fresh", 0
        remove_wavecar = wavecar_size == 0
        reason = "no usable WAVECAR (or ISTART=0): the resumed MD starts from the atomic-density guess"
    if wrapped and not has_precondition_incar and not precondition:
        reviews.append(
            f"{launcher} is precondition-wrapped but INCAR.precondition is missing; "
            "pass precondition=True (--precondition) to regenerate it"
        )

    if wavecar_size is None:
        wavecar = "none"
    elif remove_wavecar:
        wavecar = "remove (empty)" if wavecar_size == 0 else "remove"
    elif mode == "wavecar":
        wavecar = "keep"
    else:
        wavecar = "keep (unused with ISTART=0)"
    result: dict[str, Any] = {
        "mode": mode,
        "istart": new_istart,
        "istart_before": istart,
        "wavecar": wavecar,
        "remove_wavecar": remove_wavecar,
        "apply_precondition": apply,
        "launcher": launcher,
        "launcher_wrapped": wrapped,
        "requested_precondition": bool(precondition),
        "requested_fresh_start": bool(fresh_start),
        "reason": reason,
        "notes": notes,
    }
    return result, reviews


def _make_private_copy(path: Path) -> bool:
    """Replace a hard-linked ``path`` by a private copy (VASP must never write through a shared inode)."""

    try:
        links = path.stat().st_nlink
    except OSError:
        return False
    if links <= 1:
        return False
    temporary = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:12]}.private.tmp")
    try:
        shutil.copy2(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def plan_resume_run(
    run: str | Path,
    *,
    snapshot: SchedulerSnapshot,
    stale_hours: float | None,
    contcar_tolerance_angstrom: float = DEFAULT_CONTCAR_TOLERANCE_ANGSTROM,
    precondition: bool = False,
    fresh_start: bool = False,
    accept_warnings: bool = False,
    diagnostic_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan the resume of one Step1 run.  Read-only: nothing in ``run`` is created, modified or removed.

    The plan ``status`` is ``READY`` or one of the skip statuses (first match
    wins, ``skip_reason`` says why): ``ACTIVE_SLURM`` (``snapshot`` lists a job
    whose WorkDir is the run), ``ACTIVE_OR_RECENT`` (an activity file --
    OSZICAR/OUTCAR/CONTCAR/XDATCAR -- modified within ``stale_hours``; ``None``
    resolves via ``resolve_stale_hours``), ``REVIEW`` (unreadable/conflicting
    lineage or an interrupted earlier mutation), ``NOT_STARTED``,
    ``MISSING_INPUTS``, ``UNSTABLE`` (use step1-repair), ``REVIEW`` (a non-benign
    warning without ``accept_warnings``, or a VASP error marker in OUTCAR),
    ``NO_PROGRESS`` (no completed ionic step in the current segment),
    ``COMPLETE`` (the accepted total reaches the whole-run target) and finally
    ``REVIEW`` for anything else that prevents a safe resume (unparsable
    POSCAR, unknown target, a launcher that cannot be preconditioned, ...).
    Only READY plans may be executed.
    """

    tolerance_base = _validate_resume_options(contcar_tolerance_angstrom, precondition, fresh_start)
    folder = Path(run).expanduser().resolve()
    hours, _ = resolve_stale_hours(stale_hours, snapshot)
    # Fingerprint first: any change after this point invalidates the plan at execution.
    fingerprint = run_fingerprint(folder)
    diagnostic = diagnose_step1_run(folder, **_diagnostic_kwargs(diagnostic_options))

    incar = parse_incar(folder / "INCAR")
    schedule = incar_schedule(incar)
    nblock = int(schedule["nblock"])
    generation = current_generation(folder, incar)
    active_jobs = snapshot.active_jobs_for(folder)
    newest_name, age_hours = _newest_activity(folder)
    started = _nonempty(folder / "OSZICAR") or _nonempty(folder / "OUTCAR")
    missing = [name for name in RESUME_REQUIRED_INPUTS if not _nonempty(folder / name)]

    lineage_reviews: list[str] = []
    if generation.status in _BLOCKED_LINEAGE_STATUSES:
        detail = generation.conflict or f"{generation.record_path} cannot be read"
        lineage_reviews.append(f"current generation is {generation.status}: {detail}")
    interrupted = interrupted_archive(folder)
    if interrupted is not None:
        lineage_reviews.append(f"interrupted recovery mutation; inspect {interrupted}")

    outcar_tail = read_tail(folder / "OUTCAR") if _nonempty(folder / "OUTCAR") else ""
    error_markers = [marker for marker in _ERROR_MARKERS if marker in outcar_tail]
    oszicar_rows = parse_step1_oszicar(folder / "OSZICAR", nelm=int(diagnostic["scf_nelm"]))["steps"]
    n_osz = _completed_steps(oszicar_rows)

    # Current segment (what VASP actually ran) and the restart source.
    segment_potim = schedule["potim_fs"] if schedule["potim_fs"] is not None else generation.segment_potim_fs
    segment_nsw = schedule["nsw"] if schedule["nsw"] is not None else generation.segment_nsw
    segment_tebeg = float(schedule["tebeg_k"])
    segment_teend = float(schedule["teend_k"])
    reviews: list[str] = []
    if len(oszicar_rows) > n_osz and n_osz:
        reviews.append(
            f"OSZICAR has {len(oszicar_rows)} MD rows but only steps 1..{n_osz} run without a gap; it may mix two runs"
        )
    restart = _choose_restart(
        folder,
        n_osz=n_osz,
        nblock=nblock,
        # An unknown POTIM narrows the drift allowance (VASP's MD default is 0.5 fs).
        potim_fs=float(segment_potim) if segment_potim is not None else 0.5,
        tolerance_base=tolerance_base,
    )
    reviews.extend(restart["reviews"])
    segment_accepted = int(restart["segment_accepted"])

    # Cumulative accounting over generations.
    prefix = int(generation.accepted_prefix_steps)
    original_nsw = generation.original_nsw
    cumulative = prefix + segment_accepted
    remaining: int | None = None
    if original_nsw is None or original_nsw <= 0:
        reviews.append(f"no positive whole-run target NSW ({generation.generation_id}; INCAR NSW={incar.get('NSW')})")
    else:
        cumulative = min(original_nsw, cumulative)
        remaining = original_nsw - cumulative
    if segment_nsw is None or segment_nsw <= 0:
        reviews.append(f"the current segment has no positive NSW (INCAR NSW={incar.get('NSW')})")
    if segment_potim is None or not segment_potim > 0:
        reviews.append(f"the current segment has no positive POTIM (INCAR POTIM={incar.get('POTIM')})")

    # Temperature continuation: VASP ramps TEBEG->TEEND linearly over the segment NSW.
    resumed_exact = schedule_temperature(segment_tebeg, segment_teend, segment_nsw, segment_accepted)
    resumed_tebeg = float(format_temperature(resumed_exact))
    tebeg_text = format_temperature(resumed_exact) if abs(resumed_tebeg - segment_tebeg) > 1e-9 else None
    # TEEND stays explicit: an explicit TEEND line is left untouched; an absent
    # one (TEEND defaults to TEBEG in VASP) is written if TEBEG moves.
    teend_text = None
    if not schedule["teend_explicit"] and abs(resumed_tebeg - segment_teend) > 1e-9:
        teend_text = format_temperature(segment_teend)
    ramp = abs(segment_tebeg - segment_teend) > 1e-9
    rate_before = (segment_teend - segment_tebeg) / segment_nsw if segment_nsw else None
    rate_after = (segment_teend - resumed_tebeg) / remaining if remaining else None
    left_in_segment = segment_nsw - segment_accepted if segment_nsw else None
    note = None
    if ramp and remaining is not None and left_in_segment is not None and left_in_segment != remaining:
        note = (
            f"ramp rate changes from {rate_before:.4g} to {rate_after if rate_after is not None else 0.0:.4g} K/step: "
            f"the resumed segment runs NSW={remaining} while {left_in_segment} step(s) were left in the old "
            f"segment; the endpoint TEEND={format_temperature(segment_teend)} K is kept"
        )
    temperature = {
        "previous_tebeg_k": segment_tebeg,
        "previous_teend_k": segment_teend,
        "previous_nsw": segment_nsw,
        "accepted_in_segment": segment_accepted,
        "resumed_tebeg_k": resumed_tebeg,
        "teend_k": segment_teend,
        "rate_k_per_step_before": round(rate_before, 9) if rate_before is not None else None,
        "rate_k_per_step_after": round(rate_after, 9) if rate_after is not None else None,
        "note": note,
    }

    electronic, electronic_reviews = _electronic_start(
        folder, incar, precondition=precondition, fresh_start=fresh_start
    )
    reviews.extend(electronic_reviews)
    incar_changes: dict[str, Any] = {"NSW": remaining}
    if tebeg_text is not None:
        incar_changes["TEBEG"] = tebeg_text
    if teend_text is not None:
        incar_changes["TEEND"] = teend_text
    if electronic["istart_before"] != electronic["istart"]:
        incar_changes["ISTART"] = electronic["istart"]
    incar_delete = ["ICHARG"]

    # Accepted-segment ledger once this resume closes the current segment.
    if restart["source"] == "XDATCAR":
        restart_label = f"{restart['file']} frame {restart['frame']}"
    else:
        restart_label = str(restart["source"])
    closed_segment = {
        "generation": generation.generation,
        "generation_id": generation.generation_id,
        "kind": generation.kind,
        "steps": segment_accepted,
        "potim_fs": segment_potim,
        "ps": round(segment_accepted * segment_potim / 1000.0, 9) if segment_potim is not None else None,
        "tebeg_k": segment_tebeg,
        "teend_k": round(resumed_exact, 2),
        "restart_source": restart_label,
    }
    ledger = [dict(row) for row in generation.ledger] + [closed_segment]
    ledger_steps = sum(_first_int(row.get("steps"), 0) or 0 for row in ledger)
    ledger_exact = bool(generation.ledger_exact) and segment_potim is not None and ledger_steps == cumulative

    # Status: the first matching rule wins.
    warnings = diagnostic.get("warnings") or []
    warning_blocks = diagnostic.get("severity") == "warning" and not diagnostic.get("benign_warnings_only")
    if active_jobs:
        status, skip_reason = RESUME_ACTIVE_SLURM, f"active in Slurm ({_active_label(active_jobs)})"
    elif age_hours is not None and age_hours < hours:
        status, skip_reason = RESUME_ACTIVE_OR_RECENT, f"{newest_name} updated {age_hours:.2f} h ago (< {hours:g} h)"
    elif lineage_reviews:
        # Same precedence as step1_status.recovery_category: a half-done mutation or an
        # untrustworthy record is reviewed before anything else is judged.
        status, skip_reason = RESUME_REVIEW, "; ".join(lineage_reviews)
    elif not started:
        status = RESUME_NOT_STARTED
        skip_reason = "not started (OSZICAR and OUTCAR are missing or empty); launch it with step1-launch"
    elif missing:
        status, skip_reason = RESUME_MISSING_INPUTS, f"missing or empty required files: {', '.join(missing)}"
    elif diagnostic.get("unstable"):
        hard = diagnostic.get("hard_reasons") or diagnostic.get("first_bad_reasons") or ["hard-unstable trajectory"]
        status, skip_reason = RESUME_UNSTABLE, f"hard-unstable: {hard[0]}; use step1-repair"
    elif warning_blocks and not accept_warnings:
        status = RESUME_REVIEW
        skip_reason = (
            f"review-level warning: {warnings[0] if warnings else 'see diagnostic'}; inspect it, or pass "
            "accept_warnings=True (--accept-warnings) to resume anyway"
        )
    elif error_markers:
        status = RESUME_REVIEW
        skip_reason = f"OUTCAR has a VASP error marker ({', '.join(error_markers)}); inspect before resuming"
    elif n_osz == 0:
        status = RESUME_NO_PROGRESS
        skip_reason = (
            "no completed ionic step in the current segment; nothing to resume (review: inspect the Slurm logs)"
        )
    elif original_nsw is not None and original_nsw > 0 and cumulative >= original_nsw:
        status, skip_reason = RESUME_COMPLETE, f"complete: {cumulative}/{original_nsw} accepted steps"
    elif reviews:
        status, skip_reason = RESUME_REVIEW, "; ".join(reviews)
    else:
        status, skip_reason = RESUME_READY, None

    new_generation = generation.generation + 1
    return {
        "run": str(folder),
        "status": status,
        "skip_reason": skip_reason,
        "review_reasons": lineage_reviews + reviews,
        "active_jobs": active_jobs,
        "age_hours": age_hours,
        "activity_file": newest_name,
        "diagnostic": diagnostic,
        "accepted_warnings": list(warnings) if warning_blocks and accept_warnings else [],
        # Generation lineage.
        "generation": new_generation,
        "generation_id": None,
        "parent_generation": generation.generation,
        "parent_generation_id": generation.generation_id,
        "parent_segment_kind": generation.kind,
        "parent_legacy_record": bool(generation.legacy and generation.record_path is not None),
        "operation": f"step1_resume_g{new_generation}",
        "archive": None,
        # Progress accounting.
        "original_nsw": original_nsw,
        "previous_accepted_prefix_steps": prefix,
        "segment_completed_steps": n_osz,
        "segment_accepted_steps": segment_accepted,
        "accepted_prefix_steps": cumulative,
        "remaining_steps": remaining,
        "resume_nsw": remaining,
        # Restart source.
        "restart_source": restart["source"],
        "restart_file": restart["file"],
        "restart_frame": restart["frame"],
        "restart_velocities": bool(restart["velocities"]),
        "xdatcar_frames": restart["xdatcar_frames"],
        "contcar_check": restart["contcar_check"],
        # Segment settings.
        "segment_potim_fs": segment_potim,
        "previous_segment": {
            "tebeg_k": segment_tebeg,
            "teend_k": segment_teend,
            "nsw": segment_nsw,
            "potim_fs": segment_potim,
            "nblock": nblock,
            "thermostat": schedule["thermostat"],
            "ramp": ramp,
        },
        "segment_schedule": {
            "tebeg_k": resumed_tebeg,
            "teend_k": segment_teend,
            "nsw": remaining,
            "thermostat": schedule["thermostat"],
            "ramp": abs(resumed_tebeg - segment_teend) > 1e-9,
        },
        "temperature_continuation": temperature,
        "electronic_start": electronic,
        "accepted_segments": ledger,
        "accepted_ps": accepted_ps(ledger),
        "ledger_exact": ledger_exact,
        "incar_changes": incar_changes,
        "incar_delete": incar_delete,
        "fingerprint": fingerprint,
    }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _restart_frame(run: Path, plan: Mapping[str, Any]) -> list[str]:
    """The XDATCAR frame a READY plan restarts from, re-read and re-validated (raises ``SafetyError``)."""

    nblock = int(incar_schedule(parse_incar(run / "INCAR"))["nblock"])
    source = run / str(plan.get("restart_file") or "XDATCAR")
    try:
        ion_count = int(_read_structure(run / "POSCAR", check_trailing=False)["ion_count"])
        frames = _xdatcar_frames(source, ion_count) if source.is_file() else []
        index = int(plan["restart_frame"])
        if index * nblock != int(plan["segment_accepted_steps"]) or not 1 <= index <= len(frames):
            raise _StructureError(f"{source.name} has {len(frames)} frame(s) at NBLOCK={nblock}")
        frame = frames[index - 1]
        _frame_fractional(frame)
    except (_StructureError, KeyError, TypeError, ValueError) as exc:
        raise SafetyError(
            f"Refusing to mutate {run}: restart frame {plan.get('restart_frame')} cannot be restored ({exc})"
        ) from exc
    return frame


def _ensure_archived(run: Path, archive: Path, names: Iterable[str]) -> list[str]:
    """Copy any of ``names`` that the archive does not hold yet (regenerable files excepted).

    Guarantees archive-before-delete for every runtime output resume removes,
    whatever ``archive_step1_state`` copies; the manifest's file list is
    extended so it still describes the archive.
    """

    added: list[str] = []
    for name in names:
        source = run / name
        if name in NOT_ARCHIVED or not source.is_file() or (archive / name).is_file():
            continue
        shutil.copy2(source, archive / name)
        added.append(name)
    if added:
        manifest_path = archive / ARCHIVE_MANIFEST
        manifest = read_json(manifest_path)
        if not manifest:
            raise SafetyError(f"{manifest_path} is missing or unreadable; it was not written by archive_step1_state")
        files = [item for item in manifest.get("files") or [] if isinstance(item, dict)]
        files.extend({"name": name, "bytes": (archive / name).stat().st_size} for name in added)
        manifest["files"] = sorted(files, key=lambda item: str(item.get("name")))
        atomic_write_json(manifest_path, manifest)
    return added


def execute_resume_plan(
    plan: Mapping[str, Any], *, guard: SchedulerGuard, ledger_roots: Iterable[str | Path] = ()
) -> dict[str, Any]:
    """Prepare the resume segment described by a READY ``plan``; returns the updated plan.

    Every check that can refuse runs before the first write: plan status, the
    planning fingerprint, required inputs, the lineage the plan was built on,
    an interrupted earlier mutation, the restart frame, the electronic start
    (launcher and WAVECAR as planned), the launcher when preconditioning and
    NSW > 0; last and immediately before mutating, ``guard.assert_inactive(run)``
    followed by a second fingerprint comparison (a squeue call can take
    seconds).  Then: ``archive_step1_state(run, "step1_resume_g<N>")``; POSCAR
    <- CONTCAR verbatim (velocities kept), the XDATCAR frame (no velocities)
    or left unchanged; runtime outputs removed; WAVECAR handled per the
    electronic-start mode; INCAR updated (NSW, TEBEG, ISTART, ICHARG only; the
    launcher is preconditioned after it when requested); the previous segment
    record retired and its launch rows sealed; ``step1_resume.json`` written;
    the archive finalized.  An exception after archiving leaves the archive
    manifest ``IN_PROGRESS`` so the interrupted mutation stays discoverable.
    """

    if not plan.get("run"):
        raise SafetyError("Refusing to execute a resume plan that names no run directory")
    run = Path(str(plan["run"])).expanduser()
    if plan.get("status") != RESUME_READY:
        raise SafetyError(
            f"Refusing to execute the resume plan for {run}: status is {plan.get('status')!r}, not READY"
            + (f" ({plan.get('skip_reason')})" if plan.get("skip_reason") else "")
        )
    if run_fingerprint(run) != plan.get("fingerprint"):
        raise SafetyError(f"Refusing to mutate {run}: it changed since planning (file fingerprint differs); plan again")
    require_files(run, RESUME_REQUIRED_INPUTS)
    parent = current_generation(run)
    if parent.generation_id != plan.get("parent_generation_id") or parent.status in _BLOCKED_LINEAGE_STATUSES:
        raise SafetyError(
            f"Refusing to mutate {run}: its current generation changed since planning "
            f"({plan.get('parent_generation_id')} -> {parent.generation_id}, status {parent.status})"
        )
    interrupted = interrupted_archive(run)
    if interrupted is not None:
        raise SafetyError(
            f"Refusing to mutate {run}: an earlier recovery mutation was interrupted; inspect {interrupted}"
        )

    source = str(plan.get("restart_source"))
    frame: list[str] | None = None
    if source == "XDATCAR":
        frame = _restart_frame(run, plan)
    elif source == "CONTCAR":
        if not _nonempty(run / "CONTCAR"):
            raise SafetyError(f"Refusing to mutate {run}: the planned restart source CONTCAR is missing")
    elif source != "POSCAR":
        raise SafetyError(f"Refusing to mutate {run}: unknown restart source {source!r}")
    # The launcher, INCAR.precondition and WAVECAR are not fingerprinted: re-derive the
    # electronic start from the options the plan was made with and require the same actions.
    planned_start = dict(plan.get("electronic_start") or {})
    electronic, electronic_reviews = _electronic_start(
        run,
        parse_incar(run / "INCAR"),
        precondition=bool(planned_start.get("requested_precondition")),
        fresh_start=bool(planned_start.get("requested_fresh_start")),
    )
    if electronic_reviews or any(electronic.get(key) != planned_start.get(key) for key in _ELECTRONIC_ACTION_KEYS):
        raise SafetyError(
            f"Refusing to mutate {run}: its electronic start changed since planning "
            f"(planned {planned_start.get('mode')}/{planned_start.get('wavecar')}, now "
            f"{electronic.get('mode')}/{electronic.get('wavecar')}"
            + (f"; {'; '.join(electronic_reviews)}" if electronic_reviews else "")
            + "); plan again"
        )
    incar_changes = dict(plan.get("incar_changes") or {})
    try:
        nsw = int(incar_changes["NSW"])
        numbers = (
            int(plan["original_nsw"]),
            int(plan["accepted_prefix_steps"]),
            int(plan["resume_nsw"]),
            float(plan["segment_potim_fs"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyError(f"Refusing to mutate {run}: the plan lacks a resume setting ({exc})") from exc
    original_nsw, cumulative, resume_nsw, segment_potim = numbers
    if nsw <= 0 or nsw != resume_nsw or cumulative + resume_nsw != original_nsw:
        raise SafetyError(
            f"Refusing to mutate {run}: inconsistent resume segment (NSW={nsw}, accepted {cumulative}, "
            f"target {original_nsw})"
        )

    # ---- mutation starts here (Slurm and the fingerprint re-checked immediately before) ----
    guard.assert_inactive(run)
    if run_fingerprint(run) != plan.get("fingerprint"):
        # A file changed while the scheduler was being queried.
        raise SafetyError(f"Refusing to mutate {run}: it changed since planning (file fingerprint differs); plan again")
    archive = archive_step1_state(run, str(plan["operation"]))
    added_to_archive = _ensure_archived(run, archive, RESUME_RUNTIME_OUTPUTS)
    if source == "CONTCAR":
        # Verbatim: the velocity and predictor-corrector blocks continue the dynamics.
        (run / "POSCAR").write_bytes((archive / "CONTCAR").read_bytes())
    elif frame is not None:
        _write_rewind_poscar(archive / "POSCAR", frame, run / "POSCAR")
    for name in RESUME_RUNTIME_OUTPUTS:
        (run / name).unlink(missing_ok=True)

    wavecar = run / "WAVECAR"
    private_copy = False
    if electronic["remove_wavecar"]:
        wavecar.unlink(missing_ok=True)
    elif electronic["mode"] == "wavecar":
        private_copy = _make_private_copy(wavecar)
    update_incar(run / "INCAR", incar_changes, delete=list(plan.get("incar_delete") or ()))
    if electronic["apply_precondition"]:
        apply_precondition(run)  # after update_incar: INCAR.precondition derives from the final MD INCAR

    retired = retire_current_records(run, archive)
    stamp_match = _ARCHIVE_STAMP.search(archive.name)
    generation = int(plan["generation"])
    generation_id = new_generation_id("resume", generation, stamp_match.group(1) if stamp_match else None)
    sealed = seal_launch_rows(
        run,
        ledger_paths_for(run, [Path(root) for root in ledger_roots]),
        retired_generation_id=parent.generation_id,
        new_generation_id=generation_id,
    )
    prepared_at = utc_now_iso()
    electronic_record = dict(planned_start)
    electronic_record["private_copy"] = private_copy
    extra = {key: value for key, value in plan.items() if key not in _PLAN_ONLY_KEYS}
    extra.update(
        {
            "electronic_start": electronic_record,
            "retired_records": retired,
            "sealed_ledgers": sealed,
            "archived_extra_outputs": added_to_archive,
        }
    )
    record = build_segment_record(
        "resume",
        run=run.resolve(),
        generation=generation,
        generation_id=generation_id,
        parent=parent,
        prepared_at=prepared_at,
        original_nsw=original_nsw,
        accepted_prefix_steps=cumulative,
        accepted_segments=list(plan.get("accepted_segments") or []),
        ledger_exact=bool(plan.get("ledger_exact")),
        segment_nsw=resume_nsw,
        segment_potim_fs=segment_potim,
        segment_schedule=dict(plan.get("segment_schedule") or {}),
        archive=str(archive),
        extra=extra,
    )
    atomic_write_json(run / RESUME_RECORD, record)
    finalize_archive(archive, generation_id=generation_id)

    updated = dict(plan)
    updated.update(
        {
            "status": RESUME_PREPARED,
            "archive": str(archive),
            "generation_id": generation_id,
            "prepared_at": prepared_at,
            "electronic_start": electronic_record,
            "retired_records": retired,
            "sealed_ledgers": sealed,
            "archived_extra_outputs": added_to_archive,
        }
    )
    return updated


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #


def prepare_step1_resume(
    root: str | Path,
    *,
    execute: bool = False,
    stale_hours: float | None = None,
    scheduler: str | SchedulerGuard = "auto",
    contcar_tolerance_angstrom: float = DEFAULT_CONTCAR_TOLERANCE_ANGSTROM,
    precondition: bool = False,
    fresh_start: bool = False,
    accept_warnings: bool = False,
    energy_jump_ev: float = DEFAULT_ENERGY_JUMP_EV,
    max_temperature_k: float | None = None,
    startup_grace_steps: int = DEFAULT_STARTUP_GRACE_STEPS,
    catastrophic_energy_ev: float = DEFAULT_CATASTROPHIC_ENERGY_EV,
) -> dict[str, Any]:
    """Plan (default) or prepare resume segments for healthy, interrupted Step1 runs under ``root``.

    Dry run reads files and asks the scheduler, nothing else.  ``scheduler``
    is a ``--scheduler`` mode or a shared ``SchedulerGuard``; ``stale_hours=None``
    resolves to 0.1 h when Slurm is verified and 6 h otherwise.  With
    ``execute=True`` every run is planned first, then each READY run is
    prepared by ``execute_resume_plan`` in order, stopping at the first failure
    with a ``SafetyError`` that names the runs already prepared.  Skipped runs
    are never touched.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    tolerance = _validate_resume_options(contcar_tolerance_angstrom, precondition, fresh_start)
    # Validate the diagnostic options before touching the scheduler or any run.
    energy_jump, _, grace, catastrophic, _ = _diagnostic_settings(
        energy_jump_ev, max_temperature_k, startup_grace_steps, catastrophic_energy_ev, None
    )
    diagnostic_options = {
        "energy_jump_ev": energy_jump_ev,
        "max_temperature_k": max_temperature_k,
        "startup_grace_steps": startup_grace_steps,
        "catastrophic_energy_ev": catastrophic_energy_ev,
    }
    guard = as_guard(scheduler)
    snapshot = guard.snapshot
    hours, hours_reason = resolve_stale_hours(stale_hours, snapshot)

    plans: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run in _discover_runs(root_path):
        plan = plan_resume_run(
            run,
            snapshot=snapshot,
            stale_hours=hours,
            contcar_tolerance_angstrom=tolerance,
            precondition=precondition,
            fresh_start=fresh_start,
            accept_warnings=accept_warnings,
            diagnostic_options=diagnostic_options,
        )
        if plan["status"] == RESUME_READY:
            plans.append(plan)
        else:
            skipped.append(
                {
                    "run": plan["run"],
                    "relative_path": _run_label(root_path, plan["run"]) if Path(plan["run"]) != root_path else ".",
                    "status": plan["status"],
                    "skip_reason": plan["skip_reason"],
                }
            )

    if execute:
        prepared: list[str] = []
        for index, plan in enumerate(plans):
            try:
                plans[index] = execute_resume_plan(plan, guard=guard, ledger_roots=(root_path,))
            except Exception as exc:
                interrupted = interrupted_archive(plan["run"])
                state = (
                    f"its interrupted mutation is archived at {interrupted} (ARCHIVE_MANIFEST status IN_PROGRESS)"
                    if interrupted is not None
                    else "it was not modified"
                )
                done = ", ".join(_run_label(root_path, run) for run in prepared) or "none"
                untouched = ", ".join(_run_label(root_path, row["run"]) for row in plans[index + 1 :]) or "none"
                raise SafetyError(
                    f"step1-resume stopped at {_run_label(root_path, plan['run'])}: {exc}; {state}. "
                    f"Already prepared: {done}. Not attempted: {untouched}."
                ) from exc
            prepared.append(plan["run"])

    return {
        "format": "interfaceforge-step1-resume-plan",
        "schema_version": 1,
        "mode": "prepared" if execute else "dry-run",
        "root": str(root_path),
        "scheduler": snapshot.to_dict(),
        "settings": {
            "stale_hours": hours,
            "stale_hours_requested": stale_hours,
            "stale_hours_reason": hours_reason,
            "contcar_tolerance_angstrom": tolerance,
            "contcar_drift_angstrom_per_fs": CONTCAR_DRIFT_ANGSTROM_PER_FS,
            "precondition": bool(precondition),
            "fresh_start": bool(fresh_start),
            "accept_warnings": bool(accept_warnings),
            "energy_jump_ev": energy_jump,
            "max_temperature_k": max_temperature_k,
            "startup_grace_steps": grace,
            "catastrophic_energy_ev": catastrophic,
        },
        "runs": plans,
        "skipped": skipped,
        "resumable": sum(plan["status"] in {RESUME_READY, RESUME_PREPARED} for plan in plans),
        "skipped_active": sum(row["status"] in {RESUME_ACTIVE_SLURM, RESUME_ACTIVE_OR_RECENT} for row in skipped),
    }
