"""Diagnose and safely prepare recovery segments for unstable Step1 AIMD.

The recovery is deliberately dry-run first.  It never continues from the
current CONTCAR because a numerically unstable MD step may already have put
ions on top of one another.  Instead it rewinds to an XDATCAR frame before
the first energy/temperature runaway, starts the electronic state afresh,
and runs only the number of ionic steps still needed to reach the original
Step1 target.

Repeated repair and generations
-------------------------------
Each repair creates generation N+1 of the run (``step1_lineage``).  The rewind
point is found in the CURRENT segment's own OSZICAR/XDATCAR (frame ``k`` is
segment ionic step ``k * NBLOCK``), while the accounting is cumulative:
``safe_prefix_steps = accepted_prefix_steps + safe_segment_steps`` and
``repair_nsw = original_nsw - safe_prefix_steps``, whether the current segment
is the original run, an earlier repair (schema-2 or reconstructed schema-1
record) or a resume.  A run that kept 16 steps at 1.0 fs, was repaired, and
then kept 52 more steps of the 0.5 fs repair segment is therefore repaired
again with 68 accepted steps, 332 remaining and the accepted-segment ledger
``[16 @ 1.0 fs, 52 @ 0.5 fs]`` (0.042 ps).  Without ``ramp_from`` the new
``TEBEG`` is the rewound segment's schedule temperature at the rewind point, so
an interrupted TEBEG->TEEND ramp continues instead of restarting.  Execution
re-checks Slurm and the planning fingerprint immediately before mutating,
archives first (``archive_step1_state``), retires the previous segment record,
seals its launch-ledger rows and writes a schema-2 ``step1_repair.json`` that
keeps every schema-1 key with its original meaning.

Warning vs hard instability
---------------------------
``diagnose_step1_run`` reads the OSZICAR MD rows of the *current* segment and
separates hard instability (``unstable=True``, severity ``"unstable"``: the
run is eligible for automatic repair) from review-level warnings (severity
``"warning"``: never destructive on their own).  Definitions:

* ``T_target = max(TEBEG, TEEND)`` (TEEND defaults to TEBEG; TEBEG default 300);
  ``T_limit = max_temperature_k or max(1200, 4*T_target)`` (``None`` or 0 means
  the default; a negative or non-finite limit is rejected);
  ``T_warn = max(2*T_target, T_target + 300)`` clamped to ``T_limit``.
* Grace window = the first ``startup_grace_steps`` rows (default 10).
* ``F_ref`` = median of numeric F over rows ``[grace, grace + reference_window_steps)``;
  if that slice has no numeric F, the median over the first
  ``reference_window_steps`` rows; if still none, no reference.  Two guards
  keep the median from sitting on the wrong level (``reference_source``):

  - ``post_grace_leading_rows``: when the last grace row and the first
    post-grace row agree but the window median lies more than one band ABOVE
    both, the energy rose inside the reference window (a runaway or an uphill
    jump starting at steps grace+2 .. grace+window).  ``F_ref`` is then the
    median of the leading window rows that stay within one band of the first
    post-grace row, so the rise is flagged at its real onset instead of the
    healthy rows before it.
  - ``latest_rows`` (no post-grace rows yet): when every row is within one band
    of the first-rows median although the rows span more than one band, the
    median sits between two levels and would hide the excursion; ``F_ref`` is
    then the median of the trailing rows within one band of the latest row.
* A row *departs* when ``|F - F_ref| > energy_jump_ev`` (default 50 eV).

Hard signals (any one -> unstable):

* H1 non-numeric temperature at any step (``T= ******``).
* H2 non-numeric free energy at any step.
* H3 temperature > T_limit at any step.
* H4 catastrophic energy: ``|F - F_ref| > catastrophic_energy_ev`` (default
  500 eV) at any step, including the grace window.
* H5 sustained post-grace energy departure: a post-grace departure that is NOT
  an isolated spike -- it belongs to a run of >= 2 consecutive departing rows,
  or it is the last recorded row.  Grace rows that depart contiguously into
  such a departure belong to it (they are not a startup excursion), so the
  rewind anchor is the start of the whole departing run.
* H6 persistent SCF failure: NELM ceiling on >= 50 % of post-grace rows (needs
  >= 1 post-grace row).

Warning classes (review-level, never destructive on their own):

* W1 ``startup_energy_excursion``: departure only within the grace window.  It
  is *settled* (``startup_excursion_settled``) when at least one numeric
  in-band row follows the last excursion row; an excursion still departing on
  the last recorded row is unsettled.  It is *downhill*
  (``startup_excursion_downhill``) when every excursion row lies ABOVE
  ``F_ref``, i.e. the energy relaxed down into the band as a magnetic DFT+U
  start does.  Excursion rows below ``F_ref`` mean the energy rose after
  startup and stayed up (energy injection or a wrong electronic state).
* W2 ``isolated_energy_spike``: a single post-grace departing row whose
  neighbours (previous and next) are both within the band.  With
  ``startup_grace_steps=0`` a departing step 1 has no predecessor; it counts as
  isolated when step 2 is in band (the least-mutating reading).  A departing
  row next to a non-numeric energy is hard: the run is already hard through
  H2, so this only moves the rewind anchor earlier.
* W3 ``scf_elevated``: 20 % <= post-grace NELM-ceiling fraction < 50 %.
* W4 ``temperature_elevated``: any T with T_warn < T <= T_limit.

Corroboration (-> hard, reason ``"corroborated anomalies: A + B"``): W2
together with any of W1/W3/W4; or W4 together with W1 or W3.  ``W1 + W3``
alone stays a WARNING -- the expected magnetic DFT+U fresh-start pattern
(startup relaxation, then some SCF-ceiling use) that the NiO baseline shows
without runaway.

``benign_warnings_only`` is True only for a lone W1 that is settled AND
downhill on a run that is not unstable.  A completed, clean-tailed run whose
only finding is a step-1 excursion of ~77 eV above the relaxed level
therefore reports ``severity="warning"`` with ``benign_warnings_only=True``
instead of being marked unstable, while an unsettled or uphill excursion is a
review-level warning that neither resume nor repair acts on automatically.

A final OSZICAR MD line that is not newline-terminated and stops before its
``E0=`` field was torn by a kill during the write; it is dropped
(``torn_final_line``) rather than read as a non-numeric or runaway energy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .aimd import _first_float, _first_int
from .errors import SafetyError
from .step1_lineage import (
    REPAIR_RECORD,
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
    retire_current_records,
    run_fingerprint,
    schedule_temperature,
    seal_launch_rows,
    utc_now_iso,
)
from .step1_scheduler import SchedulerGuard, SchedulerSnapshot, as_guard, resolve_stale_hours
from .vasp import (
    CONSERVATIVE_ELECTRONIC_OVERRIDES,
    _poscar_elements,
    _write_poscar_without_velocities,
    build_precondition_incar,
    parse_incar,
    require_files,
    update_incar,
    wrap_launcher_with_precondition,
)

_MD_STEP = re.compile(r"^\s*(\d+)\s+.*?\bT=\s*([^\s]+)")
_FREE_ENERGY = re.compile(r"\bF=\s*([-+0-9.Ee]+)")
# Every VASP MD line prints E0= right after F=; a line that stops before it lost its F value to a torn write.
_F_FIELD_COMPLETE = re.compile(r"\bF=\s*\S+\s+E0=")
_ELECTRONIC_STEP = re.compile(r"^\s*(?:DAV|RMM|CGA|SDA|DMP):\s*(\d+)")
_XDATCAR_FRAME = re.compile(r"^\s*(?:Direct|Cartesian)\s+configuration\s*=", re.I)
_EXCLUDED = {"archive", "backup", ".interfaceforge", "precondition"}

# Diagnostic thresholds (see the module docstring for the warning/hard rules).
DEFAULT_STARTUP_GRACE_STEPS = 10
DEFAULT_REFERENCE_WINDOW_STEPS = 10
DEFAULT_ENERGY_JUMP_EV = 50.0
DEFAULT_CATASTROPHIC_ENERGY_EV = 500.0
SCF_HARD_FRACTION = 0.5
SCF_WARN_FRACTION = 0.2

_W_STARTUP = "startup_energy_excursion"
_W_SPIKE = "isolated_energy_spike"
_W_SCF = "scf_elevated"
_W_TEMPERATURE = "temperature_elevated"
# Warning pairs that corroborate each other into hard instability.  W1 + W3
# (startup relaxation, then some SCF-ceiling use) is deliberately absent.
_CORROBORATING_PAIRS = (
    (_W_SPIKE, _W_STARTUP),
    (_W_SPIKE, _W_SCF),
    (_W_SPIKE, _W_TEMPERATURE),
    (_W_TEMPERATURE, _W_STARTUP),
    (_W_TEMPERATURE, _W_SCF),
)


def _float_or_none(value: str) -> float | None:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_step1_oszicar(path: str | Path, *, nelm: int = 60, scf_skip_steps: int = 5) -> dict[str, Any]:
    """Return ionic stability and SCF-ceiling diagnostics from OSZICAR.

    The SCF-ceiling window skips the first ``scf_skip_steps`` MD rows (5 by
    default, the historical value); ``diagnose_step1_run`` passes its startup
    grace window so the SCF statistics cover exactly the post-grace rows.

    A final MD line that has no trailing newline and stops before its ``E0=``
    field was torn by a kill (e.g. wall time) while VASP was writing it: it is
    dropped and ``torn_final_line`` is True, so a healthy interrupted run is not
    read as having a non-numeric or runaway last step.
    """

    oszicar = Path(path)
    if not oszicar.is_file() or not oszicar.stat().st_size:
        return {
            "steps": [],
            "first_bad_step": None,
            "first_bad_reasons": [],
            "scf_ceiling_steps": 0,
            "scf_window_steps": 0,
            "scf_ceiling_fraction": None,
            "torn_final_line": False,
        }

    text = oszicar.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    torn_final_line = bool(
        lines
        and not text.endswith(("\n", "\r"))
        and _MD_STEP.match(lines[-1])
        and not _F_FIELD_COMPLETE.search(lines[-1])
    )
    if torn_final_line:
        lines.pop()
    steps: list[dict[str, Any]] = []
    electronic_max = 0
    for line in lines:
        electronic = _ELECTRONIC_STEP.match(line)
        if electronic:
            electronic_max = max(electronic_max, int(electronic.group(1)))
            continue
        ionic = _MD_STEP.match(line)
        if not ionic:
            continue
        energy = _FREE_ENERGY.search(line)
        steps.append(
            {
                "step": int(ionic.group(1)),
                "temperature_k": _float_or_none(ionic.group(2)),
                "free_energy_ev": _float_or_none(energy.group(1)) if energy else None,
                "electronic_iterations": electronic_max,
                "hit_nelm": electronic_max >= nelm,
            }
        )
        electronic_max = 0

    scf_window = steps[max(0, int(scf_skip_steps)) :]
    scf_ceiling_steps = sum(bool(row["hit_nelm"]) for row in scf_window)
    scf_fraction = scf_ceiling_steps / len(scf_window) if scf_window else None
    return {
        "steps": steps,
        "first_bad_step": None,
        "first_bad_reasons": [],
        "scf_ceiling_steps": scf_ceiling_steps,
        "scf_window_steps": len(scf_window),
        "scf_ceiling_fraction": scf_fraction,
        "torn_final_line": torn_final_line,
    }


def _format_ev(value: float) -> str:
    """Compact eV magnitude: ``110.0`` for ordinary jumps, ``2.1e+06`` for runaways."""

    return f"{value:.1f}" if abs(value) < 1e5 else f"{value:.1e}"


def _steps_text(steps: list[int], limit: int = 6) -> str:
    if len(steps) == 1:
        return f"step {steps[0]}"
    shown = ", ".join(str(step) for step in steps[:limit])
    if len(steps) > limit:
        shown += f", ... ({len(steps)} steps)"
    return f"steps {shown}"


def _incar_temperature(value: Any, default: float) -> float:
    number = _first_float(value, default)
    return number if number and math.isfinite(number) else default


def _diagnostic_settings(
    energy_jump_ev: float | None,
    max_temperature_k: float | None,
    startup_grace_steps: int | None,
    catastrophic_energy_ev: float | None,
    reference_window_steps: int | None,
) -> tuple[float, float | None, int, float, int]:
    """Validate the diagnostic options; ``None`` selects the documented default.

    Each rejected value would otherwise quietly mark healthy runs unstable (and
    so eligible for automatic repair), hence ``ValueError`` instead of a guess.
    Returns ``(energy_jump_ev, max_temperature_k or None, grace, catastrophic_ev, window)``.
    """

    grace = DEFAULT_STARTUP_GRACE_STEPS if startup_grace_steps is None else int(startup_grace_steps)
    window = DEFAULT_REFERENCE_WINDOW_STEPS if reference_window_steps is None else int(reference_window_steps)
    jump = DEFAULT_ENERGY_JUMP_EV if energy_jump_ev is None else float(energy_jump_ev)
    catastrophic = DEFAULT_CATASTROPHIC_ENERGY_EV if catastrophic_energy_ev is None else float(catastrophic_energy_ev)
    if grace < 0:
        raise ValueError("startup_grace_steps cannot be negative")
    if window < 1:
        raise ValueError("reference_window_steps must be at least 1")
    if not (math.isfinite(jump) and jump > 0):
        raise ValueError("energy_jump_ev must be a positive, finite energy in eV")
    if not (math.isfinite(catastrophic) and catastrophic >= jump):
        raise ValueError("catastrophic_energy_ev must be finite and at least energy_jump_ev")
    limit: float | None = None
    if max_temperature_k:  # None or 0 -> the default limit (max_temperature_k or max(1200, 4*T_target))
        limit = float(max_temperature_k)
        if not (math.isfinite(limit) and limit > 0):
            raise ValueError("max_temperature_k must be a positive, finite temperature in K")
    return jump, limit, grace, catastrophic, window


def _numeric_energies(rows: list[dict[str, Any]]) -> list[float]:
    return [row["free_energy_ev"] for row in rows if row["free_energy_ev"] is not None]


def _reference_free_energy(
    steps: list[dict[str, Any]], grace: int, window: int, jump: float
) -> tuple[float | None, list[dict[str, Any]], str | None]:
    """``(F_ref, rows it came from, reference_source)``; see the module docstring."""

    rows = steps[grace : grace + window]
    values = _numeric_energies(rows)
    if values:
        level = median(values)
        boundary = steps[grace - 1]["free_energy_ev"] if grace >= 1 else None
        first = rows[0]["free_energy_ev"]
        if (
            boundary is not None
            and first is not None
            and abs(boundary - first) <= jump
            and level - max(boundary, first) > jump
        ):
            # The energy rose by more than a band inside the reference window and
            # most of the window sits on the raised level: judge the run against
            # the level it left, so the rise is flagged at its onset.
            leading = 0
            for row in rows:
                energy = row["free_energy_ev"]
                if energy is None or abs(energy - first) > jump:
                    break
                leading += 1
            rows = rows[:leading]
            return median(_numeric_energies(rows)), rows, "post_grace_leading_rows"
        return level, rows, "post_grace_window"

    rows = steps[:window]
    values = _numeric_energies(rows)
    if not values:
        return None, [], None
    level = median(values)
    every = _numeric_energies(steps)
    if max(every) - min(every) > jump and all(abs(value - level) <= jump for value in every):
        # A short trajectory whose median sits between two levels, each within
        # one band of it, would hide the excursion: judge against the latest level.
        last = max(index for index, row in enumerate(steps) if row["free_energy_ev"] is not None)
        latest = steps[last]["free_energy_ev"]
        start = last
        while start >= 1:
            energy = steps[start - 1]["free_energy_ev"]
            if energy is None or abs(energy - latest) > jump:
                break
            start -= 1
        rows = steps[start : last + 1]
        return median(_numeric_energies(rows)), rows, "latest_rows"
    return level, rows, "first_rows"


def diagnose_step1_run(
    run: str | Path,
    *,
    energy_jump_ev: float = DEFAULT_ENERGY_JUMP_EV,
    max_temperature_k: float | None = None,
    startup_grace_steps: int = DEFAULT_STARTUP_GRACE_STEPS,
    catastrophic_energy_ev: float = DEFAULT_CATASTROPHIC_ENERGY_EV,
    reference_window_steps: int = DEFAULT_REFERENCE_WINDOW_STEPS,
) -> dict[str, Any]:
    """Classify the current Step1 segment as ``ok``, ``warning`` or ``unstable``.

    See the module docstring for the hard signals H1-H6, the warning classes
    W1-W4, the corroboration rule and when a startup excursion is benign.
    ``first_bad_step`` is the rewind anchor used by repair: the earliest row
    carrying a hard signal or a corroborating W1/W2/W4 warning.  It stays
    ``None`` when only the SCF statistic (H6) is hard, so repair rewinds to the
    segment start.  ``None`` for any threshold selects its default; settings
    that would make every run look unstable raise ``ValueError``.
    """

    energy_jump_ev, max_temperature_k, grace, catastrophic_energy_ev, window = _diagnostic_settings(
        energy_jump_ev, max_temperature_k, startup_grace_steps, catastrophic_energy_ev, reference_window_steps
    )
    folder = Path(run).expanduser().resolve()
    incar = parse_incar(folder / "INCAR")
    nelm = _first_int(incar.get("NELM"), 60) or 60
    parsed = parse_step1_oszicar(folder / "OSZICAR", nelm=nelm, scf_skip_steps=grace)
    steps = parsed["steps"]
    count = len(steps)

    # A TEBEG->TEEND ramp is judged against its hotter end point.
    tebeg = _incar_temperature(incar.get("TEBEG"), 300.0)
    target_temperature = max(tebeg, _incar_temperature(incar.get("TEEND"), tebeg))
    temperature_limit = max_temperature_k if max_temperature_k is not None else max(1200.0, 4.0 * target_temperature)
    temperature_warning = min(max(2.0 * target_temperature, target_temperature + 300.0), temperature_limit)

    # The reference free energy comes from just after the grace window, so the
    # magnetic DFT+U startup relaxation does not define the band that the rest
    # of the trajectory is judged against.
    reference_energy, reference_rows, reference_source = _reference_free_energy(steps, grace, window, energy_jump_ev)
    reference_window = [int(reference_rows[0]["step"]), int(reference_rows[-1]["step"])] if reference_rows else None
    deviations: list[float | None] = [
        abs(row["free_energy_ev"] - reference_energy)
        if reference_energy is not None and row["free_energy_ev"] is not None
        else None
        for row in steps
    ]
    departing = [value is not None and value > energy_jump_ev for value in deviations]
    in_band = [value is not None and value <= energy_jump_ev for value in deviations]
    # Grace rows that depart contiguously into the first post-grace row are the
    # start of a post-grace departure, not a startup excursion.
    continues_past_grace = [False] * count
    if grace < count and departing[grace]:
        index = grace - 1
        while index >= 0 and departing[index]:
            continues_past_grace[index] = True
            index -= 1

    # Per-row findings as (kind, text); kind is "hard" or a warning class.
    findings: list[list[tuple[str, str]]] = [[] for _ in steps]
    hard_rows: dict[str, list[int]] = {
        "temperature_nan": [],
        "energy_nan": [],
        "hot": [],
        "catastrophic": [],
        "sustained": [],
    }
    warning_rows: dict[str, list[int]] = {_W_STARTUP: [], _W_SPIKE: [], _W_TEMPERATURE: []}
    for index, row in enumerate(steps):
        temperature = row["temperature_k"]
        if temperature is None:
            findings[index].append(("hard", "non-numeric temperature"))
            hard_rows["temperature_nan"].append(index)
        elif temperature > temperature_limit:
            findings[index].append(("hard", f"temperature {temperature:.0f} K > {temperature_limit:.0f} K"))
            hard_rows["hot"].append(index)
        elif temperature > temperature_warning:
            findings[index].append(
                (_W_TEMPERATURE, f"temperature {temperature:.0f} K > {temperature_warning:.0f} K warning level")
            )
            warning_rows[_W_TEMPERATURE].append(index)

        deviation = deviations[index]
        if row["free_energy_ev"] is None:
            findings[index].append(("hard", "non-numeric free energy"))
            hard_rows["energy_nan"].append(index)
        elif deviation is not None and deviation > catastrophic_energy_ev:
            # Hard even inside the grace window: no startup transient is that large.
            text = f"|F-Fref|={_format_ev(deviation)} eV > {catastrophic_energy_ev:g} eV (catastrophic)"
            findings[index].append(("hard", text))
            hard_rows["catastrophic"].append(index)
        elif deviation is not None and departing[index]:
            band = f"|F-Fref|={_format_ev(deviation)} eV > {energy_jump_ev:g} eV"
            if index < grace and continues_past_grace[index]:
                findings[index].append(("hard", f"{band} (sustained past the grace window)"))
                hard_rows["sustained"].append(index)
            elif index < grace:
                findings[index].append((_W_STARTUP, f"{band} (startup excursion)"))
                warning_rows[_W_STARTUP].append(index)
            elif (index >= 1 and departing[index - 1]) or (index + 1 < count and departing[index + 1]):
                findings[index].append(("hard", f"{band} (sustained)"))
                hard_rows["sustained"].append(index)
            elif index == count - 1:
                findings[index].append(("hard", f"{band} (last recorded step)"))
                hard_rows["sustained"].append(index)
            elif in_band[index + 1] and (index == 0 or in_band[index - 1]):
                # Both neighbours in band; step 1 (grace 0) has no predecessor to contradict it.
                findings[index].append((_W_SPIKE, f"{band} (isolated spike)"))
                warning_rows[_W_SPIKE].append(index)
            else:
                # A neighbour has a non-numeric free energy, so the run is already
                # hard (H2); calling this row hard only moves the rewind anchor earlier.
                findings[index].append(("hard", f"{band} (not an isolated spike)"))
                hard_rows["sustained"].append(index)

    scf_fraction = parsed["scf_ceiling_fraction"]
    scf_unreliable = bool(scf_fraction is not None and scf_fraction >= SCF_HARD_FRACTION)
    scf_elevated = bool(scf_fraction is not None and SCF_WARN_FRACTION <= scf_fraction < SCF_HARD_FRACTION)
    scf_text = (
        f"NELM={nelm} reached on {parsed['scf_ceiling_steps']}/{parsed['scf_window_steps']} post-grace steps "
        f"({100.0 * scf_fraction:.0f}%)"
        if scf_fraction is not None
        else ""
    )

    classes = {name for name, rows in warning_rows.items() if rows}
    if scf_elevated:
        classes.add(_W_SCF)
    pairs = [pair for pair in _CORROBORATING_PAIRS if pair[0] in classes and pair[1] in classes]
    corroborating = {name for pair in pairs for name in pair}
    pair_reasons = [(pair, f"corroborated anomalies: {' + '.join(sorted(pair))}") for pair in pairs]

    # Rewind anchor: the earliest row with a hard finding or a corroborating
    # W1/W2/W4 warning (W3 is a whole-segment statistic and has no row).
    first_bad_step: int | None = None
    first_bad_reasons: list[str] = []
    for index, row_findings in enumerate(findings):
        reasons = [text for kind, text in row_findings if kind == "hard" or kind in corroborating]
        row_classes = {kind for kind, _ in row_findings if kind in corroborating}
        reasons += [text for pair, text in pair_reasons if row_classes & set(pair)]
        if reasons:
            first_bad_step = int(steps[index]["step"])
            first_bad_reasons = reasons
            break

    def step_numbers(indices: list[int]) -> list[int]:
        return [int(steps[index]["step"]) for index in indices]

    def worst_deviation(indices: list[int]) -> float:
        return max(float(deviations[index] or 0.0) for index in indices)

    hard_reasons: list[str] = []
    if hard_rows["temperature_nan"]:
        hard_reasons.append(f"non-numeric temperature at {_steps_text(step_numbers(hard_rows['temperature_nan']))}")
    if hard_rows["energy_nan"]:
        hard_reasons.append(f"non-numeric free energy at {_steps_text(step_numbers(hard_rows['energy_nan']))}")
    if hard_rows["hot"]:
        hottest = max(float(steps[index]["temperature_k"]) for index in hard_rows["hot"])
        hard_reasons.append(
            f"temperature up to {hottest:.0f} K > {temperature_limit:.0f} K at "
            f"{_steps_text(step_numbers(hard_rows['hot']))}"
        )
    if hard_rows["catastrophic"]:
        hard_reasons.append(
            f"|F-Fref| up to {_format_ev(worst_deviation(hard_rows['catastrophic']))} eV > "
            f"{catastrophic_energy_ev:g} eV (catastrophic) at {_steps_text(step_numbers(hard_rows['catastrophic']))}"
        )
    if hard_rows["sustained"]:
        hard_reasons.append(
            f"sustained post-grace energy departure: |F-Fref| up to "
            f"{_format_ev(worst_deviation(hard_rows['sustained']))} eV > {energy_jump_ev:g} eV at "
            f"{_steps_text(step_numbers(hard_rows['sustained']))}"
        )
    if scf_unreliable:
        hard_reasons.append(f"persistent SCF failure: {scf_text} >= {100.0 * SCF_HARD_FRACTION:.0f}%")
    hard_reasons += [text for _, text in pair_reasons]

    # A startup excursion is benign only when it settled back into the band and
    # relaxed downhill into it (every excursion row above F_ref).
    startup_rows = warning_rows[_W_STARTUP]
    startup_settled: bool | None = None
    startup_downhill: bool | None = None
    if startup_rows and reference_energy is not None:
        startup_settled = any(in_band[index] for index in range(startup_rows[-1] + 1, count))
        startup_downhill = all(steps[index]["free_energy_ev"] > reference_energy for index in startup_rows)

    warnings: list[str] = []
    spike_rows = warning_rows[_W_SPIKE]
    warm_rows = warning_rows[_W_TEMPERATURE]
    if startup_rows:
        text = (
            f"startup energy excursion within the {grace}-step grace window at "
            f"{_steps_text(step_numbers(startup_rows))}: |F-Fref| up to "
            f"{_format_ev(worst_deviation(startup_rows))} eV > {energy_jump_ev:g} eV"
        )
        if not startup_settled:
            text += "; not settled: no in-band step follows it yet"
        if not startup_downhill:
            text += "; the energy rose after startup and stayed up (not a downhill startup relaxation)"
        warnings.append(text)
    if spike_rows:
        warnings.append(
            f"isolated energy spike at {_steps_text(step_numbers(spike_rows))}: |F-Fref| up to "
            f"{_format_ev(worst_deviation(spike_rows))} eV > {energy_jump_ev:g} eV with both neighbours in band"
        )
    if scf_elevated:
        warnings.append(f"elevated SCF ceiling use: {scf_text}")
    if warm_rows:
        warmest = max(float(steps[index]["temperature_k"]) for index in warm_rows)
        warnings.append(
            f"temperature up to {warmest:.0f} K above the {temperature_warning:.0f} K warning level "
            f"(hard limit {temperature_limit:.0f} K) at {_steps_text(step_numbers(warm_rows))}"
        )

    # Largest step-to-step free-energy change once both rows are past the grace window.
    max_local_jump_ev: float | None = None
    max_local_jump_step: int | None = None
    for index in range(grace + 1, count):
        current = steps[index]["free_energy_ev"]
        previous = steps[index - 1]["free_energy_ev"]
        if current is None or previous is None:
            continue
        jump = abs(current - previous)
        if max_local_jump_ev is None or jump > max_local_jump_ev:
            max_local_jump_ev, max_local_jump_step = jump, int(steps[index]["step"])

    unstable = any(hard_rows.values()) or scf_unreliable or bool(pairs)
    warning_classes = sorted(classes)
    benign = warning_classes == [_W_STARTUP] and startup_settled is True and startup_downhill is True
    return {
        "run": str(folder),
        "md_steps": count,
        "last_step": steps[-1]["step"] if steps else None,
        "reference_free_energy_ev": reference_energy,
        "energy_jump_limit_ev": float(energy_jump_ev),
        "temperature_limit_k": temperature_limit,
        "first_bad_step": first_bad_step,
        "first_bad_reasons": first_bad_reasons,
        "scf_nelm": nelm,
        "scf_ceiling_steps": parsed["scf_ceiling_steps"],
        "scf_window_steps": parsed["scf_window_steps"],
        "scf_ceiling_fraction": scf_fraction,
        "scf_unreliable": scf_unreliable,
        "unstable": unstable,
        "severity": "unstable" if unstable else ("warning" if warning_classes else "ok"),
        "trajectory_stable": not unstable,
        "hard_reasons": hard_reasons,
        "warnings": warnings,
        "warning_classes": warning_classes,
        "benign_warnings_only": benign and not unstable,
        "startup_grace_steps": grace,
        "startup_excursion_settled": startup_settled,
        "startup_excursion_downhill": startup_downhill,
        "reference_window": reference_window,
        "reference_source": reference_source,
        "catastrophic_energy_limit_ev": float(catastrophic_energy_ev),
        "temperature_warning_k": temperature_warning,
        "startup_excursion_steps": step_numbers(startup_rows),
        "startup_max_excursion_ev": worst_deviation(startup_rows) if startup_rows else None,
        "isolated_spike_steps": step_numbers(spike_rows),
        "max_local_jump_ev": max_local_jump_ev,
        "max_local_jump_step": max_local_jump_step,
        "corroborated": sorted("+".join(sorted(pair)) for pair in pairs),
        "torn_final_line": bool(parsed.get("torn_final_line")),
    }


def _discover_runs(root: Path) -> list[Path]:
    if (root / "INCAR").is_file():
        return [root]
    runs: list[Path] = []
    for incar in sorted(root.rglob("INCAR")):
        parts = {part.lower() for part in incar.parent.relative_to(root).parts}
        if parts & _EXCLUDED or any(part.startswith("x") for part in parts):
            continue
        runs.append(incar.parent)
    return runs


def _xdatcar_frames(path: Path, ion_count: int) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames: list[list[str]] = []
    index = 7
    while index < len(lines):
        if not _XDATCAR_FRAME.match(lines[index]):
            index += 1
            continue
        block = lines[index + 1 : index + 1 + ion_count]
        if len(block) != ion_count:
            break
        if any(len(line.split()) < 3 for line in block):
            break
        frames.append(block)
        index += ion_count + 1
    return frames


def _poscar_layout(path: Path) -> tuple[list[str], int, int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8:
        raise SafetyError(f"POSCAR is too short: {path}")
    counts_line = 5 if all(token.isdigit() for token in lines[5].split()) else 6
    counts = lines[counts_line].split()
    if not counts or not all(token.isdigit() for token in counts):
        raise SafetyError(f"POSCAR has no valid ion-count line: {path}")
    ion_count = sum(int(token) for token in counts)
    mode_line = counts_line + 1
    if lines[mode_line].strip().lower().startswith("s"):
        mode_line += 1
    if not lines[mode_line].strip().lower().startswith(("d", "c", "k")):
        raise SafetyError(f"POSCAR has no Direct/Cartesian coordinate mode: {path}")
    return lines, ion_count, mode_line


def _write_rewind_poscar(original: Path, frame: list[str], destination: Path) -> None:
    lines, ion_count, mode_line = _poscar_layout(original)
    if len(frame) != ion_count:
        raise SafetyError(
            f"XDATCAR frame has {len(frame)} ions but POSCAR has {ion_count}: {original.parent}"
        )
    old_coordinates = lines[mode_line + 1 : mode_line + 1 + ion_count]
    rebuilt = lines[:mode_line] + ["Direct"]
    for old, new in zip(old_coordinates, frame, strict=True):
        xyz = new.split()[:3]
        flags = old.split()[3:6]
        rebuilt.append("  " + "  ".join(xyz + flags))
    destination.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")


def _precondition_launcher(run: Path) -> str | None:
    return next((name for name in ("runvasp.sh", "run.slurm") if (run / name).is_file()), None)


def _wrapped_precondition_launcher(run: Path) -> tuple[Path, str, str]:
    """``(launcher, current text, wrapped text)`` without touching the run; raises ``SafetyError``."""

    launcher_name = _precondition_launcher(run)
    if launcher_name is None:
        raise SafetyError(f"{run}: --precondition needs runvasp.sh or run.slurm")
    if not (run / "INCAR").is_file():
        raise SafetyError(f"{run}: --precondition needs the run's INCAR")
    launcher = run / launcher_name
    current = launcher.read_text(encoding="utf-8", errors="ignore")
    try:
        wrapped = wrap_launcher_with_precondition(current, launcher_name=launcher_name)
    except SafetyError as exc:
        raise SafetyError(f"{run}: {exc}") from exc
    return launcher, current, wrapped


def precondition_blocker(run: str | Path) -> str | None:
    """Why ``apply_precondition(run)`` would refuse, or ``None`` when it can proceed.

    Read-only: nothing in ``run`` is created or modified.  A launcher that is
    already wrapped (it carries the precondition marker) is not a blocker; a
    missing launcher or one without exactly one VASP line is.
    """

    try:
        _wrapped_precondition_launcher(Path(run))
    except SafetyError as exc:
        return str(exc)
    return None


def apply_precondition(run: Path) -> None:
    """Make the next segment start from a preconditioned ``WAVECAR``.

    Writes ``INCAR.precondition`` (an NSW=0 static built from the run's
    *current* INCAR, so call this after the segment INCAR is final), wraps
    ``runvasp.sh`` (else ``run.slurm``) so the static runs first, marks the
    launcher executable and removes ``WAVECAR`` so the wrapper's
    ``[ -s WAVECAR ]`` guard actually runs the preconditioner.  Wrapping is
    idempotent: an already wrapped launcher is left unchanged.  The MD INCAR is
    not touched: the caller sets ``ISTART=1`` so the MD reads the
    preconditioned ``WAVECAR``.  Raises ``SafetyError`` -- before writing
    anything -- when the run has no launcher or the launcher cannot be wrapped
    (see ``precondition_blocker`` for a read-only check).
    """

    run = Path(run)
    launcher, current, wrapped = _wrapped_precondition_launcher(run)
    system = f"{parse_incar(run / 'INCAR').get('SYSTEM', 'Step1')}_precondition"
    precondition_incar = build_precondition_incar((run / "INCAR").read_text(encoding="utf-8"), system=system)
    (run / "INCAR.precondition").write_text(precondition_incar, encoding="utf-8")
    if wrapped != current:
        launcher.write_text(wrapped, encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | 0o111)
    (run / "WAVECAR").unlink(missing_ok=True)


def _mtime_age_hours(path: Path) -> float | None:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    return (datetime.now(tz=timezone.utc) - modified).total_seconds() / 3600.0


# --------------------------------------------------------------------------- #
# Repair planning and execution
# --------------------------------------------------------------------------- #

# Plan statuses.  Only READY plans are executed; PREPARED is a READY plan after
# execution.  ACTIVE_SLURM / ACTIVE_OR_RECENT / REVIEW are never mutated and make
# ``prepare_step1_repair(execute=True)`` refuse the whole tree.
REPAIR_READY = "READY"
REPAIR_PREPARED = "PREPARED"
REPAIR_ACTIVE_SLURM = "ACTIVE_SLURM"
REPAIR_ACTIVE_OR_RECENT = "ACTIVE_OR_RECENT"
REPAIR_REVIEW = "REVIEW"

# Runtime outputs of the rewound segment that a repair removes once they are
# archived (unchanged from schema 1).  A current step1_resume.json is retired
# separately, through ``retire_current_records``.
REPAIR_RUNTIME_OUTPUTS = (
    "WAVECAR",
    "CHG",
    "CHGCAR",
    "CONTCAR",
    "XDATCAR",
    "XDATCAR_FINAL",
    "OSZICAR",
    "OUTCAR",
    "REPORT",
    "vasprun.xml",
    "vasp_md.dat",
    "vasp_md_FINAL.dat",
    ".vasp_md.dat",
    "MD_TempPlot.png",
)

_DIAGNOSTIC_OPTION_KEYS = (
    "energy_jump_ev",
    "max_temperature_k",
    "startup_grace_steps",
    "catastrophic_energy_ev",
    "reference_window_steps",
)
# Plan keys that describe the planning pass, not the prepared segment; they are
# kept out of the written step1_repair.json.
_PLAN_ONLY_KEYS = ("fingerprint", "skip_reason", "review_reasons", "active_jobs")
_ARCHIVE_STAMP = re.compile(r"_(\d{8}T\d{6}Z)$")
_BLOCKED_LINEAGE_STATUSES = ("UNREADABLE", "CONFLICT")


def _validate_repair_options(
    potim_fs: float, safety_steps: int, langevin_gamma: float | None, ramp_from: float | None
) -> None:
    if not (math.isfinite(potim_fs) and potim_fs > 0):
        raise SafetyError("repair POTIM must be positive and finite")
    if safety_steps < 0:
        raise SafetyError("safety_steps cannot be negative")
    if langevin_gamma is not None and not (math.isfinite(langevin_gamma) and langevin_gamma > 0):
        raise SafetyError("--langevin-gamma must be positive")
    if ramp_from is not None and not (math.isfinite(ramp_from) and ramp_from > 0):
        raise SafetyError("--ramp-from must be a positive temperature in K")


def _diagnostic_kwargs(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """``diagnose_step1_run`` keyword options; ``None`` values select the defaults there."""

    kwargs = dict(options or {})
    unknown = sorted(set(kwargs) - set(_DIAGNOSTIC_OPTION_KEYS))
    if unknown:
        raise ValueError(f"unknown Step1 diagnostic option(s): {', '.join(unknown)}")
    return kwargs


def _ledger_steps(ledger: list[dict[str, Any]]) -> int | None:
    total = 0
    for row in ledger:
        steps = _first_int(row.get("steps"))
        if steps is None:
            return None
        total += steps
    return total


def _active_label(jobs: list[dict[str, Any]]) -> str:
    return ", ".join(f"job {job.get('job_id')} {job.get('state')}" for job in jobs)


def plan_repair_run(
    run: str | Path,
    *,
    snapshot: SchedulerSnapshot,
    stale_hours: float | None,
    potim_fs: float = 0.5,
    algo: str = "Normal",
    safety_steps: int = 8,
    diagnostic_options: Mapping[str, Any] | None = None,
    langevin_gamma: float | None = None,
    ramp_from: float | None = None,
    precondition: bool = False,
) -> dict[str, Any] | None:
    """Plan the repair of one Step1 run, or ``None`` when it is not hard-unstable.

    Read-only: nothing in ``run`` is created, modified or removed.  The rewind
    point comes from the CURRENT segment's own OSZICAR/XDATCAR (XDATCAR frame
    ``k`` is segment ionic step ``k * NBLOCK``), ``safety_steps`` before the
    diagnostic's ``first_bad_step`` (segment step 0 when only the SCF statistic
    is hard).  The accounting is cumulative over generations
    (``step1_lineage.current_generation``): ``safe_prefix_steps =
    accepted_prefix_steps + safe_segment_steps`` and ``repair_nsw = original_nsw
    - safe_prefix_steps``, so a repair of a repair or of a resume never loses the
    accepted history.

    The plan ``status`` is ``READY``, ``ACTIVE_SLURM`` (``snapshot`` lists a job
    using the run as its WorkDir), ``ACTIVE_OR_RECENT`` (OSZICAR modified within
    ``stale_hours``; ``None`` resolves via ``resolve_stale_hours``) or
    ``REVIEW`` (``skip_reason``: unreadable/conflicting lineage, an interrupted
    earlier mutation, missing inputs, a launcher that cannot be preconditioned,
    or nothing left to run).  Only READY plans may be executed.
    """

    _validate_repair_options(potim_fs, safety_steps, langevin_gamma, ramp_from)
    folder = Path(run).expanduser().resolve()
    hours, _ = resolve_stale_hours(stale_hours, snapshot)
    # Fingerprint first: any change after this point invalidates the plan at execution.
    fingerprint = run_fingerprint(folder)
    diagnostic = diagnose_step1_run(folder, **_diagnostic_kwargs(diagnostic_options))
    if not diagnostic["unstable"]:
        return None

    incar = parse_incar(folder / "INCAR")
    schedule = incar_schedule(incar)
    nblock = int(schedule["nblock"])
    generation = current_generation(folder, incar)
    reviews: list[str] = []
    if generation.status in _BLOCKED_LINEAGE_STATUSES:
        detail = generation.conflict or f"{generation.record_path} cannot be read"
        reviews.append(f"current generation is {generation.status}: {detail}")
    interrupted = interrupted_archive(folder)
    if interrupted is not None:
        reviews.append(f"interrupted recovery mutation; inspect {interrupted}")
    missing = [
        name
        for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "OSZICAR")
        if not (folder / name).is_file() or not (folder / name).stat().st_size
    ]
    if missing:
        reviews.append(f"missing required files: {', '.join(missing)}")

    ion_count = 0
    if (folder / "POSCAR").is_file():
        try:
            _, ion_count, _ = _poscar_layout(folder / "POSCAR")
        except (SafetyError, IndexError, ValueError) as exc:
            # A malformed POSCAR sends this run to review; it must not abort the whole tree's plan.
            reviews.append(f"POSCAR cannot be parsed: {exc}")
    xdatcar = next(
        (
            candidate
            for candidate in (folder / "XDATCAR", folder / "XDATCAR_FINAL")
            if candidate.is_file() and candidate.stat().st_size
        ),
        None,
    )
    frames = _xdatcar_frames(xdatcar, ion_count) if xdatcar is not None and ion_count else []

    # Rewind point inside the current segment (segment step numbers).
    first_bad = diagnostic["first_bad_step"] or 1
    safe_step = max(0, first_bad - 1 - safety_steps)
    safe_step = (safe_step // nblock) * nblock
    safe_step = min(safe_step, len(frames) * nblock)
    rewind_frame = safe_step // nblock if safe_step else None
    source = xdatcar.name if safe_step and xdatcar is not None else "POSCAR"

    # Cumulative accounting over generations.
    prefix = int(generation.accepted_prefix_steps)
    original_nsw = generation.original_nsw
    if original_nsw is None or original_nsw <= 0:
        reviews.append(f"no positive whole-run target NSW ({generation.generation_id}; INCAR NSW={incar.get('NSW')})")
        cumulative = prefix + safe_step
        remaining: int | None = None
    else:
        cumulative = min(original_nsw, prefix + safe_step)
        remaining = original_nsw - cumulative
        if remaining <= 0:
            reviews.append(f"no ionic steps remain after the rewind (accepted {cumulative}/{original_nsw})")

    # Temperature schedule: continue the rewound segment's ramp from the rewind
    # point unless --ramp-from asks for a gentler restart.
    segment_tebeg = float(schedule["tebeg_k"])
    segment_teend = float(schedule["teend_k"])
    at_rewind = schedule_temperature(segment_tebeg, segment_teend, schedule["nsw"], safe_step)
    if ramp_from is not None:
        tebeg_text: str | None = f"{ramp_from:g}"
        repair_tebeg = float(ramp_from)
    else:
        repair_tebeg = float(format_temperature(at_rewind))
        tebeg_text = format_temperature(at_rewind) if abs(repair_tebeg - segment_tebeg) > 1e-9 else None
    # An absent TEEND defaults to TEBEG in VASP: write the endpoint whenever the
    # new segment ramps, so the original target temperature is kept.
    teend_text = None
    if not schedule["teend_explicit"] and abs(repair_tebeg - segment_teend) > 1e-9:
        teend_text = format_temperature(segment_teend)

    incar_changes: dict[str, Any] = {
        "ISTART": 1 if precondition else 0,
        "ALGO": algo,
        "POTIM": f"{potim_fs:g}",
        "NSW": remaining,
        **CONSERVATIVE_ELECTRONIC_OVERRIDES,
    }
    incar_delete = ["ICHARG"]
    if tebeg_text is not None:
        incar_changes["TEBEG"] = tebeg_text
    if teend_text is not None:
        incar_changes["TEEND"] = teend_text
    if langevin_gamma is not None:
        try:
            n_species = len(_poscar_elements(folder / "POSCAR"))
        except (OSError, SafetyError) as exc:
            reviews.append(f"--langevin needs the POSCAR species: {exc}")
            n_species = 0
        incar_changes["MDALGO"] = 3
        incar_changes["LANGEVIN_GAMMA"] = " ".join(f"{langevin_gamma:g}" for _ in range(n_species))
        incar_delete.append("SMASS")
    if precondition:
        blocker = precondition_blocker(folder)
        if blocker is not None:
            reviews.append(f"cannot precondition: {blocker}")

    # Accepted-segment ledger once this repair closes the current segment.
    segment_potim = schedule["potim_fs"]
    closed_segment = {
        "generation": generation.generation,
        "generation_id": generation.generation_id,
        "kind": generation.kind,
        "steps": safe_step,
        "potim_fs": segment_potim,
        "ps": round(safe_step * segment_potim / 1000.0, 9) if segment_potim is not None else None,
        "tebeg_k": segment_tebeg,
        "teend_k": round(at_rewind, 2),
        "restart_source": f"{source} frame {rewind_frame}" if rewind_frame else "POSCAR",
    }
    ledger = [dict(row) for row in generation.ledger] + [closed_segment]
    ledger_exact = bool(generation.ledger_exact) and segment_potim is not None and _ledger_steps(ledger) == cumulative

    new_generation = generation.generation + 1
    thermostat = "langevin (MDALGO=3)" if langevin_gamma is not None else schedule["thermostat"]
    repair_teend = segment_teend
    active_jobs = snapshot.active_jobs_for(folder)
    age_hours = _mtime_age_hours(folder / "OSZICAR")
    if active_jobs:
        status, skip_reason = REPAIR_ACTIVE_SLURM, f"active in Slurm ({_active_label(active_jobs)})"
    elif age_hours is None or age_hours < hours:
        age_text = "unknown" if age_hours is None else f"{age_hours:.2f} h ago"
        status, skip_reason = REPAIR_ACTIVE_OR_RECENT, f"OSZICAR updated {age_text} (< {hours:g} h)"
    elif reviews:
        status, skip_reason = REPAIR_REVIEW, "; ".join(reviews)
    else:
        status, skip_reason = REPAIR_READY, None

    return {
        "run": str(folder),
        "status": status,
        "skip_reason": skip_reason,
        "review_reasons": reviews,
        "active_jobs": active_jobs,
        "age_hours": age_hours,
        "diagnostic": diagnostic,
        # Schema-1 keys (identical meaning in step1_repair.json schema 2).
        "source": source,
        "safe_prefix_steps": cumulative,
        "safe_segment_steps": safe_step,
        "previous_safe_prefix_steps": prefix,
        "rewind_frame": rewind_frame,
        "original_nsw": original_nsw,
        "repair_nsw": remaining,
        "original_potim_fs": _first_float(incar.get("POTIM"), 1.0),
        "repair_potim_fs": float(potim_fs),
        "repair_algo": algo,
        "repair_electronic": dict(CONSERVATIVE_ELECTRONIC_OVERRIDES),
        "repair_langevin_gamma": langevin_gamma,
        "repair_ramp_from_k": ramp_from,
        "repair_precondition": bool(precondition),
        "archive": None,
        # Generation lineage.
        "generation": new_generation,
        "generation_id": None,
        "parent_generation": generation.generation,
        "parent_generation_id": generation.generation_id,
        "parent_segment_kind": generation.kind,
        "parent_legacy_record": bool(generation.legacy and generation.record_path is not None),
        "accepted_segments": ledger,
        "accepted_ps": accepted_ps(ledger),
        "ledger_exact": ledger_exact,
        "operation": f"step1_repair_g{new_generation}",
        # The segment being rewound and the one this repair prepares.
        "rewound_segment": {
            "tebeg_k": segment_tebeg,
            "teend_k": segment_teend,
            "nsw": schedule["nsw"],
            "potim_fs": segment_potim,
            "nblock": nblock,
            "thermostat": schedule["thermostat"],
            "ramp": bool(schedule["ramp"]),
            "temperature_at_rewind_k": round(at_rewind, 2),
        },
        "repair_tebeg_k": repair_tebeg,
        "repair_teend_k": repair_teend,
        "segment_schedule": {
            "tebeg_k": repair_tebeg,
            "teend_k": repair_teend,
            "nsw": remaining,
            "thermostat": thermostat,
            "ramp": abs(repair_tebeg - repair_teend) > 1e-9,
        },
        "incar_changes": incar_changes,
        "incar_delete": incar_delete,
        "fingerprint": fingerprint,
    }


def execute_repair_plan(
    plan: Mapping[str, Any], *, guard: SchedulerGuard, ledger_roots: Iterable[str | Path] = ()
) -> dict[str, Any]:
    """Prepare the repair segment described by a READY ``plan``; returns the updated plan.

    Every check that can refuse runs before the first write: plan status, the
    planning fingerprint, required inputs, the lineage the plan was built on,
    the rewind frame, the launcher (when preconditioning) and, last and
    immediately before mutating, ``guard.assert_inactive(run)`` followed by a
    second fingerprint comparison (a squeue call can take seconds).  Then:
    ``archive_step1_state(run, "step1_repair_g<N>")``; POSCAR <- the XDATCAR
    rewind frame, or the segment-start POSCAR WITHOUT its velocity block when
    rewinding to segment step 0; runtime outputs removed; INCAR updated (and
    the launcher preconditioned after it); the previous segment record retired
    and its launch rows sealed; a schema-2 ``step1_repair.json`` written; the
    archive finalized.  An exception after archiving leaves the archive
    manifest ``IN_PROGRESS`` so the interrupted mutation stays discoverable.
    """

    if not plan.get("run"):
        raise SafetyError("Refusing to execute a repair plan that names no run directory")
    run = Path(str(plan["run"])).expanduser()
    if plan.get("status") != REPAIR_READY:
        raise SafetyError(
            f"Refusing to execute the repair plan for {run}: status is {plan.get('status')!r}, not READY"
            + (f" ({plan.get('skip_reason')})" if plan.get("skip_reason") else "")
        )
    if run_fingerprint(run) != plan.get("fingerprint"):
        raise SafetyError(f"Refusing to mutate {run}: it changed since planning (file fingerprint differs); plan again")
    require_files(run, ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "OSZICAR"))
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

    safe_step = int(plan["safe_segment_steps"])
    frame: list[str] | None = None
    if safe_step:
        nblock = int(incar_schedule(parse_incar(run / "INCAR"))["nblock"])
        _, ion_count, _ = _poscar_layout(run / "POSCAR")
        xdatcar = run / str(plan["source"])
        frames = _xdatcar_frames(xdatcar, ion_count) if xdatcar.is_file() else []
        index = safe_step // nblock
        if safe_step % nblock or not 1 <= index <= len(frames):
            raise SafetyError(
                f"Refusing to mutate {run}: {xdatcar.name} has {len(frames)} frame(s) at NBLOCK={nblock}, "
                f"so segment step {safe_step} cannot be restored"
            )
        frame = frames[index - 1]
    precondition = bool(plan.get("repair_precondition"))
    if precondition:
        blocker = precondition_blocker(run)
        if blocker is not None:
            raise SafetyError(f"Refusing to mutate {run}: cannot precondition: {blocker}")
    incar_changes = dict(plan["incar_changes"])
    if incar_changes.get("NSW") is None or int(incar_changes["NSW"]) <= 0:
        raise SafetyError(f"Refusing to mutate {run}: the repair segment would have NSW={incar_changes.get('NSW')}")

    # ---- mutation starts here (Slurm and the fingerprint re-checked immediately before) ----
    guard.assert_inactive(run)
    if run_fingerprint(run) != plan.get("fingerprint"):
        # A file changed while the scheduler was being queried.
        raise SafetyError(f"Refusing to mutate {run}: it changed since planning (file fingerprint differs); plan again")
    archive = archive_step1_state(run, str(plan["operation"]))
    if frame is not None:
        _write_rewind_poscar(archive / "POSCAR", frame, run / "POSCAR")
    else:
        # Segment step 0: its POSCAR may carry a CONTCAR's velocity and
        # predictor-corrector blocks, which belong to the discarded dynamics.
        _write_poscar_without_velocities(archive / "POSCAR", run / "POSCAR")
    for name in REPAIR_RUNTIME_OUTPUTS:
        (run / name).unlink(missing_ok=True)
    update_incar(run / "INCAR", incar_changes, delete=list(plan.get("incar_delete") or ()))
    if precondition:
        apply_precondition(run)  # after update_incar: INCAR.precondition derives from the final MD INCAR

    retired = retire_current_records(run, archive)
    stamp_match = _ARCHIVE_STAMP.search(archive.name)
    generation = int(plan["generation"])
    generation_id = new_generation_id("repair", generation, stamp_match.group(1) if stamp_match else None)
    sealed = seal_launch_rows(
        run,
        ledger_paths_for(run, [Path(root) for root in ledger_roots]),
        retired_generation_id=parent.generation_id,
        new_generation_id=generation_id,
    )
    prepared_at = utc_now_iso()
    extra = {key: value for key, value in plan.items() if key not in _PLAN_ONLY_KEYS}
    extra.update({"retired_records": retired, "sealed_ledgers": sealed})
    record = build_segment_record(
        "repair",
        run=run.resolve(),
        generation=generation,
        generation_id=generation_id,
        parent=parent,
        prepared_at=prepared_at,
        original_nsw=int(plan["original_nsw"]),
        accepted_prefix_steps=int(plan["safe_prefix_steps"]),
        accepted_segments=list(plan["accepted_segments"]),
        ledger_exact=bool(plan["ledger_exact"]),
        segment_nsw=int(plan["repair_nsw"]),
        segment_potim_fs=float(plan["repair_potim_fs"]),
        segment_schedule=dict(plan["segment_schedule"]),
        archive=str(archive),
        extra=extra,
    )
    atomic_write_json(run / REPAIR_RECORD, record)
    finalize_archive(archive, generation_id=generation_id)

    updated = dict(plan)
    updated.update(
        {
            "status": REPAIR_PREPARED,
            "archive": str(archive),
            "generation_id": generation_id,
            "prepared_at": prepared_at,
            "retired_records": retired,
            "sealed_ledgers": sealed,
        }
    )
    return updated


def _run_label(root: Path, run: str | Path) -> str:
    try:
        relative = Path(run).relative_to(root).as_posix()
    except ValueError:
        return str(run)
    return relative if relative != "." else Path(run).name


def _refuse_blocked_tree(root: Path, plans: list[dict[str, Any]], stale_hours: float) -> None:
    """Tree-level rule: never partially mutate while any unstable run is not READY."""

    recent = [plan for plan in plans if plan["status"] == REPAIR_ACTIVE_OR_RECENT]
    active = [plan for plan in plans if plan["status"] == REPAIR_ACTIVE_SLURM]
    review = [plan for plan in plans if plan["status"] == REPAIR_REVIEW]
    reasons: list[str] = []
    if active:
        labels = ", ".join(f"{_run_label(root, plan['run'])} ({_active_label(plan['active_jobs'])})" for plan in active)
        reasons.append(f"unstable runs are active in Slurm: {labels}")
    if recent:
        labels = ", ".join(_run_label(root, plan["run"]) for plan in recent)
        reasons.append(f"unstable runs are still active/recent (<{stale_hours:g} h): {labels}")
    if review:
        labels = "; ".join(f"{_run_label(root, plan['run'])}: {plan['skip_reason']}" for plan in review)
        reasons.append(f"unstable runs need review before repair: {labels}")
    if reasons:
        raise SafetyError("Refusing to partially mutate the tree because " + "; ".join(reasons))


def prepare_step1_repair(
    root: str | Path,
    *,
    execute: bool = False,
    stale_hours: float | None = None,
    scheduler: str | SchedulerGuard = "auto",
    potim_fs: float = 0.5,
    algo: str = "Normal",
    safety_steps: int = 8,
    energy_jump_ev: float = DEFAULT_ENERGY_JUMP_EV,
    max_temperature_k: float | None = None,
    langevin_gamma: float | None = None,
    ramp_from: float | None = None,
    precondition: bool = False,
    startup_grace_steps: int = DEFAULT_STARTUP_GRACE_STEPS,
    catastrophic_energy_ev: float = DEFAULT_CATASTROPHIC_ENERGY_EV,
) -> dict[str, Any]:
    """Plan or prepare bounded recovery segments for unstable, inactive runs.

    The recovery segment always tightens the electronic loop
    (``EDIFF=1E-5``, ``NELM=120``, ``NELMIN=6``) on top of ``ALGO=algo`` and
    ``POTIM=potim_fs`` -- the crashes are driven by forces read off a
    sloshing SCF, not the timestep alone. ``langevin_gamma`` swaps
    ``SMASS=-1`` for a Langevin thermostat (``MDALGO=3``); ``ramp_from`` sets
    a lower initial ``TEBEG`` so the rewound geometry re-thermalises gently
    (without it, ``TEBEG`` continues the rewound segment's schedule from the
    rewind point, which leaves a constant-temperature segment unchanged).
    ``precondition`` writes an ``INCAR.precondition`` (NSW=0 static) and
    rewraps the launcher so the recovery MD restarts from a converged
    ``WAVECAR`` instead of the atomic-density guess.

    Dry run (the default) reads files and asks the scheduler, nothing else.
    ``scheduler`` is a ``--scheduler`` mode or a shared ``SchedulerGuard``;
    ``stale_hours=None`` resolves to 0.1 h when Slurm is verified and 6 h
    otherwise.  With ``execute=True`` the whole tree is refused while any
    unstable run is active in Slurm, recently modified or needs review;
    otherwise each READY run is prepared by ``execute_repair_plan``, stopping
    at the first failure with a ``SafetyError`` that names the runs already
    prepared.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    _validate_repair_options(potim_fs, safety_steps, langevin_gamma, ramp_from)
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
    for run in _discover_runs(root_path):
        plan = plan_repair_run(
            run,
            snapshot=snapshot,
            stale_hours=hours,
            potim_fs=potim_fs,
            algo=algo,
            safety_steps=safety_steps,
            diagnostic_options=diagnostic_options,
            langevin_gamma=langevin_gamma,
            ramp_from=ramp_from,
            precondition=precondition,
        )
        if plan is not None:
            plans.append(plan)

    if execute:
        _refuse_blocked_tree(root_path, plans, hours)
        prepared: list[str] = []
        for index, plan in enumerate(plans):
            try:
                plans[index] = execute_repair_plan(plan, guard=guard, ledger_roots=(root_path,))
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
                    f"step1-repair stopped at {_run_label(root_path, plan['run'])}: {exc}; {state}. "
                    f"Already prepared: {done}. Not attempted: {untouched}."
                ) from exc
            prepared.append(plan["run"])

    return {
        "format": "interfaceforge-step1-repair-plan",
        "schema_version": 1,
        "mode": "prepared" if execute else "dry-run",
        "root": str(root_path),
        "scheduler": snapshot.to_dict(),
        "settings": {
            "stale_hours": hours,
            "stale_hours_requested": stale_hours,
            "stale_hours_reason": hours_reason,
            "potim_fs": potim_fs,
            "algo": algo,
            "electronic_overrides": dict(CONSERVATIVE_ELECTRONIC_OVERRIDES),
            "langevin_gamma": langevin_gamma,
            "ramp_from_k": ramp_from,
            "precondition": bool(precondition),
            "safety_steps": safety_steps,
            "energy_jump_ev": energy_jump,
            "max_temperature_k": max_temperature_k,
            "startup_grace_steps": grace,
            "catastrophic_energy_ev": catastrophic,
        },
        "runs": plans,
        "repairable": sum(row["status"] in {REPAIR_READY, REPAIR_PREPARED} for row in plans),
        "skipped_active_or_recent": sum(
            row["status"] in {REPAIR_ACTIVE_OR_RECENT, REPAIR_ACTIVE_SLURM} for row in plans
        ),
        "skipped_review": sum(row["status"] == REPAIR_REVIEW for row in plans),
    }
