"""Diagnose and safely prepare recovery segments for unstable Step1 AIMD.

The recovery is deliberately dry-run first.  It never continues from the
current CONTCAR because a numerically unstable MD step may already have put
ions on top of one another.  Instead it rewinds to an XDATCAR frame before
the first energy/temperature runaway, starts the electronic state afresh,
and runs only the number of ionic steps still needed to reach the original
Step1 target.

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

import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .aimd import _first_float, _first_int
from .errors import SafetyError
from .vasp import (
    CONSERVATIVE_ELECTRONIC_OVERRIDES,
    _poscar_elements,
    archive_run,
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


def prepare_step1_repair(
    root: str | Path,
    *,
    execute: bool = False,
    stale_hours: float = 6.0,
    potim_fs: float = 0.5,
    algo: str = "Normal",
    safety_steps: int = 8,
    energy_jump_ev: float = 50.0,
    max_temperature_k: float | None = None,
    langevin_gamma: float | None = None,
    ramp_from: float | None = None,
    precondition: bool = False,
) -> dict[str, Any]:
    """Plan or prepare bounded recovery segments for unstable, inactive runs.

    The recovery segment always tightens the electronic loop
    (``EDIFF=1E-5``, ``NELM=120``, ``NELMIN=6``) on top of ``ALGO=algo`` and
    ``POTIM=potim_fs`` -- the crashes are driven by forces read off a
    sloshing SCF, not the timestep alone. ``langevin_gamma`` swaps
    ``SMASS=-1`` for a Langevin thermostat (``MDALGO=3``); ``ramp_from`` sets
    a lower initial ``TEBEG`` so the rewound geometry re-thermalises gently.
    ``precondition`` writes an ``INCAR.precondition`` (NSW=0 static) and
    rewraps the launcher so the recovery MD restarts from a converged
    ``WAVECAR`` instead of the atomic-density guess.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    if potim_fs <= 0 or not math.isfinite(potim_fs):
        raise SafetyError("repair POTIM must be positive and finite")
    if safety_steps < 0:
        raise SafetyError("safety_steps cannot be negative")
    if langevin_gamma is not None and langevin_gamma <= 0:
        raise SafetyError("--langevin-gamma must be positive")
    if ramp_from is not None and ramp_from <= 0:
        raise SafetyError("--ramp-from must be a positive temperature in K")

    plans: list[dict[str, Any]] = []
    for run in _discover_runs(root_path):
        require_files(run, ("INCAR", "POSCAR", "OSZICAR"))
        incar = parse_incar(run / "INCAR")
        target_nsw = _first_int(incar.get("NSW"))
        nblock = _first_int(incar.get("NBLOCK"), 1) or 1
        if target_nsw is None or target_nsw <= 0:
            raise SafetyError(f"{run}/INCAR has no positive NSW")

        previous_repair: dict[str, Any] = {}
        previous_repair_path = run / "step1_repair.json"
        if previous_repair_path.is_file():
            try:
                previous_repair = json.loads(previous_repair_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous_repair = {}
        previous_prefix_steps = _first_int(previous_repair.get("safe_prefix_steps"), 0) or 0
        original_target_nsw = (
            _first_int(previous_repair.get("original_nsw"), target_nsw) or target_nsw
        )
        diagnostic = diagnose_step1_run(
            run,
            energy_jump_ev=energy_jump_ev,
            max_temperature_k=max_temperature_k,
        )
        if not diagnostic["unstable"]:
            continue
        age_hours = _mtime_age_hours(run / "OSZICAR")
        inactive = age_hours is not None and age_hours >= stale_hours
        first_bad = diagnostic["first_bad_step"] or 1
        safe_step = max(0, first_bad - 1 - safety_steps)
        safe_step = (safe_step // nblock) * nblock

        xdatcar = next(
            (
                candidate
                for candidate in (run / "XDATCAR", run / "XDATCAR_FINAL")
                if candidate.is_file() and candidate.stat().st_size
            ),
            None,
        )
        _, ion_count, _ = _poscar_layout(run / "POSCAR")
        frames = _xdatcar_frames(xdatcar, ion_count) if xdatcar is not None else []
        safe_step = min(safe_step, len(frames) * nblock)
        safe_step = (safe_step // nblock) * nblock
        cumulative_safe_step = min(original_target_nsw, previous_prefix_steps + safe_step)
        remaining = original_target_nsw - cumulative_safe_step
        plan = {
            "run": str(run),
            "status": "READY" if inactive else "ACTIVE_OR_RECENT",
            "age_hours": age_hours,
            "diagnostic": diagnostic,
            "source": "POSCAR" if safe_step == 0 else xdatcar.name,
            "safe_prefix_steps": cumulative_safe_step,
            "safe_segment_steps": safe_step,
            "previous_safe_prefix_steps": previous_prefix_steps,
            "rewind_frame": safe_step // nblock if safe_step else None,
            "original_nsw": original_target_nsw,
            "repair_nsw": remaining,
            "original_potim_fs": _first_float(incar.get("POTIM"), 1.0),
            "repair_potim_fs": float(potim_fs),
            "repair_algo": algo,
            "repair_electronic": dict(CONSERVATIVE_ELECTRONIC_OVERRIDES),
            "repair_langevin_gamma": langevin_gamma,
            "repair_ramp_from_k": ramp_from,
            "repair_precondition": bool(precondition),
            "archive": None,
        }
        plans.append(plan)
    if execute:
        recent = [row for row in plans if row["status"] == "ACTIVE_OR_RECENT"]
        if recent:
            labels = ", ".join(Path(row["run"]).name for row in recent)
            raise SafetyError(
                "Refusing to partially mutate the tree because unstable runs are still "
                f"active/recent (<{stale_hours:g} h): {labels}"
            )

    for plan in (plans if execute else []):
        run = Path(plan["run"])
        require_files(run, ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "OSZICAR"))
        safe_segment_step = int(plan.get("safe_segment_steps", plan["safe_prefix_steps"]))
        nblock = _first_int(parse_incar(run / "INCAR").get("NBLOCK"), 1) or 1
        xdatcar = run / str(plan["source"])
        _, ion_count, _ = _poscar_layout(run / "POSCAR")
        frames = _xdatcar_frames(xdatcar, ion_count) if safe_segment_step else []
        archive = archive_run(run, "step1_repair")
        if safe_segment_step:
            frame = frames[safe_segment_step // nblock - 1]
            _write_rewind_poscar(archive / "POSCAR", frame, run / "POSCAR")
        else:
            shutil.copy2(archive / "POSCAR", run / "POSCAR")
        for name in (
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
        ):
            (run / name).unlink(missing_ok=True)
        incar_changes: dict[str, Any] = {
            "ISTART": 1 if precondition else 0,
            "ALGO": algo,
            "POTIM": f"{potim_fs:g}",
            "NSW": plan["repair_nsw"],
            **CONSERVATIVE_ELECTRONIC_OVERRIDES,
        }
        incar_delete = {"ICHARG"}
        if ramp_from is not None:
            incar_changes["TEBEG"] = f"{ramp_from:g}"
        if langevin_gamma is not None:
            n_species = len(_poscar_elements(run / "POSCAR"))
            incar_changes["MDALGO"] = 3
            incar_changes["LANGEVIN_GAMMA"] = " ".join(
                f"{langevin_gamma:g}" for _ in range(n_species)
            )
            incar_delete.add("SMASS")
        update_incar(run / "INCAR", incar_changes, delete=incar_delete)
        if precondition:
            apply_precondition(run)
        plan["repair_precondition"] = bool(precondition)
        repair_record = {
            "format": "interfaceforge-step1-repair",
            "schema_version": 1,
            **plan,
            "archive": str(archive),
            "status": "PREPARED",
        }
        (run / "step1_repair.json").write_text(
            json.dumps(repair_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plan["archive"] = str(archive)
        plan["status"] = "PREPARED"

    return {
        "format": "interfaceforge-step1-repair-plan",
        "schema_version": 1,
        "mode": "prepared" if execute else "dry-run",
        "root": str(root_path),
        "settings": {
            "stale_hours": stale_hours,
            "potim_fs": potim_fs,
            "algo": algo,
            "electronic_overrides": dict(CONSERVATIVE_ELECTRONIC_OVERRIDES),
            "langevin_gamma": langevin_gamma,
            "ramp_from_k": ramp_from,
            "precondition": bool(precondition),
            "safety_steps": safety_steps,
            "energy_jump_ev": energy_jump_ev,
            "max_temperature_k": max_temperature_k,
        },
        "runs": plans,
        "repairable": sum(row["status"] in {"READY", "PREPARED"} for row in plans),
        "skipped_active_or_recent": sum(row["status"] == "ACTIVE_OR_RECENT" for row in plans),
    }
