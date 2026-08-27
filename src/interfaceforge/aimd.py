"""AIMD protocol tooling: ``academic`` production vs ``training`` MLIP data.

InterfaceForge historically used one AIMD default for both purposes: a long
hand-curated Step1 preheat (~2 ps) followed by a dense Step2 frame
retention (every ``NBLOCK``-th step, e.g. 750 of 3000). That is right for
publication-oriented production runs and wrong for MLIP training-data
generation, where

* frames a few fs apart in one trajectory are strongly autocorrelated, so
  keeping hundreds of them just resamples one thermal basin, and
* a long discard-equilibration burn-in throws away exactly the
  transient/reactive configurations (an anchor first contacting the
  surface, a proton transfer) that matter most for training.

This module adds an explicit ``training`` profile beside the existing
``academic`` one **without changing any default behavior**:

* :func:`switch_step1_protocol` retargets a Step1 preheat INCAR's length
  and audits the result in one step.
* :func:`plan_step2_retention` / :func:`sample_step2_runs` space kept Step2
  frames at the *measured* total-energy decorrelation time instead of a
  fixed stride, targeting tens of frames per short trajectory.

The recommended use of a fixed AIMD step budget in ``training`` mode is
many short independent trajectories (different seeds / starting configs),
not a few long ones; this module only makes the per-trajectory defaults
correct so that pattern is the natural one to follow.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SafetyError
from .vasp import parse_incar, update_incar

DEFAULT_PROTOCOL = "academic"

AIMD_PROTOCOLS: dict[str, dict[str, Any]] = {
    "academic": {
        "summary": "Publication-oriented production AIMD (unchanged historical default).",
        "step1": {
            "purpose": "Full forced thermalization from the optimized AFM electronic state.",
            "nsw": 2000,
            "preheat_ps_expected": [1.0, 4.0],
        },
        "step2": {
            "retention_method": "fixed-nblock-stride",
            "retention_summary": "Keep every NBLOCK-th Step2 frame (dense, on purpose).",
        },
    },
    "training": {
        "summary": (
            "MLIP training-data generation: short burn-in, decorrelation-spaced "
            "frames, many short independent trajectories."
        ),
        "step1": {
            "purpose": (
                "Minimal burn-in only; the geometry is already pre-relaxed "
                "(classical + VASP+U opt) before AIMD starts."
            ),
            "nsw": 250,
            "preheat_ps_expected": [0.05, 0.4],
        },
        "step2": {
            "retention_method": "energy-autocorrelation",
            "retention_summary": (
                "Space kept frames at the measured total-energy decorrelation time."
            ),
            "burn_in_ps": 0.15,
            "target_frames": [15, 40],
        },
    },
}


def resolve_protocol(name: str) -> dict[str, Any]:
    """Return the profile for ``name`` or raise a clear :class:`SafetyError`."""

    try:
        return AIMD_PROTOCOLS[name]
    except KeyError:
        raise SafetyError(
            f"Unknown AIMD protocol {name!r}; choose from {sorted(AIMD_PROTOCOLS)}"
        ) from None


def _first_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def _first_int(value: Any, default: int | None = None) -> int | None:
    number = _first_float(value)
    return int(number) if number is not None else default


def preheat_ps(nsw: int | None, potim_fs: float | None) -> float | None:
    """Preheat duration in ps from ``NSW`` steps at ``POTIM`` fs/step."""

    if not nsw or potim_fs is None:
        return None
    return nsw * potim_fs / 1000.0


# --------------------------------------------------------------------------- #
# Step 1: preheat length switch + audit
# --------------------------------------------------------------------------- #

_STEP1_EXCLUDED = ("archive", "backup")


def audit_step1_incar(incar_path: str | Path, protocol: str) -> dict[str, Any]:
    """Audit a Step1 preheat INCAR against the requested protocol.

    ``issues`` (blocking) cover an INCAR that is not a fixed-temperature MD
    preheat at all; ``notes`` (non-blocking) cover a preheat whose length or
    restart hygiene does not match the protocol. The status is ``FAIL`` on
    any issue, ``WARN`` on notes only, otherwise ``PASS``.
    """

    profile = resolve_protocol(protocol)
    incar = Path(incar_path)
    parsed = parse_incar(incar)
    nsw = _first_int(parsed.get("NSW"))
    potim = _first_float(parsed.get("POTIM"), 1.0)
    ps = preheat_ps(nsw, potim)
    lo, hi = profile["step1"]["preheat_ps_expected"]

    issues: list[str] = []
    notes: list[str] = []
    if not incar.is_file():
        issues.append(f"no INCAR at {incar}")
    if parsed.get("IBRION") not in {None, "0"} and "IBRION" in parsed:
        issues.append(f"IBRION={parsed['IBRION']!r}; a preheat needs IBRION=0 (MD)")
    if nsw is None or nsw <= 0:
        issues.append(f"NSW={parsed.get('NSW')!r}; a preheat needs NSW>0")

    tebeg = _first_float(parsed.get("TEBEG"))
    teend = _first_float(parsed.get("TEEND"))
    if tebeg is not None and teend is not None and abs(tebeg - teend) > 1e-6:
        notes.append(f"TEBEG={tebeg} != TEEND={teend}; a fixed-T preheat usually sets them equal")
    if "SMASS" not in parsed and "MDALGO" not in parsed:
        notes.append("no SMASS or MDALGO: preheat has no thermostat / velocity control")
    if ps is not None and not (lo <= ps <= hi):
        notes.append(
            f"preheat {ps:.2f} ps is outside the {protocol} range {lo}-{hi} ps "
            f"(NSW={nsw}, POTIM={potim})"
        )
    if protocol == "academic" and parsed.get("LWAVE", "").upper() in {".FALSE.", "F", "FALSE"}:
        notes.append("LWAVE=.FALSE.: Step2 cannot restart the wavefunction from this preheat")

    status = "FAIL" if issues else ("WARN" if notes else "PASS")
    return {
        "incar": str(incar),
        "protocol": protocol,
        "status": status,
        "nsw": nsw,
        "potim_fs": potim,
        "preheat_ps": round(ps, 3) if ps is not None else None,
        "preheat_ps_expected": [lo, hi],
        "thermostat": parsed.get("SMASS") or parsed.get("MDALGO"),
        "issues": issues,
        "notes": notes,
    }


def _discover_step1_incars(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if (target / "INCAR").is_file():
        return [target / "INCAR"]
    found: list[Path] = []
    for incar in sorted(target.rglob("INCAR")):
        relative = incar.parent.relative_to(target)
        parts = {part.lower() for part in relative.parts}
        if parts & set(_STEP1_EXCLUDED) or any(p.startswith("x") for p in parts):
            continue
        found.append(incar)
    return found


def _switch_one(
    incar: Path, protocol: str, *, nsw: int | None, audit_only: bool, create: bool
) -> dict[str, Any]:
    profile = resolve_protocol(protocol)
    target_nsw = int(nsw) if nsw is not None else int(profile["step1"]["nsw"])
    if target_nsw <= 0:
        raise SafetyError("Step1 preheat NSW must be positive")

    before = audit_step1_incar(incar, protocol) if incar.is_file() else None
    changed = False
    if not audit_only:
        current = parse_incar(incar)
        if current.get("NSW") != str(target_nsw):
            update_incar(incar, {"NSW": target_nsw}, create=create)
            changed = True
    after = audit_step1_incar(incar, protocol)
    return {
        "incar": str(incar),
        "nsw_before": before.get("nsw") if before else None,
        "nsw_applied": target_nsw,
        "changed": changed,
        "status": after["status"],
        "audit": after,
    }


def switch_step1_protocol(
    target: str | Path,
    protocol: str = DEFAULT_PROTOCOL,
    *,
    nsw: int | None = None,
    audit_only: bool = False,
    create: bool = False,
) -> dict[str, Any]:
    """Retarget one or more Step1 preheat INCARs to ``protocol`` and audit them.

    ``target`` may be a single INCAR, a Step1 run directory, or a Step1 tree
    root (every ``INCAR`` below it is switched, skipping ``archive``/
    ``backup``/``X*`` paths). Only ``NSW`` is rewritten -- every other
    preheat choice stays exactly as curated. ``audit_only`` reports without
    writing.
    """

    profile = resolve_protocol(protocol)
    root = Path(target).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    incars = _discover_step1_incars(root)
    if not incars:
        raise SafetyError(f"No Step1 INCAR found at or below {root}")

    results = [
        _switch_one(incar, protocol, nsw=nsw, audit_only=audit_only, create=create)
        for incar in incars
    ]
    order = {"FAIL": 2, "WARN": 1, "PASS": 0}
    overall = max((r["status"] for r in results), key=lambda s: order[s])
    return {
        "mode": "audited" if audit_only else "switched",
        "protocol": protocol,
        "purpose": profile["step1"]["purpose"],
        "preheat_ps_expected": profile["step1"]["preheat_ps_expected"],
        "target": str(root),
        "incars": len(results),
        "changed": sum(r["changed"] for r in results),
        "status": overall,
        "runs": results,
    }


# --------------------------------------------------------------------------- #
# Step 2: decorrelation-based frame retention
# --------------------------------------------------------------------------- #

_OSZICAR_MD_LINE = re.compile(
    r"^\s*\d+\s+T=\s*[-+0-9.Ee]+\s+E=\s*[-+0-9.Ee]+\s+F=\s*([-+0-9.Ee]+)\s+E0=\s*([-+0-9.Ee]+)"
)


def read_md_energy_series(run: str | Path) -> list[float]:
    """Per-ionic-step total energy (``E0``, sigma->0) from a full OSZICAR."""

    run = Path(run)
    oszicar = run / "OSZICAR" if run.is_dir() else run
    if not oszicar.is_file():
        return []
    series: list[float] = []
    for line in oszicar.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _OSZICAR_MD_LINE.match(line)
        if not match:
            continue
        value = _first_float(match.group(2))
        if value is None:
            value = _first_float(match.group(1))
        if value is not None:
            series.append(value)
    return series


def integrated_autocorrelation_time(series: Sequence[float], *, c: float = 5.0) -> float:
    """Integrated autocorrelation time (in frames) with Sokal automatic windowing.

    ``tau = 1 + 2 * sum_t rho(t)`` truncated at the smallest window ``M``
    with ``M >= c * tau(M)``. Returns at least 1.0 (an uncorrelated series).
    """

    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 8:
        return 1.0
    x = x - x.mean()
    if float(np.dot(x, x)) <= 0.0:
        return 1.0

    size = 1
    while size < 2 * n:
        size *= 2
    spectrum = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:n].real
    acf /= acf[0]

    taus = 1.0 + 2.0 * np.cumsum(acf[1:])
    window = len(taus)
    for w in range(1, len(taus) + 1):
        if w >= c * taus[w - 1]:
            window = w
            break
    tau = float(taus[min(window, len(taus)) - 1])
    return max(tau, 1.0)


def select_decorrelated_frames(
    energies: Sequence[float],
    *,
    burn_in_frames: int,
    target_min: int,
    target_max: int,
) -> dict[str, Any]:
    """Drop a burn-in, then keep frames spaced ~one decorrelation time apart.

    The stride starts at ``round(tau)`` and is nudged so the kept count lands
    in ``[target_min, target_max]`` whenever the usable trajectory is long
    enough to allow it.
    """

    total = len(energies)
    burn_in_frames = max(0, min(int(burn_in_frames), max(total - target_min, 0)))
    usable = list(range(burn_in_frames, total))
    tail = [energies[i] for i in usable]
    tau = integrated_autocorrelation_time(tail)
    stride = max(1, int(round(tau)))

    def kept(step: int) -> int:
        return len(usable[::step])

    while kept(stride) > target_max:
        stride += 1
    while stride > 1 and kept(stride - 1) <= target_max:
        stride -= 1

    indices = usable[::stride]
    return {
        "trajectory_frames": total,
        "burn_in_frames": burn_in_frames,
        "autocorrelation_time_frames": round(tau, 3),
        "stride": stride,
        "kept_frames": len(indices),
        "target_frames": [target_min, target_max],
        "within_target": target_min <= len(indices) <= target_max,
        "indices": indices,
    }


def plan_step2_retention(
    protocol: str,
    energies: Sequence[float],
    *,
    potim_fs: float,
    nblock: int,
) -> dict[str, Any]:
    """Frame-retention plan for one finished Step2 trajectory under ``protocol``."""

    profile = resolve_protocol(protocol)
    total = len(energies)
    if protocol == "academic":
        stride = max(1, int(nblock or 1))
        indices = list(range(0, total, stride))
        return {
            "protocol": protocol,
            "method": "fixed-nblock-stride",
            "trajectory_frames": total,
            "burn_in_frames": 0,
            "autocorrelation_time_frames": None,
            "stride": stride,
            "kept_frames": len(indices),
            "within_target": None,
            "indices": indices,
        }

    step2 = profile["step2"]
    burn_in_frames = int(round((step2["burn_in_ps"] * 1000.0) / (potim_fs or 1.0)))
    plan = select_decorrelated_frames(
        energies,
        burn_in_frames=burn_in_frames,
        target_min=step2["target_frames"][0],
        target_max=step2["target_frames"][1],
    )
    plan.update(
        {
            "protocol": protocol,
            "method": step2["retention_method"],
            "burn_in_ps": step2["burn_in_ps"],
        }
    )
    return plan


def _sample_tsv(rows: Iterable[dict[str, Any]]) -> str:
    fields = (
        "status",
        "relative_path",
        "trajectory_frames",
        "burn_in_frames",
        "autocorrelation_time_frames",
        "stride",
        "kept_frames",
        "within_target",
    )
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(name, "")) for name in fields))
    return "\n".join(lines) + "\n"


def sample_step2_runs(
    roots: Iterable[str | Path], *, dry_run: bool = False
) -> dict[str, Any]:
    """Apply each Step2 root's recorded protocol to its finished trajectories.

    Reads ``step2_manifest.json`` for the protocol and ``NBLOCK``, then for
    every manifest run with MD steps in ``OSZICAR`` computes the retention
    plan and (unless ``dry_run``) writes ``step2_sample.json`` /
    ``step2_sample.tsv`` with per-run kept-frame counts, the decorrelation
    time, and the selected frame indices.
    """

    resolved = [Path(value).expanduser().resolve() for value in roots]
    if not resolved:
        raise SafetyError("At least one Step2 root is required")

    summaries: list[dict[str, Any]] = []
    for root in resolved:
        if not root.is_dir():
            raise FileNotFoundError(root)
        manifest_path = root / "step2_manifest.json"
        if not manifest_path.is_file():
            raise SafetyError(
                f"{root} has no step2_manifest.json; run step2-prepare first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sampling = manifest.get("sampling", {})
        protocol = sampling.get("protocol", DEFAULT_PROTOCOL)
        resolve_protocol(protocol)
        nblock = _first_int(sampling.get("nblock"), 4) or 4

        rows: list[dict[str, Any]] = []
        for run_row in manifest.get("runs", []):
            run_dir = Path(run_row.get("destination", ""))
            relative = run_row.get("relative_path") or run_dir.name
            energies = read_md_energy_series(run_dir)
            if not energies:
                rows.append(
                    {
                        "status": "PENDING",
                        "relative_path": relative,
                        "reason": "no MD steps in OSZICAR yet",
                        "kept_frames": None,
                    }
                )
                continue
            potim = _first_float(parse_incar(run_dir / "INCAR").get("POTIM"), 1.0)
            plan = plan_step2_retention(
                protocol, energies, potim_fs=potim, nblock=nblock
            )
            rows.append(
                {
                    "status": "OK",
                    "relative_path": relative,
                    "trajectory_frames": plan["trajectory_frames"],
                    "burn_in_frames": plan["burn_in_frames"],
                    "autocorrelation_time_frames": plan["autocorrelation_time_frames"],
                    "stride": plan["stride"],
                    "kept_frames": plan["kept_frames"],
                    "within_target": plan["within_target"],
                    "indices": plan["indices"],
                }
            )

        payload = {
            "format": "interfaceforge-step2-sample",
            "schema_version": 1,
            "root": str(root),
            "protocol": protocol,
            "retention": resolve_protocol(protocol)["step2"],
            "runs": rows,
        }
        json_path = root / "step2_sample.json"
        tsv_path = root / "step2_sample.tsv"
        if not dry_run:
            json_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            tsv_path.write_text(_sample_tsv(rows), encoding="utf-8")
        summaries.append(
            {
                "root": str(root),
                "protocol": protocol,
                "runs": len(rows),
                "ready": sum(row["status"] == "OK" for row in rows),
                "pending": sum(row["status"] == "PENDING" for row in rows),
                "within_target": sum(bool(row.get("within_target")) for row in rows),
                "json": None if dry_run else str(json_path),
                "tsv": None if dry_run else str(tsv_path),
            }
        )

    return {"mode": "dry-run" if dry_run else "written", "roots": summaries}
