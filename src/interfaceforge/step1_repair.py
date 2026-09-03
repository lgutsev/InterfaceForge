"""Diagnose and safely prepare recovery segments for unstable Step1 AIMD.

The recovery is deliberately dry-run first.  It never continues from the
current CONTCAR because a numerically unstable MD step may already have put
ions on top of one another.  Instead it rewinds to an XDATCAR frame before
the first energy/temperature runaway, starts the electronic state afresh,
and runs only the number of ionic steps still needed to reach the original
Step1 target.
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
_ELECTRONIC_STEP = re.compile(r"^\s*(?:DAV|RMM|CGA|SDA|DMP):\s*(\d+)")
_XDATCAR_FRAME = re.compile(r"^\s*(?:Direct|Cartesian)\s+configuration\s*=", re.I)
_EXCLUDED = {"archive", "backup", ".interfaceforge", "precondition"}


def _float_or_none(value: str) -> float | None:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_step1_oszicar(path: str | Path, *, nelm: int = 60) -> dict[str, Any]:
    """Return ionic stability and SCF-ceiling diagnostics from OSZICAR."""

    oszicar = Path(path)
    if not oszicar.is_file() or not oszicar.stat().st_size:
        return {
            "steps": [],
            "first_bad_step": None,
            "first_bad_reasons": [],
            "scf_ceiling_steps": 0,
            "scf_window_steps": 0,
            "scf_ceiling_fraction": None,
        }

    steps: list[dict[str, Any]] = []
    electronic_max = 0
    for line in oszicar.read_text(encoding="utf-8", errors="ignore").splitlines():
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

    scf_window = steps[5:]
    scf_ceiling_steps = sum(bool(row["hit_nelm"]) for row in scf_window)
    scf_fraction = scf_ceiling_steps / len(scf_window) if scf_window else None
    return {
        "steps": steps,
        "first_bad_step": None,
        "first_bad_reasons": [],
        "scf_ceiling_steps": scf_ceiling_steps,
        "scf_window_steps": len(scf_window),
        "scf_ceiling_fraction": scf_fraction,
    }


def diagnose_step1_run(
    run: str | Path,
    *,
    energy_jump_ev: float = 50.0,
    max_temperature_k: float | None = None,
) -> dict[str, Any]:
    """Diagnose runaway ionic motion and repeated electronic nonconvergence."""

    folder = Path(run).expanduser().resolve()
    incar = parse_incar(folder / "INCAR")
    nelm = _first_int(incar.get("NELM"), 60) or 60
    parsed = parse_step1_oszicar(folder / "OSZICAR", nelm=nelm)
    steps = parsed["steps"]
    target_temperature = _first_float(incar.get("TEBEG"), 300.0) or 300.0
    temperature_limit = (
        float(max_temperature_k)
        if max_temperature_k is not None
        else max(1200.0, 4.0 * target_temperature)
    )
    reference_values = [
        row["free_energy_ev"]
        for row in steps[:5]
        if row["free_energy_ev"] is not None
    ]
    reference_energy = median(reference_values) if reference_values else None

    first_bad_step: int | None = None
    first_bad_reasons: list[str] = []
    for row in steps:
        reasons: list[str] = []
        temperature = row["temperature_k"]
        energy = row["free_energy_ev"]
        if temperature is None:
            reasons.append("non-numeric temperature")
        elif temperature > temperature_limit:
            reasons.append(f"temperature {temperature:.0f} K > {temperature_limit:.0f} K")
        if energy is None:
            reasons.append("non-numeric free energy")
        elif reference_energy is not None and abs(energy - reference_energy) > energy_jump_ev:
            reasons.append(
                f"|F-Fref|={abs(energy-reference_energy):.1f} eV > {energy_jump_ev:g} eV"
            )
        if reasons:
            first_bad_step = int(row["step"])
            first_bad_reasons = reasons
            break

    scf_fraction = parsed["scf_ceiling_fraction"]
    scf_unreliable = bool(scf_fraction is not None and scf_fraction >= 0.5)
    unstable = first_bad_step is not None or scf_unreliable
    return {
        "run": str(folder),
        "md_steps": len(steps),
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
        diagnostic = diagnose_step1_run(
            run,
            energy_jump_ev=energy_jump_ev,
            max_temperature_k=max_temperature_k,
        )
        if diagnostic["md_steps"] >= target_nsw or not diagnostic["unstable"]:
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
        remaining = target_nsw - safe_step
        plan = {
            "run": str(run),
            "status": "READY" if inactive else "ACTIVE_OR_RECENT",
            "age_hours": age_hours,
            "diagnostic": diagnostic,
            "source": "POSCAR" if safe_step == 0 else xdatcar.name,
            "safe_prefix_steps": safe_step,
            "rewind_frame": safe_step // nblock if safe_step else None,
            "original_nsw": target_nsw,
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
        safe_step = int(plan["safe_prefix_steps"])
        nblock = _first_int(parse_incar(run / "INCAR").get("NBLOCK"), 1) or 1
        xdatcar = run / str(plan["source"])
        _, ion_count, _ = _poscar_layout(run / "POSCAR")
        frames = _xdatcar_frames(xdatcar, ion_count) if safe_step else []
        archive = archive_run(run, "step1_repair")
        if safe_step:
            frame = frames[safe_step // nblock - 1]
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
            launcher_name = next(
                (n for n in ("runvasp.sh", "run.slurm") if (run / n).is_file()), None
            )
            if launcher_name is None:
                raise SafetyError(f"{run}: --precondition needs runvasp.sh or run.slurm")
            system = f"{parse_incar(run / 'INCAR').get('SYSTEM', 'Step1')}_precondition"
            (run / "INCAR.precondition").write_text(
                build_precondition_incar((run / "INCAR").read_text(encoding="utf-8"), system=system),
                encoding="utf-8",
            )
            launcher = run / launcher_name
            launcher.write_text(
                wrap_launcher_with_precondition(
                    launcher.read_text(encoding="utf-8", errors="ignore"),
                    launcher_name=launcher_name,
                ),
                encoding="utf-8",
            )
            launcher.chmod(launcher.stat().st_mode | 0o111)
            (run / "WAVECAR").unlink(missing_ok=True)
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
