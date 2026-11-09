"""Read converged VASP results and audit the magnetic branch after an initialized run.

Pure-text parsers (no pymatgen) for exactly the quantities the density-init
benchmark compares: final energies, forces, stress, SCF iteration counts,
electronic convergence, wall time and final local moments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..vasp import parse_incar
from .inputs import expand_vasp_list, nonempty

AUDIT_NAME = "density_init_audit.json"

_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)")
_SIGMA0 = re.compile(r"energy\(sigma->0\)\s*=\s*([-+0-9.Ee]+)")
_NELM = re.compile(r"\bNELM\s*=\s*(\d+)")
_ELAPSED = re.compile(r"Elapsed time \(sec\):\s*([0-9.]+)")
_OSZ_SCF = re.compile(r"^\s*(DAV|RMM|CG|SDA|DIA|ALG|EDD|DMP)\s*:\s*\d+")
_OSZ_IONIC = re.compile(r"^\s*\d+\s+F=\s*([-+0-9.Ee]+)(?:.*?E0=\s*([-+0-9.Ee]+))?(?:.*?mag=\s*([-+0-9.Ee]+))?")


def _floats(text: str) -> list[float]:
    return [float(token) for token in text.split()]


def parse_oszicar(path: Path) -> dict[str, Any]:
    """Electronic steps per ionic step, plus the final F/E0/mag line."""

    per_ionic: list[int] = []
    current = 0
    last: dict[str, float | None] = {}
    if not path.is_file():
        return {"scf_steps_per_ionic": [], "final": {}}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if _OSZ_SCF.match(line):
            current += 1
            continue
        match = _OSZ_IONIC.match(line)
        if match:
            per_ionic.append(current)
            current = 0
            last = {
                "F": float(match.group(1)),
                "E0": float(match.group(2)) if match.group(2) else None,
                "mag": float(match.group(3)) if match.group(3) else None,
            }
    if current:
        per_ionic.append(current)  # an interrupted last ionic step
    return {"scf_steps_per_ionic": per_ionic, "final": last}


def _last_block(lines: list[str], header: str, start_after: int = 2) -> tuple[int, list[str]] | None:
    index = max((i for i, line in enumerate(lines) if header in line), default=-1)
    if index < 0:
        return None
    return index, lines[index + start_after :]


def parse_outcar(path: Path) -> dict[str, Any]:
    if not nonempty(path):
        return {"present": False}
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out: dict[str, Any] = {"present": True}
    toten = _TOTEN.findall(text)
    sigma0 = _SIGMA0.findall(text)
    out["toten_ev"] = float(toten[-1]) if toten else None
    out["energy_sigma0_ev"] = float(sigma0[-1]) if sigma0 else None
    nelm = _NELM.search(text)
    out["nelm"] = int(nelm.group(1)) if nelm else None
    elapsed = _ELAPSED.search(text)
    out["elapsed_s"] = float(elapsed.group(1)) if elapsed else None
    out["finished"] = "General timing and accounting informations for this job" in text
    out["ediff_reached"] = text.count("aborting loop because EDIFF is reached")
    out["ediff_not_reached"] = text.count("aborting loop EDIFF was not reached")

    forces: list[list[float]] | None = None
    block = _last_block(lines, "TOTAL-FORCE (eV/Angst)")
    if block is not None:
        forces = []
        for line in block[1]:
            if line.strip().startswith("---"):
                break
            values = line.split()
            if len(values) >= 6:
                forces.append([float(v) for v in values[3:6]])
    out["forces"] = forces

    stress = None
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("in kB"):
            try:
                stress = _floats(stripped[len("in kB") :])[:6]
            except ValueError:
                stress = None
            break
    out["stress_kb"] = stress

    moments = None
    block = _last_block(lines, "magnetization (x)", start_after=1)
    if block is not None:
        moments = []
        started = False
        for line in block[1]:
            stripped = line.strip()
            if stripped.startswith("---"):
                if started:
                    break
                started = True
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if started:
                values = stripped.split()
                if values[0].isdigit():
                    moments.append(float(values[-1]))
    out["local_moments"] = moments or None
    return out


def electronic_status(outcar: dict[str, Any], oszicar: dict[str, Any]) -> dict[str, Any]:
    steps = oszicar.get("scf_steps_per_ionic") or []
    nelm = outcar.get("nelm")
    if not outcar.get("present"):
        state = "NOT_RUN"
    elif outcar.get("ediff_not_reached"):
        state = "UNCONVERGED"
    elif steps and nelm and steps[-1] >= nelm and not outcar.get("ediff_reached"):
        state = "UNCONVERGED"
    elif not outcar.get("finished"):
        state = "INCOMPLETE"
    else:
        state = "CONVERGED"
    return {
        "state": state,
        "scf_steps_first_ionic": steps[0] if steps else None,
        "scf_steps_total": sum(steps) if steps else None,
        "ionic_steps": len(steps),
        "nelm": nelm,
    }


def magnetic_pattern(
    final: list[float] | None,
    reference: list[float] | None,
    *,
    reference_threshold: float = 0.5,
    min_moment: float = 0.5,
) -> dict[str, Any]:
    """Compare converged local-moment signs with the intended (INCAR) pattern.

    Sites whose reference ``|MAGMOM| >= reference_threshold`` define the
    pattern.  A global flip of every sign is the same collinear state and is
    reported as preserved.  A site whose converged ``|m| < min_moment``
    counts as quenched, which breaks the pattern.
    """

    if reference is None:
        return {"status": "NOT_APPLICABLE", "detail": "no reference MAGMOM"}
    if final is None:
        return {"status": "UNKNOWN", "detail": "no final local moments (set LORBIT >= 10)"}
    if len(final) != len(reference):
        return {"status": "UNKNOWN", "detail": f"{len(final)} final moments for {len(reference)} MAGMOM entries"}
    sites = [i for i, value in enumerate(reference) if abs(value) >= reference_threshold]
    if not sites:
        return {"status": "NOT_APPLICABLE", "detail": "no site above the reference threshold"}
    expected = [1 if reference[i] > 0 else -1 for i in sites]
    observed = [0 if abs(final[i]) < min_moment else (1 if final[i] > 0 else -1) for i in sites]
    quenched = [i + 1 for i, sign in zip(sites, observed, strict=True) if sign == 0]
    direct = all(o == e for o, e in zip(observed, expected, strict=True))
    flipped = all(o == -e for o, e in zip(observed, expected, strict=True))
    mismatched = [i + 1 for i, o, e in zip(sites, observed, expected, strict=True) if o != e]
    status = "PRESERVED" if direct else "PRESERVED_GLOBAL_FLIP" if flipped else "BROKEN"
    mixed = len(set(expected)) > 1
    return {
        "status": status,
        "pattern": "mixed-sign" if mixed else "sign-uniform",
        "reference_sites": [i + 1 for i in sites],
        "mismatched_sites": [] if status != "BROKEN" else mismatched,
        "quenched_sites": quenched,
        "final_abs_moment_range": [
            round(min(abs(final[i]) for i in sites), 4),
            round(max(abs(final[i]) for i in sites), 4),
        ],
        "net_final_moment": round(sum(final), 4),
        "thresholds": {"reference": reference_threshold, "min_moment": min_moment},
    }


def summarize_run(run: Path) -> dict[str, Any]:
    outcar = parse_outcar(run / "OUTCAR")
    oszicar = parse_oszicar(run / "OSZICAR")
    incar = parse_incar(run / "INCAR")
    magmom = expand_vasp_list(incar["MAGMOM"]) if "MAGMOM" in incar else None
    report_path = run / "density_init.json"
    init: dict[str, Any] = {}
    if report_path.is_file():
        try:
            init = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError:
            init = {}
    timing = init.get("timing") or {}
    return {
        "run": str(run),
        "electronic": electronic_status(outcar, oszicar),
        "energy_sigma0_ev": outcar.get("energy_sigma0_ev"),
        "toten_ev": outcar.get("toten_ev"),
        "forces": outcar.get("forces"),
        "stress_kb": outcar.get("stress_kb"),
        "local_moments": outcar.get("local_moments"),
        "oszicar_mag": (oszicar.get("final") or {}).get("mag"),
        "vasp_wall_s": outcar.get("elapsed_s"),
        "ispin": int(float(incar.get("ISPIN", "1"))),
        "incar_magmom": magmom,
        "density_init": {
            "status": init.get("status"),
            "backend": init.get("backend"),
            "active": init.get("active"),
            "overhead_s": timing.get("overhead_s", timing.get("total_s")),
            "inference_s": timing.get("inference_s"),
            "model_load_s": timing.get("model_load_s"),
            "grid_discovery_s": timing.get("grid_discovery_s"),
        },
    }


def audit_initialized_run(
    run_dir: str | Path, *, reference_threshold: float = 0.5, min_moment: float = 0.5, write: bool = True
) -> dict[str, Any]:
    """Post-VASP audit: did the run converge, and on the intended magnetic branch?"""

    run = Path(run_dir).expanduser().resolve()
    summary = summarize_run(run)
    reference = summary["incar_magmom"] if summary["ispin"] == 2 else None
    pattern = magnetic_pattern(
        summary["local_moments"], reference, reference_threshold=reference_threshold, min_moment=min_moment
    )
    electronic = summary["electronic"]["state"]
    if electronic in {"NOT_RUN", "INCOMPLETE"}:
        status = "INCOMPLETE"
    elif electronic != "CONVERGED" or pattern["status"] == "BROKEN":
        status = "FAIL"
    elif pattern["status"] == "UNKNOWN":
        status = "WARN"
    else:
        status = "PASS"
    payload = {
        "format": "interfaceforge-density-init-audit",
        "schema_version": 1,
        "status": status,
        "run_dir": str(run),
        "electronic": summary["electronic"],
        "magnetic_pattern": pattern,
        "final_local_moments": summary["local_moments"],
        "incar_magmom": summary["incar_magmom"],
        "density_init": summary["density_init"],
        "energy_sigma0_ev": summary["energy_sigma0_ev"],
    }
    if write:
        (run / AUDIT_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
