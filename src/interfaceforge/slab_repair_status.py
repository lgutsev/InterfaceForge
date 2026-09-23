"""Live status summary for slab tight-SCF and relaxation-restart campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import SafetyError
from .slab_tight_scf import scf_diagnostics_from_outcar, selective_free_mask

FAMILIES = ("tight_scf", "relax_continue")


def _status_for(family: str, calc: Path, force_target: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "folder": calc.name,
        "path": str(calc),
        "status": "NOT_STARTED",
        "finished": False,
        "ionic_steps": 0,
        "NSW": None,
        "progress_fraction": None,
        "final_scf_converged": None,
        "unconverged_scf_steps": [],
        "ionic_converged": None,
        "hit_nsw_limit": None,
        "final_max_force_eV_per_A": None,
        "force_target_eV_per_A": force_target,
        "LOCPOT": (calc / "LOCPOT").is_file(),
        "CHGCAR": (calc / "CHGCAR").is_file(),
        "ready_for_workfunction_audit": False,
        "ready_for_final_static": False,
        "note": "",
    }
    outcar = calc / "OUTCAR"
    if not outcar.is_file() or outcar.stat().st_size == 0:
        return row
    try:
        mask = selective_free_mask(calc / "CONTCAR") if (calc / "CONTCAR").is_file() else None
    except (OSError, ValueError, IndexError, SafetyError):
        mask = None
    try:
        diag = scf_diagnostics_from_outcar(outcar, free_mask=mask)
    except (OSError, ValueError, SafetyError) as exc:
        row["status"] = "PARSE_ERROR"
        row["note"] = str(exc)
        return row

    progress = None
    if diag.NSW and diag.NSW > 0:
        progress = min(1.0, diag.ionic_steps / diag.NSW)
    row.update(
        {
            "finished": diag.finished,
            "ionic_steps": diag.ionic_steps,
            "NSW": diag.NSW,
            "progress_fraction": progress,
            "final_scf_converged": diag.final_scf_converged,
            "unconverged_scf_steps": diag.unconverged_scf_steps,
            "ionic_converged": diag.ionic_converged,
            "hit_nsw_limit": diag.hit_nsw_limit,
            "final_max_force_eV_per_A": diag.final_max_force_eV_per_A,
        }
    )

    if family == "tight_scf":
        if not diag.finished:
            row["status"] = "RUNNING_STATIC"
        elif diag.final_scf_converged is False:
            row["status"] = "STATIC_SCF_FAILED"
        else:
            row["status"] = "STATIC_CONVERGED"
            row["ready_for_workfunction_audit"] = row["LOCPOT"]
            if not row["LOCPOT"]:
                row["note"] = "static SCF finished but LOCPOT is missing"
        return row

    if not diag.finished:
        row["status"] = "RUNNING_RELAX"
    elif diag.hit_nsw_limit:
        row["status"] = "RELAX_NSW_LIMIT"
    elif diag.ionic_converged is False:
        row["status"] = "RELAX_NOT_CONVERGED"
    elif (
        diag.final_max_force_eV_per_A is not None
        and diag.final_max_force_eV_per_A > force_target
    ):
        row["status"] = "RELAX_FORCE_HIGH"
        row["note"] = (
            f"VASP stopped but Fmax={diag.final_max_force_eV_per_A:.4f} eV/A "
            f"> target {force_target:g}"
        )
    else:
        row["status"] = "RELAX_CONVERGED"
        row["ready_for_final_static"] = True
    return row


def slab_repair_status(
    root: str | Path = ".",
    *,
    force_target: float = 0.03,
    write_json: str | Path | None = "slab_repair_status.json",
    write_text: str | Path | None = "slab_repair_status.txt",
) -> dict[str, Any]:
    """Summarize an in-progress slab repair campaign without reading LOCPOT data."""

    if force_target <= 0:
        raise SafetyError("force_target must be positive")
    root_path = Path(root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_root = root_path / family
        if not family_root.is_dir():
            continue
        for calc in sorted(path for path in family_root.iterdir() if path.is_dir()):
            if not ((calc / "INCAR").is_file() or (calc / "OUTCAR").is_file()):
                continue
            rows.append(_status_for(family, calc, force_target))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    payload = {
        "root": str(root_path),
        "force_target_eV_per_A": force_target,
        "count": len(rows),
        "counts": counts,
        "workfunction_ready": sum(bool(row["ready_for_workfunction_audit"]) for row in rows),
        "final_static_ready": sum(bool(row["ready_for_final_static"]) for row in rows),
        "rows": rows,
    }

    if write_json:
        output = Path(write_json).expanduser()
        if not output.is_absolute():
            output = root_path / output
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["json_output"] = str(output)
    if write_text:
        output = Path(write_text).expanduser()
        if not output.is_absolute():
            output = root_path / output
        lines = [
            "Slab repair live status",
            "=======================",
            f"root: {root_path}",
            f"force target: {force_target:g} eV/A",
            "",
            f"{'family':15s} {'folder':42s} {'status':20s} {'steps':>10s} {'Fmax':>8s} {'LOCPOT':>6s}",
        ]
        for row in rows:
            steps = (
                f"{row['ionic_steps']}/{row['NSW']}"
                if row["NSW"]
                else str(row["ionic_steps"])
            )
            fmax = row["final_max_force_eV_per_A"]
            fmax_text = f"{fmax:.4f}" if isinstance(fmax, float) else "--"
            lines.append(
                f"{row['family'][:15]:15s} {row['folder'][:42]:42s} "
                f"{row['status'][:20]:20s} {steps:>10s} {fmax_text:>8s} "
                f"{('yes' if row['LOCPOT'] else 'no'):>6s}"
            )
            if row["note"]:
                lines.append(f"  note: {row['note']}")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        payload["text_output"] = str(output)
    return payload
