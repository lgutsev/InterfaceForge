"""Prepare tight static SCF reruns for slabs flagged by ``slab-align``.

The vacuum-flatness audit can flag a slab whose LOCPOT vacuum is sloped even
though ``LDIPOL``/``IDIPOL`` are correct.  A loosely converged relaxation
(``EDIFF=1E-4``) combined with a large ``AMIN`` on a long cell is a common
cause: VASP warns that charge sloshing along the long vector can spoil
convergence and recommends ``AMIN=0.01`` together with tighter ``EDIFF``.

This module reads the ``band_edge_alignment.json`` audit, inspects every
daughter's OUTCAR for its electronic settings and per-ionic-step SCF
convergence, and writes a *separate* tree of static single-point inputs at
the final geometry (CONTCAR -> POSCAR).  Only electronic-convergence and
potential-output tags change; the functional, cutoff, k-points, LREAL, and
DIPOL are preserved so that any change in the vacuum slope can be attributed
to electronic convergence.  Nothing is submitted and parent folders are never
modified.

A tight static SCF cannot repair the geometry itself, so the parent relaxation
is audited too: a run that exhausted ``NSW`` without VASP's "reached required
accuracy", or whose final forces on free atoms are large (``EDIFFG > 0`` is an
energy criterion and never checks forces), is reported with a warning.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import SafetyError
from .slab_alignment import parse_incar, parse_poscar_lines, vacuum_warning_from_outcar

AUDIT_JSON = "band_edge_alignment.json"
PROVENANCE_NAME = "TIGHT_SCF_PROVENANCE.json"
PARENT_INCAR_NAME = "INCAR.parent"
OPTIONAL_INPUTS = ("vdw_kernel.bindat",)

PLAN_FIELDS = [
    "folder",
    "role",
    "action",
    "reason",
    "flatness_status",
    "selected_swing_eV",
    "vasp_vacuum_warning",
    "parent_finished",
    "parent_ionic_steps",
    "parent_unconverged_scf_steps",
    "parent_final_scf_converged",
    "parent_EDIFF",
    "parent_NELM",
    "parent_AMIN",
    "parent_charge_sloshing_warning",
    "parent_IBRION",
    "parent_NSW",
    "parent_EDIFFG",
    "parent_ionic_converged",
    "parent_hit_nsw_limit",
    "parent_final_max_force_eV_per_A",
    "warnings",
    "wavecar",
    "incar_changes",
    "destination",
]

_STATEMENT = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=(.*)$")
_ITERATION = re.compile(r"Iteration\s+(\d+)\(\s*(\d+)\)")


@dataclass
class ScfDiagnostics:
    """Electronic-convergence facts recovered from a parent OUTCAR."""

    finished: bool = False
    ionic_steps: int = 0
    unconverged_scf_steps: list[int] = field(default_factory=list)
    final_scf_converged: bool | None = None
    EDIFF: float | None = None
    NELM: int | None = None
    AMIN: float | None = None
    ISTART: int | None = None
    ICHARG: int | None = None
    charge_sloshing_warning: bool = False
    vacuum_warning: str = ""
    IBRION: int | None = None
    NSW: int | None = None
    EDIFFG: float | None = None
    reached_required_accuracy: bool = False
    ionic_converged: bool | None = None
    hit_nsw_limit: bool | None = None
    final_max_force_eV_per_A: float | None = None
    force_atoms: str = "all"

    @property
    def is_relaxation(self) -> bool:
        return self.IBRION in (1, 2, 3) and bool(self.NSW)


def _first_float(pattern: str, line: str) -> float | None:
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None


def scf_diagnostics_from_outcar(path: str | Path, free_mask: list[bool] | None = None) -> ScfDiagnostics:
    """Parse settings, SCF convergence, and ionic convergence from an OUTCAR.

    An ionic step counts as converged only when VASP printed ``aborting loop
    because EDIFF is reached`` for it; a step that ends without that line
    (NELM exhausted, or VASP 6's explicit "not reached" message) is
    unconverged.  For relaxations, ionic convergence requires "reached
    required accuracy"; the final force maximum uses only atoms free in
    ``free_mask`` (selective dynamics) when it is supplied.
    """

    diag = ScfDiagnostics()
    outcar = Path(path)
    if not outcar.is_file():
        raise SafetyError("OUTCAR is missing")
    current_step: int | None = None
    step_converged = False
    step_status: dict[int, bool] = {}
    forces: list[float] = []
    last_forces: list[float] = []
    force_state = 0  # 0 idle, 1 expecting dashes, 2 reading rows

    def close_step() -> None:
        if current_step is not None:
            step_status[current_step] = step_converged

    with outcar.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if force_state:
                if line.lstrip().startswith("---"):
                    if force_state == 2:
                        last_forces, force_state = forces, 0
                    else:
                        forces, force_state = [], 2
                    continue
                if force_state == 2:
                    fields = line.split()
                    if len(fields) >= 6:
                        fx, fy, fz = (float(value) for value in fields[3:6])
                        forces.append((fx * fx + fy * fy + fz * fz) ** 0.5)
                        continue
                    force_state = 0
            if "TOTAL-FORCE" in line:
                force_state = 1
                continue
            if diag.NSW is None and re.match(r"^\s*NSW\s*=", line):
                value = _first_float(r"NSW\s*=\s*([0-9]+)", line)
                diag.NSW = int(value) if value is not None else None
            elif diag.IBRION is None and re.match(r"^\s*IBRION\s*=", line):
                value = _first_float(r"IBRION\s*=\s*(-?[0-9]+)", line)
                diag.IBRION = int(value) if value is not None else None
            elif diag.EDIFFG is None and re.match(r"^\s*EDIFFG\s*=", line):
                match = re.search(r"EDIFFG\s*=\s*([-+0-9.EeDd]+)", line)
                diag.EDIFFG = float(match.group(1).upper().replace("D", "E")) if match else None
            elif "reached required accuracy" in line:
                diag.reached_required_accuracy = True
            if diag.EDIFF is None and re.match(r"^\s*EDIFF\s*=", line):
                match = re.search(r"EDIFF\s*=\s*([-+0-9.EeDd]+)", line)
                diag.EDIFF = float(match.group(1).upper().replace("D", "E")) if match else None
            elif diag.NELM is None and re.match(r"^\s*NELM\s*=", line):
                value = _first_float(r"NELM\s*=\s*([0-9]+)", line)
                diag.NELM = int(value) if value is not None else None
            elif diag.AMIN is None and re.match(r"^\s*AMIN\s*=", line):
                diag.AMIN = _first_float(r"AMIN\s*=\s*([-+0-9.Ee]+)", line)
            elif diag.ISTART is None and re.match(r"^\s*ISTART\s*=", line):
                value = _first_float(r"ISTART\s*=\s*(-?[0-9]+)", line)
                diag.ISTART = int(value) if value is not None else None
            elif diag.ICHARG is None and re.match(r"^\s*ICHARG\s*=", line):
                value = _first_float(r"ICHARG\s*=\s*(-?[0-9]+)", line)
                diag.ICHARG = int(value) if value is not None else None
            if "charge sloshing" in line.lower() or "AMIN is rather" in line:
                diag.charge_sloshing_warning = True
            match = _ITERATION.search(line)
            if match:
                step = int(match.group(1))
                if step != current_step:
                    close_step()
                    current_step = step
                    step_converged = False
                continue
            if "aborting loop because EDIFF is reached" in line:
                step_converged = True
            elif "EDIFF was not reached" in line:
                step_converged = False
            elif "General timing and accounting" in line:
                diag.finished = True
    close_step()
    diag.ionic_steps = len(step_status)
    diag.unconverged_scf_steps = sorted(step for step, ok in step_status.items() if not ok)
    if step_status:
        diag.final_scf_converged = step_status[max(step_status)]
    diag.vacuum_warning = vacuum_warning_from_outcar(outcar)
    if diag.is_relaxation:
        diag.ionic_converged = diag.reached_required_accuracy
        diag.hit_nsw_limit = not diag.reached_required_accuracy and diag.ionic_steps >= int(diag.NSW or 0)
    if last_forces:
        if free_mask is not None and len(free_mask) == len(last_forces):
            selected = [force for force, free in zip(last_forces, free_mask, strict=True) if free]
            diag.force_atoms = "free"
        else:
            selected = last_forces
        diag.final_max_force_eV_per_A = max(selected) if selected else 0.0
    return diag


def selective_free_mask(path: str | Path) -> list[bool] | None:
    """Per-atom "any direction free" flags from a POSCAR/CONTCAR, or None."""

    structure_path = Path(path)
    if not structure_path.is_file():
        return None
    lines = structure_path.read_text(encoding="utf-8", errors="replace").splitlines()
    structure = parse_poscar_lines(lines)
    count = sum(structure.counts)
    nonempty = [line.strip() for line in lines[: structure.coordinate_end_line] if line.strip()]
    coordinates, preamble = nonempty[-count:], nonempty[:-count]
    # The mode line (Direct/Cartesian) directly precedes the coordinates;
    # "Selective dynamics" can only be the line before it.
    if len(preamble) < 2 or not preamble[-2].lower().startswith("s"):
        return None
    return [any(flag.upper().startswith("T") for flag in fields.split()[3:6]) for fields in coordinates]


def relaxation_warnings(diag: ScfDiagnostics, force_warn: float) -> list[str]:
    """Human-readable reasons to distrust the parent geometry."""

    warnings: list[str] = []
    if diag.hit_nsw_limit:
        warnings.append(
            f"PARENT_HIT_NSW_LIMIT: relaxation used all NSW={diag.NSW} ionic steps without "
            "'reached required accuracy'"
        )
    elif diag.ionic_converged is False:
        warnings.append(
            f"PARENT_IONIC_NOT_CONVERGED: no 'reached required accuracy' after {diag.ionic_steps} "
            f"of NSW={diag.NSW} steps"
        )
    if diag.final_max_force_eV_per_A is not None and diag.final_max_force_eV_per_A > force_warn:
        note = (
            "; EDIFFG>0 is an energy criterion, so forces were never checked"
            if diag.is_relaxation and diag.EDIFFG is not None and diag.EDIFFG > 0
            else ""
        )
        warnings.append(
            f"PARENT_FORCES_HIGH: final max force on {diag.force_atoms} atoms "
            f"{diag.final_max_force_eV_per_A:.3f} eV/A > {force_warn:g}{note}"
        )
    if diag.final_scf_converged is False:
        warnings.append("PARENT_FINAL_SCF_UNCONVERGED: the last ionic step did not reach EDIFF")
    return warnings


def _split_comment(line: str) -> tuple[str, str]:
    cut = min((index for index in (line.find("#"), line.find("!")) if index >= 0), default=-1)
    if cut < 0:
        return line, ""
    return line[:cut], line[cut:]


def rewrite_incar(text: str, overrides: dict[str, str]) -> tuple[str, list[dict[str, str]]]:
    """Apply tag overrides to INCAR text, preserving every other line.

    Handles ``;``-separated statements and inline ``#``/``!`` comments.  A
    duplicated overridden tag keeps only its first (rewritten) occurrence so
    the resulting INCAR is unambiguous.  Tags absent from the parent are
    appended under a marker comment.
    """

    wanted = {key.upper(): value for key, value in overrides.items()}
    written: set[str] = set()
    old_values: dict[str, str] = {}
    output: list[str] = []
    for line in text.splitlines():
        code, comment = _split_comment(line)
        statements = code.split(";")
        kept: list[str] = []
        touched = False
        for statement in statements:
            match = _STATEMENT.match(statement)
            key = match.group(1).upper() if match else None
            if key not in wanted:
                if statement.strip():
                    kept.append(statement.strip())
                continue
            touched = True
            old_values.setdefault(key, match.group(2).strip())
            if key not in written:
                kept.append(f"{key} = {wanted[key]}")
                written.add(key)
        if not touched:
            output.append(line)
            continue
        rebuilt = "; ".join(kept)
        if comment:
            rebuilt = f"{rebuilt}  {comment}".strip()
        if rebuilt:
            output.append(rebuilt)
    missing = [key for key in wanted if key not in written]
    if missing:
        output.extend(["", "# Tight static SCF settings added by InterfaceForge slab-tight-scf"])
        output.extend(f"{key} = {wanted[key]}" for key in missing)
    changes = []
    for key, value in wanted.items():
        old = old_values.get(key, "")
        if _normalized(old) != _normalized(value):
            changes.append({"tag": key, "old": old or "(unset)", "new": value})
    return "\n".join(output).rstrip() + "\n", changes


def _normalized(value: str) -> str:
    text = value.strip().upper().strip(".")
    if text in ("T", "TRUE"):
        return "TRUE"
    if text in ("F", "FALSE"):
        return "FALSE"
    try:
        return repr(float(text.replace("D", "E")))
    except ValueError:
        return " ".join(text.split())


def tight_scf_overrides(
    *,
    axis: str,
    ediff: float,
    nelm: int,
    amin: float,
    reuse_wavecar: bool,
    dipol: list[float] | None,
) -> dict[str, str]:
    if axis not in ("x", "y", "z"):
        raise SafetyError("Surface-normal axis must be x, y, or z")
    overrides = {
        "NSW": "0",
        "IBRION": "-1",
        # Self-consistent from the parent wavefunctions (or atoms); never the
        # non-self-consistent ICHARG=11 path.
        "ISTART": "1" if reuse_wavecar else "0",
        "ICHARG": "0" if reuse_wavecar else "2",
        "EDIFF": f"{ediff:G}",
        "NELM": str(nelm),
        "AMIN": f"{amin:g}",
        "LDIPOL": ".TRUE.",
        "IDIPOL": str("xyz".index(axis) + 1),
        "LVHAR": ".TRUE.",
        "LVACPOTAV": ".TRUE.",
        "LCHARG": ".TRUE.",
    }
    if dipol is not None:
        overrides["DIPOL"] = " ".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in dipol)
    return overrides


def _potcar_elements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:", text)


def _check_geometry(parent: Path) -> None:
    contcar = parent / "CONTCAR"
    if not contcar.is_file() or contcar.stat().st_size == 0:
        raise SafetyError("CONTCAR is missing or empty")
    final = parse_poscar_lines(contcar.read_text(encoding="utf-8", errors="replace").splitlines())
    poscar = parent / "POSCAR"
    if poscar.is_file():
        initial = parse_poscar_lines(poscar.read_text(encoding="utf-8", errors="replace").splitlines())
        if initial.counts != final.counts:
            raise SafetyError(f"CONTCAR counts {final.counts} differ from POSCAR counts {initial.counts}")
    potcar = parent / "POTCAR"
    if potcar.is_file() and not final.species[0].startswith("X"):
        elements = _potcar_elements(potcar)
        if elements and elements != final.species:
            raise SafetyError(f"POTCAR order {elements} differs from CONTCAR species {final.species}")


def _select(
    rows: list[dict[str, Any]],
    select: str,
    only: set[str] | None,
    with_references: bool,
) -> dict[str, str]:
    """Return ``{folder: role}`` for the daughters to prepare."""

    chosen: dict[str, str] = {}
    for row in rows:
        name = row["folder"]
        if only is not None and name not in only:
            continue
        flatness = str(row.get("flatness_status", ""))
        flagged = (
            flatness.startswith("SUSPECT")
            or flatness == "FAILED_FLATNESS"
            or bool(row.get("vasp_vacuum_warning"))
        )
        if select == "all" or flagged:
            chosen[name] = "flagged" if flagged else "requested"
    if with_references:
        names = {row["folder"] for row in rows}
        for row in rows:
            reference = row.get("reference", "")
            if row["folder"] in chosen and reference in names and reference not in chosen:
                chosen[reference] = "reference_control"
    return chosen


def _skip_reason(row: dict[str, Any]) -> str:
    flatness = str(row.get("flatness_status", ""))
    if flatness == "OK":
        return "vacuum passed the flatness audit"
    if flatness == "FAILED_ANALYSIS":
        return "LOCPOT audit failed (" + str(row.get("error", "")) + "); use --select all to include"
    return f"flatness {flatness or 'unknown'} not selected"


def prepare_tight_scf(
    root: str | Path = ".",
    *,
    output: str | Path = "tight_scf",
    audit: str | Path = AUDIT_JSON,
    config: str | Path | None = "slab_alignment.json",
    select: str = "flagged",
    only: list[str] | None = None,
    with_references: bool = True,
    ediff: float = 1e-7,
    nelm: int = 200,
    amin: float = 0.01,
    wavecar: str = "copy",
    dipol: str = "keep",
    copy_patterns: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    force_warn: float = 0.05,
    require_relaxed: bool = False,
) -> dict[str, Any]:
    """Write tight static-SCF copies of audited slab calculations.

    Every immediate daughter listed in the slab-align audit is inspected.
    Selected daughters get ``<output>/<folder>/`` containing POSCAR (from
    CONTCAR), KPOINTS, POTCAR, a rewritten INCAR, the parent INCAR as
    ``INCAR.parent``, optionally WAVECAR, and a provenance JSON.  Parent
    relaxations that did not converge ionically are reported as warnings, or
    blocked when ``require_relaxed`` is set.
    """

    if select not in ("flagged", "all"):
        raise SafetyError("select must be 'flagged' or 'all'")
    if wavecar not in ("copy", "none"):
        raise SafetyError("wavecar must be 'copy' or 'none'")
    if dipol not in ("keep", "suggested"):
        raise SafetyError("dipol must be 'keep' or 'suggested'")
    if not 0 < ediff < 1e-4 or nelm < 1 or not 0 < amin <= 0.1:
        raise SafetyError("Require 0 < EDIFF < 1E-4, NELM >= 1, and 0 < AMIN <= 0.1")

    root_path = Path(root).expanduser().resolve()
    audit_path = Path(audit).expanduser()
    if not audit_path.is_absolute():
        audit_path = root_path / audit_path
    if not audit_path.is_file():
        raise SafetyError(f"Missing {audit_path}; run 'iface vasp slab-align' first")
    rows = json.loads(audit_path.read_text(encoding="utf-8")).get("rows", [])
    if not rows:
        raise SafetyError(f"{audit_path} contains no audited folders")
    out_path = Path(output).expanduser()
    if not out_path.is_absolute():
        out_path = root_path / out_path
    out_path = out_path.resolve()
    if out_path == root_path:
        raise SafetyError("Output directory must differ from the calculation root")

    only_set = set(only) if only else None
    if only_set:
        unknown = only_set - {row["folder"] for row in rows}
        if unknown:
            raise SafetyError("Not in the audit: " + ", ".join(sorted(unknown)))
    chosen = _select(rows, select, only_set, with_references)

    plan: list[dict[str, Any]] = []
    for row in rows:
        name = row["folder"]
        parent = root_path / name
        entry: dict[str, Any] = {
            "folder": name,
            "role": chosen.get(name, ""),
            "action": "",
            "reason": "",
            "flatness_status": row.get("flatness_status", ""),
            "selected_swing_eV": row.get("selected_swing_eV", ""),
            "vasp_vacuum_warning": row.get("vasp_vacuum_warning", ""),
            "wavecar": "",
            "incar_changes": [],
            "destination": "",
            "warnings": [],
        }
        try:
            mask = selective_free_mask(parent / "CONTCAR") if (parent / "CONTCAR").is_file() else None
        except (OSError, ValueError, IndexError, SafetyError):
            mask = None
        try:
            diag = scf_diagnostics_from_outcar(parent / "OUTCAR", free_mask=mask)
            entry.update(
                {
                    "parent_finished": diag.finished,
                    "parent_ionic_steps": diag.ionic_steps,
                    "parent_unconverged_scf_steps": diag.unconverged_scf_steps,
                    "parent_final_scf_converged": diag.final_scf_converged,
                    "parent_EDIFF": diag.EDIFF,
                    "parent_NELM": diag.NELM,
                    "parent_AMIN": diag.AMIN,
                    "parent_charge_sloshing_warning": diag.charge_sloshing_warning,
                    "parent_IBRION": diag.IBRION,
                    "parent_NSW": diag.NSW,
                    "parent_EDIFFG": diag.EDIFFG,
                    "parent_ionic_converged": diag.ionic_converged,
                    "parent_hit_nsw_limit": diag.hit_nsw_limit,
                    "parent_final_max_force_eV_per_A": diag.final_max_force_eV_per_A,
                    "warnings": relaxation_warnings(diag, force_warn),
                }
            )
            diag_error = ""
        except (OSError, SafetyError) as exc:
            diag = None
            diag_error = str(exc)
        plan.append(entry)
        if name not in chosen:
            entry["action"] = "SKIPPED"
            reason = "not in --only" if only_set is not None and name not in only_set else _skip_reason(row)
            entry["reason"] = "; ".join(item for item in (reason, diag_error) if item)
            continue
        destination = out_path / name
        entry["destination"] = str(destination)
        try:
            if diag is None:
                raise SafetyError(diag_error)
            if not diag.finished:
                raise SafetyError("parent OUTCAR has no final timing block; the run may be incomplete")
            if require_relaxed and diag.ionic_converged is False:
                raise SafetyError("parent relaxation did not reach required accuracy (--require-relaxed)")
            for required in ("INCAR", "KPOINTS", "POTCAR"):
                if not (parent / required).is_file():
                    raise SafetyError(f"parent {required} is missing")
            _check_geometry(parent)
            axis = str(row.get("axis") or "z")
            incar_values = parse_incar(parent / "INCAR")
            if dipol == "suggested" and row.get("suggested_DIPOL_normal") not in (None, ""):
                vector = list(incar_values.get("DIPOL") or [0.5, 0.5, 0.5])
                vector["xyz".index(axis)] = float(row["suggested_DIPOL_normal"])
                dipol_vector: list[float] | None = vector
            else:
                dipol_vector = None
            reuse = wavecar == "copy" and (parent / "WAVECAR").is_file() and (parent / "WAVECAR").stat().st_size > 0
            overrides = tight_scf_overrides(
                axis=axis, ediff=ediff, nelm=nelm, amin=amin, reuse_wavecar=reuse, dipol=dipol_vector
            )
            parent_incar = (parent / "INCAR").read_text(encoding="utf-8", errors="replace")
            new_incar, changes = rewrite_incar(parent_incar, overrides)
            entry["incar_changes"] = changes
            if reuse:
                entry["wavecar"] = "copied"
            else:
                entry["wavecar"] = "absent; fresh start" if wavecar == "copy" else "not copied; fresh start"
            if destination.exists():
                if not overwrite:
                    entry["action"] = "SKIPPED_EXISTS"
                    entry["reason"] = "destination exists; pass --overwrite to refresh inputs"
                    continue
                if (destination / "OUTCAR").exists():
                    raise SafetyError("destination already contains OUTCAR; move it aside manually")
            if dry_run:
                entry["action"] = "WOULD_PREPARE"
                continue
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / "INCAR").open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(new_incar)
            shutil.copy2(parent / "INCAR", destination / PARENT_INCAR_NAME)
            shutil.copy2(parent / "CONTCAR", destination / "POSCAR")
            copied = ["CONTCAR->POSCAR", "KPOINTS", "POTCAR"]
            for name_to_copy in ("KPOINTS", "POTCAR", *OPTIONAL_INPUTS):
                if (parent / name_to_copy).is_file():
                    shutil.copy2(parent / name_to_copy, destination / name_to_copy)
                    if name_to_copy in OPTIONAL_INPUTS:
                        copied.append(name_to_copy)
            for pattern in copy_patterns or []:
                for match in sorted(parent.glob(pattern)):
                    if match.is_file() and match.name not in ("INCAR", "POSCAR", "OUTCAR", "WAVECAR"):
                        shutil.copy2(match, destination / match.name)
                        copied.append(match.name)
            if reuse:
                shutil.copy2(parent / "WAVECAR", destination / "WAVECAR")
                copied.append("WAVECAR")
            elif (destination / "WAVECAR").exists():
                (destination / "WAVECAR").unlink()
            provenance = {
                "parent": str(parent),
                "purpose": "tight static SCF at the parent's final geometry to test whether the "
                "sloped vacuum is an electronic-convergence artefact",
                "audit_row": {key: row.get(key) for key in (
                    "flatness_status", "selected_side", "selected_slope_eV_per_A", "selected_swing_eV",
                    "selected_std_eV", "vasp_vacuum_crosscheck", "vasp_vacuum_warning",
                    "vacuum_minus_ef_eV", "axis", "current_DIPOL", "suggested_DIPOL_normal",
                )},
                "parent_scf": asdict(diag),
                "parent_geometry_warnings": entry["warnings"],
                "overrides": overrides,
                "incar_changes": changes,
                "copied": copied,
            }
            (destination / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
            entry["action"] = "PREPARED"
        except (OSError, ValueError, SafetyError) as exc:
            entry["action"] = "BLOCKED"
            entry["reason"] = str(exc)

    config_copied = ""
    if config and not dry_run and any(entry["action"] == "PREPARED" for entry in plan):
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = root_path / config_path
        if config_path.is_file():
            shutil.copy2(config_path, out_path / config_path.name)
            config_copied = str(out_path / config_path.name)

    outputs = _write_plan(root_path, out_path, plan, dry_run=dry_run)
    counts: dict[str, int] = {}
    for entry in plan:
        counts[entry["action"]] = counts.get(entry["action"], 0) + 1
    return {
        "root": str(root_path),
        "output": str(out_path),
        "audit": str(audit_path),
        "dry_run": dry_run,
        "settings": {"EDIFF": ediff, "NELM": nelm, "AMIN": amin, "wavecar": wavecar, "dipol": dipol},
        "counts": counts,
        "geometry_warnings": sum(bool(entry["warnings"]) for entry in plan),
        "config_copied": config_copied,
        "outputs": outputs,
        "plan": plan,
    }


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(f"{item['tag']}:{item['old']}->{item['new']}" for item in value)
        if value and isinstance(value[0], str):
            return " | ".join(value)
        return " ".join(str(item) for item in value)
    return str(value)


def _ionic_label(entry: dict[str, Any]) -> str:
    if entry.get("parent_hit_nsw_limit"):
        return "NSW_LIMIT"
    converged = entry.get("parent_ionic_converged")
    if converged is None:
        return "static" if "parent_finished" in entry else "--"
    return "converged" if converged else "NOT_CONV"


def _write_plan(root: Path, out_path: Path, plan: list[dict[str, Any]], *, dry_run: bool) -> dict[str, str]:
    target = root if dry_run else out_path
    target.mkdir(parents=True, exist_ok=True)
    tsv_path = target / "tight_scf_plan.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS, delimiter="\t")
        writer.writeheader()
        for entry in plan:
            writer.writerow({key: _format(entry.get(key, "")) for key in PLAN_FIELDS})
    lines = [
        "Tight static SCF preparation" + (" (dry run: nothing written)" if dry_run else ""),
        "=" * 60,
        "Parent folders were not modified and no job was submitted.",
        "",
        f"{'folder':36s} {'role':18s} {'action':15s} {'EDIFF':>8s} {'AMIN':>6s} {'unconv':>6s} "
        f"{'ionic':>9s} {'Fmax':>7s}",
    ]
    for entry in plan:
        unconverged = entry.get("parent_unconverged_scf_steps")
        lines.append(
            f"{entry['folder'][:36]:36s} {entry['role'][:18] or '-':18s} {entry['action'][:15]:15s} "
            f"{_format(entry.get('parent_EDIFF')):>8s} {_format(entry.get('parent_AMIN')):>6s} "
            f"{(str(len(unconverged)) if isinstance(unconverged, list) else '--'):>6s} "
            f"{_ionic_label(entry):>9s} {_format(entry.get('parent_final_max_force_eV_per_A')) or '--':>7s}"
        )
        if entry.get("reason"):
            lines.append(f"  note: {entry['reason']}")
        for warning in entry.get("warnings") or []:
            lines.append(f"  warn: {warning}")
        if entry.get("incar_changes") and entry["action"] in ("PREPARED", "WOULD_PREPARE"):
            lines.append("  INCAR: " + _format(entry["incar_changes"]))
    lines.extend(
        [
            "",
            "After the runs finish, re-audit the new tree with the same configuration:",
            f"  iface vasp slab-align {out_path} --config <your slab_alignment config> --no-write-dipole-fixes",
            "Compare selected_swing_eV and vacuum_minus_ef_eV against the parent audit.",
        ]
    )
    text_path = target / "tight_scf_plan.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"tsv": str(tsv_path), "text": str(text_path)}
