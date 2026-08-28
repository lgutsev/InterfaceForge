"""VASP and VASP-MLFF preparation, continuation, and recovery helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import DependencyError, SafetyError

_INCAR_LINE = re.compile(r"^(\s*)([A-Za-z][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$")
_ARCHIVE_FILES = (
    "run.slurm",
    "runvasp.sh",
    "INCAR",
    "KPOINTS",
    "POTCAR",
    "POSCAR",
    "CONTCAR",
    "XDATCAR",
    "XDATCAR_FINAL",
    "OSZICAR",
    "OUTCAR",
    "ML_LOGFILE",
    "REPORT",
    "ML_AB",
    "ML_ABN",
    "ML_FF",
    "ML_FFN",
    "vasprun.xml",
)

INCAR_PRESETS = ("static", "relax", "md", "dos")
MLFF_ACCURACY_PROFILES = ("accurate",)
STEP2_HUBBARD_TAG_PREFIX = "LDAU"
STEP2_HUBBARD_TAGS = ("LMAXMIX",)
STEP2_INHERITED_FILES = ("KPOINTS", "POTCAR", "runvasp.sh", "run.slurm")
STEP2_NBLOCK = 4


def _step2_nsw(protocol: str) -> int:
    """Step2 MD length in steps for ``protocol`` (academic 5000, training 1000)."""

    from .aimd import resolve_protocol

    return int(resolve_protocol(protocol)["step2"]["nsw"])


def _step2_dense_frames(protocol: str) -> int:
    return _step2_nsw(protocol) // STEP2_NBLOCK

# Active spin/electronic tags copied verbatim from each Step1 INCAR (like the
# LDAU* set) so a fixed-temperature Step2 MD keeps the same magnetic ground
# state. MAGMOM is per-atom and Step2's POSCAR is Step1's CONTCAR, so the
# atom order matches; ISTART is not set because Step2 does not inherit a
# WAVECAR, so the moments must still be initialised from MAGMOM.
STEP2_INHERITED_ELECTRONIC_TAGS = ("ISPIN", "MAGMOM", "LASPH", "LNONCOLLINEAR")

# training protocol only: trim per-step OUTCAR bloat that is useless for
# force/energy training labels, and drop the slab dipole correction (add
# vacuum instead). academic leaves the template untouched.
STEP2_TRAINING_FORCED_TAGS = {"NWRITE": "1", "LDAUPRINT": "0"}
STEP2_TRAINING_STRIPPED_TAGS = ("LORBIT", "LDIPOL", "IDIPOL", "DIPOL")

# Audit note (never a FAIL) when the promoted Step1 slab has less than this
# much vacuum between the slab+adsorbate stack and its periodic image -- a
# tall passivant eats headroom that the bare-slab cell had to spare.
STEP2_MIN_VACUUM_A = 12.0


def _step2_slab_vacuum(structure: Path) -> dict[str, Any] | None:
    """Best-effort slab-to-image vacuum of a POSCAR; ``None`` if ASE is absent."""

    try:
        from .geometry import slab_vacuum

        return slab_vacuum(structure, axis="auto")
    except DependencyError:
        return None
    except Exception:  # best-effort note only; never block the audit
        return None


def _step2_sampling_block(protocol: str) -> dict[str, Any]:
    """Sampling summary recorded in the Step2 manifest and audit.

    ``academic`` runs a long trajectory (5 ps) and keeps the dense NBLOCK
    stride; ``training`` runs a short trajectory (1 ps) and defers frame
    selection to ``iface vasp step2-sample``, which spaces frames at the
    measured energy decorrelation time. The training-set diversity is meant
    to come from many independent short runs (different pre-relaxed start
    structures), not from one long trajectory.
    """

    from .aimd import resolve_protocol

    block: dict[str, Any] = {
        "protocol": protocol,
        "nsw": _step2_nsw(protocol),
        "nblock": STEP2_NBLOCK,
    }
    if protocol == "academic":
        block["training_frames_per_run"] = _step2_dense_frames(protocol)
        block["retention"] = "fixed NBLOCK stride (dense)"
    else:
        step2 = resolve_protocol(protocol)["step2"]
        block["training_frames_per_run"] = None
        block["retention"] = {
            "method": step2["retention_method"],
            "target_frames": step2["target_frames"],
            "burn_in_ps": step2["burn_in_ps"],
            "applied_by": "iface vasp step2-sample",
        }
    return block


def _step2_sampling_markdown(block: dict[str, Any]) -> str:
    if isinstance(block.get("retention"), dict):
        retention = block["retention"]
        lo, hi = retention["target_frames"]
        return (
            f"- Sampling: `NSW={block['nsw']}`, `NBLOCK={block['nblock']}`; retention by "
            f"**{retention['method']}** ({retention['burn_in_ps']} ps burn-in, target "
            f"**{lo}-{hi} frames/run**, applied by `{retention['applied_by']}`)"
        )
    return (
        f"- Sampling: `NSW={block['nsw']}`, `NBLOCK={block['nblock']}` "
        f"→ **{block['training_frames_per_run']} labeled frames per run** ({block['retention']})"
    )


def parse_incar(path: str | Path) -> dict[str, str]:
    """Parse the last active value of each INCAR tag."""

    parsed: dict[str, str] = {}
    incar = Path(path)
    if not incar.is_file():
        return parsed
    for raw in incar.read_text(encoding="utf-8", errors="ignore").splitlines():
        active = re.split(r"[!#]", raw, maxsplit=1)[0]
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", active)
        if match:
            parsed[match.group(1).upper()] = match.group(2).strip()
    return parsed


def _incar_assignment(line: str) -> tuple[str, str] | None:
    """Return an active INCAR assignment without interpreting its value."""

    active = re.split(r"[!#]", line, maxsplit=1)[0]
    match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", active)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).strip()


def _is_step2_hubbard_tag(tag: str) -> bool:
    normalized = tag.upper()
    return normalized.startswith(STEP2_HUBBARD_TAG_PREFIX) or normalized in STEP2_HUBBARD_TAGS


def _is_step2_electronic_tag(tag: str) -> bool:
    return tag.upper() in STEP2_INHERITED_ELECTRONIC_TAGS


def _temperature_label(temperature: float) -> str:
    value = float(temperature)
    if not value > 0:
        raise SafetyError(f"Temperature must be positive, got {temperature}")
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _preflight_step2_reprotocol(plans: list[dict[str, Any]], output_roots: list[Path]) -> None:
    """Refuse to rewrite an INCAR under a Step2 run that has already started."""

    runtime_markers = ("OUTCAR", "OSZICAR", "vasprun.xml", "WAVECAR", "XDATCAR")
    for output in output_roots:
        launch = output / "step2_launch.json"
        if launch.is_file():
            payload = json.loads(launch.read_text(encoding="utf-8"))
            if any(row.get("status") == "SUBMITTED" for row in payload.get("runs", [])):
                raise SafetyError(
                    f"{output} records submitted jobs in step2_launch.json; "
                    "refusing to change a launched run's INCAR"
                )
    for plan in plans:
        destination: Path = plan["destination"]
        if not (destination / "INCAR").is_file():
            raise SafetyError(f"{destination} has no INCAR to re-protocol; run step2-prepare first")
        present = [name for name in runtime_markers if (destination / name).exists()]
        if present:
            raise SafetyError(
                f"{destination} already has runtime output ({', '.join(present)}); "
                "re-protocol before launching, then gzip the finished OUTCAR instead"
            )


def _collect_verbatim_source_tags(
    source_text: str, predicate: Callable[[str], bool], kind: str
) -> tuple[list[str], dict[str, str]]:
    lines: list[str] = []
    values: dict[str, str] = {}
    for line in source_text.splitlines():
        assignment = _incar_assignment(line)
        if assignment is None or not predicate(assignment[0]):
            continue
        tag, value = assignment
        if tag in values:
            raise SafetyError(
                f"Source INCAR contains duplicate active {kind} tag {tag}; "
                "refusing an ambiguous Step2 inheritance"
            )
        values[tag] = value
        lines.append(line)
    return lines, values


def _render_step2_incar(
    source_text: str,
    template_text: str,
    temperature: float,
    *,
    protocol: str = "academic",
) -> dict[str, Any]:
    """Render a Step2 INCAR with a deliberately narrow precedence rule.

    The Step2 template is authoritative for every ordinary setting, except:

    * every active ``LDAU*`` / ``LMAXMIX`` line is copied byte-for-byte from
      that Step1 run (avoids rebuilding species-length DFT+U arrays);
    * every active spin/electronic tag in
      :data:`STEP2_INHERITED_ELECTRONIC_TAGS` (``ISPIN``, ``MAGMOM``,
      ``LASPH``, ``LNONCOLLINEAR``) is likewise copied verbatim, so a Step2
      MD keeps Step1's magnetic ground state instead of silently running
      non-spin-polarised;
    * the requested temperature overrides ``SYSTEM`` / ``TEBEG`` / ``TEEND``
      and sampling is pinned to ``NSW`` / ``NBLOCK``.

    Under ``protocol="training"`` the render additionally forces
    :data:`STEP2_TRAINING_FORCED_TAGS` (``NWRITE=1``, ``LDAUPRINT=0``) and
    drops :data:`STEP2_TRAINING_STRIPPED_TAGS` (``LORBIT`` and the slab
    dipole correction) to keep OUTCAR small; academic leaves the template
    untouched.
    """

    hubbard_lines, hubbard_values = _collect_verbatim_source_tags(
        source_text, _is_step2_hubbard_tag, "Hubbard"
    )
    electronic_lines, electronic_values = _collect_verbatim_source_tags(
        source_text, _is_step2_electronic_tag, "spin/electronic"
    )

    training = protocol == "training"
    stripped = set(STEP2_TRAINING_STRIPPED_TAGS) if training else set()

    if training:
        # A forced training tag (e.g. LDAUPRINT, which matches the LDAU
        # prefix) overrides any inherited copy without touching the real
        # DFT+U parameters.
        for tag in STEP2_TRAINING_FORCED_TAGS:
            hubbard_values.pop(tag, None)
            electronic_values.pop(tag, None)
        hubbard_lines = [
            line
            for line in hubbard_lines
            if (_incar_assignment(line) or ("",))[0] not in STEP2_TRAINING_FORCED_TAGS
        ]

    temperature_text = _temperature_label(float(temperature)).replace("p", ".")
    replacements = {
        "SYSTEM": f"Step2_DFT_MD_{_temperature_label(float(temperature))}K",
        "TEBEG": temperature_text,
        "TEEND": temperature_text,
        "NSW": str(_step2_nsw(protocol)),
        "NBLOCK": str(STEP2_NBLOCK),
    }
    if training:
        replacements.update(STEP2_TRAINING_FORCED_TAGS)

    found: set[str] = set()
    output: list[str] = []
    template_active: set[str] = set()
    for line in template_text.splitlines():
        assignment = _incar_assignment(line)
        if assignment is None:
            output.append(line)
            continue
        tag, _ = assignment
        if tag in template_active:
            raise SafetyError(
                f"Step2 template contains duplicate active INCAR tag {tag}; "
                "refusing ambiguous template precedence"
            )
        template_active.add(tag)
        if tag in replacements:
            # SYSTEM/TEBEG/TEEND/NSW/NBLOCK, plus the training-forced tags.
            prefix = line[: len(line) - len(line.lstrip())]
            output.append(f"{prefix}{tag} = {replacements[tag]}")
            found.add(tag)
            continue
        if _is_step2_hubbard_tag(tag):
            continue  # the source run is the only authority for Hubbard settings
        if tag in electronic_values:
            continue  # replaced by the verbatim Step1 value below
        if tag in stripped:
            continue  # dropped for the training protocol
        output.append(line)

    for tag, value in replacements.items():
        if tag not in found:
            output.extend(("", f"{tag} = {value}"))

    if hubbard_lines:
        output.extend(
            ("", "# DFT+U settings inherited verbatim from this run's Step1 INCAR", *hubbard_lines)
        )
    if electronic_lines:
        output.extend(
            (
                "",
                "# Spin/electronic settings inherited verbatim from this run's Step1 INCAR",
                *electronic_lines,
            )
        )
    if training:
        output.extend(
            (
                "",
                "# training protocol: LORBIT and the slab dipole correction are dropped;"
                " use a large vacuum instead",
            )
        )

    text = "\n".join(output).rstrip() + "\n"
    rendered_values: dict[str, str] = {}
    for line in text.splitlines():
        assignment = _incar_assignment(line)
        if assignment is not None:
            rendered_values[assignment[0]] = assignment[1]
    for tag, value in {**hubbard_values, **electronic_values}.items():
        if rendered_values.get(tag) != value:
            raise SafetyError(f"Internal error: Step2 did not preserve source {tag} exactly")
    return {
        "text": text,
        "hubbard_tags": hubbard_values,
        "electronic_tags": electronic_values,
        "stripped_tags": sorted(stripped & template_active),
        "forced_tags": dict(STEP2_TRAINING_FORCED_TAGS) if training else {},
        "template_tags": sorted(
            tag
            for tag in rendered_values
            if not _is_step2_hubbard_tag(tag) and not _is_step2_electronic_tag(tag)
        ),
    }


def _step2_excluded(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if (
            part.startswith("X")
            or "backup" in lowered
            or lowered in {".interfaceforge", "archive"}
            or lowered.startswith(("restart_archive_", "refit_archive_", "stability_archive_"))
        ):
            return True
    return False


def _resolve_step2_input(run: Path, source_root: Path, name: str) -> Path | None:
    """Resolve a run-specific input first, then a shared ancestor input."""

    current = run
    while True:
        candidate = current / name
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
        if current == source_root:
            return None
        current = current.parent


def _vasp_list_length(value: str) -> int:
    count = 0
    for token in value.split():
        repeat = re.fullmatch(r"(\d+)\*.+", token)
        count += int(repeat.group(1)) if repeat else 1
    return count


def prepare_step2_series(
    source: str | Path,
    *,
    temperatures: Iterable[float] = (300.0, 450.0, 600.0),
    output_root: str | Path | None = None,
    template: str | Path | None = None,
    source_structure: str = "CONTCAR",
    protocol: str = "academic",
    dry_run: bool = False,
    audit_only: bool = False,
    reprotocol: bool = False,
) -> dict[str, Any]:
    """Promote a recursive Step1 tree into fixed-temperature Step2 DFT-MD runs.

    Common INCAR controls come from the Step2 template. The complete active
    ``LDAU*``/``LMAXMIX`` set comes only from each source run. ``CONTCAR`` is
    promoted to ``POSCAR`` and run-specific/shared VASP inputs are copied into
    a sibling ``Step2_<temperature>K`` tree. Existing destination roots are
    never overwritten.
    """

    from .aimd import resolve_protocol

    resolve_protocol(protocol)
    source_root = Path(source).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    destination_parent = (
        Path(output_root).expanduser().resolve() if output_root is not None else source_root.parent
    )
    if template is None:
        template_label = "packaged INCAR.step2_dft_md"
        template_text = resources.files("interfaceforge").joinpath(
            "templates/INCAR.step2_dft_md"
        ).read_text(encoding="utf-8")
    else:
        template_path = Path(template).expanduser().resolve()
        if not template_path.is_file() or not template_path.stat().st_size:
            raise FileNotFoundError(template_path)
        template_label = str(template_path)
        template_text = template_path.read_text(encoding="utf-8", errors="ignore")

    normalized_temperatures = [float(value) for value in temperatures]
    labels = [_temperature_label(value) for value in normalized_temperatures]
    if len(set(labels)) != len(labels):
        raise SafetyError("Temperatures must be unique")
    if audit_only and dry_run:
        raise SafetyError("--audit-only and --dry-run cannot be combined")

    runs: list[Path] = []
    for incar in sorted(source_root.rglob("INCAR")):
        relative = incar.parent.relative_to(source_root)
        if _step2_excluded(relative):
            continue
        structure = incar.parent / source_structure
        if structure.is_file() and structure.stat().st_size:
            runs.append(incar.parent)
    if not runs:
        raise SafetyError(
            f"No Step1 runs with nonempty INCAR and {source_structure} found below {source_root}"
        )

    if audit_only and reprotocol:
        raise SafetyError("--audit-only and --set-protocol cannot be combined")

    output_roots = [destination_parent / f"Step2_{label}K" for label in labels]
    existing = [path for path in output_roots if path.exists()]
    if audit_only or reprotocol:
        verb = "audit" if audit_only else "re-protocol"
        missing_outputs = [path for path in output_roots if not path.is_dir()]
        if missing_outputs:
            raise SafetyError(
                f"Cannot {verb} missing Step2 destination(s): "
                + ", ".join(str(path) for path in missing_outputs)
            )
    elif existing:
        raise SafetyError(
            "Refusing to overwrite existing Step2 destination(s): "
            + ", ".join(str(path) for path in existing)
        )

    plans: list[dict[str, Any]] = []
    for run in runs:
        relative = run.relative_to(source_root)
        source_incar = run / "INCAR"
        structure = run / source_structure
        elements = _poscar_elements(structure)
        resolved_inputs = {
            name: path
            for name in STEP2_INHERITED_FILES
            if (path := _resolve_step2_input(run, source_root, name)) is not None
        }
        if "KPOINTS" not in resolved_inputs:
            raise SafetyError(f"No nonempty KPOINTS found for Step1 run {run}")
        if "POTCAR" not in resolved_inputs:
            raise SafetyError(f"No nonempty POTCAR found for Step1 run {run}")
        if not ({"runvasp.sh", "run.slurm"} & resolved_inputs.keys()):
            raise SafetyError(f"No runvasp.sh or run.slurm found for Step1 run {run}")
        source_text = source_incar.read_text(encoding="utf-8", errors="ignore")
        ion_count = _poscar_ion_count(structure)
        for temperature, label, output in zip(
            normalized_temperatures, labels, output_roots, strict=True
        ):
            rendered = _render_step2_incar(
                source_text, template_text, temperature, protocol=protocol
            )
            for tag in ("LDAUL", "LDAUU", "LDAUJ"):
                value = rendered["hubbard_tags"].get(tag)
                if value is not None and _vasp_list_length(value) != len(elements):
                    raise SafetyError(
                        f"{source_incar}: {tag} has {_vasp_list_length(value)} entries but "
                        f"{source_structure} has {len(elements)} species ({' '.join(elements)})"
                    )
            magmom = rendered["electronic_tags"].get("MAGMOM")
            if magmom is not None and _vasp_list_length(magmom) != ion_count:
                raise SafetyError(
                    f"{source_incar}: MAGMOM has {_vasp_list_length(magmom)} entries but "
                    f"{source_structure} has {ion_count} ions"
                )
            plans.append(
                {
                    "source": run,
                    "relative": relative,
                    "destination": output / relative,
                    "temperature_k": temperature,
                    "temperature_label": label,
                    "structure": structure,
                    "elements": elements,
                    "inputs": resolved_inputs,
                    "incar_text": rendered["text"],
                    "hubbard_tags": rendered["hubbard_tags"],
                    "electronic_tags": rendered["electronic_tags"],
                    "stripped_tags": rendered["stripped_tags"],
                }
            )

    if reprotocol and not dry_run:
        _preflight_step2_reprotocol(plans, output_roots)

    manifest_rows: list[dict[str, Any]] = []
    if not dry_run and (not audit_only or reprotocol):
        created_roots: list[Path] = []
        try:
            if not reprotocol:
                for output in output_roots:
                    output.mkdir(parents=True, exist_ok=False)
                    created_roots.append(output)
            for plan in plans:
                destination: Path = plan["destination"]
                if not reprotocol:
                    if destination not in output_roots:
                        destination.mkdir(parents=True, exist_ok=False)
                    shutil.copy2(plan["structure"], destination / "POSCAR")
                    for name, source_path in plan["inputs"].items():
                        target = destination / name
                        shutil.copy2(source_path, target)
                        if name in {"runvasp.sh", "run.slurm"}:
                            # Guarantee the inherited launcher is executable even if the
                            # Step1 copy never had its execute bit set; the auditor and
                            # sbatch both require this.
                            target.chmod(target.stat().st_mode | 0o111)
                (destination / "INCAR").write_text(plan["incar_text"], encoding="utf-8")
                manifest_rows.append(
                    {
                        "source": str(plan["source"]),
                        "relative_path": plan["relative"].as_posix() or ".",
                        "destination": str(destination),
                        "temperature_k": plan["temperature_k"],
                        "elements": plan["elements"],
                        "source_structure": source_structure,
                        "hubbard_tags": plan["hubbard_tags"],
                        "electronic_tags": plan["electronic_tags"],
                        "stripped_tags": plan["stripped_tags"],
                        "inherited_files": sorted(plan["inputs"]),
                        "source_incar_sha256": _sha256_file(plan["source"] / "INCAR"),
                        "step2_incar_sha256": _sha256_file(destination / "INCAR"),
                        "source_structure_sha256": _sha256_file(plan["structure"]),
                        "step2_poscar_sha256": _sha256_file(destination / "POSCAR"),
                    }
                )
            for output in output_roots:
                rows = [row for row in manifest_rows if Path(row["destination"]).is_relative_to(output)]
                manifest = {
                    "format": "interfaceforge-step2-series",
                    "schema_version": 1,
                    "source_root": str(source_root),
                    "template": template_label,
                    "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
                    "precedence": {
                        "ordinary_incar_tags": "Step2 template",
                        "hubbard_tags": "exact active LDAU* and LMAXMIX lines from each Step1 INCAR",
                        "electronic_tags": (
                            "exact active "
                            + ", ".join(STEP2_INHERITED_ELECTRONIC_TAGS)
                            + " lines from each Step1 INCAR (no MAGMOM re-init, no ISTART)"
                        ),
                        "temperature_tags": "requested temperature overrides SYSTEM, TEBEG, and TEEND",
                        "sampling_tags": (
                            f"{protocol} Step2 fixes NSW={_step2_nsw(protocol)} and "
                            f"NBLOCK={STEP2_NBLOCK}; the {protocol} retention policy "
                            "governs which frames are kept"
                        ),
                        "training_trim": (
                            "training strips "
                            + ", ".join(STEP2_TRAINING_STRIPPED_TAGS)
                            + " and forces "
                            + ", ".join(f"{k}={v}" for k, v in STEP2_TRAINING_FORCED_TAGS.items())
                            if protocol == "training"
                            else "n/a (academic leaves the template untouched)"
                        ),
                    },
                    "sampling": _step2_sampling_block(protocol),
                    "runs": rows,
                }
                (output / "step2_manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        except Exception:
            for output in reversed(created_roots):
                shutil.rmtree(output, ignore_errors=True)
            raise

    audit_payload: dict[str, Any] | None = None
    if not dry_run:
        audit_payload = _audit_step2_plans(
            plans,
            output_roots=output_roots,
            source_root=source_root,
            template_label=template_label,
            template_text=template_text,
            source_structure=source_structure,
            protocol=protocol,
        )
        failed_reports = [
            report["markdown"]
            for report in audit_payload["reports"]
            if report["status"] != "PASS"
        ]
        if failed_reports:
            raise SafetyError(
                "Step2 preparation audit FAILED; no jobs were submitted. Review: "
                + ", ".join(failed_reports)
            )

    return {
        "mode": (
            "dry-run"
            if dry_run
            else "audited"
            if audit_only
            else "reprotocoled-and-audited"
            if reprotocol
            else "prepared-and-audited"
        ),
        "source_root": str(source_root),
        "template": template_label,
        "temperatures_k": normalized_temperatures,
        "output_roots": [str(path) for path in output_roots],
        "source_runs": len(runs),
        "prepared_runs": len(plans),
        "source_structure": source_structure,
        "hubbard_rule": "preserve exact active LDAU* and LMAXMIX assignments per source run",
        "protocol": protocol,
        "sampling": _step2_sampling_block(protocol),
        "planned": [
            {
                "source": str(plan["source"]),
                "destination": str(plan["destination"]),
                "temperature_k": plan["temperature_k"],
                "elements": plan["elements"],
                "hubbard_tags": plan["hubbard_tags"],
                "electronic_tags": plan["electronic_tags"],
                "inherited_files": sorted(plan["inputs"]),
            }
            for plan in plans
        ],
        "audit": audit_payload,
    }


def _load_step2_launch_root(
    root: Path,
    launcher: str | None,
    emit: Callable[[str], None] = lambda _message: None,
) -> list[dict[str, Any]]:
    emit(f"[{root.name}] reading step2_audit.json + step2_manifest.json")
    audit_path = root / "step2_audit.json"
    manifest_path = root / "step2_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise SafetyError(
            f"{root} is missing step2_audit.json or step2_manifest.json; "
            "run step2-prepare (or step2-prepare --audit-only) first"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise SafetyError(f"{audit_path} is not PASS; refusing to launch")
    previous_launch = root / "step2_launch.json"
    if previous_launch.is_file():
        previous = json.loads(previous_launch.read_text(encoding="utf-8"))
        submitted = [
            row for row in previous.get("runs", []) if row.get("status") == "SUBMITTED"
        ]
        if submitted:
            raise SafetyError(
                f"{root} already records {len(submitted)} submitted Step2 job(s) in "
                f"{previous_launch}; refusing a possible duplicate launch"
            )
    manifest_rows = {
        str(row.get("relative_path")): row for row in manifest.get("runs", [])
    }
    audit_rows = {
        str(row.get("relative_path")): row for row in audit.get("runs", [])
    }
    if not manifest_rows or set(manifest_rows) != set(audit_rows):
        raise SafetyError(f"Manifest/audit run sets do not match in {root}")

    planned: list[dict[str, Any]] = []
    for relative in sorted(manifest_rows):
        manifest_row = manifest_rows[relative]
        audit_row = audit_rows[relative]
        if audit_row.get("status") != "PASS":
            raise SafetyError(f"Audit row is not PASS: {root / relative}")
        run = (root / relative).resolve() if relative != "." else root
        if not run.is_relative_to(root):
            raise SafetyError(f"Unsafe destination outside launch root: {run}")
        for name, hash_key in (
            ("INCAR", "step2_incar_sha256"),
            ("POSCAR", "step2_poscar_sha256"),
        ):
            path = run / name
            expected = manifest_row.get(hash_key)
            if not path.is_file() or not expected or _sha256_file(path) != expected:
                raise SafetyError(
                    f"{path} changed after preparation; rerun step2-prepare --audit-only "
                    "and inspect the audit before launching"
                )
        inherited_hashes = dict(audit_row.get("inherited_sha256") or {})
        for name, expected in inherited_hashes.items():
            path = run / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise SafetyError(
                    f"{path} changed after audit; refusing to launch the folder"
                )
        forbidden = [
            name
            for name in ("OUTCAR", "OSZICAR", "WAVECAR", "CHGCAR", "vasprun.xml")
            if (run / name).exists()
        ]
        if forbidden:
            raise SafetyError(
                f"{run} already contains runtime outputs ({', '.join(forbidden)}); "
                "refusing a possible duplicate launch"
            )
        script = resolve_launcher(run, launcher)
        if script.name not in inherited_hashes:
            raise SafetyError(
                f"Launcher {script.name} was not part of the PASS audit for {run}"
            )
        emit(
            f"[{root.name}] preflight OK: {relative} "
            f"(T={audit_row.get('temperature_k')}K, launcher={script.name})"
        )
        planned.append(
            {
                "root": str(root),
                "relative_path": relative,
                "directory": str(run),
                "launcher": script.name,
                "temperature_k": audit_row.get("temperature_k"),
            }
        )
    return planned


def launch_step2_runs(
    roots: Iterable[str | Path],
    *,
    execute: bool = False,
    launcher: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Launch only unchanged, PASS-audited Step2 daughter runs.

    The default is a non-mutating launch plan. ``execute=True`` is the sole
    path that calls ``sbatch``. All roots are fully preflighted before the
    first job is submitted.
    """

    emit = progress or (lambda _message: None)
    resolved_roots = [Path(value).expanduser().resolve() for value in roots]
    if not resolved_roots:
        raise SafetyError("At least one Step2 root is required")
    emit(
        f"Preflighting {len(resolved_roots)} Step2 root(s): "
        + ", ".join(root.name for root in resolved_roots)
    )
    planned: list[dict[str, Any]] = []
    for root in resolved_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        planned.extend(_load_step2_launch_root(root, launcher, emit))
    if not planned:
        raise SafetyError("No PASS-audited Step2 daughter runs were found")
    emit(f"Preflight PASS: {len(planned)} run(s) ready to submit")

    if not execute:
        emit("Dry run only; no jobs submitted. Re-run with --execute to submit.")
        return {
            "mode": "dry-run",
            "roots": [str(root) for root in resolved_roots],
            "runs": len(planned),
            "preflight": "PASS",
            "submission": "not performed; pass --execute after review",
            "planned": planned,
        }

    rows: list[dict[str, Any]] = []
    failure: str | None = None
    for index, item in enumerate(planned, start=1):
        emit(
            f"[{index}/{len(planned)}] sbatch {item['relative_path']} "
            f"in {item['directory']}"
        )
        try:
            job_id = submit_run(item["directory"], item["launcher"])
            rows.append({**item, "status": "SUBMITTED", "job_id": job_id, "detail": ""})
            emit(f"    submitted job {job_id}")
        except Exception as exc:
            failure = f"{item['directory']}: {exc}"
            rows.append(
                {**item, "status": "FAILED", "job_id": "", "detail": str(exc)}
            )
            emit(f"    FAILED: {exc}")
            break

    report_paths: list[str] = []
    for root in resolved_roots:
        root_rows = [row for row in rows if row["root"] == str(root)]
        if any(row["status"] == "FAILED" for row in root_rows):
            root_status = "FAILED"
        elif root_rows:
            root_status = "SUBMITTED"
        else:
            root_status = "NOT_SUBMITTED"
        payload = {
            "format": "interfaceforge-step2-launch",
            "schema_version": 1,
            "status": root_status,
            "root": str(root),
            "preflight": "PASS",
            "runs": root_rows,
        }
        json_path = root / "step2_launch.json"
        tsv_path = root / "step2_launch.tsv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "status",
                    "job_id",
                    "temperature_k",
                    "relative_path",
                    "directory",
                    "launcher",
                    "detail",
                ),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(
                {key: row.get(key, "") for key in writer.fieldnames} for row in root_rows
            )
        report_paths.extend((str(json_path), str(tsv_path)))
        emit(f"[{root.name}] {root_status}; wrote {json_path.name}, {tsv_path.name}")
    if failure is not None:
        raise SafetyError(
            f"Step2 launch stopped after a submission failure ({failure}). "
            f"Review partial launch records: {', '.join(report_paths)}"
        )
    emit(f"Done: {len(rows)} job(s) submitted across {len(resolved_roots)} root(s)")
    return {
        "mode": "submitted",
        "roots": [str(root) for root in resolved_roots],
        "runs": len(rows),
        "preflight": "PASS",
        "submitted": len(rows),
        "reports": report_paths,
        "jobs": rows,
    }


def _audit_step2_plans(
    plans: list[dict[str, Any]],
    *,
    output_roots: list[Path],
    source_root: Path,
    template_label: str,
    template_text: str,
    source_structure: str,
    protocol: str = "academic",
) -> dict[str, Any]:
    """Independently verify prepared Step2 inputs and write human-readable audits."""

    from .aimd import _overconverged_notes, preheat_ps, resolve_protocol

    def _num(value: str | None) -> float | None:
        try:
            return float(str(value).split()[0])
        except (TypeError, ValueError, IndexError):
            return None

    preheat_lo, preheat_hi = resolve_protocol(protocol)["step1"]["preheat_ps_expected"]
    sampling_block = _step2_sampling_block(protocol)

    forbidden_runtime_files = (
        "OUTCAR",
        "OSZICAR",
        "WAVECAR",
        "CHGCAR",
        "CHG",
        "XDATCAR",
        "vasprun.xml",
        "REPORT",
    )
    rows: list[dict[str, Any]] = []
    for plan in plans:
        destination: Path = plan["destination"]
        issues: list[str] = []
        notes: list[str] = []
        source_incar = parse_incar(plan["source"] / "INCAR")
        _source_nsw = _num(source_incar.get("NSW"))
        source_ps = preheat_ps(
            int(_source_nsw) if _source_nsw is not None else None,
            _num(source_incar.get("POTIM")) or 1.0,
        )
        if source_ps is not None and not (preheat_lo <= source_ps <= preheat_hi):
            notes.append(
                f"Step1 preheat {source_ps:.2f} ps is outside the {protocol} range "
                f"{preheat_lo}-{preheat_hi} ps"
            )
        incar_path = destination / "INCAR"
        poscar_path = destination / "POSCAR"
        if not incar_path.is_file():
            issues.append("missing INCAR")
            parsed: dict[str, str] = {}
        else:
            actual_text = incar_path.read_text(encoding="utf-8", errors="ignore")
            if actual_text != plan["incar_text"]:
                issues.append("INCAR differs from deterministic rendered template")
            parsed = parse_incar(incar_path)
        expected_temperature = plan["temperature_label"].replace("p", ".")
        expected_system = f"Step2_DFT_MD_{plan['temperature_label']}K"
        for tag, expected in (
            ("SYSTEM", expected_system),
            ("TEBEG", expected_temperature),
            ("TEEND", expected_temperature),
            ("NSW", str(_step2_nsw(protocol))),
            ("NBLOCK", str(STEP2_NBLOCK)),
        ):
            if parsed.get(tag) != expected:
                issues.append(f"{tag}={parsed.get(tag)!r}, expected {expected!r}")
        for tag, expected in {**plan["hubbard_tags"], **plan["electronic_tags"]}.items():
            if parsed.get(tag) != expected:
                issues.append(f"{tag} changed from Step1")
        for tag in ("LDAUL", "LDAUU", "LDAUJ"):
            value = parsed.get(tag)
            if value is not None and _vasp_list_length(value) != len(plan["elements"]):
                issues.append(
                    f"{tag} has {_vasp_list_length(value)} values for {len(plan['elements'])} species"
                )
        # Spin coherence: a spin-polarised Step1 must not become a
        # non-spin-polarised Step2 (silently wrong for AFM oxides).
        if source_incar.get("ISPIN") == "2" and parsed.get("ISPIN") != "2":
            issues.append("Step1 is spin-polarised (ISPIN=2) but Step2 INCAR is not")
        magmom = parsed.get("MAGMOM")
        if magmom is not None and _vasp_list_length(magmom) != _poscar_ion_count(plan["structure"]):
            issues.append(
                f"MAGMOM has {_vasp_list_length(magmom)} values for "
                f"{_poscar_ion_count(plan['structure'])} ions"
            )
        if protocol == "training":
            for tag, forced in STEP2_TRAINING_FORCED_TAGS.items():
                if parsed.get(tag) != forced:
                    issues.append(f"training protocol requires {tag}={forced}, found {parsed.get(tag)!r}")
            still_present = [tag for tag in STEP2_TRAINING_STRIPPED_TAGS if tag in parsed]
            if still_present:
                issues.append("training protocol should strip: " + ", ".join(still_present))
            notes.extend(_overconverged_notes(parsed))
        if not poscar_path.is_file():
            issues.append("missing POSCAR")
        elif _sha256_file(poscar_path) != _sha256_file(plan["structure"]):
            issues.append(f"POSCAR does not match Step1 {source_structure}")
        vacuum_a = _step2_slab_vacuum(plan["structure"])
        if vacuum_a is not None and vacuum_a["vacuum_a"] < STEP2_MIN_VACUUM_A:
            notes.append(
                f"thin vacuum: {vacuum_a['vacuum_a']:.1f} A between the slab stack and its "
                f"{vacuum_a['axis']}-image; extend with: iface vasp geom vacuum "
                f"{plan['source'] / source_structure} --extend 18 -o {poscar_path}"
            )
        inherited_hashes: dict[str, str] = {}
        for name, source_path in plan["inputs"].items():
            target = destination / name
            if not target.is_file():
                issues.append(f"missing inherited {name}")
                continue
            source_hash = _sha256_file(source_path)
            inherited_hashes[name] = source_hash
            if _sha256_file(target) != source_hash:
                issues.append(f"inherited {name} differs from source")
            if name in {"runvasp.sh", "run.slurm"} and not os.access(target, os.X_OK):
                issues.append(f"inherited launcher {name} is not executable")
        unexpected_runtime = [name for name in forbidden_runtime_files if (destination / name).exists()]
        if unexpected_runtime:
            issues.append("runtime outputs present: " + ", ".join(unexpected_runtime))
        rows.append(
            {
                "status": "PASS" if not issues else "FAIL",
                "source": str(plan["source"]),
                "relative_path": plan["relative"].as_posix() or ".",
                "destination": str(destination),
                "temperature_k": plan["temperature_k"],
                "nsw": _step2_nsw(protocol),
                "nblock": STEP2_NBLOCK,
                "training_frames": sampling_block["training_frames_per_run"],
                "protocol": protocol,
                "step1_preheat_ps": round(source_ps, 3) if source_ps is not None else None,
                "slab_vacuum_a": (
                    {
                        "axis": vacuum_a["axis"],
                        "gap_to_image": round(vacuum_a["vacuum_a"], 2),
                        "slab_span": round(vacuum_a["slab_span_a"], 2),
                    }
                    if vacuum_a is not None
                    else None
                ),
                "elements": plan["elements"],
                "hubbard_tags": plan["hubbard_tags"],
                "electronic_tags": plan["electronic_tags"],
                "stripped_tags": plan["stripped_tags"],
                "inherited_files": sorted(plan["inputs"]),
                "inherited_sha256": inherited_hashes,
                "issues": issues,
                "notes": notes,
            }
        )

    reports: list[dict[str, Any]] = []
    for output in output_roots:
        output_rows = [row for row in rows if Path(row["destination"]).is_relative_to(output)]
        status = "PASS" if output_rows and all(row["status"] == "PASS" for row in output_rows) else "FAIL"
        payload = {
            "format": "interfaceforge-step2-preparation-audit",
            "schema_version": 1,
            "status": status,
            "source_root": str(source_root),
            "output_root": str(output),
            "template": template_label,
            "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
            "protocol": protocol,
            "sampling": sampling_block,
            "runs": output_rows,
        }
        json_path = output / "step2_audit.json"
        tsv_path = output / "step2_audit.tsv"
        markdown_path = output / "step2_audit.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "status",
                    "temperature_k",
                    "nsw",
                    "nblock",
                    "training_frames",
                    "relative_path",
                    "elements",
                    "LDAUL",
                    "LDAUU",
                    "LDAUJ",
                    "ISPIN",
                    "inherited_files",
                    "issues",
                    "notes",
                ),
                delimiter="\t",
            )
            writer.writeheader()
            for row in output_rows:
                writer.writerow(
                    {
                        "status": row["status"],
                        "temperature_k": row["temperature_k"],
                        "nsw": row["nsw"],
                        "nblock": row["nblock"],
                        "training_frames": row["training_frames"],
                        "relative_path": row["relative_path"],
                        "elements": " ".join(row["elements"]),
                        "LDAUL": row["hubbard_tags"].get("LDAUL", ""),
                        "LDAUU": row["hubbard_tags"].get("LDAUU", ""),
                        "LDAUJ": row["hubbard_tags"].get("LDAUJ", ""),
                        "ISPIN": row["electronic_tags"].get("ISPIN", ""),
                        "inherited_files": ",".join(row["inherited_files"]),
                        "issues": "; ".join(row["issues"]),
                        "notes": "; ".join(row.get("notes", [])),
                    }
                )
        markdown_lines = [
            "# Step2 preparation audit",
            "",
            f"**Status:** {status}",
            "",
            f"- Source: `{source_root}`",
            f"- Output: `{output}`",
            f"- Template: `{template_label}`",
            f"- Runs: {len(output_rows)}",
            f"- Protocol: **{protocol}**",
            _step2_sampling_markdown(sampling_block),
            (
                "- Spin/electronic tags inherited verbatim from Step1: "
                + (
                    ", ".join(sorted(output_rows[0]["electronic_tags"]))
                    if output_rows and output_rows[0]["electronic_tags"]
                    else "none in this Step1 tree"
                )
            ),
            "- Submission: **not performed**",
            "",
            "| Status | Run | Species | LDAUU | ISPIN | Inherited inputs | Issues | Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in output_rows:
            markdown_lines.append(
                "| "
                + " | ".join(
                    (
                        row["status"],
                        row["relative_path"],
                        " ".join(row["elements"]),
                        row["hubbard_tags"].get("LDAUU", "—"),
                        row["electronic_tags"].get("ISPIN", "—"),
                        ", ".join(row["inherited_files"]),
                        "; ".join(row["issues"]) or "—",
                        "; ".join(row.get("notes", [])) or "—",
                    )
                )
                + " |"
            )
        markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
        reports.append(
            {
                "status": status,
                "output_root": str(output),
                "json": str(json_path),
                "tsv": str(tsv_path),
                "markdown": str(markdown_path),
            }
        )
    return {
        "status": "PASS" if reports and all(report["status"] == "PASS" for report in reports) else "FAIL",
        "runs": len(rows),
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] == "FAIL" for row in rows),
        "reports": reports,
    }


STEP1_INHERITED_FILES = ("KPOINTS", "POTCAR", "runvasp.sh", "run.slurm")
STEP1_EXTRA_INHERITED_TAGS = ("GGA", "ISPIN", "LASPH", "LNONCOLLINEAR", "MAGMOM")


def _is_step1_inherited_tag(tag: str) -> bool:
    normalized = tag.upper()
    if normalized == "LDAUPRINT":
        return False
    return (
        normalized.startswith("LDAU")
        or normalized in {"LMAXMIX", *STEP1_EXTRA_INHERITED_TAGS}
    )


def _verified_hardlink(source: Path, target: Path) -> str:
    """Hard-link ``source`` to ``target``; fall back to copy if the FS refuses.

    Returns ``"hardlink"`` or ``"copy"``. A hard link that silently became a
    copy (some network filesystems) is detected by device/inode identity and
    reported as ``"copy"`` rather than pretended to be a link.
    """

    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(source, target)
        return "copy"
    if (source.stat().st_dev, source.stat().st_ino) != (
        target.stat().st_dev,
        target.stat().st_ino,
    ):
        return "copy"
    return "hardlink"


def _render_step1_incar(
    opt_incar_text: str, template_text: str, *, temperature: float, nsw: int
) -> dict[str, Any]:
    """Render a Step1 preheat INCAR from an OPT run.

    The packaged template owns every ordinary tag (electronic convergence,
    the preheat MD block, output). ``GGA``, ``ISPIN``, ``LASPH``,
    ``LNONCOLLINEAR``, ``MAGMOM`` and the full active ``LDAU*`` / ``LMAXMIX``
    set are copied byte-for-byte from the OPT INCAR; the requested
    temperature overrides ``SYSTEM`` / ``TEBEG`` / ``TEEND`` and ``NSW`` is
    set from the protocol. ``ISTART=1`` stays (the OPT WAVECAR is inherited).
    """

    inherited_lines, inherited_values = _collect_verbatim_source_tags(
        opt_incar_text, _is_step1_inherited_tag, "OPT"
    )
    label = _temperature_label(float(temperature))
    text_temp = label.replace("p", ".")
    replacements = {
        "SYSTEM": f"Step1_preheat_{label}K",
        "TEBEG": text_temp,
        "TEEND": text_temp,
        "NSW": str(int(nsw)),
    }

    found: set[str] = set()
    output: list[str] = []
    seen: set[str] = set()
    for line in template_text.splitlines():
        assignment = _incar_assignment(line)
        if assignment is None:
            output.append(line)
            continue
        tag = assignment[0]
        if tag in seen:
            raise SafetyError(f"Step1 template has a duplicate active tag {tag}")
        seen.add(tag)
        if tag in replacements:
            prefix = line[: len(line) - len(line.lstrip())]
            output.append(f"{prefix}{tag} = {replacements[tag]}")
            found.add(tag)
            continue
        if _is_step1_inherited_tag(tag):
            continue  # the OPT run is the only authority for these
        output.append(line)

    for tag, value in replacements.items():
        if tag not in found:
            output.extend(("", f"{tag} = {value}"))
    if inherited_lines:
        output.extend(
            ("", "# Inherited verbatim from this run's OPT INCAR", *inherited_lines)
        )

    rendered_text = "\n".join(output).rstrip() + "\n"
    rendered = {
        assignment[0]: assignment[1]
        for line in rendered_text.splitlines()
        if (assignment := _incar_assignment(line)) is not None
    }
    for tag, value in inherited_values.items():
        if rendered.get(tag) != value:
            raise SafetyError(f"Internal error: Step1 did not preserve OPT {tag} exactly")
    return {"text": rendered_text, "inherited_tags": inherited_values}


def prepare_step1_series(
    source: str | Path,
    *,
    temperature: float = 300.0,
    output_root: str | Path | None = None,
    template: str | Path | None = None,
    source_structure: str = "CONTCAR",
    protocol: str = "academic",
    dry_run: bool = False,
    audit_only: bool = False,
) -> dict[str, Any]:
    """Promote a recursive OPT tree into a sibling ``Step1`` preheat tree.

    Each finished OPT run (local ``INCAR`` + nonempty ``CONTCAR`` + nonempty
    ``WAVECAR``) becomes a fixed-temperature preheat MD: ``CONTCAR`` is
    promoted to ``POSCAR``, the OPT ``WAVECAR`` is hard-linked in for the
    ``ISTART=1`` restart, ``GGA``/spin/``LDAU*`` are inherited verbatim, and
    the rest comes from the packaged ``INCAR.step1_preheat`` template.
    """

    from .aimd import resolve_protocol

    nsw = int(resolve_protocol(protocol)["step1"]["nsw"])
    source_root = Path(source).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if audit_only and dry_run:
        raise SafetyError("--audit-only and --dry-run cannot be combined")
    destination_parent = (
        Path(output_root).expanduser().resolve() if output_root is not None else source_root.parent
    )
    if template is None:
        template_label = "packaged INCAR.step1_preheat"
        template_text = (
            resources.files("interfaceforge")
            .joinpath("templates/INCAR.step1_preheat")
            .read_text(encoding="utf-8")
        )
    else:
        template_path = Path(template).expanduser().resolve()
        if not template_path.is_file() or not template_path.stat().st_size:
            raise FileNotFoundError(template_path)
        template_label = str(template_path)
        template_text = template_path.read_text(encoding="utf-8", errors="ignore")

    output_root_path = destination_parent / "Step1"
    if audit_only:
        if not output_root_path.is_dir():
            raise SafetyError(f"Cannot audit missing Step1 tree: {output_root_path}")
    elif output_root_path.exists():
        raise SafetyError(f"Refusing to overwrite existing Step1 tree: {output_root_path}")

    runs: list[Path] = []
    for incar in sorted(source_root.rglob("INCAR")):
        relative = incar.parent.relative_to(source_root)
        if _step2_excluded(relative):
            continue
        structure = incar.parent / source_structure
        if structure.is_file() and structure.stat().st_size:
            runs.append(incar.parent)
    if not runs:
        raise SafetyError(
            f"No OPT runs with a nonempty INCAR and {source_structure} found below {source_root}"
        )

    plans: list[dict[str, Any]] = []
    for run in runs:
        relative = run.relative_to(source_root)
        structure = run / source_structure
        elements = _poscar_elements(structure)
        ion_count = _poscar_ion_count(structure)
        wavecar = run / "WAVECAR"
        if not (wavecar.is_file() and wavecar.stat().st_size):
            raise SafetyError(
                f"OPT run {run} has no nonempty WAVECAR for the ISTART=1 restart"
            )
        resolved_inputs = {
            name: path
            for name in STEP1_INHERITED_FILES
            if (path := _resolve_step2_input(run, source_root, name)) is not None
        }
        for required in ("KPOINTS", "POTCAR"):
            if required not in resolved_inputs:
                raise SafetyError(f"No nonempty {required} found for OPT run {run}")
        if not ({"runvasp.sh", "run.slurm"} & resolved_inputs.keys()):
            raise SafetyError(f"No runvasp.sh or run.slurm found for OPT run {run}")

        rendered = _render_step1_incar(
            (run / "INCAR").read_text(encoding="utf-8", errors="ignore"),
            template_text,
            temperature=temperature,
            nsw=nsw,
        )
        for tag in ("LDAUL", "LDAUU", "LDAUJ"):
            value = rendered["inherited_tags"].get(tag)
            if value is not None and _vasp_list_length(value) != len(elements):
                raise SafetyError(
                    f"{run}/INCAR: {tag} has {_vasp_list_length(value)} entries but "
                    f"{source_structure} has {len(elements)} species"
                )
        magmom = rendered["inherited_tags"].get("MAGMOM")
        if magmom is not None and _vasp_list_length(magmom) != ion_count:
            raise SafetyError(
                f"{run}/INCAR: MAGMOM has {_vasp_list_length(magmom)} entries but "
                f"{source_structure} has {ion_count} ions"
            )
        plans.append(
            {
                "source": run,
                "relative": relative,
                "destination": output_root_path / relative,
                "structure": structure,
                "wavecar": wavecar,
                "elements": elements,
                "inputs": resolved_inputs,
                "incar_text": rendered["text"],
                "inherited_tags": rendered["inherited_tags"],
            }
        )

    link_modes: dict[str, str] = {}
    if not dry_run and not audit_only:
        try:
            output_root_path.mkdir(parents=True, exist_ok=False)
            for plan in plans:
                destination: Path = plan["destination"]
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(plan["structure"], destination / "POSCAR")
                link_modes[plan["relative"].as_posix() or "."] = _verified_hardlink(
                    plan["wavecar"], destination / "WAVECAR"
                )
                for name, path in plan["inputs"].items():
                    target = destination / name
                    shutil.copy2(path, target)
                    if name in {"runvasp.sh", "run.slurm"}:
                        target.chmod(target.stat().st_mode | 0o111)
                (destination / "INCAR").write_text(plan["incar_text"], encoding="utf-8")
            manifest = {
                "format": "interfaceforge-step1-series",
                "schema_version": 1,
                "source_root": str(source_root),
                "template": template_label,
                "protocol": protocol,
                "temperature_k": float(temperature),
                "nsw": nsw,
                "wavecar_link_mode": link_modes,
                "precedence": {
                    "ordinary_incar_tags": "Step1 template",
                    "inherited_tags": (
                        "exact active LDAU*/LMAXMIX plus "
                        + ", ".join(STEP1_EXTRA_INHERITED_TAGS)
                        + " from each OPT INCAR; ISTART=1 restart from the OPT WAVECAR"
                    ),
                },
                "runs": [
                    {
                        "source": str(plan["source"]),
                        "relative_path": plan["relative"].as_posix() or ".",
                        "destination": str(plan["destination"]),
                        "elements": plan["elements"],
                        "inherited_tags": plan["inherited_tags"],
                        "source_incar_sha256": _sha256_file(plan["source"] / "INCAR"),
                        "step1_incar_sha256": _sha256_file(plan["destination"] / "INCAR"),
                        "step1_poscar_sha256": _sha256_file(plan["destination"] / "POSCAR"),
                    }
                    for plan in plans
                ],
            }
            (output_root_path / "step1_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(output_root_path, ignore_errors=True)
            raise

    audit_payload = None
    if not dry_run:
        audit_payload = _audit_step1_plans(plans, output_root_path, protocol, source_structure)
        if audit_payload["status"] != "PASS":
            raise SafetyError(
                f"Step1 preparation audit is {audit_payload['status']}; review "
                f"{output_root_path / 'step1_audit.md'}"
            )

    return {
        "mode": "dry-run" if dry_run else "audited" if audit_only else "prepared-and-audited",
        "source_root": str(source_root),
        "output_root": str(output_root_path),
        "template": template_label,
        "protocol": protocol,
        "temperature_k": float(temperature),
        "nsw": nsw,
        "prepared_runs": len(plans),
        "wavecar_link_mode": link_modes,
        "audit": audit_payload,
    }


def _audit_step1_plans(
    plans: list[dict[str, Any]], output_root: Path, protocol: str, source_structure: str
) -> dict[str, Any]:
    """Independently verify a prepared Step1 tree and write the audit files."""

    from .aimd import audit_step1_incar

    rows: list[dict[str, Any]] = []
    for plan in plans:
        destination: Path = plan["destination"]
        issues: list[str] = []
        notes: list[str] = []
        incar_path = destination / "INCAR"
        if not incar_path.is_file():
            issues.append("missing INCAR")
        elif incar_path.read_text(encoding="utf-8", errors="ignore") != plan["incar_text"]:
            issues.append("INCAR differs from the deterministic render")
        parsed = parse_incar(incar_path)
        for tag, value in plan["inherited_tags"].items():
            if parsed.get(tag) != value:
                issues.append(f"{tag} not inherited from OPT")
        if parsed.get("IBRION") != "0" or parsed.get("SMASS") != "-1":
            issues.append("Step1 INCAR is not a velocity-rescaled preheat (IBRION=0, SMASS=-1)")
        if parsed.get("ISTART") != "1":
            issues.append("ISTART is not 1 (WAVECAR restart)")
        poscar = destination / "POSCAR"
        if not poscar.is_file() or _sha256_file(poscar) != _sha256_file(plan["structure"]):
            issues.append(f"POSCAR does not match OPT {source_structure}")
        wavecar = destination / "WAVECAR"
        if not (wavecar.is_file() and wavecar.stat().st_size):
            issues.append("missing WAVECAR")
        for name in plan["inputs"]:
            target = destination / name
            if not target.is_file() or _sha256_file(target) != _sha256_file(plan["inputs"][name]):
                issues.append(f"inherited {name} missing or altered")
            elif name in {"runvasp.sh", "run.slurm"} and not os.access(target, os.X_OK):
                issues.append(f"inherited launcher {name} is not executable")

        step1_audit = audit_step1_incar(incar_path, protocol)
        notes.extend(step1_audit["notes"])
        vacuum = _step2_slab_vacuum(plan["structure"])
        if vacuum is not None and vacuum["vacuum_a"] < STEP2_MIN_VACUUM_A:
            notes.append(
                f"thin vacuum: {vacuum['vacuum_a']:.1f} A to the {vacuum['axis']}-image; "
                f"iface vasp geom vacuum {plan['source'] / source_structure} --extend 18"
            )
        rows.append(
            {
                "status": "FAIL" if issues else ("WARN" if notes else "PASS"),
                "relative_path": plan["relative"].as_posix() or ".",
                "preheat_ps": step1_audit["preheat_ps"],
                "ISPIN": plan["inherited_tags"].get("ISPIN", ""),
                "LDAUU": plan["inherited_tags"].get("LDAUU", ""),
                "issues": issues,
                "notes": notes,
            }
        )

    status = (
        "FAIL"
        if any(r["status"] == "FAIL" for r in rows)
        else "WARN"
        if any(r["status"] == "WARN" for r in rows)
        else "PASS"
    )
    payload = {
        "format": "interfaceforge-step1-preparation-audit",
        "schema_version": 1,
        "status": status,
        "output_root": str(output_root),
        "protocol": protocol,
        "runs": rows,
    }
    (output_root / "step1_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    header = "| Status | Run | Preheat ps | ISPIN | LDAUU | Issues | Notes |\n|---|---|---|---|---|---|---|"
    body = "\n".join(
        "| "
        + " | ".join(
            (
                r["status"],
                r["relative_path"],
                str(r["preheat_ps"]),
                r["ISPIN"] or "—",
                r["LDAUU"] or "—",
                "; ".join(r["issues"]) or "—",
                "; ".join(r["notes"]) or "—",
            )
        )
        + " |"
        for r in rows
    )
    (output_root / "step1_audit.md").write_text(
        f"# Step1 preparation audit\n\n**Status:** {status}  ·  protocol **{protocol}**\n\n"
        f"{header}\n{body}\n",
        encoding="utf-8",
    )
    with (output_root / "step1_audit.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("status", "relative_path", "preheat_ps", "ISPIN", "LDAUU", "issues", "notes"),
            delimiter="\t",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **{k: r[k] for k in ("status", "relative_path", "preheat_ps", "ISPIN", "LDAUU")},
                    "issues": "; ".join(r["issues"]),
                    "notes": "; ".join(r["notes"]),
                }
            )
    return payload


def update_incar(
    path: str | Path,
    changes: Mapping[str, Any],
    *,
    delete: Iterable[str] = (),
    create: bool = False,
) -> Path:
    """Update tags while preserving unrelated comments and ordering."""

    incar = Path(path)
    if not incar.exists() and not create:
        raise FileNotFoundError(incar)
    original = incar.read_text(encoding="utf-8", errors="ignore") if incar.exists() else ""
    normalized = {str(key).upper(): str(value) for key, value in changes.items()}
    deleted = {str(key).upper() for key in delete}
    found: set[str] = set()
    output: list[str] = []

    for line in original.splitlines(keepends=True):
        match = _INCAR_LINE.match(line)
        if not match:
            output.append(line)
            continue
        key = match.group(2).upper()
        if key in deleted:
            continue
        if key in normalized:
            ending = match.group(5) or "\n"
            output.append(f"{match.group(1)}{key}{match.group(3)}{normalized[key]}{ending}")
            found.add(key)
        else:
            output.append(line)

    missing = [key for key in normalized if key not in found]
    if missing:
        if output and not output[-1].endswith("\n"):
            output[-1] += "\n"
        if output and output[-1].strip():
            output.append("\n")
        output.extend(f"{key} = {normalized[key]}\n" for key in missing)

    temporary = incar.with_suffix(incar.suffix + ".tmp")
    temporary.write_text("".join(output), encoding="utf-8")
    temporary.replace(incar)
    return incar


def incar_preset(
    name: str,
    *,
    temperature: float = 300.0,
    nsw: int = 3000,
    potim: float = 1.0,
    workfunction: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    """Return a small, explicit ionic/electronic-control preset.

    The presets intentionally avoid system-dependent convergence choices such
    as ENCUT, KSPACING, EDIFF, spin, and dispersion corrections.

    ``workfunction=True`` additionally sets ``LVHAR = True``. VASP only
    writes the local electrostatic potential (not just the charge density)
    into LOCPOT when this tag is set, and that potential is required by any
    work-function analysis, including :mod:`interfaceforge.workfunction` and
    the standalone ``examples/vasp/workfunction/plot_workfunc.py`` script.
    """

    if name == "static":
        changes, delete = (
            {"IBRION": -1, "NSW": 0},
            {"MDALGO", "SMASS", "TEBEG", "TEEND", "POTIM", "EDIFFG"},
        )
    elif name == "relax":
        changes, delete = (
            {"IBRION": 2, "NSW": nsw, "ISIF": 3, "EDIFFG": -0.02},
            {"MDALGO", "SMASS", "TEBEG", "TEEND", "POTIM"},
        )
    elif name == "md":
        changes, delete = (
            {
                "IBRION": 0,
                "NSW": nsw,
                "POTIM": potim,
                "MDALGO": 2,
                "SMASS": 1.0,
                "TEBEG": temperature,
                "TEEND": temperature,
                "ISIF": 2,
            },
            {"EDIFFG"},
        )
    elif name == "dos":
        changes, delete = (
            {
                "IBRION": -1,
                "NSW": 0,
                "ICHARG": 11,
                "LORBIT": 11,
                "NEDOS": 2000,
                "ISMEAR": -5,
            },
            {"MDALGO", "SMASS", "TEBEG", "TEEND", "POTIM", "EDIFFG"},
        )
    else:
        raise ValueError(f"Unknown INCAR preset {name!r}; choose from {INCAR_PRESETS}")
    if workfunction:
        changes = {**changes, "LVHAR": ".TRUE."}
        delete = delete - {"LVHAR"}
    return changes, delete


def apply_incar_preset(
    path: str | Path,
    preset: str,
    *,
    temperature: float = 300.0,
    nsw: int = 3000,
    potim: float = 1.0,
    workfunction: bool = False,
    create: bool = False,
) -> dict[str, Any]:
    """Apply one conservative preset without overwriting unrelated settings."""

    changes, delete = incar_preset(
        preset,
        temperature=temperature,
        nsw=nsw,
        potim=potim,
        workfunction=workfunction,
    )
    output = update_incar(path, changes, delete=delete, create=create)
    return {
        "incar": str(output.resolve()),
        "preset": preset,
        "workfunction": workfunction,
        "changes": changes,
        "removed_tags": sorted(delete),
    }


def stage_tags(
    stage: str, *, temperature: float, nsw: int, potim: float, teend: float | None = None
) -> tuple[dict[str, Any], set[str]]:
    """Return conservative VASP MLFF tags for one stage.

    ``teend`` optionally sets a training temperature ramp (``TEBEG=temperature``,
    ``TEEND=teend``) instead of a single fixed temperature; VASP's MLFF
    best-practices guidance suggests training somewhat above the highest
    application temperature for coverage. Ignored for stages other than
    "train"; when omitted, TEBEG=TEEND=temperature as before.
    """

    common = {
        "ML_LMLFF": ".TRUE.",
        "ML_WTSIF": "1E-10",
        "ISYM": "0",
    }
    if stage == "train":
        return (
            {
                **common,
                "ML_MODE": "train",
                "IBRION": "0",
                "NSW": nsw,
                "POTIM": potim,
                "MDALGO": "2",
                "SMASS": "1.0",
                "TEBEG": temperature,
                "TEEND": teend if teend is not None else temperature,
                "ISIF": "2",
            },
            {"ML_ISTART", "ML_ESTBLOCK"},
        )
    if stage == "refit":
        return ({**common, "ML_MODE": "refit"}, {"ML_ISTART", "IBRION", "NSW", "ML_ESTBLOCK"})
    if stage == "stability":
        return (
            {
                **common,
                "ML_MODE": "run",
                "IBRION": "0",
                "NSW": nsw,
                "POTIM": potim,
                "MDALGO": "2",
                "SMASS": "1.0",
                "TEBEG": temperature,
                "TEEND": temperature,
                "ISIF": "2",
                "ML_ESTBLOCK": "50",
            },
            {
                "ML_ISTART",
                "ML_MB",
                "ML_MCONF",
                "ML_LBASIS_DISCARD",
                "ML_CDOUB",
                "ML_CSLOPE",
                "ML_ICRITERIA",
                "ML_NMDINT",
            },
        )
    if stage == "validation_dft":
        return (
            {
                "ML_LMLFF": ".FALSE.",
                "IBRION": "-1",
                "NSW": "0",
            },
            {"ML_MODE", "ML_ISTART", "ML_ESTBLOCK"},
        )
    if stage == "validation_ml":
        return (
            {
                **common,
                "ML_MODE": "run",
                "IBRION": "-1",
                "NSW": "0",
                "ML_ESTBLOCK": "1",
            },
            {"ML_ISTART"},
        )
    raise ValueError(f"Unknown VASP stage: {stage}")


def mlff_accuracy_profile_tags(profile: str, stage: str) -> dict[str, Any]:
    """Return opt-in VASP best-practice tags for accuracy-oriented MLFF stages."""

    if profile not in MLFF_ACCURACY_PROFILES:
        raise ValueError(
            f"Unknown MLFF accuracy profile {profile!r}; choose from {MLFF_ACCURACY_PROFILES}"
        )
    if stage == "train":
        return {
            "ML_IALGO_LINREG": "1",
            "ML_SION1": "0.3",
            "ML_MRB2": "12",
        }
    if stage == "refit":
        return {
            "ML_IALGO_LINREG": "4",
            "ML_SION1": "0.5",
            "ML_MRB2": "12",
            "ML_EPS_LOW": "1E-11",
        }
    return {}


def require_files(folder: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (folder / name).is_file() or not (folder / name).stat().st_size]
    if missing:
        raise SafetyError(f"Missing required files in {folder}: {', '.join(missing)}")


def archive_run(folder: str | Path, operation: str) -> Path:
    """Create a recoverable archive before mutating a run folder."""

    run = Path(folder).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = run / ".interfaceforge" / "archive" / f"{operation}_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    copied = 0
    for name in _ARCHIVE_FILES:
        source = run / name
        if source.is_file():
            shutil.copy2(source, archive / name)
            copied += 1
    for pattern in ("slurm-*.out", "slurm-*.err", "vasp.*.out", "vasp.*.err"):
        for source in run.glob(pattern):
            if source.is_file():
                shutil.copy2(source, archive / source.name)
                copied += 1
    if copied == 0:
        raise SafetyError(f"No run state was available to archive in {run}")
    return archive


def _clean_outputs(folder: Path) -> None:
    for name in (
        "WAVECAR",
        "CHGCAR",
        "EIGENVAL",
        "DOSCAR",
        "PROCAR",
        "PCDAT",
        "REPORT",
        "vasprun.xml",
        "OSZICAR",
        "OUTCAR",
        "ML_LOGFILE",
    ):
        (folder / name).unlink(missing_ok=True)


def _continue_source(folder: Path) -> Path:
    source = folder / "ML_ABN"
    if source.is_file() and source.stat().st_size:
        return source
    source = folder / "ML_AB"
    if source.is_file() and source.stat().st_size:
        return source
    raise SafetyError(f"Neither ML_ABN nor ML_AB is available in {folder}")


def prepare_recovery(
    folder: str | Path,
    operation: str,
    *,
    temperature: float | None = None,
    nsw: int | None = None,
    ml_mb: int | None = None,
    ml_mconf: int | None = None,
    force_expand: bool = False,
    increase_eps_low: bool = False,
    ml_outblock: int = 1,
) -> dict[str, Any]:
    """Safely mutate one VASP MLFF folder for a well-defined recovery operation."""

    run = Path(folder).resolve()
    require_files(run, ("INCAR", "POTCAR", "KPOINTS"))
    if operation not in {"continue", "discard", "expand", "refit", "stability", "heat"}:
        raise ValueError(f"Unknown recovery operation: {operation}")
    if increase_eps_low and operation != "discard":
        raise SafetyError("ML_EPS_LOW adjustment is supported only with discard recovery")
    if operation == "heat" and ml_outblock < 1:
        raise SafetyError("ML_OUTBLOCK must be a positive integer for heat-flux production")

    if operation in {"continue", "discard", "expand"}:
        require_files(run, ("CONTCAR",))
        source = _continue_source(run)
    elif operation == "refit":
        source = _continue_source(run)
    else:
        require_files(run, ("ML_FFN", "CONTCAR"))
        header = (run / "ML_FFN").open("rb").read(4096).decode("utf-8", errors="ignore")
        if not re.search(r"ML_LFAST.{0,20}(true|T)", header, re.I):
            raise SafetyError("ML_FFN does not report ML_LFAST=true")
        source = run / "ML_FFN"

    if operation in {"discard", "expand"}:
        outcar_text = (run / "OUTCAR").read_text(errors="ignore") if (run / "OUTCAR").is_file() else ""
        capacity_stop = re.search(r"increase\s+ML_M(B|CONF)|ML_M(B|CONF).*(too small|exceed)", outcar_text, re.I)
        capacity_stop = capacity_stop or re.search(
            r"not enough storage reserved for local reference configurations",
            outcar_text,
            re.I,
        )
        if not capacity_stop and not force_expand:
            raise SafetyError(
                "OUTCAR does not show a recognized ML_MB/ML_MCONF capacity stop; "
                "use --force-expand only after manual confirmation"
            )
    if operation == "expand" and (ml_mb is None or ml_mb < 1):
        raise SafetyError("--ml-mb must be a positive integer for expansion")

    archive = archive_run(run, operation)
    if (run / "CONTCAR").is_file() and (run / "CONTCAR").stat().st_size:
        shutil.copy2(run / "CONTCAR", run / "POSCAR")
    _clean_outputs(run)

    if operation in {"continue", "discard", "expand", "refit"}:
        destination = run / "ML_AB"
        if source != destination:
            shutil.copy2(source, destination)
        for name in ("ML_ABN", "ML_FF", "ML_FFN"):
            (run / name).unlink(missing_ok=True)
    else:
        shutil.copy2(source, run / "ML_FF")

    current = parse_incar(run / "INCAR")
    selected_temperature = float(
        temperature if temperature is not None else current.get("TEBEG", 300)
    )
    selected_nsw = int(nsw if nsw is not None else current.get("NSW", 3000))
    selected_potim = float(current.get("POTIM", 1.0))

    if operation in {"continue", "discard", "expand"}:
        changes, delete = stage_tags(
            "train",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
        if operation == "discard":
            changes["ML_LBASIS_DISCARD"] = ".TRUE."
            if increase_eps_low:
                try:
                    old_eps_low = float(current.get("ML_EPS_LOW", "1E-9"))
                except ValueError as error:
                    raise SafetyError("INCAR contains an invalid ML_EPS_LOW value") from error
                new_eps_low = old_eps_low * 10.0
                if not 0.0 < new_eps_low < 1.0e-7:
                    raise SafetyError(
                        "Tenfold ML_EPS_LOW increase must remain strictly below 1E-7; "
                        f"current value is {old_eps_low:g}"
                    )
                changes["ML_EPS_LOW"] = f"{new_eps_low:.0E}"
        elif operation == "expand":
            changes["ML_MB"] = int(ml_mb)
            if ml_mconf is not None:
                changes["ML_MCONF"] = int(ml_mconf)
            changes["ML_LBASIS_DISCARD"] = ".FALSE."
    elif operation == "refit":
        changes, delete = stage_tags(
            "refit",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
    else:
        changes, delete = stage_tags(
            "stability",
            temperature=selected_temperature,
            nsw=selected_nsw,
            potim=selected_potim,
        )
        if operation == "heat":
            # Green-Kubo heat-flux production: identical validated settings
            # as "stability", plus ML_LHEAT to write ML_HEAT for
            # postprocessing. See https://vasp.at/wiki/ML_LHEAT
            changes["ML_LHEAT"] = ".TRUE."
            changes["ML_OUTBLOCK"] = int(ml_outblock)
    changes["ISTART"] = 0
    delete.add("ICHARG")
    update_incar(run / "INCAR", changes, delete=delete)

    return {
        "folder": str(run),
        "operation": operation,
        "archive": str(archive),
        "temperature": selected_temperature,
        "nsw": selected_nsw if operation != "refit" else None,
        "ml_mb": ml_mb,
        "ml_mconf": ml_mconf,
        "ml_eps_low": parse_incar(run / "INCAR").get("ML_EPS_LOW"),
        "ml_outblock": ml_outblock if operation == "heat" else None,
    }


def resolve_launcher(folder: str | Path, launcher: str | None = None) -> Path:
    """Resolve an explicit launcher or prefer the standalone VASP launcher."""

    run = Path(folder).resolve()
    if launcher:
        script = run / launcher
        if not script.is_file():
            raise FileNotFoundError(script)
        return script
    for name in ("runvasp.sh", "run.slurm"):
        script = run / name
        if script.is_file():
            return script
    raise FileNotFoundError(
        f"No VASP launcher found in {run}; expected runvasp.sh or run.slurm"
    )


def submit_run(
    folder: str | Path,
    launcher: str | None = None,
    *,
    potcar_root: str | Path | None = None,
    potcar_mapping: str | Path | None = None,
) -> str:
    """Submit one prepared run and return the scheduler job id."""

    run = Path(folder).resolve()
    ensure_run_potcar(
        run,
        pseudopotential_root=potcar_root,
        mapping_file=potcar_mapping,
    )
    script = resolve_launcher(run, launcher)
    result = subprocess.run(
        ["sbatch", script.name],
        cwd=run,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _poscar_elements(poscar: Path) -> list[str]:
    """Read VASP 5+ species, with the legacy first-line convention as fallback."""

    lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise SafetyError(f"POSCAR is too short: {poscar}")
    symbols = lines[5].split()
    if symbols and all(re.fullmatch(r"\d+", token) for token in symbols):
        counts = symbols
        symbols = lines[0].split()
    else:
        counts = lines[6].split()
    if not symbols or any(not re.fullmatch(r"[A-Z][a-z]?", token) for token in symbols):
        raise SafetyError(
            "POSCAR contains neither a valid VASP 5+ species line nor a legacy "
            "first-line element list"
        )
    if len(counts) != len(symbols) or not all(re.fullmatch(r"\d+", token) for token in counts):
        raise SafetyError(f"POSCAR element and count lines are inconsistent: {poscar}")
    return symbols


def _poscar_ion_count(poscar: Path) -> int:
    """Total number of ions from a VASP 5+ or legacy POSCAR counts line."""

    lines = poscar.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise SafetyError(f"POSCAR is too short: {poscar}")
    counts = lines[5].split()
    if not all(re.fullmatch(r"\d+", token) for token in counts):
        counts = lines[6].split()
    if not counts or not all(re.fullmatch(r"\d+", token) for token in counts):
        raise SafetyError(f"POSCAR has no valid ion-count line: {poscar}")
    return sum(int(token) for token in counts)


def assemble_potcar(
    poscar: str | Path,
    output: str | Path,
    *,
    pseudopotential_root: str | Path,
    mapping_file: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble POTCAR from an explicit licensed local pseudopotential tree."""

    poscar_path = Path(poscar).resolve()
    output_path = Path(output).resolve()
    root = Path(pseudopotential_root).expanduser().resolve()
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing POTCAR: {output_path}")
    if mapping_file is None:
        mapping_label = "built-in POTCAR_DEFS mapping"
        mapping_text = resources.files("interfaceforge").joinpath(
            "templates/potcar_pbe_54.yaml"
        ).read_text(encoding="utf-8")
    else:
        mapping_path = Path(mapping_file).resolve()
        mapping_label = str(mapping_path)
        mapping_text = mapping_path.read_text(encoding="utf-8")
    mapping = yaml.safe_load(mapping_text)
    if not isinstance(mapping, Mapping):
        raise SafetyError(f"Invalid POTCAR map: {mapping_label}")
    elements = _poscar_elements(poscar_path)
    source_files: list[Path] = []
    variants: list[str] = []
    for element in elements:
        variant = str(mapping.get(element, "")).strip()
        if not variant:
            raise SafetyError(f"No POTCAR mapping for element {element}")
        source = root / variant / "POTCAR"
        if not source.is_file() or not source.stat().st_size:
            raise SafetyError(f"Missing licensed POTCAR source: {source}")
        source_files.append(source)
        variants.append(variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("wb") as destination:
        for source in source_files:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, destination)
    temporary.replace(output_path)
    return {
        "output": str(output_path),
        "elements": elements,
        "variants": variants,
        "sources": [str(path) for path in source_files],
        "mapping": mapping_label,
    }


def resolve_potcar_root(explicit: str | Path | None = None) -> Path:
    """Find the licensed local PBE PAW tree without bundling POTCAR data."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    if value := os.environ.get("IFACE_POTCAR_ROOT"):
        candidates.append(Path(value).expanduser())
    if value := os.environ.get("VASP_PP_PATH"):
        base = Path(value).expanduser()
        candidates.extend((base / "potpaw_PBE", base))
    candidates.append(Path.home() / "pot" / "potpaw_PBE")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise SafetyError(
        "POTCAR is missing and no licensed PBE PAW tree was found. "
        "Pass --potcar-root, set IFACE_POTCAR_ROOT, or set VASP_PP_PATH. "
        f"Searched: {searched}"
    )


def ensure_run_potcar(
    folder: str | Path,
    *,
    pseudopotential_root: str | Path | None = None,
    mapping_file: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a missing/empty run POTCAR from POSCAR before submission."""

    run = Path(folder).resolve()
    output = run / "POTCAR"
    if output.is_file() and output.stat().st_size:
        return {"status": "existing", "output": str(output)}
    require_files(run, ("POSCAR",))
    root = resolve_potcar_root(pseudopotential_root)
    payload = assemble_potcar(
        run / "POSCAR",
        output,
        pseudopotential_root=root,
        mapping_file=mapping_file,
        force=output.exists(),
    )
    payload["status"] = "generated"
    return payload


def prepare_standard_restart(
    folder: str | Path,
    *,
    from_contcar: bool = True,
    clean_electronic: bool = False,
) -> dict[str, Any]:
    """Prepare an ordinary VASP restart without submitting or regenerating POTCAR."""

    run = Path(folder).resolve()
    require_files(run, ("INCAR", "POSCAR", "POTCAR", "KPOINTS"))
    if from_contcar:
        require_files(run, ("CONTCAR",))
    archive = archive_run(run, "vasp_restart")
    if from_contcar:
        shutil.copy2(run / "CONTCAR", run / "POSCAR")
    delete = {"ICHARG"}
    changes: dict[str, Any] = {}
    if clean_electronic:
        delete.add("ISTART")
        for name in ("WAVECAR", "CHG", "CHGCAR"):
            (run / name).unlink(missing_ok=True)
        changes["ISTART"] = 0
    update_incar(run / "INCAR", changes, delete=delete)
    return {
        "folder": str(run),
        "archive": str(archive),
        "from_contcar": from_contcar,
        "clean_electronic": clean_electronic,
        "submitted": False,
    }


def prepare_band_run(
    source_folder: str | Path,
    destination: str | Path,
    *,
    line_kpoints: str | Path,
    lmaxmix: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare a non-self-consistent line-mode band calculation."""

    source = Path(source_folder).resolve()
    target = Path(destination).resolve()
    kpoints = Path(line_kpoints).resolve()
    require_files(source, ("INCAR", "POSCAR", "POTCAR", "CHGCAR"))
    if not kpoints.is_file() or not kpoints.stat().st_size:
        raise SafetyError(f"Missing line-mode KPOINTS: {kpoints}")
    if target.exists() and any(target.iterdir()) and not force:
        raise SafetyError(f"Refusing to replace nonempty band directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("INCAR", "POSCAR", "POTCAR", "CHGCAR", "run.slurm"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, target / name)
    if (source / "CONTCAR").is_file() and (source / "CONTCAR").stat().st_size:
        shutil.copy2(source / "CONTCAR", target / "POSCAR")
    shutil.copy2(kpoints, target / "KPOINTS")
    update_incar(
        target / "INCAR",
        {
            "ICHARG": 11,
            "NSW": 0,
            "IBRION": -1,
            "ALGO": "Normal",
            "LREAL": "Auto",
            "LMAXMIX": int(lmaxmix),
            "ISMEAR": 0,
        },
        delete={"KPAR", "NCORE"},
    )
    return {"source": str(source), "destination": str(target), "kpoints": str(kpoints)}


def package_outputs(
    root: str | Path,
    output: str | Path,
    *,
    include_large: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Create a lightweight, non-destructive archive of reproducibility outputs."""

    source_root = Path(root).resolve()
    output_path = Path(output).resolve()
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing archive: {output_path}")
    names = {
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "CONTCAR",
        "OSZICAR",
        "ML_LOGFILE",
        "run.slurm",
        "EIGENVAL",
        "DOSCAR",
    }
    if include_large:
        names.update({"OUTCAR", "vasprun.xml", "XDATCAR", "LOCPOT"})
    # POTCAR is intentionally excluded from a portable archive.
    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.name in names
        and ".interfaceforge/archive" not in path.as_posix()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(source_root))
    return {
        "root": str(source_root),
        "output": str(output_path),
        "files": len(files),
        "include_large": include_large,
        "potcar_excluded": True,
    }


_MODEL_ARCHIVE_FILES = {
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "CONTCAR",
    "OSZICAR",
    "ML_LOGFILE",
    "REPORT",
    "run.slurm",
    "runvasp.sh",
    "XDATCAR_FINAL",
    "vasp_md_FINAL.dat",
    "ML_AB",
    "ML_ABN",
    "ML_FF",
    "ML_FFN",
}
_MODEL_ARCHIVE_LARGE_FILES = {"OUTCAR", "vasprun.xml", "XDATCAR", "LOCPOT"}
_MODEL_ARCHIVE_MANIFEST = "interfaceforge-model-archive.json"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_model_directory_name(name: str, excluded_names: set[str]) -> bool:
    return (
        "backup" in name.casefold()
        or name.startswith("X")
        or name in excluded_names
    )


def _discover_ml_ab_files(
    source_root: Path,
    *,
    excluded_names: set[str],
    recursive: bool,
) -> tuple[list[Path], list[str]]:
    """Find ML_AB files while pruning backup and generated-package trees."""

    if _excluded_model_directory_name(source_root.name, excluded_names):
        return [], ["."]

    models: list[Path] = []
    excluded_directories: list[str] = []
    for current, directory_names, file_names in os.walk(source_root, topdown=True):
        current_path = Path(current)
        if not recursive and current_path != source_root:
            directory_names[:] = []
        else:
            retained_directories: list[str] = []
            for name in directory_names:
                candidate = current_path / name
                if name == ".interfaceforge" or candidate.is_symlink():
                    continue
                if _excluded_model_directory_name(name, excluded_names):
                    excluded_directories.append(candidate.relative_to(source_root).as_posix())
                    continue
                retained_directories.append(name)
            directory_names[:] = retained_directories

        if "ML_AB" not in file_names:
            continue
        model = current_path / "ML_AB"
        if model.is_file() and not model.is_symlink() and model.stat().st_size:
            models.append(model)
    return sorted(models), sorted(excluded_directories)


def archive_mlff_models(
    root: str | Path,
    output: str | Path | None = None,
    *,
    include_large: bool = False,
    exclude_folders: Iterable[str] = (),
    recursive: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Archive user-selected VASP-MLFF runs containing a nonempty ML_AB.

    Model presence is used only for discovery. The command intentionally does
    not claim that a model is scientifically validated; callers should point it
    at runs they have already accepted. POTCAR is never included.
    """

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise SafetyError(f"Model archive root is not a directory: {source_root}")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder_name = source_root.name or "root"
        output_path = (Path.cwd() / f"MLFF_Models_{folder_name}_{stamp}.zip").resolve()
    else:
        output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing archive: {output_path}")

    excluded_names = {str(name).strip() for name in exclude_folders if str(name).strip()}
    model_paths, excluded_directories = _discover_ml_ab_files(
        source_root,
        excluded_names=excluded_names,
        recursive=recursive,
    )
    if not model_paths:
        raise SafetyError(
            f"No nonempty ML_AB files found below {source_root} after excluding "
            "configured directories and applying the selected scan depth"
        )

    selected_names = set(_MODEL_ARCHIVE_FILES)
    if include_large:
        selected_names.update(_MODEL_ARCHIVE_LARGE_FILES)

    runs: list[dict[str, Any]] = []
    files_to_archive: list[tuple[Path, Path]] = []
    for model_path in model_paths:
        run = model_path.parent
        relative_run = run.relative_to(source_root)
        artifacts: list[dict[str, Any]] = []
        for name in sorted(selected_names):
            path = run / name
            if not path.is_file() or path.is_symlink() or not path.stat().st_size:
                continue
            archive_name = relative_run / name
            artifacts.append(
                {
                    "path": archive_name.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
            files_to_archive.append((path, archive_name))
        runs.append(
            {
                "relative_path": relative_run.as_posix() or ".",
                "artifacts": artifacts,
            }
        )

    manifest = {
        "format": "interfaceforge-vasp-mlff-model-archive",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "selection": "directories containing a nonempty ML_AB; scientific acceptance is user-supplied",
        "include_large": include_large,
        "recursive": recursive,
        "potcar_excluded": True,
        "large_restart_files_excluded": ["CHG", "CHGCAR", "WAVECAR"],
        "directory_exclusions": {
            "rules": [
                "name contains 'backup' (case-insensitive)",
                "name starts with 'X'",
                "exact user-supplied folder names",
            ],
            "requested_names": sorted(excluded_names),
            "excluded": excluded_directories,
        },
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path, archive_name in files_to_archive:
                archive.write(path, archive_name.as_posix())
            archive.writestr(
                _MODEL_ARCHIVE_MANIFEST,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    largest_files = sorted(
        (
            {"path": archive_name.as_posix(), "size_bytes": path.stat().st_size}
            for path, archive_name in files_to_archive
        ),
        key=lambda item: int(item["size_bytes"]),
        reverse=True,
    )[:10]
    return {
        "root": str(source_root),
        "output": str(output_path),
        "archive_sha256": _sha256_file(output_path),
        "archive_size_bytes": output_path.stat().st_size,
        "total_uncompressed_bytes": sum(path.stat().st_size for path, _ in files_to_archive),
        "runs": len(runs),
        "files": len(files_to_archive),
        "excluded_directories": excluded_directories,
        "requested_exclude_folders": sorted(excluded_names),
        "largest_files": largest_files,
        "include_large": include_large,
        "recursive": recursive,
        "manifest": _MODEL_ARCHIVE_MANIFEST,
        "potcar_excluded": True,
        "chg_chgcar_wavecar_excluded": True,
    }
