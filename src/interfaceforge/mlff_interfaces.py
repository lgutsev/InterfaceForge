"""Bulk VASP-MLFF training-campaign generation for a family x termination x
composition grid of interface structures (e.g. Real/Ideal x N_Term/Ti_Term x
x=0,0.25,0.5,0.75,1.0), plus a throttled mass-launcher and a grid-aware audit
rollup.

This module deliberately does not duplicate the existing campaign machinery
(``prepare_campaign``/``submit_campaign``/``stage_tags``/``run_audit``): its
job is (1) turning a reviewed source manifest into a correct ``campaign.yaml``
that machinery already knows how to prepare, submit, and audit, and (2) the
two genuinely new pieces that machinery does not provide -- Slurm-array
throttled mass submission, and a rollup of ``iface audit``'s flat run table
grouped by the family/termination/x grid instead of by individual run.

Source discovery is intentionally two-step (discover -> human review of the
written CSV -> build) rather than one-shot: this repo has no visibility into
how any particular Step2-style source tree is actually laid out, so
auto-discovery is best-effort and reported, never silently trusted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .audit import find_runs, run_audit
from .campaign import submission_candidates
from .config import Campaign
from .errors import SafetyError
from .scheduler import render_job, write_job
from .vasp import parse_incar

DEFAULT_FAMILIES = ("Real", "Ideal")
DEFAULT_TERMS = ("N_Term", "Ti_Term")
DEFAULT_X_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SOURCE_INPUTS = ("CONTCAR", "INCAR", "KPOINTS", "POTCAR")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ancestor_input(leaf: Path, name: str, boundary: Path) -> Path | None:
    """Resolve a leaf input without escaping the reviewed source tree."""

    current = leaf
    while current == boundary or boundary in current.parents:
        candidate = current / name
        if candidate.is_file() and candidate.stat().st_size:
            return candidate.resolve()
        if current == boundary:
            break
        current = current.parent
    return None


def _mlff_is_fast(path: Path) -> bool:
    if not path.is_file() or not path.stat().st_size:
        return False
    with path.open("rb") as handle:
        header = handle.read(4096).decode("utf-8", errors="ignore")
    return bool(re.search(r"ML_LFAST.{0,20}(true|T)", header, re.I))


def _x_label(value: float) -> str:
    return f"{value:g}"


def discover_mlff_interface_sources(
    source_root: str | Path,
    output_manifest: str | Path,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    x_values: tuple[float, ...] = DEFAULT_X_VALUES,
    structure_name: str = "CONTCAR",
) -> dict[str, Any]:
    """Best-effort discovery of one equilibrated structure per grid cell.

    Matches each ``structure_name`` file found under ``source_root`` against
    every (family, term, x) combination by simple case-insensitive token
    matching against its path components -- "real"/"ideal",
    "n_term"/"nterm"/"n-term" (and the ti_term equivalents), and an
    ``x``-prefixed or bare numeric directory matching that x value. A cell
    with zero or more than one match is reported, never guessed: the written
    CSV's ``match_status`` is ``matched``, ``missing``, or ``ambiguous``
    (with candidates listed), and this function does not raise on an
    incomplete grid -- review and hand-fix the CSV, then pass it to
    ``generate_mlff_interfaces_campaign``.
    """

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise SafetyError(f"Source root is not a directory: {root}")
    candidates = sorted(root.rglob(structure_name))

    def _term_tokens(term: str) -> list[str]:
        base = term.lower()
        return [base, base.replace("_", "-"), base.replace("_", "")]

    # Match x by numeric value rather than a single formatted string: a
    # directory can spell x=1.0 as "x1", "x1.0", "x=1.00", etc.
    number_token = re.compile(r"(?<![a-z0-9.])x?[-_=]?(\d+(?:\.\d+)?)(?![a-z0-9])", re.I)

    def _path_x_values(text: str) -> list[float]:
        values = []
        for match in number_token.finditer(text):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
        return values

    rows: list[dict[str, Any]] = []
    for family in families:
        for term in terms:
            for x in x_values:
                matches: list[Path] = []
                for path in candidates:
                    text = str(path.relative_to(root)).lower()
                    if not re.search(rf"(?<![a-z]){re.escape(family.lower())}(?![a-z])", text):
                        continue
                    term_hit = any(
                        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text)
                        for token in _term_tokens(term)
                    )
                    if not term_hit:
                        continue
                    path_x_values = _path_x_values(text)
                    is_x_match = any(math.isclose(candidate, x, abs_tol=1e-6) for candidate in path_x_values)
                    # A real observed convention: the x=0 (oxygen-free) baseline
                    # carries no numeric suffix at all (e.g. "SiN_TiN_N-term"),
                    # unlike the oxygen-substituted cells ("..._O_x0.25"). When
                    # x==0 and nothing else under this family/term has *any*
                    # x-like numeric token, treat the bare leaf as the match.
                    is_bare_zero = x == 0.0 and not path_x_values
                    if not (is_x_match or is_bare_zero):
                        continue
                    matches.append(path)
                if len(matches) == 1:
                    status = "matched"
                    # Reuse the source tree's own leaf directory name so the
                    # generated run tree ("runs/vasp/<system_id>/<stage>") is
                    # literally family/term/<the same leaf name Step2 uses>,
                    # not a synthesized id -- this is what lets the generated
                    # tree read as "Step2, but for MLFF training" rather than
                    # a flattened restructuring of it.
                    system_id = f"{family}/{term}/{matches[0].parent.name}"
                elif not matches:
                    status = "missing"
                    system_id = f"{family}/{term}/x{_x_label(x)}"
                else:
                    status = "ambiguous"
                    system_id = f"{family}/{term}/x{_x_label(x)}"
                row = {
                        "system_id": system_id,
                        "family": family,
                        "term": term,
                        "x": _x_label(x),
                        "structure_path": str(matches[0]) if len(matches) == 1 else "",
                        "match_status": status,
                        "candidates": "; ".join(str(m) for m in matches),
                    }
                if len(matches) == 1:
                    leaf = matches[0].parent
                    missing_inputs: list[str] = []
                    for name in SOURCE_INPUTS:
                        source = matches[0] if name == structure_name else _ancestor_input(leaf, name, root)
                        row[f"{name.lower()}_path"] = str(source) if source else ""
                        row[f"{name.lower()}_sha256"] = _sha256(source) if source else ""
                        if source is None:
                            missing_inputs.append(name)
                    row["inputs_status"] = "complete" if not missing_inputs else "missing"
                    row["missing_inputs"] = ";".join(missing_inputs)
                else:
                    for name in SOURCE_INPUTS:
                        row[f"{name.lower()}_path"] = ""
                        row[f"{name.lower()}_sha256"] = ""
                    row["inputs_status"] = "unresolved"
                    row["missing_inputs"] = ";".join(SOURCE_INPUTS)
                rows.append(row)

    manifest_path = Path(output_manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["system_id", "family", "term", "x", "structure_path", "match_status", "candidates"]
        fieldnames += [field for name in SOURCE_INPUTS for field in (f"{name.lower()}_path", f"{name.lower()}_sha256")]
        fieldnames += ["inputs_status", "missing_inputs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["match_status"] for row in rows)
    return {
        "source_root": str(root),
        "manifest": str(manifest_path),
        "grid_size": len(rows),
        "structure_files_seen": len(candidates),
        "status_counts": dict(counts),
        "rows": rows,
    }


def generate_mlff_interfaces_campaign(
    manifest_source: str | Path,
    campaign_root: str | Path,
    *,
    profile_path: str | Path,
    profile_name: str = "vasp_train",
    encut: float = 520.0,
    ivdw: int = 11,
    tebeg: float = 300.0,
    teend: float = 600.0,
    train_nsw: int = 3000,
    refit_nsw: int = 0,
    stability_nsw: int = 3000,
    potim: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    """Build a ready-to-``iface prepare`` campaign.yaml from a reviewed manifest.

    Every row must have ``match_status == "matched"`` (see
    ``discover_mlff_interface_sources``); this refuses to generate a
    campaign from an incomplete or ambiguous grid rather than silently
    dropping cells. Exact per-system VASP inputs are snapshotted and used
    during preparation; oxygen-free and oxygen-containing interfaces never
    share or regenerate a POTCAR.
    """

    manifest_path = Path(manifest_source).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SafetyError(f"No rows in manifest: {manifest_path}")
    bad = [
        row["system_id"]
        for row in rows
        if row.get("match_status") != "matched" or row.get("inputs_status") != "complete"
    ]
    if bad:
        raise SafetyError(
            f"Manifest has {len(bad)} unresolved grid cell(s), refusing to generate a "
            f"campaign: {', '.join(bad)}. Resolve every structure and its exact "
            f"INCAR/KPOINTS/POTCAR in {manifest_path} first."
        )

    root = Path(campaign_root).resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise SafetyError(f"Refusing to write into a nonempty campaign root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    systems: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for row in rows:
        system_id = row["system_id"]
        snapshot = root / "inputs" / "systems" / system_id
        snapshot.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for name in SOURCE_INPUTS:
            source = Path(row[f"{name.lower()}_path"])
            expected_hash = row[f"{name.lower()}_sha256"]
            if _sha256(source) != expected_hash:
                raise SafetyError(f"Source changed since discovery: {source}")
            destination = snapshot / name
            shutil.copy2(source, destination)
            copied[name] = str(destination.relative_to(root))
            provenance_rows.append(
                {
                    "system_id": system_id,
                    "name": name,
                    "source": str(source),
                    "snapshot": copied[name],
                    "sha256": expected_hash,
                }
            )
        source_tags = parse_incar(snapshot / "INCAR")
        for tag, expected in (("ENCUT", encut), ("IVDW", ivdw), ("POTIM", potim)):
            actual = source_tags.get(tag)
            try:
                matches = math.isclose(float(actual), float(expected), abs_tol=1e-10)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise SafetyError(
                    f"{system_id}: source INCAR {tag}={actual!r}, expected {expected!r}; "
                    "fix or explicitly regenerate the reviewed manifest/campaign"
                )
        systems.append(
            {
                "id": system_id,
                "kind": "interface",
                "structure": copied["CONTCAR"],
                "inputs": {name: copied[name] for name in ("INCAR", "KPOINTS", "POTCAR")},
                "tags": {"family": row["family"], "term": row["term"], "x": row["x"]},
            }
        )

    provenance_path = root / "inputs" / "source_provenance.json"
    provenance_path.write_text(json.dumps(provenance_rows, indent=2) + "\n", encoding="utf-8")

    campaign_data = {
        "schema_version": 1,
        "project": {
            "name": "mlff-thermal-conductivity-grid",
            "description": (
                "Real/Ideal x N_Term/Ti_Term x composition MLFF training grid "
                "for VASP ML_LHEAT Green-Kubo thermal conductivity."
            ),
        },
        "profile": str(Path(profile_path).resolve()),
        "systems": systems,
        "reference": {"engine": "vasp", "inputs": {}},
        "stages": {
            "vasp_mlff": {
                "enabled": True,
                "train": {
                    "temperature": tebeg,
                    "teend": teend,
                    "nsw": train_nsw,
                    "potim": potim,
                    "profile": profile_name,
                },
                "refit": {"nsw": refit_nsw, "potim": potim, "profile": profile_name},
                "stability": {
                    "temperature": tebeg,
                    "nsw": stability_nsw,
                    "potim": potim,
                    "profile": profile_name,
                },
            }
        },
        "dataset": {"strategy": "grouped", "ratios": [0.8, 0.1, 0.1]},
        "models": {},
        "active_learning": {"enabled": False},
        "exploration": {},
        "validation": {},
    }
    campaign_path = root / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign_data, sort_keys=False), encoding="utf-8")

    return {
        "campaign_root": str(root),
        "campaign": str(campaign_path),
        "systems": len(systems),
        "grid": sorted({(row["family"], row["term"]) for row in rows}),
        "x_values": sorted({row["x"] for row in rows}),
        "provenance": str(provenance_path),
        "note": "Exact per-system CONTCAR/INCAR/KPOINTS/POTCAR snapshots are immutable campaign inputs.",
    }


def write_throttled_array_launcher(
    campaign: Campaign,
    *,
    stage: str = "train",
    concurrency: int = 4,
    array_profile_name: str | None = None,
    output: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one Slurm array job that runs VASP directly in every leaf,
    throttled to ``concurrency`` concurrent tasks via Slurm's native
    ``--array=0-N%K`` syntax.

    Requires ``iface prepare`` to have already run. The array uses the selected
    stage profile's VASP command directly; it never executes a nested
    ``run.slurm`` whose ``SLURM_SUBMIT_DIR`` would point at the array's launch
    directory instead of the intended leaf.
    """

    if concurrency < 1:
        raise SafetyError("concurrency must be at least 1")
    launchers = submission_candidates(campaign, stage=stage)
    if not launchers:
        raise SafetyError(
            f"No prepared run.slurm files found for stage {stage!r}; run iface prepare first"
        )
    array_dir = Path(output).resolve() if output else campaign.root / "runs" / "vasp" / "_arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = array_dir / f"{stage}_array_manifest.txt"
    run_directories = [path.parent for path in launchers]
    manifest_path.write_text("\n".join(str(path) for path in run_directories) + "\n", encoding="utf-8")

    profile = yaml.safe_load(campaign.profile_path.read_text(encoding="utf-8"))
    stage_settings = dict(campaign.stages.get("vasp_mlff", {}).get(stage, {}))
    execution_profile_name = str(stage_settings.get("profile", "vasp_workq"))
    jobs = profile.get("jobs", {})
    if execution_profile_name not in jobs:
        raise SafetyError(f"Profile has no stage job named {execution_profile_name!r}")
    resource_profile_name = array_profile_name or execution_profile_name
    if resource_profile_name not in jobs:
        raise SafetyError(
            f"Profile has no job named {resource_profile_name!r}; add one sized for a single "
            f"{stage} leaf (the array multiplies it by concurrency, not by task count)"
        )
    engine_command = str(jobs[execution_profile_name].get("command", "")).strip()
    if not engine_command:
        raise SafetyError(f"Profile job {execution_profile_name!r} has no VASP command")
    execution_job = dict(jobs[execution_profile_name])
    resource_job = dict(jobs[resource_profile_name])
    for runtime_key in ("modules", "preamble", "environment"):
        if runtime_key in execution_job:
            resource_job[runtime_key] = execution_job[runtime_key]
    render_profile = {**profile, "jobs": {**jobs, resource_profile_name: resource_job}}
    command = (
        f'MANIFEST="{manifest_path}"\n'
        'LEAF_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")\n'
        'if [[ -z "$LEAF_DIR" ]]; then echo "No manifest line for task $SLURM_ARRAY_TASK_ID" >&2; exit 2; fi\n'
        'cd "$LEAF_DIR"\n'
        'for required in INCAR KPOINTS POSCAR POTCAR; do\n'
        '  [[ -s "$required" ]] || { echo "Missing $LEAF_DIR/$required" >&2; exit 3; }\n'
        'done\n'
        f'{engine_command}\n'
    )
    script = render_job(
        render_profile,
        resource_profile_name,
        command=command,
        job_name=f"{campaign.name}_{stage}_array",
        array=f"0-{len(launchers) - 1}%{concurrency}",
        working_directory=str(array_dir),
    )
    script_path = array_dir / f"{stage}_array.slurm"
    write_job(script_path, script, force=force)
    return {
        "stage": stage,
        "leaves": len(launchers),
        "concurrency": concurrency,
        "manifest": str(manifest_path),
        "launcher": str(script_path),
        "profile": resource_profile_name,
        "submitted": False,
    }


def mass_audit_mlff_interfaces(
    campaign: Campaign,
    *,
    readiness_profile: str = "general",
) -> dict[str, Any]:
    """Audit the whole grid via the existing ``iface audit`` engine, then
    roll the flat per-run table up by (family, term, x) -- one row per grid
    cell aggregating its train/refit/stability runs, plus totals per family
    and per term.

    Family/term/x for each run come from ``campaign.systems[i].tags`` (set
    by ``generate_mlff_interfaces_campaign``), looked up by system id -- the
    run's ``relative_path`` with its last path component (the stage) split
    off, since a system id can itself be "/"-nested (e.g.
    "Real/N_Term/SiN_TiN_N-term") to mirror a source tree's own directory
    structure, rather than by re-deriving family/term/x from the id text.
    """

    runs_root = campaign.root / "runs" / "vasp"
    if not find_runs(runs_root):
        raise SafetyError(f"No run directories found below {runs_root}; run iface prepare first")
    payload = run_audit(runs_root, readiness_profile=readiness_profile)

    provenance_path = campaign.root / "inputs" / "source_provenance.json"
    provenance_records = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else []
    )
    provenance_checks: list[dict[str, Any]] = []
    for record in provenance_records:
        source = Path(record["source"])
        snapshot = campaign.root / record["snapshot"]
        expected = record["sha256"]
        source_hash = _sha256(source) if source.is_file() else None
        snapshot_hash = _sha256(snapshot) if snapshot.is_file() else None
        provenance_checks.append(
            {
                **record,
                "source_hash": source_hash,
                "snapshot_hash": snapshot_hash,
                "source_unchanged": source_hash == expected,
                "snapshot_verified": snapshot_hash == expected,
            }
        )
    source_incars = {
        system.id: parse_incar(system.inputs["INCAR"])
        for system in campaign.systems
        if "INCAR" in system.inputs and system.inputs["INCAR"].is_file()
    }
    all_incar_tags = sorted({tag for tags in source_incars.values() for tag in tags})
    incar_differences = {
        tag: {
            str(value): sorted(
                system_id
                for system_id, tags in source_incars.items()
                if tags.get(tag, "<missing>") == value
            )
            for value in sorted(
                {tags.get(tag, "<missing>") for tags in source_incars.values()}, key=str
            )
        }
        for tag in all_incar_tags
        if len({tags.get(tag, "<missing>") for tags in source_incars.values()}) > 1
    }

    tags_by_system = {system.id: system.tags for system in campaign.systems}
    cells: dict[str, dict[str, Any]] = {}
    unparsed: list[str] = []
    for row in payload["runs"]:
        relative = row["relative_path"].replace("\\", "/")
        if "/" not in relative:
            unparsed.append(row["relative_path"])
            continue
        # stage is always the last path component; system_id is everything
        # before it and may itself be "/"-nested (e.g. "Real/N_Term/leaf"),
        # so this must split off the *last* segment, not the first.
        system_id, stage = relative.rsplit("/", 1)
        tags = tags_by_system.get(system_id)
        if tags is None:
            unparsed.append(row["relative_path"])
            continue
        cell = cells.setdefault(
            system_id,
            {"family": tags.get("family"), "term": tags.get("term"), "x": tags.get("x"), "stages": {}},
        )
        run = runs_root / row["relative_path"]
        expected_inputs = {
            "POSCAR": campaign.systems[[item.id for item in campaign.systems].index(system_id)].structure,
            **campaign.systems[[item.id for item in campaign.systems].index(system_id)].inputs,
        }
        input_checks = {
            name: target.is_file() and source.is_file() and _sha256(target) == _sha256(source)
            for name, source in expected_inputs.items()
            if name in {"POSCAR", "KPOINTS", "POTCAR"}
            for target in (run / name,)
        }
        incar_tags = parse_incar(run / "INCAR") if (run / "INCAR").is_file() else {}
        setting_checks = {
            "ENCUT": incar_tags.get("ENCUT"),
            "IVDW": incar_tags.get("IVDW"),
            "POTIM": incar_tags.get("POTIM"),
            "TEBEG": incar_tags.get("TEBEG"),
            "TEEND": incar_tags.get("TEEND"),
            "ML_MODE": incar_tags.get("ML_MODE"),
            "ML_LHEAT": incar_tags.get("ML_LHEAT"),
        }
        cell["stages"][stage] = {
            "health": row["health"],
            "next_action": row["next_action"],
            "inputs_verified": input_checks,
            "settings": setting_checks,
            "ml_ab_available": any(
                (run / name).is_file() and (run / name).stat().st_size for name in ("ML_ABN", "ML_AB")
            ),
            "fast_mlff_available": _mlff_is_fast(run / "ML_FFN"),
            "ml_heat_available": (run / "ML_HEAT").is_file() and (run / "ML_HEAT").stat().st_size > 0,
        }

    by_family: dict[str, Counter] = defaultdict(Counter)
    by_term: dict[str, Counter] = defaultdict(Counter)
    for cell in cells.values():
        train_health = cell["stages"].get("train", {}).get("health", "not started")
        by_family[cell["family"]][train_health] += 1
        by_term[cell["term"]][train_health] += 1

    all_train = [cell["stages"].get("train", {}) for cell in cells.values()]
    all_grid_cells_present = len(cells) == len(campaign.systems)
    result = {
        "schema_version": 1,
        "campaign_root": str(campaign.root),
        "readiness_profile": readiness_profile,
        "grid_cells": len(cells),
        "cells": dict(sorted(cells.items())),
        "train_health_by_family": {family: dict(counts) for family, counts in by_family.items()},
        "train_health_by_term": {term: dict(counts) for term, counts in by_term.items()},
        "unparsed_runs": unparsed,
        "provenance": {
            "records": len(provenance_checks),
            "passed": bool(provenance_checks)
            and all(
                row["source_unchanged"] and row["snapshot_verified"]
                for row in provenance_checks
            ),
            "checks": provenance_checks,
        },
        "incar_comparison": {
            "systems": len(source_incars),
            "all_identical": not incar_differences,
            "differing_tags": incar_differences,
        },
        "gates": {
            "refit_ready": all_grid_cells_present
            and all(stage.get("ml_ab_available", False) for stage in all_train),
            "heat_ready": all_grid_cells_present
            and all(
                any(stage.get("fast_mlff_available", False) for stage in cell["stages"].values())
                for cell in cells.values()
            ),
        },
        "flat_audit_outputs": payload["outputs"],
    }
    report_root = campaign.root / "reports" / "mlff_interfaces"
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "audit.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = report_root / "audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system_id",
                "family",
                "term",
                "x",
                "stage",
                "health",
                "inputs_verified",
                "ml_ab_available",
                "fast_mlff_available",
                "ml_heat_available",
            ],
        )
        writer.writeheader()
        for system_id, cell in sorted(cells.items()):
            for stage, details in sorted(cell["stages"].items()):
                writer.writerow(
                    {
                        "system_id": system_id,
                        "family": cell["family"],
                        "term": cell["term"],
                        "x": cell["x"],
                        "stage": stage,
                        "health": details["health"],
                        "inputs_verified": all(details["inputs_verified"].values()),
                        "ml_ab_available": details["ml_ab_available"],
                        "fast_mlff_available": details["fast_mlff_available"],
                        "ml_heat_available": details["ml_heat_available"],
                    }
                )
    result["outputs"] = {"json": str(json_path), "csv": str(csv_path)}
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
