"""Campaign planning, directory preparation, and controlled submission."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import Campaign, load_profile
from .errors import SafetyError
from .scheduler import render_job, write_job
from .state import StateStore
from .vasp import mlff_accuracy_profile_tags, stage_tags, update_incar

_LAYOUT = ("inputs", "structures", "runs/vasp", "datasets", "models", "validation", "reports", "logs")
_STAGES = ("train", "refit", "stability")


def build_plan(campaign: Campaign) -> dict[str, Any]:
    """Return a deterministic, serializable campaign plan."""

    vasp_settings = dict(campaign.stages.get("vasp_mlff", {}))
    # Generating new VASP-MLFF labels is an explicit opt-in.  Campaigns built
    # from existing DFT trajectories must never acquire MLFF jobs merely
    # because the stages block was omitted.
    vasp_enabled = bool(vasp_settings.get("enabled", False))
    tasks: list[dict[str, Any]] = []
    if vasp_enabled:
        for system in campaign.systems:
            for stage in _STAGES:
                settings = dict(vasp_settings.get(stage, {}))
                tasks.append(
                    {
                        "engine": "vasp_mlff",
                        "system": system.id,
                        "kind": system.kind,
                        "stage": stage,
                        "profile": settings.get("profile", "vasp_workq"),
                        "directory": str(
                            campaign.root / "runs" / "vasp" / system.id / stage
                        ),
                    }
                )
    for model_name in ("mace", "deepmd"):
        settings = dict(campaign.models.get(model_name, {}))
        if settings.get("enabled", False):
            tasks.append(
                {
                    "engine": model_name,
                    "stage": "train",
                    "profile": settings.get("profile", f"{model_name}_gpu"),
                    "directory": str(campaign.root / "models" / model_name),
                }
            )
    return {
        "schema_version": 1,
        "project": campaign.name,
        "campaign_file": str(campaign.path),
        "profile": str(campaign.profile_path),
        "systems": [
            {
                "id": item.id,
                "kind": item.kind,
                "structure": str(item.structure),
                "temperature": item.temperature,
                "tags": item.tags,
            }
            for item in campaign.systems
        ],
        "tasks": tasks,
    }


def _copy_if_present(source: Path, destination: Path, *, force: bool) -> bool:
    if not source.is_file():
        return False
    if destination.exists() and not force:
        return False
    shutil.copy2(source, destination)
    return True


def _resolve_reference_input(campaign: Campaign, name: str) -> Path | None:
    value = campaign.reference.get("inputs", {}).get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (campaign.root / path).resolve()


def prepare_campaign(campaign: Campaign, *, force: bool = False) -> dict[str, Any]:
    """Scaffold restartable VASP-MLFF runs without submitting jobs."""

    for relative in _LAYOUT:
        (campaign.root / relative).mkdir(parents=True, exist_ok=True)
    profile = load_profile(campaign.profile_path)
    plan = build_plan(campaign)
    vasp_settings = dict(campaign.stages.get("vasp_mlff", {}))
    vasp_enabled = bool(vasp_settings.get("enabled", False))
    accuracy_profile = vasp_settings.get("accuracy_profile")
    base_incar = _resolve_reference_input(campaign, "INCAR")
    kpoints = _resolve_reference_input(campaign, "KPOINTS")
    potcar = _resolve_reference_input(campaign, "POTCAR")
    prepared: list[dict[str, Any]] = []
    warnings: list[str] = []

    for system in campaign.systems if vasp_enabled else ():
        system_root = campaign.root / "runs" / "vasp" / system.id
        for stage in _STAGES:
            settings = dict(vasp_settings.get(stage, {}))
            run = system_root / stage
            run.mkdir(parents=True, exist_ok=True)
            created: list[str] = []

            if _copy_if_present(system.structure, run / "POSCAR", force=force):
                created.append("POSCAR")
            elif not (run / "POSCAR").is_file():
                warnings.append(f"{system.id}/{stage}: missing structure {system.structure}")
            for name, source in (("KPOINTS", kpoints), ("POTCAR", potcar)):
                if source is not None and _copy_if_present(source, run / name, force=force):
                    created.append(name)
                elif not (run / name).is_file():
                    warnings.append(f"{system.id}/{stage}: missing reference input {name}")

            incar = run / "INCAR"
            if base_incar and base_incar.is_file() and (force or not incar.exists()):
                shutil.copy2(base_incar, incar)
                created.append("INCAR")
            elif not incar.exists():
                incar.write_text(
                    "# InterfaceForge generated shell; add converged electronic-structure settings.\n",
                    encoding="utf-8",
                )
                created.append("INCAR")

            temperature = float(
                settings.get("temperature", system.temperature if system.temperature is not None else 300)
            )
            nsw = int(settings.get("nsw", 3000))
            potim = float(settings.get("potim", 1.0))
            teend_setting = settings.get("teend")
            teend = float(teend_setting) if teend_setting is not None else None
            changes, delete = stage_tags(
                stage, temperature=temperature, nsw=nsw, potim=potim, teend=teend
            )
            if accuracy_profile:
                changes.update(
                    mlff_accuracy_profile_tags(str(accuracy_profile), stage)
                )
            update_incar(incar, changes, delete=delete)

            profile_name = str(settings.get("profile", "vasp_workq"))
            launcher = render_job(
                profile,
                profile_name,
                job_name=f"{campaign.name}_{system.id}_{stage}",
            )
            write_job(run / "run.slurm", launcher, force=force)
            created.append("run.slurm")
            prepared.append(
                {
                    "system": system.id,
                    "stage": stage,
                    "directory": str(run),
                    "created": created,
                }
            )

    plan_path = campaign.root / ".interfaceforge" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    state = StateStore(campaign.root)
    state.event("prepare", campaign=str(campaign.path), runs=len(prepared), warnings=warnings)
    state.artifact("plan", plan_path)
    return {"prepared": prepared, "warnings": warnings, "plan": str(plan_path)}


def submission_candidates(
    campaign: Campaign,
    *,
    system: str | None = None,
    stage: str | None = None,
) -> list[Path]:
    plan = build_plan(campaign)
    candidates: list[Path] = []
    for task in plan["tasks"]:
        if task["engine"] != "vasp_mlff":
            continue
        if system and task["system"] != system:
            continue
        if stage and task["stage"] != stage:
            continue
        launcher = Path(task["directory"]) / "run.slurm"
        if launcher.is_file():
            candidates.append(launcher)
    return candidates


def submit_campaign(
    campaign: Campaign,
    *,
    system: str | None = None,
    stage: str | None = None,
    execute: bool = False,
) -> list[dict[str, str]]:
    """List or submit selected run scripts. Execution is explicit."""

    candidates = submission_candidates(campaign, system=system, stage=stage)
    if not candidates:
        raise SafetyError("No matching prepared run.slurm files were found")
    results: list[dict[str, str]] = []
    for launcher in candidates:
        if not execute:
            results.append({"launcher": str(launcher), "status": "dry-run"})
            continue
        from .vasp import submit_run

        job_id = submit_run(launcher.parent, launcher.name)
        (launcher.parent / ".interfaceforge.jobid").write_text(job_id + "\n", encoding="utf-8")
        results.append({"launcher": str(launcher), "status": "submitted", "job_id": job_id})
    StateStore(campaign.root).event("submit", execute=execute, jobs=results)
    return results
