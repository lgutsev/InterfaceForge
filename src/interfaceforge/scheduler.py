"""Render scheduler jobs from portable YAML profiles."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, SafetyError


def sanitize_job_name(name: str, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.")
    return (cleaned or "interfaceforge")[:limit]


def _positive_int(job: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(job.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Scheduler field {key} must be an integer") from exc
    if value < 1:
        raise ConfigurationError(f"Scheduler field {key} must be positive")
    return value


def render_job(
    profile: Mapping[str, Any],
    job_key: str,
    *,
    command: str | None = None,
    job_name: str = "interfaceforge",
    array: str | None = None,
    working_directory: str = "${SLURM_SUBMIT_DIR:-$PWD}",
) -> str:
    """Render one local or Slurm job script."""

    jobs = profile.get("jobs", {})
    if job_key not in jobs:
        raise ConfigurationError(f"Profile has no job named {job_key!r}")
    job = dict(jobs[job_key])
    scheduler = str(profile.get("scheduler", "")).lower()
    selected_command = command or str(job.get("command", "")).strip()
    if not selected_command:
        raise ConfigurationError(f"No command configured for profile job {job_key!r}")

    if scheduler == "local":
        if array is not None:
            raise ConfigurationError(
                f"Job {job_key!r} requests a Slurm array ({array!r}), but the local "
                "scheduler has no SLURM_ARRAY_TASK_ID; use a Slurm profile instead"
            )
        return (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"cd {shlex.quote(working_directory)}\n"
            f"{selected_command}\n"
        )
    if scheduler != "slurm":
        raise ConfigurationError(f"Unsupported scheduler: {scheduler}")

    nodes = _positive_int(job, "nodes", 1)
    ntasks = _positive_int(job, "ntasks", 1)
    cpus = _positive_int(job, "cpus_per_task", 1)
    partition = str(job.get("partition", "")).strip()
    account = str(job.get("account", "")).strip()
    walltime = str(job.get("time", "01:00:00")).strip()
    if not partition or not account:
        raise ConfigurationError(f"Slurm job {job_key!r} needs partition and account")

    safe_name = sanitize_job_name(job_name)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={safe_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={account}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={safe_name}.%A_%a.out" if array else f"#SBATCH --output={safe_name}.%j.out",
        f"#SBATCH --error={safe_name}.%A_%a.err" if array else f"#SBATCH --error={safe_name}.%j.err",
    ]
    gpus = job.get("gpus")
    if gpus is not None and int(gpus) > 0:
        lines.append(f"#SBATCH --gres=gpu:{int(gpus)}")
    if array:
        lines.append(f"#SBATCH --array={array}")

    lines.extend(["", "set -euo pipefail", f'cd "{working_directory}"'])
    modules = job.get("modules", [])
    if modules:
        lines.extend(["module purge", *[f"module load {shlex.quote(str(item))}" for item in modules]])
    lines.extend(str(item) for item in job.get("preamble", []))
    for key, value in dict(job.get("environment", {})).items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
            raise ConfigurationError(f"Unsafe environment-variable name: {key!r}")
        lines.append(f"export {key}={shlex.quote(str(value))}")

    # Do not use str.format here: generated shell programs legitimately contain
    # `${VARIABLE}`, function bodies, and Python dictionaries. Replace only the
    # four documented scheduler tokens.
    formatted = selected_command
    for token, value in {
        "nodes": nodes,
        "ntasks": ntasks,
        "cpus_per_task": cpus,
        "gpus": int(gpus or 0),
    }.items():
        formatted = formatted.replace("{" + token + "}", str(value))
    lines.extend(["", formatted, ""])
    return "\n".join(lines)


def write_job(path: str | Path, content: str, *, force: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not force:
        raise SafetyError(f"Refusing to overwrite existing launcher: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    output.chmod(0o750)
    return output
