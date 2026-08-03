"""Optional, process-isolated adapter for AI2-Kit 1.0.9 CLL workflows."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import __version__
from .config import Campaign, load_profile
from .errors import ConfigurationError, DependencyError, SafetyError
from .state import StateStore, sha256_file

TARGET_VERSION = "1.0.9"
GENERATED_NAMES = ("artifacts.yml", "executor.yml", "workflow.yml", "adapter_manifest.json")
SAFE_REMOTE_TOKEN = re.compile(r"^[A-Za-z0-9_./+@:-]+$")

# IUPAC conventional atomic weights (or standard representative values for
# elements without a standard interval). Fail closed for elements not reviewed
# here instead of silently inventing a LAMMPS mass.
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "K": 39.0983, "Ca": 40.078,
    "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.921595,
    "Se": 78.971, "Br": 79.904, "Rb": 85.4678, "Sr": 87.62,
    "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Ru": 101.07,
    "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Cs": 132.90545196, "Ba": 137.327, "Hf": 178.49,
    "Ta": 180.94788, "W": 183.84, "Re": 186.207, "Os": 190.23,
    "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98040,
}


@dataclass(frozen=True)
class AdapterPaths:
    root: Path
    generated: Path
    checkpoints: Path
    logs: Path
    status: Path
    imports: Path
    work: Path
    manifest: Path


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SafetyError(f"Required adapter file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"Could not read adapter JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SafetyError(f"Adapter JSON root must be an object: {path}")
    return value


def _settings(campaign: Campaign) -> dict[str, Any]:
    active = campaign.active_learning
    if not active.get("enabled", False):
        raise ConfigurationError("active_learning.enabled is false")
    return dict(active["ai2kit"])


def adapter_paths(campaign: Campaign, output: str | Path | None = None) -> AdapterPaths:
    configured = output if output is not None else campaign.active_learning.get(
        "output_root", "runs/active_learning/ai2kit"
    )
    candidate = Path(configured).expanduser()
    root = candidate.resolve() if candidate.is_absolute() else (campaign.root / candidate).resolve()
    try:
        relative = root.relative_to(campaign.root.resolve())
    except ValueError as exc:
        raise SafetyError(
            f"AI2-kit output root must stay below the campaign root: {root}"
        ) from exc
    if not relative.parts or len(root.parts) < 3:
        raise SafetyError(f"Unsafe AI2-kit output root: {root}")
    generated = root / "generated"
    return AdapterPaths(
        root=root,
        generated=generated,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        status=root / "status",
        imports=root / "imports",
        work=root / "work",
        manifest=generated / "adapter_manifest.json",
    )


def _resolve_inputs(campaign: Campaign, values: list[str], label: str) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        path = path.resolve() if path.is_absolute() else (campaign.root / path).resolve()
        if not path.exists() or not os.access(path, os.R_OK):
            raise SafetyError(f"Unreadable {label} artifact: {path}")
        paths.append(path)
    return paths


def _profile_settings(profile: dict[str, Any], type_map: list[str]) -> dict[str, Any]:
    raw = profile.get("ai2kit")
    if not isinstance(raw, dict):
        raise ConfigurationError("Scheduler profile requires an ai2kit mapping")
    forbidden = {"password", "private_key", "token", "secret"}

    def scan(value: Any, prefix: str = "ai2kit") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in forbidden:
                    raise SafetyError(f"Secrets are not allowed in scheduler profiles: {prefix}.{key}")
                scan(item, f"{prefix}.{key}")

    scan(raw)
    required = ("ssh", "work_dir", "python_cmd", "commands", "jobs", "potcar_source")
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise ConfigurationError(f"profile.ai2kit is missing: {', '.join(missing)}")
    ssh = raw["ssh"]
    if not isinstance(ssh, dict) or not str(ssh.get("host", "")).strip():
        raise ConfigurationError("profile.ai2kit.ssh.host is required")
    commands = raw["commands"]
    jobs = raw["jobs"]
    potcars = raw["potcar_source"]
    if not isinstance(commands, dict) or any(not commands.get(key) for key in ("ai2kit", "deepmd", "lammps", "vasp")):
        raise ConfigurationError("profile.ai2kit.commands requires ai2kit, deepmd, lammps, and vasp")
    if not isinstance(jobs, dict) or any(not jobs.get(key) for key in ("train", "explore", "label")):
        raise ConfigurationError("profile.ai2kit.jobs requires train, explore, and label job names")
    if not isinstance(potcars, dict) or any(not potcars.get(element) for element in type_map):
        raise ConfigurationError("profile.ai2kit.potcar_source must cover the complete dataset.type_map")
    for key in ("ai2kit",):
        if not SAFE_REMOTE_TOKEN.fullmatch(str(commands[key])):
            raise ConfigurationError(f"profile.ai2kit.commands.{key} must be one executable token")
    return dict(raw)


def _job_template(profile: dict[str, Any], job_name: str) -> dict[str, str]:
    jobs = profile["jobs"]
    if job_name not in jobs:
        raise ConfigurationError(f"Profile has no AI2-kit job template {job_name!r}")
    job = dict(jobs[job_name])
    required = ("partition", "account", "nodes", "ntasks", "cpus_per_task", "time")
    missing = [key for key in required if job.get(key) in (None, "")]
    if missing:
        raise ConfigurationError(f"Profile job {job_name!r} is missing: {', '.join(missing)}")
    header = [
        f"#SBATCH --partition={job['partition']}",
        f"#SBATCH --account={job['account']}",
        f"#SBATCH --nodes={int(job['nodes'])}",
        f"#SBATCH --ntasks={int(job['ntasks'])}",
        f"#SBATCH --cpus-per-task={int(job['cpus_per_task'])}",
        f"#SBATCH --time={job['time']}",
    ]
    if int(job.get("gpus", 0)) > 0:
        header.append(f"#SBATCH --gres=gpu:{int(job['gpus'])}")
    setup = ["set -euo pipefail"]
    if job.get("modules"):
        setup.append("module purge")
        setup.extend(f"module load {shlex.quote(str(value))}" for value in job["modules"])
    setup.extend(str(value) for value in job.get("preamble", []))
    for key, value in dict(job.get("environment", {})).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ConfigurationError(f"Unsafe environment variable in {job_name}: {key!r}")
        setup.append(f"export {key}={shlex.quote(str(value))}")
    return {"header": "\n".join(header) + "\n", "setup": "\n".join(setup) + "\n"}


def _deepmd_template(campaign: Campaign, settings: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    model = campaign.models["deepmd"]
    architecture = settings["architecture"]
    if architecture == "se_e2_a":
        descriptor: dict[str, Any] = {
            "type": "se_e2_a", "sel": "auto:1.20", "rcut_smth": 0.5,
            "rcut": float(model.get("rcut", 6.0)), "neuron": [25, 50, 100],
            "axis_neuron": 16, "resnet_dt": False, "seed": seeds[0],
        }
    else:
        # Experimental pass-through uses InterfaceForge's standalone generator
        # vocabulary, but is never represented as engine-validated.
        from .training import _deepmd_descriptor

        descriptor = _deepmd_descriptor(architecture, settings["backend"], seeds[0])
    return {
        "model": {
            "descriptor": descriptor,
            "fitting_net": {
                "neuron": [240, 240, 240], "resnet_dt": True,
                "seed": seeds[0] + 1000,
            },
        },
        "learning_rate": {"type": "exp", "start_lr": 0.001, "stop_lr": 1e-6, "decay_steps": 5000},
        "loss": {
            "start_pref_e": 0.02, "limit_pref_e": 1.0,
            "start_pref_f": 1000.0, "limit_pref_f": 1.0,
            "start_pref_v": 0.0, "limit_pref_v": 0.0,
        },
        "training": {
            "numb_steps": int(model.get("numb_steps", 500000)),
            "seed": seeds[0] + 2000, "disp_freq": 1000,
            "save_freq": 20000, "disp_training": True,
            "time_training": True, "profiling": False,
        },
    }


def _seed_modifier(seeds: list[int]) -> str:
    return (
        f"_interfaceforge_seeds = iter({seeds!r})\n"
        "def input_modifier_fn(data):\n"
        "    seed = next(_interfaceforge_seeds)\n"
        "    descriptor = data['model']['descriptor']\n"
        "    if descriptor.get('type') == 'hybrid':\n"
        "        for offset, item in enumerate(descriptor['list']): item['seed'] = seed + offset\n"
        "    else:\n"
        "        descriptor['seed'] = seed\n"
        "    data['model']['fitting_net']['seed'] = seed + 1000\n"
        "    data['training']['seed'] = seed + 2000\n"
        "    return data\n"
    )


def _file_or_directory_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "kind": "directory" if path.is_dir() else "file"}
    if path.is_file():
        record["sha256"] = sha256_file(path)
    return record


def _export_fingerprint(campaign: Campaign, paths: AdapterPaths) -> tuple[str, dict[str, str]]:
    files = {
        "campaign": sha256_file(campaign.path),
        "profile": sha256_file(campaign.profile_path),
    }
    for name in GENERATED_NAMES[:3]:
        path = paths.generated / name
        if not path.is_file():
            raise SafetyError(f"Missing generated AI2-kit file: {path}")
        files[name] = sha256_file(path)
    encoded = json.dumps(files, sort_keys=True).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest(), files


def build_ai2kit_command(campaign: Campaign, paths: AdapterPaths, profile_ai2kit: dict[str, Any]) -> list[str]:
    command = str(profile_ai2kit["commands"]["ai2kit"])
    return [
        command, "workflow", "cll-mlp-training",
        str(paths.generated / "artifacts.yml"),
        str(paths.generated / "executor.yml"),
        str(paths.generated / "workflow.yml"),
        "--executor", str(_settings(campaign)["executor_name"]),
        "--path-prefix", f"{campaign.name}/interfaceforge",
        "--checkpoint", str(paths.checkpoints / "cll"),
    ]


def export_adapter(campaign: Campaign, *, output: str | Path | None = None, force: bool = False) -> dict[str, Any]:
    settings = _settings(campaign)
    paths = adapter_paths(campaign, output)
    if paths.root.exists() and any(paths.root.iterdir()) and not force:
        raise SafetyError(
            f"AI2-kit output directory is not empty: {paths.root}; use --force to replace generated files"
        )
    for directory in (paths.generated, paths.checkpoints, paths.logs, paths.status, paths.imports, paths.work):
        directory.mkdir(parents=True, exist_ok=True)

    type_map = list(campaign.dataset.get("type_map", []))
    if not type_map:
        raise ConfigurationError("dataset.type_map must be explicit for AI2-kit export")
    try:
        mass_map = [ATOMIC_MASSES[element] for element in type_map]
    except KeyError as exc:
        raise ConfigurationError(f"No reviewed atomic mass for {exc.args[0]!r}") from exc
    profile = load_profile(campaign.profile_path)
    profile_ai2kit = _profile_settings(profile, type_map)
    training = _resolve_inputs(campaign, settings["training_artifacts"], "training")
    validation = _resolve_inputs(campaign, settings["validation_artifacts"], "validation")
    exploration = _resolve_inputs(campaign, settings["exploration_artifacts"], "exploration")
    strains = [float(value) for value in campaign.exploration.get("strains", [0.0])]
    if any(abs(value) > 1e-12 for value in strains):
        raise ConfigurationError(
            "AI2-kit MVP does not guess a LAMMPS strain transform; set exploration.strains to [0.0]"
        )
    temperatures = [float(value) for value in campaign.exploration.get("temperatures", [])]
    if not temperatures:
        raise ConfigurationError("exploration.temperatures must be non-empty")

    artifacts: dict[str, Any] = {"artifacts": {}}
    train_names: list[str] = []
    for index, path in enumerate(training):
        name = f"if-train-{index:03d}"
        train_names.append(name)
        artifacts["artifacts"][name] = {"url": str(path), "attrs": {"interfaceforge": {"role": "training"}}}
    for index, path in enumerate(validation):
        name = f"if-validation-{index:03d}"
        train_names.append(name)
        artifacts["artifacts"][name] = {
            "url": str(path),
            "attrs": {"deepmd": {"validation_data": True}, "interfaceforge": {"role": "validation"}},
        }
    explore_names: list[str] = []
    for index, path in enumerate(exploration):
        name = f"if-explore-{index:03d}"
        explore_names.append(name)
        entry: dict[str, Any] = {"url": str(path), "attrs": {"interfaceforge": {"role": "exploration"}}}
        if path.is_dir():
            entry["includes"] = "*"
        artifacts["artifacts"][name] = entry

    executor_name = settings["executor_name"]
    contexts = {
        "train": _job_template(profile, str(profile_ai2kit["jobs"]["train"])),
        "explore": _job_template(profile, str(profile_ai2kit["jobs"]["explore"])),
        "label": _job_template(profile, str(profile_ai2kit["jobs"]["label"])),
    }
    ssh = {"host": str(profile_ai2kit["ssh"]["host"])}
    gateway = profile_ai2kit["ssh"].get("gateway")
    if isinstance(gateway, dict) and gateway.get("host"):
        ssh["gateway"] = {"host": str(gateway["host"])}
    commands = profile_ai2kit["commands"]
    executor = {
        "executors": {
            executor_name: {
                "ssh": ssh,
                "queue_system": {"slurm": {}},
                "work_dir": str(profile_ai2kit["work_dir"]),
                "python_cmd": str(profile_ai2kit["python_cmd"]),
                "context": {
                    "train": {"deepmd": {"script_template": contexts["train"], "dp_cmd": str(commands["deepmd"])}},
                    "explore": {"lammps": {
                        "script_template": contexts["explore"], "lammps_cmd": str(commands["lammps"]),
                        "concurrency": int(profile_ai2kit.get("explore_concurrency", 1)),
                    }},
                    "label": {"vasp": {
                        "script_template": contexts["label"], "vasp_cmd": str(commands["vasp"]),
                        "concurrency": int(profile_ai2kit.get("label_concurrency", 1)),
                    }},
                },
            }
        }
    }
    seeds = [int(value) for value in campaign.models["deepmd"]["seeds"]][: settings["model_count"]]
    reference_inputs = campaign.reference["inputs"]
    incar = Path(reference_inputs.get("INCAR", ""))
    kpoints = Path(reference_inputs.get("KPOINTS", ""))
    incar = incar.resolve() if incar.is_absolute() else (campaign.root / incar).resolve()
    kpoints = kpoints.resolve() if kpoints.is_absolute() else (campaign.root / kpoints).resolve()
    for path, label in ((incar, "INCAR"), (kpoints, "KPOINTS")):
        if not path.is_file() or not path.stat().st_size:
            raise SafetyError(f"AI2-kit requires a readable VASP {label}: {path}")
    workflow = {
        "workflow": {
            "general": {
                "type_map": type_map,
                "mass_map": mass_map,
                "max_iters": campaign.active_learning["max_iterations"],
            },
            "train": {"deepmd": {
                "model_num": settings["model_count"], "init_dataset": train_names,
                "input_template": _deepmd_template(campaign, settings, seeds),
                "input_modifier_fn": _seed_modifier(seeds),
            }},
            "label": {"vasp": {
                "limit": settings["selection_limit"], "init_system_files": [],
                "input_template": incar.read_text(encoding="utf-8"),
                "kpoints_template": kpoints.read_text(encoding="utf-8"),
                "potcar_source": {element: str(profile_ai2kit["potcar_source"][element]) for element in type_map},
            }},
            "explore": {"lammps": {
                "timestep": float(profile_ai2kit.get("timestep", 0.001)),
                "sample_freq": int(profile_ai2kit.get("sample_freq", 100)),
                "nsteps": int(profile_ai2kit.get("nsteps", 10000)),
                "ensemble": "nvt", "system_files": explore_names,
                "explore_vars": {"TEMP": temperatures, "PRES": [1.0]},
            }},
            "select": {"model_devi": {
                "f_trust_lo": settings["trust_force_low"],
                "f_trust_hi": settings["trust_force_high"],
            }},
            "update": {"walkthrough": {"table": []}},
        }
    }
    _write_yaml(paths.generated / "artifacts.yml", artifacts)
    _write_yaml(paths.generated / "executor.yml", executor)
    _write_yaml(paths.generated / "workflow.yml", workflow)
    fingerprint, hashes = _export_fingerprint(campaign, paths)
    manifest = {
        "schema_version": 1, "interfaceforge_version": __version__,
        "ai2kit_version": TARGET_VERSION, "campaign": str(campaign.path),
        "profile": str(campaign.profile_path), "output_root": str(paths.root),
        "checkpoint": str(paths.checkpoints / "cll"),
        "architecture": settings["architecture"], "backend": settings["backend"],
        "model_count": settings["model_count"], "seeds": seeds,
        "experimental_compatibility": settings["experimental_compatibility"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_hashes": hashes, "export_fingerprint": fingerprint,
        "inputs": {
            "training": [_file_or_directory_record(path) for path in training],
            "validation": [_file_or_directory_record(path) for path in validation],
            "exploration": [_file_or_directory_record(path) for path in exploration],
        },
        "command": build_ai2kit_command(campaign, paths, profile_ai2kit),
        "execution_state": "exported",
        "compatibility_evidence": (
            "configuration generated from AI2-kit 1.0.9 public CLL schema; "
            "engine execution not yet proven"
        ),
    }
    _write_json(paths.manifest, manifest)
    state = StateStore(campaign.root)
    state.event("ai2kit_export", output=str(paths.root), fingerprint=fingerprint)
    for name in GENERATED_NAMES:
        state.artifact(f"ai2kit_{Path(name).stem}", paths.generated / name)
    return manifest


def _run_probe(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, shell=False)


def _remote_probe(host: str, arguments: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    if not SAFE_REMOTE_TOKEN.fullmatch(host):
        raise ConfigurationError("Unsafe SSH host token in profile.ai2kit.ssh.host")
    remote_command = shlex.join(arguments)
    return _run_probe(["ssh", host, remote_command], timeout=timeout)


def preflight_adapter(campaign: Campaign, *, output_root: str | Path | None = None, remote: bool = False,
                      report_output: str | Path | None = None) -> dict[str, Any]:
    paths = adapter_paths(campaign, output_root)
    manifest = _load_json(paths.manifest)
    profile = load_profile(campaign.profile_path)
    type_map = list(campaign.dataset["type_map"])
    profile_ai2kit = _profile_settings(profile, type_map)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail[:800]})

    try:
        fingerprint, _ = _export_fingerprint(campaign, paths)
        add("export_fingerprint", fingerprint == manifest.get("export_fingerprint"), fingerprint)
    except SafetyError as exc:
        fingerprint = ""
        add("export_fingerprint", False, str(exc))
    add("python", sys.version_info >= (3, 9), sys.version.split()[0])
    for group in ("training", "validation", "exploration"):
        for record in manifest.get("inputs", {}).get(group, []):
            path = Path(record["path"])
            add(f"artifact:{group}:{path.name}", path.exists() and os.access(path, os.R_OK), str(path))
    ai2kit_cmd = str(profile_ai2kit["commands"]["ai2kit"])
    executable = shutil.which(ai2kit_cmd)
    add("ai2kit_command", executable is not None, executable or f"not found: {ai2kit_cmd}")
    version_text = ""
    help_text = ""
    if executable:
        version_probe = _run_probe([executable, "--version"])
        version_text = (version_probe.stdout + "\n" + version_probe.stderr).strip()
        try:
            installed = importlib.metadata.version("ai2-kit")
        except importlib.metadata.PackageNotFoundError:
            match = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", version_text)
            installed = match.group(1) if match else "unknown"
        add("ai2kit_version", installed == TARGET_VERSION, f"installed={installed}; target={TARGET_VERSION}")
        feature = _run_probe([executable, "workflow", "cll-mlp-training", "--help"])
        help_text = feature.stdout + "\n" + feature.stderr
        add("ai2kit_cll_cli", feature.returncode == 0 and "checkpoint" in help_text.lower(), help_text[-800:])
    else:
        add("ai2kit_version", False, "AI2-kit command unavailable")
        add("ai2kit_cll_cli", False, "AI2-kit command unavailable")
    if os.environ.get("SLURM_JOB_ID"):
        add(
            "controller_context",
            True,
            "inside Slurm allocation; export/preflight allowed, execution will be refused",
            required=False,
        )
    else:
        add("controller_context", True, "external/login controller context")

    remote_checked = False
    if remote:
        remote_checked = True
        host = str(profile_ai2kit["ssh"]["host"])
        add("ssh_command", shutil.which("ssh") is not None, shutil.which("ssh") or "ssh not found")
        python_cmd = str(profile_ai2kit["python_cmd"])
        for token in (python_cmd, *[str(profile_ai2kit["commands"][key]) for key in ("deepmd", "lammps", "vasp")]):
            if not SAFE_REMOTE_TOKEN.fullmatch(token):
                raise ConfigurationError(f"Unsafe remote command token: {token!r}")
        python_probe = _remote_probe(host, [python_cmd, "--version"])
        add("remote_python", python_probe.returncode == 0, (python_probe.stdout + python_probe.stderr).strip())
        for command in ("sbatch", "squeue", "sacct"):
            probe = _remote_probe(host, ["command", "-v", command])
            add(f"remote_{command}", probe.returncode == 0, (probe.stdout + probe.stderr).strip())
        for key in ("deepmd", "lammps", "vasp"):
            command = str(profile_ai2kit["commands"][key])
            probe = _remote_probe(host, ["command", "-v", command])
            add(f"remote_{key}", probe.returncode == 0, (probe.stdout + probe.stderr).strip())
        lammps_help = _remote_probe(host, [str(profile_ai2kit["commands"]["lammps"]), "-h"])
        lammps_text = (lammps_help.stdout + lammps_help.stderr).lower()
        add("remote_lammps_deepmd", lammps_help.returncode == 0 and "deepmd" in lammps_text, lammps_text[-800:])
        for element, potcar in profile_ai2kit["potcar_source"].items():
            probe = _remote_probe(host, ["test", "-r", str(potcar)])
            add(f"remote_potcar:{element}", probe.returncode == 0, str(potcar))
        work_dir = str(profile_ai2kit["work_dir"])
        probe = _remote_probe(host, ["test", "-d", work_dir, "-a", "-w", work_dir])
        add("remote_work_dir", probe.returncode == 0, work_dir)
    else:
        add("remote_engine_checks", True, "not requested; --remote is mandatory before execution", required=False)

    passed = all(check["ok"] for check in checks if check["required"])
    report = {
        "schema_version": 1, "passed": passed, "remote_checked": remote_checked,
        "time": datetime.now(timezone.utc).isoformat(), "export_fingerprint": fingerprint,
        "checks": checks,
    }
    destination = Path(report_output).resolve() if report_output else paths.status / "preflight.json"
    _write_json(destination, report)
    manifest["preflight"] = {"passed": passed, "remote_checked": remote_checked, "report": str(destination),
                             "export_fingerprint": fingerprint, "time": report["time"]}
    manifest["execution_state"] = "preflight_passed" if passed else "preflight_failed"
    _write_json(paths.manifest, manifest)
    StateStore(campaign.root).event("ai2kit_preflight", passed=passed, remote=remote_checked)
    return report


def run_adapter(campaign: Campaign, *, output_root: str | Path | None = None, execute: bool = False,
                resume: bool = False, allow_multiple_iterations: bool = False) -> dict[str, Any]:
    paths = adapter_paths(campaign, output_root)
    manifest = _load_json(paths.manifest)
    profile_ai2kit = _profile_settings(load_profile(campaign.profile_path), list(campaign.dataset["type_map"]))
    command = build_ai2kit_command(campaign, paths, profile_ai2kit)
    if not execute:
        return {"executed": False, "state": manifest.get("execution_state", "exported"), "command": command}
    if os.environ.get("SLURM_JOB_ID"):
        raise SafetyError(
            "AI2-kit controller execution from a Slurm compute job is unsupported because it would submit child jobs; "
            "run the controller from an external workstation or permitted login/service environment"
        )
    if campaign.active_learning["max_iterations"] > 1 and not allow_multiple_iterations:
        raise SafetyError("Multiple AI2-kit iterations require --allow-multiple-iterations")
    fingerprint, _ = _export_fingerprint(campaign, paths)
    preflight = manifest.get("preflight", {})
    if not preflight.get("passed") or preflight.get("export_fingerprint") != fingerprint:
        raise SafetyError("A successful current AI2-kit preflight is required before --execute")
    if profile_ai2kit.get("ssh") and not preflight.get("remote_checked"):
        raise SafetyError("Run `iface active-learning ai2kit preflight --remote` before execution")
    checkpoint = paths.checkpoints / "cll"
    if resume and not checkpoint.exists():
        raise SafetyError(f"Cannot resume because the adapter checkpoint does not exist: {checkpoint}")
    if not resume and checkpoint.exists() and any(checkpoint.iterdir() if checkpoint.is_dir() else [checkpoint]):
        raise SafetyError("Checkpoint state already exists; use --resume after confirming campaign identity")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stdout_path = paths.logs / f"run_{stamp}.out"
    stderr_path = paths.logs / f"run_{stamp}.err"
    manifest.update({"execution_state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                     "last_command": command, "stdout": str(stdout_path), "stderr": str(stderr_path)})
    _write_json(paths.manifest, manifest)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=paths.generated, stdout=stdout, stderr=stderr,
                                    text=True, check=False, shell=False)
    except KeyboardInterrupt:
        manifest["execution_state"] = "interrupted"
        manifest["interrupted_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(paths.manifest, manifest)
        raise
    manifest["returncode"] = result.returncode
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["execution_state"] = "awaiting_approval" if result.returncode == 0 else "failed"
    _write_json(paths.manifest, manifest)
    StateStore(campaign.root).event("ai2kit_run", returncode=result.returncode, resume=resume)
    if result.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
        raise SafetyError(
            f"AI2-kit exited with code {result.returncode}; checkpoint and logs were preserved. stderr tail:\n{tail}"
        )
    return {"executed": True, "returncode": 0, "state": manifest["execution_state"],
            "stdout": str(stdout_path), "stderr": str(stderr_path), "checkpoint": str(checkpoint)}


def adapter_status(campaign: Campaign, *, output_root: str | Path | None = None) -> dict[str, Any]:
    paths = adapter_paths(campaign, output_root)
    if not paths.manifest.is_file():
        return {"state": "not_exported", "output_root": str(paths.root)}
    manifest = _load_json(paths.manifest)
    state = str(manifest.get("execution_state", "unknown"))
    fingerprint_ok = False
    try:
        fingerprint, _ = _export_fingerprint(campaign, paths)
        fingerprint_ok = fingerprint == manifest.get("export_fingerprint")
    except SafetyError:
        pass
    logs = sorted(str(path) for path in paths.logs.glob("run_*.*")) if paths.logs.is_dir() else []
    report = {
        "state": state, "output_root": str(paths.root), "fingerprint_current": fingerprint_ok,
        "checkpoint_present": (paths.checkpoints / "cll").exists(), "returncode": manifest.get("returncode"),
        "logs": logs[-10:], "note": "checkpoint treated as opaque; no pickle deserialization performed",
    }
    _write_json(paths.status / "status.json", report)
    return report


def _structure_id(symbols: list[str], cell: np.ndarray, positions: np.ndarray) -> str:
    import hashlib

    payload = {"symbols": symbols, "cell": np.round(cell, 8).tolist(), "positions": np.round(positions, 8).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_digest(files: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def stage_import(campaign: Campaign, *, round_number: int, result_root: str | Path,
                 output_root: str | Path | None = None) -> dict[str, Any]:
    if round_number < 0:
        raise ConfigurationError("--round must be non-negative")
    try:
        from ase.io import iread, write
    except ModuleNotFoundError as exc:
        raise DependencyError("ASE is required for AI2-kit import; install interfaceforge[vasp]") from exc
    paths = adapter_paths(campaign, output_root)
    manifest = _load_json(paths.manifest)
    source_root = Path(result_root).expanduser().resolve()
    if not source_root.is_dir():
        raise SafetyError(f"AI2-kit result root is not a directory: {source_root}")
    source_files = sorted({*source_root.rglob("*.extxyz"), *source_root.rglob("OUTCAR")})
    if not source_files:
        raise SafetyError(f"No extxyz or OUTCAR label results found below {source_root}")
    source_digest = _source_digest(source_files)
    destination = paths.imports / f"round_{round_number:03d}"
    existing_manifest = destination / "import_manifest.json"
    if existing_manifest.is_file():
        existing = _load_json(existing_manifest)
        if existing.get("source_digest") == source_digest:
            return {**existing, "idempotent": True}
        raise SafetyError(f"Import round already contains different source data: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    type_map = list(campaign.dataset["type_map"])
    known_ids: set[str] = set()
    for canonical in (campaign.root / "datasets" / "canonical").glob("*.extxyz"):
        try:
            for atoms in iread(str(canonical), index=":"):
                known_ids.add(
                    _structure_id(
                        atoms.get_chemical_symbols(),
                        np.asarray(atoms.cell),
                        np.asarray(atoms.positions),
                    )
                )
        except Exception:
            continue
    accepted_atoms: list[Any] = []
    accepted_ids: set[str] = set()
    rejected: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for source in source_files:
        if source.name == "OUTCAR":
            text = source.read_text(encoding="utf-8", errors="ignore")
            if "General timing and accounting informations" not in text:
                rejected.append({"source": str(source), "frame": "*", "reason": "OUTCAR lacks VASP completion marker"})
                continue
        try:
            frames = list(iread(str(source), index=":"))
        except Exception as exc:
            rejected.append({"source": str(source), "frame": "*", "reason": f"parse failure: {exc}"})
            continue
        for frame_index, atoms in enumerate(frames):
            try:
                symbols = atoms.get_chemical_symbols()
                unexpected = sorted(set(symbols) - set(type_map))
                if unexpected:
                    raise ValueError(f"unexpected species: {unexpected}")
                positions = np.asarray(atoms.positions, dtype=float)
                cell = np.asarray(atoms.cell.array, dtype=float)
                if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
                    raise ValueError("invalid or non-finite coordinates")
                if cell.shape != (3, 3) or not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) < 1e-6:
                    raise ValueError("invalid or degenerate cell")
                energy_value = atoms.info.get("REF_energy")
                if energy_value is None:
                    energy_value = atoms.get_potential_energy()
                energy = float(energy_value)
                forces_value = atoms.arrays.get("REF_forces")
                if forces_value is None:
                    forces_value = atoms.get_forces(apply_constraint=False)
                forces = np.asarray(forces_value, dtype=float)
                if not math.isfinite(energy):
                    raise ValueError("non-finite energy")
                if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
                    raise ValueError("invalid or non-finite force array")
                structure_id = _structure_id(symbols, cell, positions)
                if structure_id in known_ids or structure_id in accepted_ids:
                    raise ValueError("duplicate structure identity")
                clean = atoms.copy()
                clean.calc = None
                clean.info["REF_energy"] = energy
                clean.info["interfaceforge_structure_id"] = structure_id
                clean.info["source_path"] = str(source)
                clean.info["source_frame"] = frame_index
                clean.arrays["REF_forces"] = forces.copy()
                accepted_atoms.append(clean)
                accepted_ids.add(structure_id)
                lineage.append({
                    "structure_id": structure_id, "source_path": str(source),
                    "source_sha256": sha256_file(source), "source_frame": frame_index,
                    "round": round_number, "raw_forces_preserved": True,
                })
            except Exception as exc:
                rejected.append({"source": str(source), "frame": frame_index, "reason": str(exc)})
    accepted_path = destination / "accepted.extxyz"
    if accepted_atoms:
        write(str(accepted_path), accepted_atoms, format="extxyz")
    else:
        _atomic_text(accepted_path, "")
    with (destination / "rejected.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "frame", "reason"])
        writer.writeheader()
        writer.writerows(rejected)
    with (destination / "lineage.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["structure_id", "source_path", "source_sha256", "source_frame", "round", "raw_forces_preserved"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lineage)
    payload = {
        "schema_version": 1, "round": round_number, "result_root": str(source_root),
        "source_digest": source_digest, "accepted": len(accepted_atoms), "rejected": len(rejected),
        "duplicates": sum("duplicate" in row["reason"] for row in rejected),
        "accepted_extxyz": str(accepted_path), "rejected_csv": str(destination / "rejected.csv"),
        "lineage_csv": str(destination / "lineage.csv"), "canonical_dataset_modified": False,
        "unit_policy": "ASE eV and eV/angstrom; no conversion applied",
        "duplicate_policy": "SHA-256 of species plus cell and Cartesian positions rounded to 1e-8",
    }
    _write_json(existing_manifest, payload)
    approval = {
        "round": round_number, "approved": False, "accepted": payload["accepted"],
        "rejected": payload["rejected"], "duplicates": payload["duplicates"],
        "trust_force_low": _settings(campaign)["trust_force_low"],
        "trust_force_high": _settings(campaign)["trust_force_high"],
        "import_manifest_sha256": sha256_file(existing_manifest),
        "next_round": "blocked pending explicit approval",
    }
    _write_json(destination / "approval.json", approval)
    manifest["execution_state"] = "awaiting_approval"
    manifest["latest_import"] = str(existing_manifest)
    _write_json(paths.manifest, manifest)
    StateStore(campaign.root).artifact(f"ai2kit_import_round_{round_number:03d}", existing_manifest)
    return payload


def approve_round(campaign: Campaign, *, round_number: int, output_root: str | Path | None = None) -> dict[str, Any]:
    paths = adapter_paths(campaign, output_root)
    destination = paths.imports / f"round_{round_number:03d}"
    import_manifest = _load_json(destination / "import_manifest.json")
    approval_path = destination / "approval.json"
    approval = _load_json(approval_path)
    current_hash = sha256_file(destination / "import_manifest.json")
    if approval.get("import_manifest_sha256") != current_hash:
        raise SafetyError(
            "The staged import manifest changed after validation; inspect it and stage the round again"
        )
    approval.update({
        "approved": True, "approved_at": datetime.now(timezone.utc).isoformat(),
        "import_manifest_sha256": current_hash,
    })
    _write_json(approval_path, approval)
    manifest = _load_json(paths.manifest)
    manifest["execution_state"] = "approved"
    manifest["approved_round"] = round_number
    _write_json(paths.manifest, manifest)
    StateStore(campaign.root).event("ai2kit_approve", round=round_number, accepted=import_manifest["accepted"])
    return approval
