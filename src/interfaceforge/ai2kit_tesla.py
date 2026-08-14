"""Example-driven AI2-Kit TESLA adapter for an existing MACE committee.

This backend intentionally generates a transparent shell workflow.  AI2-Kit
performs chemistry-aware conversion and model-deviation screening, while
oh-my-batch expands the exploration matrix and owns Slurm submission/recovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import Campaign, load_profile
from .errors import ConfigurationError, SafetyError
from .state import StateStore, sha256_file

TARGET_AI2KIT_VERSION = "1.0.9"
TARGET_OMB_VERSION = "0.7.2"
SAFE_COMMAND = re.compile(r"^[A-Za-z0-9_./+:-]+$")


def _atomic_text(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    if executable:
        path.chmod(0o750)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SafetyError(f"Required TESLA manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"Could not read TESLA manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SafetyError(f"TESLA manifest root must be an object: {path}")
    return payload


def _settings(campaign: Campaign) -> dict[str, Any]:
    settings = dict(campaign.active_learning.get("ai2kit", {}))
    if settings.get("workflow") != "tesla_mace":
        raise ConfigurationError("AI2-Kit workflow is not tesla_mace")
    return settings


def _root(campaign: Campaign, output_root: str | Path | None = None) -> Path:
    configured = output_root or campaign.active_learning.get(
        "output_root", "runs/active_learning/ai2kit"
    )
    value = Path(configured).expanduser()
    path = value.resolve() if value.is_absolute() else (campaign.root / value).resolve()
    try:
        relative = path.relative_to(campaign.root.resolve())
    except ValueError as exc:
        raise SafetyError(f"AI2-Kit output must stay below the campaign root: {path}") from exc
    if not relative.parts:
        raise SafetyError("AI2-Kit output cannot be the campaign root")
    return path


def _resolve(
    campaign: Campaign,
    value: str | Path,
    label: str,
    *,
    allow_directory: bool = False,
) -> Path:
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (campaign.root / candidate).resolve()
    valid = path.exists() if allow_directory else path.is_file()
    if not valid or not os.access(path, os.R_OK):
        raise SafetyError(f"Unreadable {label}: {path}")
    return path


def _profile(campaign: Campaign) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = load_profile(campaign.profile_path)
    adapter = profile.get("ai2kit")
    if not isinstance(adapter, dict):
        raise ConfigurationError("Scheduler profile requires an ai2kit mapping")
    commands = adapter.get("commands")
    jobs = adapter.get("jobs")
    if not isinstance(commands, dict):
        raise ConfigurationError("profile.ai2kit.commands must be a mapping")
    required_commands = ("ai2kit", "omb", "python", "mace")
    missing = [key for key in required_commands if not str(commands.get(key, "")).strip()]
    if missing:
        raise ConfigurationError(
            f"profile.ai2kit.commands is missing: {', '.join(missing)}"
        )
    for key in required_commands:
        if not SAFE_COMMAND.fullmatch(str(commands[key])):
            raise ConfigurationError(f"Unsafe profile.ai2kit.commands.{key} token")
    if not isinstance(jobs, dict) or any(not jobs.get(key) for key in ("train", "explore", "label")):
        raise ConfigurationError("profile.ai2kit.jobs requires train, explore, and label")
    if str(profile.get("scheduler", "")).lower() != "slurm":
        raise ConfigurationError("TESLA MACE currently requires a Slurm profile")
    return profile, dict(adapter)


def _slurm_header(profile: dict[str, Any], job_key: str, job_name: str) -> str:
    jobs = profile.get("jobs", {})
    if job_key not in jobs:
        raise ConfigurationError(f"Profile has no job named {job_key!r}")
    job = dict(jobs[job_key])
    required = ("partition", "account", "nodes", "ntasks", "cpus_per_task", "time")
    missing = [key for key in required if job.get(key) in (None, "")]
    if missing:
        raise ConfigurationError(f"Profile job {job_key!r} is missing: {', '.join(missing)}")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", job_name)[:80]
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={safe_name}",
        f"#SBATCH --partition={job['partition']}",
        f"#SBATCH --account={job['account']}",
        f"#SBATCH --nodes={int(job['nodes'])}",
        f"#SBATCH --ntasks={int(job['ntasks'])}",
        f"#SBATCH --cpus-per-task={int(job['cpus_per_task'])}",
        f"#SBATCH --time={job['time']}",
        f"#SBATCH --output={safe_name}.%j.out",
        f"#SBATCH --error={safe_name}.%j.err",
    ]
    if int(job.get("gpus", 0) or 0) > 0:
        lines.append(f"#SBATCH --gres=gpu:{int(job['gpus'])}")
    lines.extend(["", "set -euo pipefail"])
    modules = list(job.get("modules", []))
    if modules:
        lines.append("module purge")
        lines.extend(f"module load {shlex.quote(str(module))}" for module in modules)
    lines.extend(str(item) for item in job.get("preamble", []))
    for key, value in dict(job.get("environment", {})).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ConfigurationError(f"Unsafe environment variable in profile job: {key}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(lines) + "\n"


def _job_command(profile: dict[str, Any], job_key: str) -> str:
    job = dict(profile["jobs"][job_key])
    command = str(job.get("command", "")).strip()
    if not command:
        raise ConfigurationError(f"Profile job {job_key!r} requires a command")
    replacements = {
        "nodes": int(job["nodes"]),
        "ntasks": int(job["ntasks"]),
        "cpus_per_task": int(job["cpus_per_task"]),
        "gpus": int(job.get("gpus", 0) or 0),
    }
    for token, value in replacements.items():
        command = command.replace("{" + token + "}", str(value))
    return command


def _input_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        record: dict[str, Any] = {
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file",
        }
        if path.is_file():
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def _expand_xyz_artifacts(paths: list[Path], label: str) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_file():
            expanded.append(path)
        else:
            expanded.extend(sorted({*path.rglob("*.xyz"), *path.rglob("*.extxyz")}))
    unique = list(dict.fromkeys(expanded))
    if not unique:
        raise SafetyError(f"No xyz/extxyz files found in {label} artifacts")
    return unique


def _generated_hashes(generated: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(generated.rglob("*")):
        if path.is_file() and path.name != "adapter_manifest.json":
            hashes[str(path.relative_to(generated))] = sha256_file(path)
    return hashes


def _fingerprint(campaign: Campaign, generated: Path) -> tuple[str, dict[str, str]]:
    hashes = {
        "campaign": sha256_file(campaign.path),
        "profile": sha256_file(campaign.profile_path),
        **{f"generated:{key}": value for key, value in _generated_hashes(generated).items()},
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def _model_deviation_py() -> str:
    return '''#!/usr/bin/env python
import sys
import numpy as np
import ase.io
from mace.calculators import MACECalculator

traj_file, output_file, dtype, *model_paths = sys.argv[1:]
if len(model_paths) < 2:
    raise SystemExit("At least two MACE models are required")
frames = ase.io.read(traj_file, index=":", format="extxyz")
calculators = [
    MACECalculator(model_paths=[path], device="cuda", default_dtype=dtype)
    for path in model_paths
]
rows = []
for step, atoms in enumerate(frames):
    forces = []
    for calculator in calculators:
        probe = atoms.copy()
        probe.calc = calculator
        forces.append(probe.get_forces())
    sigma = np.std(np.asarray(forces), axis=0)
    per_atom = np.linalg.norm(sigma, axis=-1)
    rows.append((step, 0.0, 0.0, 0.0, per_atom.max(), per_atom.min(), per_atom.mean()))
with open(output_file, "w", encoding="utf-8") as handle:
    handle.write("# step max_devi_v min_devi_v avg_devi_v max_devi_f min_devi_f avg_devi_f\\n")
    for row in rows:
        handle.write(f"{row[0]:12d}" + "".join(f" {value:18.8e}" for value in row[1:]) + "\\n")
'''


def _openmm_py() -> str:
    return '''#!/usr/bin/env python
import math
import sys
import numpy as np
import ase.io
import openmm
from openmm import app, unit
from openmmml import MLPotential

structure, model, steps, temperature, sample_frequency, timestep_fs, friction_ps, max_force, output, seed = sys.argv[1:]
steps, sample_frequency, seed = int(steps), int(sample_frequency), int(seed)
temperature, timestep_fs, friction_ps, max_force = map(float, (temperature, timestep_fs, friction_ps, max_force))
atoms = ase.io.read(structure, format="extxyz")
topology = app.Topology()
chain = topology.addChain()
residue = topology.addResidue("SYS", chain)
for symbol in atoms.get_chemical_symbols():
    topology.addAtom(symbol, app.Element.getBySymbol(symbol), residue)
cell_nm = np.asarray(atoms.cell) * 0.1
topology.setPeriodicBoxVectors([openmm.Vec3(*vector) * unit.nanometer for vector in cell_nm])
potential = MLPotential("mace", modelPath=model)
system = potential.createSystem(topology)
integrator = openmm.LangevinMiddleIntegrator(
    temperature * unit.kelvin,
    friction_ps / unit.picosecond,
    timestep_fs * unit.femtoseconds,
)
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName("CUDA")
simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(np.asarray(atoms.positions) * 0.1 * unit.nanometer)
simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
frames = []
for frame_index in range(steps // sample_frequency):
    simulation.step(sample_frequency)
    state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True, enforcePeriodicBox=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * 10.0
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    ) * 0.0010364269656262175
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * 0.010364269656262175
    observed = float(np.max(np.linalg.norm(forces, axis=1)))
    if not math.isfinite(energy) or not np.isfinite(positions).all() or not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite state at saved frame {frame_index}")
    frame = atoms.copy()
    frame.positions = positions
    frame.info["MACE_energy"] = energy
    frame.info["driver_max_force"] = observed
    frame.arrays["MACE_forces"] = forces
    frames.append(frame)
    if observed > max_force:
        print(f"Safety stop: max force {observed:.6f} eV/A exceeds {max_force:.6f}")
        break
ase.io.write(output, frames, format="extxyz")
'''


def _prepare_inputs_py() -> str:
    return '''#!/usr/bin/env python
import sys
from pathlib import Path
import ase.io

output_dir = Path(sys.argv[1])
strains = [float(value) for value in sys.argv[2].split(",")]
sources = [Path(value) for value in sys.argv[3:]]
output_dir.mkdir(parents=True, exist_ok=True)
index = 0
for source in sources:
    atoms = ase.io.read(source)
    for strain in strains:
        frame = atoms.copy()
        scale = 1.0 + strain
        cell = frame.cell.array.copy()
        cell[0] *= scale
        cell[1] *= scale
        frame.set_cell(cell, scale_atoms=True)
        frame.info["IF_source"] = str(source)
        frame.info["IF_inplane_strain"] = strain
        ase.io.write(output_dir / f"structure-{index:04d}.xyz", frame, format="extxyz")
        index += 1
'''


def _normalize_labels_py() -> str:
    return '''#!/usr/bin/env python
import sys
import numpy as np
import ase.io

source, destination = sys.argv[1:]
frames = ase.io.read(source, index=":", format="extxyz")
normalized = []
for atoms in frames:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    clean = atoms.copy()
    clean.calc = None
    clean.info["REF_energy"] = energy
    clean.arrays["REF_forces"] = forces
    normalized.append(clean)
ase.io.write(destination, normalized, format="extxyz")
'''


def _mace_train_template(command: str, model: dict[str, Any]) -> str:
    options = [
        command,
        "--name=mace_model",
        "--train_file=../all.xyz",
        "--valid_file=../valid.xyz",
        "--model=MACE",
        f"--r_max={float(model.get('r_max', 6.0))}",
        f"--batch_size={int(model.get('batch_size', 16))}",
        f"--max_num_epochs={int(model.get('max_num_epochs', 200))}",
        f"--patience={int(model.get('patience', 30))}",
        f"--energy_key={model.get('energy_key', 'REF_energy')}",
        f"--forces_key={model.get('forces_key', 'REF_forces')}",
        f"--energy_weight={float(model.get('stage2_energy_weight', 10.0))}",
        f"--forces_weight={float(model.get('stage2_forces_weight', 50.0))}",
        "--E0s=average",
        "--device=cuda",
        "--seed=@SEED",
        "--swa",
        "--save_cpu",
    ]
    separator = " \\\n  "
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + separator.join(options) + "\n"


def _openmm_run_template(python_cmd: str, dtype: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
MACE_MODELS_ARR=(@MACE_MODELS)
MD_MODEL="${{MACE_MODELS_ARR[0]}}"
{python_cmd} @SCRIPT_DIR/openmm-run.py @STRUCTURE_FILE "$MD_MODEL" \\
  @STEPS @TEMP @SAMPLE_FREQ @TIMESTEP_FS @FRICTION_PS @MAX_FORCE traj.xyz @SEED
{python_cmd} @SCRIPT_DIR/model-devi.py traj.xyz model_devi.out {dtype} @MACE_MODELS
'''


def _vasp_run_template(incar: Path, kpoints: Path, potcars: dict[str, str], command: str) -> str:
    cases = "\n".join(
        f"    {symbol}) cat {shlex.quote(path)} >> POTCAR ;;"
        for symbol, path in sorted(potcars.items())
    )
    return f'''#!/usr/bin/env bash
set -euo pipefail
cp @POSCAR_FILE POSCAR
cp {shlex.quote(str(incar))} INCAR
cp {shlex.quote(str(kpoints))} KPOINTS
: > POTCAR
read -r -a SPECIES < <(sed -n '6p' POSCAR)
for symbol in "${{SPECIES[@]}}"; do
  case "$symbol" in
{cases}
    *) echo "No POTCAR configured for $symbol" >&2; exit 2 ;;
  esac
done
{command}
'''


def _iteration_script(ai2kit: str, omb: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
: "${{ITER_NAME:?}}" "${{CONFIG_DIR:?}}" "${{WORK_DIR:?}}"
ITER_DIR="$WORK_DIR/iter-$ITER_NAME"
MACE_DIR="$ITER_DIR/mace"
OPENMM_DIR="$ITER_DIR/openmm"
SCREENING_DIR="$ITER_DIR/screening"
LABELING_DIR="$ITER_DIR/vasp"
mkdir -p "$MACE_DIR" "$OPENMM_DIR" "$SCREENING_DIR" "$LABELING_DIR"
[[ -f "$ITER_DIR/iter.done" ]] && exit 0

if [[ "$ITER_NAME" == "000" ]]; then
  mkdir -p "$MACE_DIR/external"
  index=0
  while IFS= read -r model; do
    ln -sfn "$model" "$MACE_DIR/external/model-$index.model"
    index=$((index + 1))
  done < "$CONFIG_DIR/committee-models.txt"
  MACE_MODEL_GLOB="$MACE_DIR/external/model-*.model"
else
  cat "$CONFIG_DIR"/training/*.xyz "$WORK_DIR"/iter-*/new-dataset/dataset.xyz > "$MACE_DIR/all.xyz"
  cat "$CONFIG_DIR"/validation/*.xyz > "$MACE_DIR/valid.xyz"
  if [[ ! -f "$MACE_DIR/setup.done" ]]; then
    {omb} combo add_var SEED $COMMITTEE_SEEDS - \\
      make_files "$MACE_DIR/model-{{i}}/run.sh" --template "$CONFIG_DIR/mace/run.sh" --mode 755 - done
    {omb} batch add_work_dirs "$MACE_DIR/model-*" - \\
      add_header_files "$CONFIG_DIR/mace/slurm-header.sh" - \\
      add_cmds "bash ./run.sh" - make "$MACE_DIR/mace-train-{{i}}.slurm"
    touch "$MACE_DIR/setup.done"
  fi
  {omb} job slurm submit "$MACE_DIR"/mace-train-*.slurm --max_tries 2 --wait --recovery "$MACE_DIR/slurm-recovery.json"
  MACE_MODEL_GLOB="$MACE_DIR/model-*/mace_model_stagetwo.model"
fi

if [[ ! -f "$OPENMM_DIR/setup.done" ]]; then
  {omb} combo add_files STRUCTURE_FILE "$WORK_DIR/openmm-data/*" --abs - \\
    add_file_set MACE_MODELS "$MACE_MODEL_GLOB" --abs - \\
    add_var TEMP $MD_TEMPERATURES - add_var STEPS "$MD_STEPS" - \\
    add_var SAMPLE_FREQ "$SAMPLE_FREQUENCY" - add_var TIMESTEP_FS "$TIMESTEP_FS" - \\
    add_var FRICTION_PS "$FRICTION_PS" - add_var MAX_FORCE "$MAX_FORCE_EV_ANG" - \\
    add_var SCRIPT_DIR "$(realpath "$CONFIG_DIR/openmm")" - \\
    add_randint SEED -n "$MD_REPLICAS" -a 1 -b 2147483646 --uniq - \\
    make_files "$OPENMM_DIR/job-{{TEMP}}K-{{i:04d}}/run.sh" --template "$CONFIG_DIR/openmm/run.sh" --mode 755 - done
  {omb} batch add_work_dirs "$OPENMM_DIR/job-*" - \\
    add_header_files "$CONFIG_DIR/openmm/slurm-header.sh" - add_cmds "bash ./run.sh" - \\
    make "$OPENMM_DIR/openmm-{{i}}.slurm" --concurrency "$MD_WORKERS"
  touch "$OPENMM_DIR/setup.done"
fi
{omb} job slurm submit "$OPENMM_DIR"/openmm-*.slurm --max_tries 2 --wait --recovery "$OPENMM_DIR/slurm-recovery.json"

{ai2kit} tool model_devi read "$OPENMM_DIR/job-*/" --traj_file traj.xyz \\
  --md_file model_devi.out --format extxyz --ignore_error - \\
  slice "$EQUILIBRATION_FRAMES:" - grade --lo "$TRUST_FORCE_LOW" --hi "$TRUST_FORCE_HIGH" --col max_devi_f - \\
  dump_stats "$SCREENING_DIR/stats.tsv" - write "$SCREENING_DIR/good.xyz" --level good - \\
  write "$SCREENING_DIR/decent.xyz" --level decent - write "$SCREENING_DIR/poor.xyz" --level poor - done

if [[ ! -s "$SCREENING_DIR/decent.xyz" ]]; then
  echo "No candidate frames in the trust window"
  touch "$ITER_DIR/iter.done"
  exit 0
fi

{ai2kit} tool ase read "$SCREENING_DIR/decent.xyz" --format extxyz - sample "$MAX_LABEL" - \\
  write_frames "$LABELING_DIR/data/{{i:04d}}.vasp" --format vasp
if [[ "$USE_POOR_FRAMES" -gt 0 && -s "$SCREENING_DIR/poor.xyz" ]]; then
  {ai2kit} tool ase read "$SCREENING_DIR/poor.xyz" --format extxyz - sample "$USE_POOR_FRAMES" - \\
    write_frames "$LABELING_DIR/data/poor-{{i:04d}}.vasp" --format vasp
fi
{omb} combo add_files POSCAR_FILE "$LABELING_DIR/data/*" --abs - \\
  make_files "$LABELING_DIR/job-{{i:04d}}/run.sh" --template "$CONFIG_DIR/vasp/run.sh" --mode 755 - done
{omb} batch add_work_dirs "$LABELING_DIR/job-*" - add_header_files "$CONFIG_DIR/vasp/slurm-header.sh" - \\
  add_cmds "bash ./run.sh" - make "$LABELING_DIR/vasp-{{i}}.slurm" --concurrency "$LABEL_WORKERS"
{omb} job slurm submit "$LABELING_DIR"/vasp-*.slurm --max_tries 2 --wait --recovery "$LABELING_DIR/slurm-recovery.json"
{ai2kit} tool ase read "$LABELING_DIR/job-*/OUTCAR" --format vasp-out --ignore_error - \\
  write "$ITER_DIR/new-dataset/raw-labels.data" --format extxyz
$PYTHON_CMD "$CONFIG_DIR/normalize-labels.py" \\
  "$ITER_DIR/new-dataset/raw-labels.data" "$ITER_DIR/new-dataset/dataset.xyz"
if [[ "$UPDATE_MD_STRUCTURES" -gt 0 && -s "$SCREENING_DIR/good.xyz" ]]; then
  rm -f "$WORK_DIR"/openmm-data/*.xyz
  {ai2kit} tool ase read "$SCREENING_DIR/good.xyz" --format extxyz - \\
    sample "$UPDATE_MD_STRUCTURES" - write_frames "$WORK_DIR/openmm-data/{{i:04d}}.xyz" --format extxyz
fi
touch "$ITER_DIR/iter.done"
'''


def _run_script(settings: dict[str, Any], campaign: Campaign, commands: dict[str, Any]) -> str:
    temperatures = " ".join(str(float(value)) for value in campaign.exploration.get("temperatures", [300]))
    strains = ",".join(str(float(value)) for value in campaign.exploration.get("strains", [0.0]))
    replicas = int(campaign.exploration.get("replicas", 1))
    iterations = int(campaign.active_learning["max_iterations"])
    return f'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export CONFIG_DIR="$ROOT/00-config"
export WORK_DIR="$ROOT/../work"
mkdir -p "$WORK_DIR/openmm-data"
if [[ ! -f "$WORK_DIR/setup.done" ]]; then
  mapfile -t SOURCES < "$CONFIG_DIR/exploration-files.txt"
  {commands['python']} "$CONFIG_DIR/prepare-inputs.py" "$WORK_DIR/openmm-data" {shlex.quote(strains)} "${{SOURCES[@]}}"
  touch "$WORK_DIR/setup.done"
fi
export COMMITTEE_SEEDS={shlex.quote(' '.join(str(value) for value in settings['committee_seeds']))}
export MD_TEMPERATURES={shlex.quote(temperatures)}
export MD_REPLICAS={replicas}
export MD_STEPS={settings['md_steps']}
export SAMPLE_FREQUENCY={settings['sample_frequency']}
export TIMESTEP_FS={settings['timestep_fs']}
export FRICTION_PS={settings['friction_ps']}
export MAX_FORCE_EV_ANG={settings['max_force_ev_ang']}
export EQUILIBRATION_FRAMES={settings['equilibration_frames']}
export TRUST_FORCE_LOW={settings['trust_force_low']}
export TRUST_FORCE_HIGH={settings['trust_force_high']}
export MAX_LABEL={settings['selection_limit']}
export USE_POOR_FRAMES={settings['use_poor_frames']}
export UPDATE_MD_STRUCTURES={settings['update_md_structures']}
export PYTHON_CMD={shlex.quote(str(commands['python']))}
export MD_WORKERS={int(_settings_value(settings, 'md_workers', 1))}
export LABEL_WORKERS={int(_settings_value(settings, 'label_workers', 1))}
for ((iteration=0; iteration<{iterations}; iteration++)); do
  ITER_NAME="$(printf '%03d' "$iteration")" "$CONFIG_DIR/../01-workflow/iter-mace-openmm-vasp.sh"
done
'''


def _settings_value(settings: dict[str, Any], key: str, default: Any) -> Any:
    return settings.get(key, default)


def export_tesla_adapter(
    campaign: Campaign, *, output: str | Path | None = None, force: bool = False
) -> dict[str, Any]:
    settings = _settings(campaign)
    root = _root(campaign, output)
    generated = root / "generated"
    manifest_path = generated / "adapter_manifest.json"
    if root.exists() and any(root.iterdir()) and not force:
        raise SafetyError(
            f"AI2-Kit output directory is not empty: {root}; use --force to replace generated files"
        )
    profile, adapter_profile = _profile(campaign)
    commands = dict(adapter_profile["commands"])
    job_names = dict(adapter_profile["jobs"])
    models = [_resolve(campaign, value, "MACE committee model") for value in settings["committee_models"]]
    if any(not path.is_file() for path in models):
        raise SafetyError("Every MACE committee model must be a file")
    training_roots = [
        _resolve(campaign, value, "MACE training artifact", allow_directory=True)
        for value in settings["training_artifacts"]
    ]
    validation_roots = [
        _resolve(campaign, value, "MACE validation artifact", allow_directory=True)
        for value in settings["validation_artifacts"]
    ]
    training = _expand_xyz_artifacts(training_roots, "training")
    validation = _expand_xyz_artifacts(validation_roots, "validation")
    exploration = [_resolve(campaign, value, "exploration structure") for value in settings["exploration_artifacts"]]
    reference_inputs = dict(campaign.reference.get("inputs", {}))
    incar = _resolve(campaign, reference_inputs.get("INCAR", ""), "VASP INCAR")
    kpoints = _resolve(campaign, reference_inputs.get("KPOINTS", ""), "VASP KPOINTS")
    type_map = list(campaign.dataset.get("type_map", []))
    potcar_source = adapter_profile.get("potcar_source")
    if not type_map or not isinstance(potcar_source, dict) or any(element not in potcar_source for element in type_map):
        raise ConfigurationError("profile.ai2kit.potcar_source must cover dataset.type_map")
    for element in type_map:
        _resolve(campaign, str(potcar_source[element]), f"POTCAR source for {element}")

    config = generated / "00-config"
    workflow = generated / "01-workflow"
    directories = (
        config / "mace",
        config / "openmm",
        config / "vasp",
        config / "training",
        config / "validation",
        workflow,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(config / "committee-models.txt", "\n".join(str(path) for path in models) + "\n")
    _atomic_text(config / "exploration-files.txt", "\n".join(str(path) for path in exploration) + "\n")
    for index, path in enumerate(training):
        target = config / "training" / f"base-{index:04d}.xyz"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path)
    for index, path in enumerate(validation):
        target = config / "validation" / f"valid-{index:04d}.xyz"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path)
    _atomic_text(config / "prepare-inputs.py", _prepare_inputs_py(), executable=True)
    _atomic_text(config / "normalize-labels.py", _normalize_labels_py(), executable=True)
    _atomic_text(config / "openmm" / "openmm-run.py", _openmm_py(), executable=True)
    _atomic_text(config / "openmm" / "model-devi.py", _model_deviation_py(), executable=True)
    _atomic_text(
        config / "openmm" / "run.sh",
        _openmm_run_template(str(commands["python"]), settings["default_dtype"]),
        executable=True,
    )
    _atomic_text(
        config / "mace" / "run.sh",
        _mace_train_template(str(commands["mace"]), dict(campaign.models["mace"])),
        executable=True,
    )
    _atomic_text(
        config / "vasp" / "run.sh",
        _vasp_run_template(
            incar,
            kpoints,
            {key: str(value) for key, value in potcar_source.items()},
            _job_command(profile, str(job_names["label"])),
        ),
        executable=True,
    )
    _atomic_text(
        config / "mace" / "slurm-header.sh",
        _slurm_header(profile, str(job_names["train"]), f"{campaign.name}_mace_train"),
    )
    _atomic_text(
        config / "openmm" / "slurm-header.sh",
        _slurm_header(profile, str(job_names["explore"]), f"{campaign.name}_mace_explore"),
    )
    _atomic_text(
        config / "vasp" / "slurm-header.sh",
        _slurm_header(profile, str(job_names["label"]), f"{campaign.name}_vasp_label"),
    )
    _atomic_text(
        workflow / "iter-mace-openmm-vasp.sh",
        _iteration_script(str(commands["ai2kit"]), str(commands["omb"])),
        executable=True,
    )
    _atomic_text(generated / "run.sh", _run_script(settings, campaign, commands), executable=True)

    fingerprint, hashes = _fingerprint(campaign, generated)
    manifest = {
        "schema_version": 1,
        "interfaceforge_version": __version__,
        "workflow": "tesla_mace",
        "ai2kit_version": TARGET_AI2KIT_VERSION,
        "omb_version": settings.get("omb_version", TARGET_OMB_VERSION),
        "campaign": str(campaign.path),
        "profile": str(campaign.profile_path),
        "output_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_fingerprint": fingerprint,
        "generated_hashes": hashes,
        "committee": _input_records(models),
        "training": _input_records(training),
        "validation": _input_records(validation),
        "exploration": _input_records(exploration),
        "command": ["bash", str(generated / "run.sh")],
        "execution_state": "exported",
        "verification": "generated and unit-tested; awaiting a human-supervised LONI round",
    }
    _write_json(manifest_path, manifest)
    state = StateStore(campaign.root)
    state.event("ai2kit_tesla_export", output=str(root), fingerprint=fingerprint)
    state.artifact("ai2kit_tesla_manifest", manifest_path)
    return manifest


def preflight_tesla_adapter(
    campaign: Campaign,
    *,
    output_root: str | Path | None = None,
    remote: bool = False,
    report_output: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(campaign, output_root)
    generated = root / "generated"
    manifest_path = generated / "adapter_manifest.json"
    manifest = _load_json(manifest_path)
    _, adapter_profile = _profile(campaign)
    commands = dict(adapter_profile["commands"])
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail[:800]})

    fingerprint, _ = _fingerprint(campaign, generated)
    add("export_fingerprint", fingerprint == manifest.get("export_fingerprint"), fingerprint)
    for record in manifest.get("committee", []):
        path = Path(record["path"])
        add(f"committee:{path.name}", path.is_file() and sha256_file(path) == record.get("sha256"), str(path))
    executables: dict[str, str | None] = {}
    for key in ("ai2kit", "omb", "python", "mace"):
        executable = shutil.which(str(commands[key]))
        executables[key] = executable
        add(f"command:{key}", executable is not None, executable or str(commands[key]))
    python_cmd = str(commands["python"])
    if executables["python"]:
        version_probe = subprocess.run(
            [
                python_cmd,
                "-c",
                (
                    "import importlib.metadata as m; "
                    "print(m.version('ai2-kit')); print(m.version('oh-my-batch'))"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        version_lines = version_probe.stdout.splitlines()
        ai2kit_version = (
            version_lines[0].strip() if len(version_lines) > 0 else "not-installed"
        )
        omb_version = (
            version_lines[1].strip() if len(version_lines) > 1 else "not-installed"
        )
    else:
        ai2kit_version = "not-installed"
        omb_version = "not-installed"
    add(
        "ai2kit_version",
        ai2kit_version == TARGET_AI2KIT_VERSION,
        f"installed={ai2kit_version}; target={TARGET_AI2KIT_VERSION}",
    )
    add(
        "omb_version",
        omb_version == str(_settings(campaign).get("omb_version", TARGET_OMB_VERSION)),
        f"installed={omb_version}",
    )
    runtime_probe = (
        subprocess.run(
            [python_cmd, "-c", "import ase, mace, openmm, openmmml"],
            capture_output=True,
            text=True,
            check=False,
        )
        if executables["python"]
        else None
    )
    add(
        "mace_openmm_runtime",
        runtime_probe is not None and runtime_probe.returncode == 0,
        (
            runtime_probe.stderr
            if runtime_probe is not None and runtime_probe.stderr
            else "ase, mace, openmm and openmmml import successfully"
            if runtime_probe is not None
            else "configured Python command is unavailable"
        ),
    )
    shell = subprocess.run(["bash", "-n", str(generated / "run.sh")], capture_output=True, text=True, check=False)
    add("run_shell_syntax", shell.returncode == 0, shell.stderr or "ok")
    if os.environ.get("SLURM_JOB_ID"):
        add("controller_context", False, "TESLA controller must run on a login/service host, not inside a Slurm job")
    else:
        add("controller_context", True, "not inside a Slurm allocation")
    add("remote_checks", not remote, "TESLA uses the current Slurm host; --remote is not used", required=False)
    passed = all(item["ok"] for item in checks if item["required"])
    report = {
        "schema_version": 1,
        "workflow": "tesla_mace",
        "passed": passed,
        "remote_checked": False,
        "time": datetime.now(timezone.utc).isoformat(),
        "export_fingerprint": fingerprint,
        "checks": checks,
    }
    destination = Path(report_output).resolve() if report_output else root / "status" / "preflight.json"
    _write_json(destination, report)
    manifest["preflight"] = report
    manifest["execution_state"] = "preflight_passed" if passed else "preflight_failed"
    _write_json(manifest_path, manifest)
    return report


def run_tesla_adapter(
    campaign: Campaign,
    *,
    output_root: str | Path | None = None,
    execute: bool = False,
    resume: bool = False,
    allow_multiple_iterations: bool = False,
) -> dict[str, Any]:
    root = _root(campaign, output_root)
    generated = root / "generated"
    manifest_path = generated / "adapter_manifest.json"
    manifest = _load_json(manifest_path)
    command = ["bash", str(generated / "run.sh")]
    if not execute:
        return {"executed": False, "state": manifest.get("execution_state"), "command": command}
    if os.environ.get("SLURM_JOB_ID"):
        raise SafetyError("Run the TESLA controller from a LONI login/service host, not a compute job")
    if campaign.active_learning["max_iterations"] > 1 and not allow_multiple_iterations:
        raise SafetyError("Multiple TESLA iterations require --allow-multiple-iterations")
    fingerprint, _ = _fingerprint(campaign, generated)
    preflight = manifest.get("preflight", {})
    if not preflight.get("passed") or preflight.get("export_fingerprint") != fingerprint:
        raise SafetyError("A successful current TESLA preflight is required before --execute")
    work = root / "work"
    if work.exists() and any(work.iterdir()) and not resume:
        raise SafetyError("TESLA work state already exists; use --resume after reviewing status")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stdout_path = logs / f"tesla_{stamp}.out"
    stderr_path = logs / f"tesla_{stamp}.err"
    manifest.update({"execution_state": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    _write_json(manifest_path, manifest)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=generated, stdout=stdout, stderr=stderr, text=True, check=False)
    manifest.update(
        {
            "returncode": result.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "execution_state": "awaiting_approval" if result.returncode == 0 else "failed",
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    )
    _write_json(manifest_path, manifest)
    if result.returncode:
        tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
        raise SafetyError(f"TESLA exited with {result.returncode}; recovery files were preserved:\n{tail}")
    return {"executed": True, "returncode": 0, "state": manifest["execution_state"], "work": str(work)}


def status_tesla_adapter(campaign: Campaign, *, output_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(campaign, output_root)
    manifest_path = root / "generated" / "adapter_manifest.json"
    if not manifest_path.is_file():
        return {"workflow": "tesla_mace", "state": "not_exported", "output_root": str(root)}
    manifest = _load_json(manifest_path)
    iterations = []
    for path in sorted((root / "work").glob("iter-*")) if (root / "work").is_dir() else []:
        iterations.append(
            {
                "iteration": path.name,
                "done": (path / "iter.done").is_file(),
                "stats": (
                    str(path / "screening" / "stats.tsv")
                    if (path / "screening" / "stats.tsv").is_file()
                    else None
                ),
                "candidates": (
                    str(path / "screening" / "decent.xyz")
                    if (path / "screening" / "decent.xyz").is_file()
                    else None
                ),
                "labels": (
                    str(path / "new-dataset" / "dataset.xyz")
                    if (path / "new-dataset" / "dataset.xyz").is_file()
                    else None
                ),
            }
        )
    return {
        "workflow": "tesla_mace",
        "state": manifest.get("execution_state", "unknown"),
        "output_root": str(root),
        "iterations": iterations,
        "recovery_files": (
            sorted(str(path) for path in (root / "work").rglob("slurm-recovery.json"))
            if (root / "work").exists()
            else []
        ),
    }
