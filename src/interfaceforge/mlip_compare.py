# ruff: noqa: E501
"""Matched-frame, cross-backend MACE / DeePMD-DPA / NequIP accuracy audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DependencyError, SafetyError

DEFAULT_SEEDS = (11, 23, 37, 53)
ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"

# The DeePMD committee that MACE is compared against. The internal engine key
# stays "DPA2" for every architecture; only the rendered label changes.
DEEPMD_DISPLAY = {
    "dpa2": "DPA-2",
    "dpa2_ft": "DPA-2 (fine-tuned)",
    "dpa3": "DPA-3",
    "dpa3_ft": "DPA-3 (fine-tuned)",
    "dpa4": "DPA-4",
}

MACE_EVALUATOR = r"""#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--task", type=int)
args = parser.parse_args()
task = args.task if args.task is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
with (args.root / "mace_models.tsv").open(newline="", encoding="utf-8") as handle:
    models = list(csv.DictReader(handle, delimiter="\t"))
if task < 0 or task >= len(models):
    raise SystemExit(f"Invalid model task {task}")
row = models[task]
model_path = Path(row["model_path"])
if not model_path.is_file():
    raise SystemExit(f"Missing MACE model: {model_path}")
systems = json.loads((args.root / "systems.json").read_text(encoding="utf-8"))
target_root = args.root / "predictions" / "mace" / row["model"]
target_root.mkdir(parents=True, exist_ok=True)
calculator = MACECalculator(
    model_paths=str(model_path), device="cuda", default_dtype="float32"
)
for system in systems:
    target = target_root / f'{system["system_id"]}.npz'
    if target.is_file() and target.stat().st_size:
        continue
    frames = read(system["mace_input"], index=":")
    energies, forces = [], []
    for atoms in frames:
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        # Frames read from extxyz carry move_mask as FixAtoms; the reference labels
        # keep raw DFT forces on frozen atoms, so predictions must be raw as well.
        forces.append(np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64))
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            energy=np.asarray(energies, dtype=np.float64),
            forces=np.asarray(forces, dtype=np.float64),
        )
    os.replace(temporary, target)
    print(f'{row["model"]} {system["system_id"]}: {len(frames)} frames', flush=True)
"""


# Internal engine keys. "DPA2" is the legacy key for whichever DeePMD/DPA
# architecture is compared (its display name comes from DEEPMD_DISPLAY).
ENGINE_ORDER = ("MACE", "DPA2", "NEQUIP")
BACKEND_ENGINE = {"mace": "MACE", "deepmd": "DPA2", "nequip": "NEQUIP"}
ENGINE_COLORS = {"MACE": "#0072B2", "DPA2": "#D55E00", "NEQUIP": "#009E73"}
DEFAULT_BACKENDS = ("mace", "deepmd")
ENERGY_NORMALIZATION = (
    "(E_pred - E_DFT) / N_atoms in meV/atom on total energies against the canonical REF_energy "
    "labels shared by every backend; no per-backend reference shift. 'centered' additionally "
    "removes each system's mean offset."
)
STRESS_POLICY = (
    "not compared: the canonical labels exclude virials and not every backend was trained on "
    "stress, so no scientifically comparable stress value exists"
)

NEQUIP_EVALUATOR = r"""#!/usr/bin/env python3
# InterfaceForge mlip-compare NequIP evaluator (generated). Mirrors evaluate_mace.py:
# one committee member per array task, predictions for every matched system.
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--task", type=int)
parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
args = parser.parse_args()
task = args.task if args.task is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
with (args.root / "nequip_models.tsv").open(newline="", encoding="utf-8") as handle:
    models = list(csv.DictReader(handle, delimiter="\t"))
if task < 0 or task >= len(models):
    raise SystemExit(f"Invalid model task {task}")
row = models[task]
model_path = Path(row["model_path"])
if not model_path.is_file():
    raise SystemExit(f"Missing compiled NequIP model: {model_path}")
try:
    from nequip.integrations.ase import NequIPCalculator
except ImportError:
    from nequip.ase import NequIPCalculator
calculator = NequIPCalculator.from_compiled_model(str(model_path), device=args.device)
systems = json.loads((args.root / "systems.json").read_text(encoding="utf-8"))
target_root = args.root / "predictions" / "nequip" / row["model"]
target_root.mkdir(parents=True, exist_ok=True)
for system in systems:
    target = target_root / f'{system["system_id"]}.npz'
    if target.is_file() and target.stat().st_size:
        continue
    frames = read(system["mace_input"], index=":")
    energies, forces = [], []
    for atoms in frames:
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64))
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            energy=np.asarray(energies, dtype=np.float64),
            forces=np.asarray(forces, dtype=np.float64),
        )
    os.replace(temporary, target)
    print(f'{row["model"]} {system["system_id"]}: {len(frames)} frames', flush=True)
"""


def engine_display(engine: str, deepmd_arch: str = "dpa2") -> str:
    return {"MACE": "MACE", "NEQUIP": "NequIP"}.get(
        engine, DEEPMD_DISPLAY.get(deepmd_arch, deepmd_arch.upper()) if engine == "DPA2" else engine
    )


def _discover_nequip_models(root: Path, seeds: tuple[int, ...] | None) -> list[dict[str, Any]]:
    """Compiled NequIP committee members, labelled model_000.. in seed order."""

    from .nequip import discover_members

    if not root.is_dir():
        raise SafetyError(f"NequIP committee root not found: {root}")
    state = discover_members(root)
    by_seed = {member["seed"]: member for member in state["members"]}
    order = list(seeds) if seeds else [member["seed"] for member in state["members"]]
    if not order:
        raise SafetyError(f"No NequIP seed_* members under {root}")
    rows, missing = [], []
    for index, seed in enumerate(order):
        member = by_seed.get(seed)
        if member is None or not member["compiled_model"]:
            missing.append(seed)
            continue
        rows.append({"model": f"model_{index:03d}", "seed": seed, "model_path": member["compiled_model"]})
    if missing:
        ready = [member["seed"] for member in state["members"] if member["compiled_model"]]
        raise SafetyError(
            f"No compiled NequIP model for seed(s) {missing} under {root}. "
            f"Seeds with a compiled model: {ready or 'none'} (train/finalize first, or pass --nequip-seeds)"
        )
    return rows


def _ase_io() -> tuple[Any, Any]:
    try:
        from ase.io import iread, write
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "ASE is required; install InterfaceForge with interfaceforge[vasp]"
        ) from exc
    return iread, write


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_output(root: Path, campaign: Path, force: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not force:
            raise SafetyError(f"Comparison output is not empty: {root}")
        try:
            root.relative_to(campaign)
        except ValueError as exc:
            raise SafetyError(f"Output is outside campaign root: {root}") from exc
        if root == campaign:
            raise SafetyError(f"Refusing broad output replacement: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _frame_key(atoms: Any) -> tuple[str, int]:
    leaf = str(atoms.info.get("IF_leaf", "")).strip()
    if not leaf or "source_frame" not in atoms.info:
        raise SafetyError("MACE frame lacks IF_leaf/source_frame provenance")
    return leaf, int(atoms.info["source_frame"])


def _symbols(system: Path) -> list[str]:
    type_map = (system / "type_map.raw").read_text(encoding="utf-8").split()
    atom_types = [int(value) for value in (system / "type.raw").read_text(encoding="utf-8").split()]
    try:
        return [type_map[index] for index in atom_types]
    except IndexError as exc:
        raise SafetyError(f"Invalid type mapping in {system}") from exc


def _oxidation(leaf: str, heritage: str) -> str:
    """Canonical interface oxidation coordinate.

    Bulk heritage has no oxidation coordinate (``NA``). An interface leaf with
    no ``O_x`` token is unoxidized (``0``), not unknown; ``O_x1.0`` and
    ``O_x1.00`` collapse to the same ``1`` so the fully oxidized systems are one
    group rather than two.
    """

    if heritage == "bulk":
        return "NA"
    match = re.search(r"o[_-]?x[_=-]?([01](?:\.\d+)?)", leaf.lower())
    return f"{float(match.group(1)):g}" if match else "0"


def _groups(leaf: str) -> dict[str, str]:
    lower = leaf.lower()
    heritage = "bulk" if leaf.startswith("bulk/") else "interface"
    temperature = re.search(r"(?<!\d)(300|450|600)k", lower)
    return {
        "heritage": heritage,
        "temperature": f"{temperature.group(1)}K" if temperature else "NA",
        "family": "Ideal" if "ideal" in lower else ("Real" if "real" in lower else "NA"),
        "termination": "Ti_Term" if "ti_term" in lower else ("N_Term" if "n_term" in lower else "NA"),
        "oxidation": _oxidation(leaf, heritage),
    }


def validate_membership(
    mace_test: str | Path,
    deepmd_test: str | Path,
    grouped_root: str | Path,
    *,
    atol: float = 1.0e-7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Require identical identities, order, geometries, and reference labels."""

    mace_path = Path(mace_test).expanduser().resolve()
    deepmd_root = Path(deepmd_test).expanduser().resolve()
    grouped = Path(grouped_root).expanduser().resolve()
    grouped.mkdir(parents=True, exist_ok=True)
    iread, write = _ase_io()
    frames: dict[tuple[str, int], Any] = {}
    for atoms in iread(str(mace_path), index=":"):
        key = _frame_key(atoms)
        if key in frames:
            raise SafetyError(f"Duplicate MACE frame identity: {key}")
        frames[key] = atoms
    if not frames:
        raise SafetyError(f"No MACE frames in {mace_path}")

    system_paths = sorted({path.parent for path in deepmd_root.rglob("set.000")})
    if not system_paths:
        raise SafetyError(f"No DeePMD systems below {deepmd_root}")
    used: set[tuple[str, int]] = set()
    rows, max_delta = [], {"position": 0.0, "cell": 0.0, "energy": 0.0, "force": 0.0}
    atom_frames = 0
    for index, system in enumerate(system_paths):
        with (system / "frame_map.csv").open(newline="", encoding="utf-8") as handle:
            mapping = list(csv.DictReader(handle))
        if not mapping:
            raise SafetyError(f"Empty frame map: {system}")
        set_dir = system / "set.000"
        coord = np.load(set_dir / "coord.npy")
        box = np.load(set_dir / "box.npy")
        energy = np.load(set_dir / "energy.npy").reshape(-1)
        force = np.load(set_dir / "force.npy")
        symbols = _symbols(system)
        nframes, natoms = len(mapping), len(symbols)
        if coord.shape != (nframes, 3 * natoms) or force.shape != (nframes, 3 * natoms):
            raise SafetyError(f"Unexpected coordinate/force shape in {system}")
        if box.shape != (nframes, 9) or energy.shape != (nframes,):
            raise SafetyError(f"Unexpected box/energy shape in {system}")

        ordered, leaf = [], ""
        for local, item in enumerate(mapping):
            if int(item["local_frame"]) != local:
                raise SafetyError(f"Non-contiguous frame map in {system}")
            leaf = item["relative_leaf"]
            key = (leaf, int(item["source_frame"]))
            if key not in frames:
                raise SafetyError(f"DeePMD frame missing from MACE: {key}")
            if key in used:
                raise SafetyError(f"Duplicate DeePMD frame: {key}")
            used.add(key)
            atoms = frames[key]
            if atoms.get_chemical_symbols() != symbols:
                raise SafetyError(f"Atom order mismatch for {key}")
            delta = {
                "position": float(np.max(np.abs(atoms.positions.reshape(-1) - coord[local]))),
                "cell": float(np.max(np.abs(atoms.cell.array.reshape(-1) - box[local]))),
                "energy": abs(float(atoms.info[ENERGY_KEY]) - float(energy[local])),
                "force": float(np.max(np.abs(np.asarray(atoms.arrays[FORCES_KEY]).reshape(-1) - force[local]))),
            }
            for name, value in delta.items():
                max_delta[name] = max(max_delta[name], value)
            if max(delta.values()) > atol:
                raise SafetyError(f"Canonical data mismatch for {key}: {delta}")
            ordered.append(atoms)

        system_id = f"system_{index:03d}"
        mace_input = grouped / f"{system_id}.extxyz"
        write(str(mace_input), ordered, format="extxyz")
        first = ordered[0].info
        metadata = {
            key: first[info_key]
            for key, info_key in (
                ("stage", "IF_stage"),
                ("temperature_k", "IF_temperature_k"),
                ("case", "IF_case"),
                ("ligand", "IF_ligand"),
                ("coverage_pct", "IF_coverage_pct"),
            )
            if info_key in first
        }
        if "case" in metadata:
            # Canonical NiO frames omit IF_ligand for ligand-free cases: that is "none", not unknown.
            metadata.setdefault("ligand", "none")
        rows.append(
            {
                "system_id": system_id,
                "system_index": index,
                "relative_leaf": leaf,
                "deepmd_system": str(system.resolve()),
                "mace_input": str(mace_input.resolve()),
                "frames": nframes,
                "natoms": natoms,
                **_groups(leaf),
                "frame_ids": [
                    str(atoms.info.get("frame_id") or f"{leaf}:{int(atoms.info['source_frame'])}")
                    for atoms in ordered
                ],
                "source_frames": [int(atoms.info["source_frame"]) for atoms in ordered],
                "metadata": {key: (value.item() if hasattr(value, "item") else value) for key, value in metadata.items()},
            }
        )
        atom_frames += nframes * natoms

    missing = sorted(set(frames) - used)
    if missing:
        raise SafetyError(f"MACE frames absent from DeePMD: {missing[:5]}")
    summary = {
        "systems": len(rows),
        "frames": len(used),
        "atom_frames": atom_frames,
        "exact_membership": used == set(frames),
        "duplicate_frame_ids": 0,
        "max_absolute_delta": max_delta,
        "atol": atol,
    }
    return rows, summary


# Same priority as launch_scripts/separation_energy_common.sh:sep_mace_member --
# a stage-two / SWA export wins, otherwise the single-stage export is used. A
# naive foundation-model fine-tune (EMA, no SWA) only writes "<name>.model".
_STAGE_TWO_SUFFIXES = ("_stagetwo", "_stage_two", "_stage2", "_swa")


def _select_seed_model(mace_model_dir: Path, seed: int) -> tuple[Path, str]:
    directory = mace_model_dir if mace_model_dir.is_dir() else mace_model_dir.parent
    candidates = [
        path
        for path in sorted(directory.glob("*.model"))
        if path.is_file() and path.stat().st_size and "_compiled" not in path.name
    ]
    if not candidates:
        raise SafetyError(
            f"No usable MACE model for seed {seed} in {directory}: expected a "
            "non-empty, uncompiled *.model export (stage-two preferred)"
        )
    stage_two = [
        path
        for path in candidates
        if any(path.stem.endswith(suffix) for suffix in _STAGE_TWO_SUFFIXES)
    ]
    pool, tier = (stage_two, "stage-two") if stage_two else (candidates, "single-stage")
    chosen = max(pool, key=lambda path: path.stat().st_mtime)
    if len(pool) > 1:
        note = f"seed {seed}: {len(pool)} {tier} exports in {directory}; using newest {chosen.name}"
    elif tier == "single-stage":
        note = f"seed {seed}: no stage-two export in {directory}; using {chosen.name}"
    else:
        note = ""
    return chosen.resolve(), note


def _has_committee_layout(directory: Path) -> bool:
    """A ``seed_*`` directory that actually holds a non-empty ``.model`` file."""

    for seed_dir in directory.glob("seed_*"):
        if not seed_dir.is_dir():
            continue
        for search in (seed_dir / "mace_model", seed_dir):
            if search.is_dir() and any(
                path.is_file() and path.stat().st_size for path in search.glob("*.model")
            ):
                return True
    return False


def _resolve_committee_root(campaign: Path, mace_models_root: str | Path | None) -> Path:
    if not mace_models_root:
        return campaign / "models" / "mace_committee_520eV" / "mace_committee"
    given = Path(mace_models_root).expanduser()
    if given.is_absolute():
        return given.resolve()
    encut_parent = campaign / "models" / "mace_committee_520eV"
    if len(given.parts) == 1:
        # A bare name ('mace_finetune_committee'): the committees live under the
        # ENCUT-tagged parent -- check there first so a stale empty tree beside
        # the campaign root does not shadow the real one.
        bases = (encut_parent, campaign / "models", campaign, Path.cwd())
    else:
        bases = (campaign, Path.cwd())
    matches = [base / given for base in bases if (base / given).is_dir()]
    if not matches:
        raise SafetyError(
            f"MACE committee root not found: {mace_models_root!r}. Looked under "
            f"{', '.join(str(base) for base in bases)}. Pass an absolute path or a "
            "name like 'mace_finetune_committee'."
        )
    for candidate in matches:
        if _has_committee_layout(candidate):
            return candidate.resolve()
    return matches[0].resolve()


def _usable_seeds(root: Path) -> list[int]:
    found = []
    for seed_dir in root.glob("seed_*"):
        suffix = seed_dir.name.removeprefix("seed_")
        if not seed_dir.is_dir() or not suffix.isdigit():
            continue
        try:
            _select_seed_model(seed_dir / "mace_model", int(suffix))
        except SafetyError:
            continue
        found.append(int(suffix))
    return sorted(found)


def _discover_models(
    root: Path, seeds: tuple[int, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, notes, missing = [], [], []
    for index, seed in enumerate(seeds):
        try:
            chosen, note = _select_seed_model(root / f"seed_{seed}" / "mace_model", seed)
        except SafetyError:
            missing.append(seed)
            continue
        if note:
            notes.append(note)
        rows.append(
            {"model": f"model_{index:03d}", "seed": seed, "model_path": str(chosen)}
        )
    if missing:
        available = _usable_seeds(root)
        raise SafetyError(
            f"No usable MACE export for seed(s) {missing} under {root}. "
            + (
                f"Seeds with a usable export: {available}; re-run with "
                f"--seeds {' '.join(map(str, available))}"
                if available
                else "No seed has a usable export here."
            )
        )
    return rows, notes


def _slurm(root: Path, nmodels: int) -> str:
    return f"""#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J mlip.mace.audit
#SBATCH --array=0-{nmodels - 1}%2
#SBATCH -o mlip.mace.audit.%A_%a.out
#SBATCH -e mlip.mace.audit.%A_%a.err
set -eo pipefail
module purge
set +u
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate /project/lgutsev/env/mace_env
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python {root / "evaluate_mace.py"} --root {root}
"""


def _resolve_nequip_root(campaign: Path, value: str | Path | None) -> Path:
    if not value:
        return campaign / "models" / "nequip"
    given = Path(value).expanduser()
    return given.resolve() if given.is_absolute() else (campaign / given).resolve()


def _nequip_launcher(root: Path, nmodels: int, profile_path: str | Path | None, profile_job: str) -> str:
    """Render the NequIP inference array from the campaign's scheduler profile.

    Unlike the historical MACE launcher, no account, partition or environment is
    hard-coded here: everything comes from ``profile_job`` in the profile.
    """

    from .config import load_profile
    from .scheduler import render_job

    if profile_path is None:
        raise SafetyError("NequIP comparison needs the campaign scheduler profile (run via 'iface mlip-compare')")
    profile = load_profile(profile_path)
    job = dict(profile.get("jobs", {}).get(profile_job, {}))
    if not job:
        raise SafetyError(f"Scheduler profile has no job {profile_job!r} for NequIP inference")
    device = "cuda" if int(job.get("gpus", 0) or 0) > 0 else "cpu"
    command = f"python {shlex.quote(str(root / 'evaluate_nequip.py'))} --root {shlex.quote(str(root))} --device {device}"
    if str(profile.get("scheduler")) == "local":
        command = f"for TASK in $(seq 0 {nmodels - 1}); do {command} --task \"$TASK\"; done"
        return render_job(profile, profile_job, command=command, job_name="mlip_nequip_audit", working_directory=str(root))
    return render_job(
        profile,
        profile_job,
        command=command,
        job_name="mlip_nequip_audit",
        array=f"0-{nmodels - 1}%2",
        working_directory=str(root),
    )


def prepare_comparison(
    campaign_root: str | Path,
    *,
    output_root: str | Path | None = None,
    mace_models_root: str | Path | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    deepmd_arch: str = "dpa2",
    force: bool = False,
    backends: tuple[str, ...] | None = None,
    nequip_models_root: str | Path | None = None,
    nequip_seeds: tuple[int, ...] | None = None,
    profile_path: str | Path | None = None,
    nequip_profile: str = "nequip_gpu",
) -> dict[str, Any]:
    """Validate matched canonical test frames and stage inference for each backend.

    ``backends`` defaults to MACE + DeePMD (the historical comparison) and gains
    ``nequip`` automatically when ``nequip_models_root`` is given. Every backend is
    evaluated on the *same* ``inputs/system_XXX.extxyz`` frames, whose identity,
    geometry and labels were proven equal to the DeePMD systems.
    """

    if deepmd_arch not in DEEPMD_DISPLAY:
        raise SafetyError(
            f"Unknown DeePMD architecture {deepmd_arch!r}; expected one of "
            f"{sorted(DEEPMD_DISPLAY)}"
        )
    selected = tuple(backends) if backends else DEFAULT_BACKENDS + (("nequip",) if nequip_models_root else ())
    unknown = sorted(set(selected) - set(BACKEND_ENGINE))
    if unknown or not selected or len(set(selected)) != len(selected):
        raise SafetyError(f"backends must be distinct values from {sorted(BACKEND_ENGINE)}; got {list(selected)}")
    campaign = Path(campaign_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve() if output_root else campaign / "audit" / "mlip_compare"
    _prepare_output(output, campaign, force)
    canonical = campaign / "datasets" / "canonical"
    systems, validation = validate_membership(
        canonical / "test.extxyz", canonical / "deepmd" / "test", output / "inputs"
    )
    _write_json(output / "systems.json", systems)
    engines: dict[str, list[dict[str, Any]]] = {}
    launchers: dict[str, str] = {}
    model_notes: list[str] = []
    if "mace" in selected:
        model_root = _resolve_committee_root(campaign, mace_models_root)
        mace_models, model_notes = _discover_models(model_root, seeds)
        (output / "evaluate_mace.py").write_text(MACE_EVALUATOR, encoding="utf-8")
        (output / "evaluate_mace.py").chmod(0o755)
        with (output / "mace_models.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("model", "seed", "model_path"), delimiter="\t")
            writer.writeheader()
            writer.writerows(mace_models)
        launcher = output / "run_mace_evaluate.slurm"
        launcher.write_text(_slurm(output, len(mace_models)), encoding="utf-8")
        launchers["MACE"] = str(launcher)
        engines["MACE"] = mace_models
    if "deepmd" in selected:
        engines["DPA2"] = [{"model": f"model_{index:03d}", "seed": seed} for index, seed in enumerate(seeds)]
    if "nequip" in selected:
        nequip_root = _resolve_nequip_root(campaign, nequip_models_root)
        nequip_models = _discover_nequip_models(nequip_root, nequip_seeds)
        (output / "evaluate_nequip.py").write_text(NEQUIP_EVALUATOR, encoding="utf-8")
        (output / "evaluate_nequip.py").chmod(0o755)
        with (output / "nequip_models.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("model", "seed", "model_path"), delimiter="\t")
            writer.writeheader()
            writer.writerows(nequip_models)
        launcher = output / "run_nequip_evaluate.slurm"
        launcher.write_text(_nequip_launcher(output, len(nequip_models), profile_path, nequip_profile), encoding="utf-8")
        launcher.chmod(0o750)
        launchers["NEQUIP"] = str(launcher)
        engines["NEQUIP"] = nequip_models
    legacy_models = engines.get("MACE") or engines.get("DPA2") or engines.get("NEQUIP") or []
    payload = {
        "schema_version": 1,
        "status": "READY",
        "benchmark_scope": "in-distribution interpolation",
        "mace_inference_dtype": "float32",
        "deepmd_architecture": deepmd_arch,
        "campaign_root": str(campaign),
        "output_root": str(output),
        "backends": list(selected),
        "engines": engines,
        "engine_display": {engine: engine_display(engine, deepmd_arch) for engine in engines},
        "models": legacy_models,
        "model_selection_notes": model_notes,
        "systems": systems,
        "validation": validation,
        "energy_normalization": ENERGY_NORMALIZATION,
        "stress_comparison": STRESS_POLICY,
        "launchers": launchers,
        "launcher": launchers.get("MACE") or next(iter(launchers.values()), None),
        "next": f"sbatch {launchers.get('MACE') or next(iter(launchers.values()), '')}".strip(),
        "next_steps": [f"sbatch {path}" for path in launchers.values()]
        + ([f"run the DeePMD `dp test` evaluation job for {deepmd_arch}"] if "DPA2" in engines else []),
    }
    _write_json(output / "comparison_manifest.json", payload)
    return payload


def _latest_deepmd_eval(campaign: Path, arch: str = "dpa2") -> Path | None:
    """Return the most recent evaluation job directory for a DeePMD architecture.

    Slurm job IDs are not zero-padded, so a lexical sort would place ``job_998``
    after ``job_1002``. Order by the integer job ID when every candidate has one
    and fall back to modification time otherwise.
    """

    roots = [
        path
        for path in (campaign / "models" / "deepmd" / "evaluation" / arch).glob("job_*")
        if path.is_dir()
    ]
    if not roots:
        return None
    job_ids = [path.name.removeprefix("job_") for path in roots]
    if all(job_id.isdigit() for job_id in job_ids):
        return max(roots, key=lambda path: int(path.name.removeprefix("job_")))
    return max(roots, key=lambda path: path.stat().st_mtime)


def _manifest_engines(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Engines of a comparison manifest; pre-NequIP manifests were MACE + DeePMD."""

    engines = manifest.get("engines")
    if isinstance(engines, dict) and engines:
        return {engine: list(models) for engine, models in engines.items()}
    return {"MACE": list(manifest["models"]), "DPA2": list(manifest["models"])}


def _prediction_file(output: Path, engine: str, label: str, system_id: str) -> Path:
    return output / "predictions" / engine.lower() / label / f"{system_id}.npz"


def comparison_status(
    campaign_root: str | Path,
    *,
    output_root: str | Path | None = None,
    deepmd_eval_root: str | Path | None = None,
) -> dict[str, Any]:
    campaign = Path(campaign_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve() if output_root else campaign / "audit" / "mlip_compare"
    manifest_path = output / "comparison_manifest.json"
    if not manifest_path.is_file():
        raise SafetyError(f"Run mlip-compare prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    systems = manifest["systems"]
    engines = _manifest_engines(manifest)
    arch = str(manifest.get("deepmd_architecture", "dpa2"))
    dpa_root = None
    if "DPA2" in engines:
        dpa_root = (
            Path(deepmd_eval_root).expanduser().resolve()
            if deepmd_eval_root
            else _latest_deepmd_eval(campaign, arch)
        )
    counts: dict[str, dict[str, int]] = {}
    for engine, models in engines.items():
        counts[engine] = {}
        for model in models:
            label = model["model"]
            if engine == "DPA2":
                done = 0
                if dpa_root:
                    for system in systems:
                        prefix = dpa_root / "by_system" / system["system_id"] / f"{label}_detail"
                        if Path(str(prefix) + ".e_peratom.out").is_file() and Path(str(prefix) + ".f.out").is_file():
                            done += 1
                counts[engine][label] = done
            else:
                counts[engine][label] = sum(
                    _prediction_file(output, engine, label, system["system_id"]).is_file() for system in systems
                )
    expected = len(systems)
    ready = {
        engine: bool(values) and all(value == expected for value in values.values())
        for engine, values in counts.items()
    }
    hints: list[str] = []
    launchers = manifest.get("launchers", {})
    if "MACE" in engines and not ready["MACE"]:
        hints.append(
            f"MACE inference incomplete -- (re-)submit {output / 'run_mace_evaluate.slurm'}"
        )
    if "NEQUIP" in engines and not ready["NEQUIP"]:
        hints.append(
            f"NequIP inference incomplete -- (re-)submit {launchers.get('NEQUIP', output / 'run_nequip_evaluate.slurm')}"
        )
    if "DPA2" in engines and not ready["DPA2"]:
        if deepmd_eval_root and not (dpa_root and dpa_root.is_dir()):
            hints.append(f"--deepmd-eval-root does not exist: {dpa_root}")
        elif dpa_root is None:
            hints.append(
                f"no job_* evaluation under {campaign / 'models' / 'deepmd' / 'evaluation' / arch}/"
                " -- run the DeePMD `dp test` job for this architecture first"
            )
        elif all(value == 0 for value in counts["DPA2"].values()):
            hints.append(
                f"{dpa_root} has no by_system/*/model_XXX_detail.*.out files for this test set"
            )
        else:
            hints.append(f"DeePMD evaluation partial under {dpa_root}")
    return {
        "schema_version": 1,
        "status": "READY_TO_FINALIZE" if all(ready.values()) else "INCOMPLETE",
        "expected_systems_per_model": expected,
        "deepmd_architecture": arch,
        "engines": list(engines),
        "mace": counts.get("MACE", {}),
        "deepmd": counts.get("DPA2", {}),
        "nequip": counts.get("NEQUIP", {}),
        "ready": ready,
        "deepmd_eval_root": str(dpa_root) if dpa_root else None,
        "deepmd_eval_root_exists": bool(dpa_root and dpa_root.is_dir()),
        "hints": hints,
    }


def _numeric(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append([float(value) for value in line.split()])
    if not rows:
        raise SafetyError(f"No numeric predictions in {path}")
    return np.asarray(rows, dtype=float)


def _metrics(
    ref_e: np.ndarray,
    pred_e: np.ndarray,
    ref_f: np.ndarray,
    pred_f: np.ndarray,
    *,
    center_groups: list[slice] | None = None,
) -> dict[str, float]:
    e_error = np.asarray(pred_e) - np.asarray(ref_e)
    f_error = np.asarray(pred_f) - np.asarray(ref_f)
    if center_groups:
        centered = np.concatenate(
            [e_error[group] - np.mean(e_error[group]) for group in center_groups]
        )
    else:
        centered = e_error - np.mean(e_error)
    vectors = f_error.reshape(-1, 3)
    force_std = float(np.std(ref_f))
    return {
        "energy_mae_mev_per_atom": float(np.mean(np.abs(e_error)) * 1000.0),
        "energy_rmse_mev_per_atom": float(np.sqrt(np.mean(e_error**2)) * 1000.0),
        "energy_centered_rmse_mev_per_atom": float(np.sqrt(np.mean(centered**2)) * 1000.0),
        "force_mae_mev_per_angstrom": float(np.mean(np.abs(f_error)) * 1000.0),
        "force_rmse_mev_per_angstrom": float(np.sqrt(np.mean(f_error**2)) * 1000.0),
        "force_vector_rmse_mev_per_angstrom": float(
            np.sqrt(np.mean(np.sum(vectors**2, axis=1))) * 1000.0
        ),
        "force_relative_rmse_percent": (
            float(np.sqrt(np.mean(f_error**2)) / force_std * 100.0)
            if force_std
            else math.nan
        ),
    }


METADATA_GROUP_FIELDS = ("stage", "temperature_k", "ligand", "coverage_pct")


def _system_row(
    engine: str,
    model: str,
    seed: int | str,
    system: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "engine": engine,
        "model": model,
        "seed": seed,
        "system_id": system["system_id"],
        "relative_leaf": system["relative_leaf"],
        "frames": system["frames"],
        "natoms": system["natoms"],
        "heritage": system["heritage"],
        "temperature": system["temperature"],
        "family": system["family"],
        "termination": system["termination"],
        "oxidation": system["oxidation"],
        **{key: (system.get("metadata") or {}).get(key, "NA") for key in METADATA_GROUP_FIELDS},
        **metrics,
    }


def _overall(
    engine: str,
    model: str,
    seed: int | str,
    entries: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    ref_e = np.concatenate([entry[1] for entry in entries])
    pred_e = np.concatenate([entry[2] for entry in entries])
    ref_f = np.concatenate([entry[3].reshape(-1) for entry in entries])
    pred_f = np.concatenate([entry[4].reshape(-1) for entry in entries])
    cursor, groups = 0, []
    for entry in entries:
        groups.append(slice(cursor, cursor + len(entry[1])))
        cursor += len(entry[1])
    micro = _metrics(ref_e, pred_e, ref_f, pred_f, center_groups=groups)
    per_system = [_metrics(*entry[1:]) for entry in entries]
    macro = {}
    for key in micro:
        values = np.asarray([row[key] for row in per_system])
        if "mae" in key or key.endswith("_percent"):
            macro[key] = float(np.nanmean(values))
        else:
            macro[key] = float(np.sqrt(np.nanmean(values**2)))
    base = {"engine": engine, "model": model, "seed": seed, "systems": len(entries)}
    return [{**base, "averaging": "micro", **micro}, {**base, "averaging": "macro", **macro}]


def _uncertainty(
    engine: str,
    refs: list[np.ndarray],
    predictions: list[list[np.ndarray]],
    quantity: str,
) -> dict[str, Any]:
    reference = np.concatenate([value.reshape(-1) for value in refs])
    members = np.stack(
        [np.concatenate([value.reshape(-1) for value in member]) for member in predictions]
    )
    mean, spread = np.mean(members, axis=0), np.std(members, axis=0)
    error = np.abs(mean - reference)
    correlation = (
        float(np.corrcoef(spread, error)[0, 1])
        if np.std(spread) > 0 and np.std(error) > 0
        else None
    )
    denominator = float(np.mean(spread**2))
    scale = (
        float(np.sqrt(np.mean((mean - reference) ** 2) / denominator))
        if denominator > 0
        else None
    )
    calibrated = spread * scale if scale is not None else spread
    return {
        "engine": engine,
        "quantity": quantity,
        "observations": int(reference.size),
        "spread_error_pearson": correlation,
        "rmse_to_rms_spread_scale": scale,
        "raw_coverage_1sigma": float(np.mean(error <= spread)),
        "raw_coverage_2sigma": float(np.mean(error <= 2.0 * spread)),
        "calibrated_coverage_1sigma": float(np.mean(error <= calibrated)),
        "calibrated_coverage_2sigma": float(np.mean(error <= 2.0 * calibrated)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _write_svg(
    path: Path, overall: list[dict[str, Any]], *, deepmd_display: str = "DPA-2"
) -> None:
    rows = {
        row["engine"]: row
        for row in overall
        if row["model"] == "ensemble_mean" and row["averaging"] == "micro"
    }
    engines = [name for name in ENGINE_ORDER if name in rows]
    labels = {"MACE": "MACE", "DPA2": deepmd_display, "NEQUIP": "NequIP"}
    colors = {"MACE": "#2563eb", "DPA2": "#dc2626", "NEQUIP": "#059669"}
    metrics = (
        ("Energy RMSE (meV/atom)", "energy_rmse_mev_per_atom"),
        ("Force RMSE (meV/A)", "force_rmse_mev_per_angstrom"),
    )
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:28px;font-weight:700}.label{font-size:15px}.value{font-size:14px;font-weight:700}</style>',
        '<text x="55" y="52" class="title">Matched-frame MLIP comparison</text>',
        '<text x="55" y="80" class="label">Ensemble-mean micro RMSE on identical test configurations</text>',
    ]
    for panel, (title, key) in enumerate(metrics):
        x0 = 70 + panel * 480
        values = [float(rows[engine][key]) for engine in engines]
        maximum = max(values, default=1.0) or 1.0
        svg.append(f'<text x="{x0}" y="135" class="label">{title}</text>')
        for index, engine in enumerate(engines):
            y = 180 + index * 105
            width = 350.0 * values[index] / maximum
            svg.append(f'<text x="{x0}" y="{y + 25}" class="label">{labels[engine]}</text>')
            svg.append(
                f'<rect x="{x0 + 70}" y="{y}" width="{width:.1f}" height="38" '
                f'rx="5" fill="{colors[engine]}"/>'
            )
            svg.append(
                f'<text x="{x0 + 80 + width:.1f}" y="{y + 25}" class="value">'
                f'{values[index]:.3f}</text>'
            )
    svg.append(
        '<text x="55" y="560" class="label">Scope: interpolation; use an independent challenge set for transferability claims.</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _heatmap_label(row: dict[str, Any]) -> str:
    """Return a compact, unique row label for a canonical test system."""

    if row["heritage"] == "bulk":
        return f'bulk / {row["relative_leaf"].removeprefix("bulk/")}'
    if row.get("family", "NA") == "NA" and row.get("termination", "NA") == "NA":
        leaf = str(row["relative_leaf"])
        return leaf if len(leaf) <= 70 else "…" + leaf[-69:]
    return (
        f'interface / {row["temperature"]} / {row["family"]} / '
        f'{row["termination"]} / O={row["oxidation"]}'
    )


def _write_force_heatmaps(
    output: Path,
    system_rows: list[dict[str, Any]],
    *,
    deepmd_display: str = "DPA-2",
) -> dict[str, Path]:
    """Plot member-by-system force RMSE on one shared scale for every engine."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "MLIP comparison heatmaps require matplotlib; "
            "install InterfaceForge with interfaceforge[report]"
        ) from exc

    rows = [row for row in system_rows if row["model"] != "ensemble_mean"]
    engines = tuple(engine for engine in ENGINE_ORDER if any(row["engine"] == engine for row in rows))
    if not engines:
        raise SafetyError("Cannot plot force RMSE heatmaps without member rows")
    system_ids = sorted(
        {str(row["system_id"]) for row in rows},
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    lookup = {
        (str(row["engine"]), str(row["model"]), str(row["system_id"])): row
        for row in rows
    }
    model_names = {
        engine: sorted({str(row["model"]) for row in rows if row["engine"] == engine}) for engine in engines
    }
    template: dict[str, dict[str, Any]] = {}
    for engine in engines:
        for model in model_names[engine]:
            for system_id in system_ids:
                row = lookup.get((engine, model, system_id))
                if row is None:
                    raise SafetyError(
                        "Cannot plot force RMSE heatmaps from an incomplete system/member matrix"
                    )
                template.setdefault(system_id, row)

    matrices: dict[str, np.ndarray] = {
        engine: np.asarray(
            [
                [
                    float(lookup[(engine, model, system_id)]["force_rmse_mev_per_angstrom"]) / 1000.0
                    for model in model_names[engine]
                ]
                for system_id in system_ids
            ],
            dtype=float,
        )
        for engine in engines
    }
    labels = [_heatmap_label(template[system_id]) for system_id in system_ids]
    shared_min = min(float(np.min(matrix)) for matrix in matrices.values())
    shared_max = max(float(np.max(matrix)) for matrix in matrices.values())
    if not math.isfinite(shared_min) or not math.isfinite(shared_max):
        raise SafetyError("Non-finite force RMSE cannot be plotted")
    if shared_max <= shared_min:
        shared_max = shared_min + 1.0e-12
    display = {"MACE": "MACE", "DPA2": deepmd_display, "NEQUIP": "NequIP"}

    def render(path_stem: str, selected: tuple[str, ...]) -> tuple[Path, Path]:
        height = max(10.0, 0.31 * len(system_ids) + 2.0)
        width = 12.0 if len(selected) == 1 else 9.0 * len(selected)
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=(width, height),
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        image = None
        for panel, engine in enumerate(selected):
            ax = axes[0, panel]
            matrix = matrices[engine]
            names = model_names[engine]
            image = ax.imshow(
                matrix,
                aspect="auto",
                cmap="viridis",
                vmin=shared_min,
                vmax=shared_max,
            )
            ax.set_title(display[engine], fontsize=14)
            ax.set_xticks(range(len(names)), labels=names, rotation=28, ha="right")
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.tick_params(axis="y", labelsize=7.2, labelleft=panel == 0)
            ax.tick_params(axis="x", labelsize=8.5)
            threshold = shared_min + 0.55 * (shared_max - shared_min)
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = float(matrix[row_index, column_index])
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.4f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color="black" if value >= threshold else "white",
                    )
            for row_index in range(len(system_ids) + 1):
                ax.axhline(row_index - 0.5, color="white", linewidth=0.25, alpha=0.45)
            for column_index in range(len(names) + 1):
                ax.axvline(column_index - 0.5, color="white", linewidth=0.25, alpha=0.45)

        assert image is not None
        colorbar = fig.colorbar(image, ax=list(axes[0]), shrink=0.78, pad=0.02)
        colorbar.set_label("Force RMSE (eV/Å)")
        title = "Per-system force RMSE"
        if len(selected) > 1:
            names = [display[engine] for engine in selected]
            joined = " and ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + f" and {names[-1]}"
            title += f" — matched {joined} committees (shared scale)"
        fig.suptitle(title, fontsize=16)
        png = output / f"{path_stem}.png"
        svg = output / f"{path_stem}.svg"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        return png, svg

    outputs: dict[str, Path] = {}
    for engine in engines:
        png, svg = render(f"force_rmse_heatmap_{engine.lower()}", (engine,))
        outputs[f"force_heatmap_{engine.lower()}_png"] = png
        outputs[f"force_heatmap_{engine.lower()}_svg"] = svg
    if len(engines) > 1:
        png, svg = render("force_rmse_heatmaps", engines)
        outputs["force_heatmaps_png"] = png
        outputs["force_heatmaps_svg"] = svg
    return outputs


PUBLICATION_GROUP_ORDER = (
    "Overall",
    "Bulk SiN",
    "Bulk TiN",
    "Bulk TiO",
    "Ideal / N-terminated interface",
    "Ideal / Ti-terminated interface",
    "Real / N-terminated interface",
    "Real / Ti-terminated interface",
)

TEMPERATURE_GROUP_ORDER = ("Overall", "300 K", "450 K", "600 K")

OXIDATION_GROUP_ORDER = (
    "Overall",
    "Bulk",
    "O = 0",
    "O = 0.25",
    "O = 0.5",
    "O = 0.75",
    "O = 1",
)


def _publication_group(row: dict[str, Any]) -> str:
    """Collapse the 48 trajectories into chemically interpretable figure bins."""

    if row["heritage"] == "bulk":
        leaf = str(row["relative_leaf"])
        for material in ("SiN", "TiN", "TiO"):
            if f"/{material}-Bulk" in leaf:
                return f"Bulk {material}"
        raise SafetyError(f"Cannot assign publication bulk group: {leaf}")
    family = str(row["family"])
    termination = {
        "N_Term": "N-terminated",
        "Ti_Term": "Ti-terminated",
    }.get(str(row["termination"]))
    if family not in {"Ideal", "Real"} or termination is None:
        raise SafetyError(
            "Cannot assign publication interface group: "
            f'{row["relative_leaf"]}'
        )
    return f"{family} / {termination} interface"


def _publication_summary_rows(
    system_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact pooled RMSEs for the overall set and physical-system bins."""

    return _pooled_summary_rows(
        system_rows,
        group_key="physical_group",
        group_order=PUBLICATION_GROUP_ORDER,
        group_for=_publication_group,
    )


def _temperature_group(row: dict[str, Any]) -> str:
    exported = row.get("temperature_k", "NA")
    if exported not in (None, "", "NA"):
        try:
            return f"{float(exported):g} K"
        except (TypeError, ValueError):
            pass
    temperature = str(row["temperature"])
    if not re.fullmatch(r"\d+K", temperature):
        raise SafetyError(
            f'Cannot assign temperature group: {row["relative_leaf"]}'
        )
    return temperature.removesuffix("K") + " K"


def _temperature_summary_rows(
    system_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact pooled RMSEs for the overall, 300 K, and 450 K sets."""

    return _pooled_summary_rows(
        system_rows,
        group_key="temperature_group",
        group_order=TEMPERATURE_GROUP_ORDER,
        group_for=_temperature_group,
    )


def _oxidation_group(row: dict[str, Any]) -> str:
    """Bin trajectories by interface oxygen coverage; bulk is its own bin."""

    if row["heritage"] == "bulk":
        return "Bulk"
    oxidation = str(row["oxidation"])
    if oxidation == "NA":
        raise SafetyError(
            f'Interface trajectory has no oxidation coordinate: {row["relative_leaf"]}'
        )
    return f"O = {oxidation}"


def _oxidation_summary_rows(
    system_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact pooled RMSEs for the overall set, bulk, and each coverage.

    Bulk trajectories have no oxidation coordinate; they are pooled into a
    single ``Bulk`` bin so every test system still lands in exactly one bin and
    the ``Overall`` row stays identical to the other summary figures.
    """

    return _pooled_summary_rows(
        system_rows,
        group_key="oxidation_group",
        group_order=OXIDATION_GROUP_ORDER,
        group_for=_oxidation_group,
    )


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def _pooled_summary_rows(
    system_rows: list[dict[str, Any]],
    *,
    group_key: str,
    group_order: tuple[str, ...],
    group_for: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """Return exact pooled RMSEs for an overall set and requested bins.

    RMSEs are pooled from squared per-system RMSEs with their correct numbers
    of observations. This is algebraically identical to recomputing the metric
    from all frame predictions in a bin; it is not an average of RMSEs.
    """

    augmented = [(row, group_for(row)) for row in system_rows]
    groups_present = {group for _, group in augmented}
    ordered_groups = [
        group
        for group in group_order
        if group == "Overall" or group in groups_present
    ] + sorted(groups_present - set(group_order), key=_natural_key)
    rows: list[dict[str, Any]] = []
    engines = [
        engine
        for engine in ENGINE_ORDER
        if any(row["engine"] == engine for row in system_rows)
    ]
    for engine in engines:
        model_order = list(
            dict.fromkeys(
                str(row["model"])
                for row in system_rows
                if row["engine"] == engine
            )
        )
        for model in model_order:
            candidates = [
                (row, group)
                for row, group in augmented
                if row["engine"] == engine and str(row["model"]) == model
            ]
            for group in ordered_groups:
                selected = [
                    row
                    for row, physical_group in candidates
                    if group == "Overall" or physical_group == group
                ]
                if not selected:
                    continue
                energy_observations = sum(int(row["frames"]) for row in selected)
                force_observations = sum(
                    int(row["frames"]) * int(row["natoms"]) * 3
                    for row in selected
                )
                energy_rmse = math.sqrt(
                    sum(
                        int(row["frames"])
                        * float(row["energy_rmse_mev_per_atom"]) ** 2
                        for row in selected
                    )
                    / energy_observations
                )
                force_rmse = math.sqrt(
                    sum(
                        int(row["frames"])
                        * int(row["natoms"])
                        * 3
                        * float(row["force_rmse_mev_per_angstrom"]) ** 2
                        for row in selected
                    )
                    / force_observations
                )
                rows.append(
                    {
                        "engine": engine,
                        "model": model,
                        "seed": selected[0]["seed"],
                        group_key: group,
                        "systems": len(selected),
                        "frames": energy_observations,
                        "atom_frames": force_observations // 3,
                        "energy_rmse_mev_per_atom": energy_rmse,
                        "force_rmse_mev_per_angstrom": force_rmse,
                    }
                )
    return rows


# Ordered qualitative palette (Okabe-Ito, then two extras) for N-family figures.
FAMILY_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
    "#7F3C8D",  # violet
    "#11A579",  # teal
)


def _family_offsets(families: list[str]) -> dict[str, float]:
    """Symmetric vertical offsets within a group row; +/-0.13 for two families."""

    n = len(families)
    if n <= 1:
        return {families[0]: 0.0} if families else {}
    spacing = 0.26 if n == 2 else min(0.22, 0.62 / (n - 1))
    start = -spacing * (n - 1) / 2.0
    return {family: start + index * spacing for index, family in enumerate(families)}


def _render_rmse_summary(
    output: Path,
    summary_rows: list[dict[str, Any]],
    *,
    group_key: str = "physical_group",
    group_order: tuple[str, ...] = PUBLICATION_GROUP_ORDER,
    path_stem: str = "publication_rmse_summary",
    output_key: str = "publication_rmse",
    figure_height: float = 4.0,
    families: list[str] | tuple[str, ...] | None = None,
    family_key: str = "engine",
    family_display: dict[str, str] | None = None,
    family_colors: dict[str, str] | None = None,
    members: bool | None = None,
) -> dict[str, Path]:
    """Plot compact energy and force RMSE panels for one or more model families.

    ``families`` are the ordered ``family_key`` values to draw (default: the
    ``MACE``/``DPA2`` engines present). Each family/group draws its committee
    member range (open circles + a connecting line) and the committee-averaged
    (``ensemble_mean``) diamond. ``members=False`` drops the per-member circles
    for a legible many-family plot; the default keeps them for up to four.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "MLIP comparison figures require matplotlib; "
            "install InterfaceForge with interfaceforge[report]"
        ) from exc

    present = list(dict.fromkeys(str(row[family_key]) for row in summary_rows))
    families = list(families) if families is not None else [
        name for name in ENGINE_ORDER if name in present
    ] or present
    missing = [name for name in families if name not in present]
    if missing:
        raise SafetyError(f"RMSE figure: no rows for {family_key} {missing}")
    display = dict(family_display or {})
    colors = dict(family_colors) if family_colors else {
        family: FAMILY_PALETTE[index % len(FAMILY_PALETTE)]
        for index, family in enumerate(families)
    }
    offsets = _family_offsets(families)
    show_members = members if members is not None else len(families) <= 4

    groups = [
        group
        for group in group_order
        if any(row[group_key] == group for row in summary_rows)
    ]
    groups += sorted(
        {str(row[group_key]) for row in summary_rows} - set(groups), key=_natural_key
    )
    if not groups:
        raise SafetyError("No groups available for the RMSE figure")
    group_counts = {
        group: int(
            next(row["systems"] for row in summary_rows if row[group_key] == group)
        )
        for group in groups
    }
    y_positions = np.arange(len(groups), dtype=float)
    metrics = (
        ("(a) Energy", "energy_rmse_mev_per_atom", r"RMSE (meV atom$^{-1}$)"),
        ("(b) Force", "force_rmse_mev_per_angstrom", r"RMSE (meV $\AA^{-1}$)"),
    )
    scale = 1.0 + 0.13 * max(len(families) - 2, 0)
    width = 7.2 if len(families) <= 2 else 8.0
    member_size = 13.0 if len(families) <= 3 else 8.0

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(width, figure_height * scale),
            sharey=True,
            layout="constrained",
        )
        banded = len(families) > 2
        for ax, (title, metric, xlabel) in zip(axes, metrics, strict=True):
            if banded:
                for group_index in range(len(groups)):
                    if group_index % 2 == 0:
                        ax.axhspan(
                            group_index - 0.5,
                            group_index + 0.5,
                            color="#F1F3F5",
                            zorder=0,
                        )
            maximum = 0.0
            for family in families:
                for group_index, group in enumerate(groups):
                    selected = [
                        row
                        for row in summary_rows
                        if str(row[family_key]) == family and row[group_key] == group
                    ]
                    if not selected:
                        continue
                    member_values = [
                        float(row[metric])
                        for row in selected
                        if row["model"] != "ensemble_mean"
                    ]
                    ensemble_values = [
                        float(row[metric])
                        for row in selected
                        if row["model"] == "ensemble_mean"
                    ]
                    if len(ensemble_values) != 1 or not member_values:
                        raise SafetyError(
                            "RMSE figure needs >=1 member and exactly one ensemble "
                            f"mean for {family} / {group}"
                        )
                    y = y_positions[group_index] + offsets[family]
                    maximum = max(maximum, *member_values, ensemble_values[0])
                    if show_members:
                        ax.plot(
                            [min(member_values), max(member_values)],
                            [y, y],
                            color=colors[family],
                            linewidth=1.0,
                            zorder=2,
                        )
                        ax.scatter(
                            member_values,
                            [y] * len(member_values),
                            marker="o",
                            s=member_size,
                            facecolors="white",
                            edgecolors=colors[family],
                            linewidths=0.75,
                            zorder=3,
                        )
                    ax.scatter(
                        [ensemble_values[0]],
                        [y],
                        marker="D",
                        s=22,
                        color=colors[family],
                        edgecolors="white",
                        linewidths=0.45,
                        zorder=4,
                    )
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_xlim(left=0.0, right=maximum * 1.08 if maximum else 1.0)
            ax.set_ylim(len(groups) - 0.5, -0.5)
            ax.grid(axis="x", color="#D1D5DB", linewidth=0.45, alpha=0.75)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if "Overall" in groups and len(groups) > 1:
                ax.axhline(0.5, color="#6B7280", linewidth=0.65)

        axes[0].set_yticks(
            y_positions,
            labels=[f"{group}  ($n$={group_counts[group]})" for group in groups],
        )
        handles = [
            Line2D([0], [0], color=colors[family], linewidth=1.5,
                   label=display.get(family, family))
            for family in families
        ]
        if show_members:
            handles.append(
                Line2D([0], [0], marker="o", color="#4B5563", markerfacecolor="white",
                       linewidth=0, markersize=4.2, label="committee member")
            )
        handles.append(
            Line2D([0], [0], marker="D", color="#4B5563", markerfacecolor="#4B5563",
                   linewidth=0, markersize=4.6, label="committee-averaged prediction")
        )
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncols=min(len(handles), 4),
            frameon=False,
            handlelength=1.5,
            columnspacing=1.2,
        )
        png = output / f"{path_stem}.png"
        svg = output / f"{path_stem}.svg"
        pdf = output / f"{path_stem}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
    return {
        f"{output_key}_png": png,
        f"{output_key}_svg": svg,
        f"{output_key}_pdf": pdf,
    }


def _write_publication_rmse_figure(
    output: Path,
    summary_rows: list[dict[str, Any]],
    *,
    group_key: str = "physical_group",
    group_order: tuple[str, ...] = PUBLICATION_GROUP_ORDER,
    path_stem: str = "publication_rmse_summary",
    output_key: str = "publication_rmse",
    figure_height: float = 4.0,
    deepmd_display: str = "DPA-2",
) -> dict[str, Path]:
    """Engine-family wrapper (MACE / DeePMD arch / NequIP) over ``_render_rmse_summary``."""

    present = {str(row["engine"]) for row in summary_rows}
    families = tuple(engine for engine in ENGINE_ORDER if engine in present)
    return _render_rmse_summary(
        output,
        summary_rows,
        group_key=group_key,
        group_order=group_order,
        path_stem=path_stem,
        output_key=output_key,
        figure_height=figure_height,
        families=families,
        family_key="engine",
        family_display={"MACE": "MACE", "DPA2": deepmd_display, "NEQUIP": "NequIP"},
        family_colors={engine: ENGINE_COLORS[engine] for engine in families},
        members=True,
    )


ENGINE_COLUMN = {"MACE": "mace", "DPA2": "deepmd", "NEQUIP": "nequip"}


def _load_engine_prediction(
    engine: str,
    *,
    output: Path,
    dpa_root: Path | None,
    label: str,
    system: dict[str, Any],
    ref_e: np.ndarray,
    ref_f: np.ndarray,
    ref_delta: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom energies and raw forces of one member on one matched system."""

    natoms = int(system["natoms"])
    if engine == "DPA2":
        assert dpa_root is not None
        prefix = dpa_root / "by_system" / system["system_id"] / f"{label}_detail"
        e_detail = _numeric(Path(str(prefix) + ".e_peratom.out"))
        f_detail = _numeric(Path(str(prefix) + ".f.out"))
        if e_detail.shape != (len(ref_e), 2):
            raise SafetyError(f"Unexpected DeePMD energy detail shape: {e_detail.shape}")
        if f_detail.shape != (len(ref_e) * natoms, 6):
            raise SafetyError(f"Unexpected DeePMD force detail shape: {f_detail.shape}")
        dpa_ref_e, dpa_e = e_detail[:, 0], e_detail[:, 1]
        dpa_ref_f = f_detail[:, :3].reshape(ref_f.shape)
        dpa_f = f_detail[:, 3:].reshape(ref_f.shape)
        ref_delta["energy"] = max(ref_delta["energy"], float(np.max(np.abs(dpa_ref_e - ref_e))))
        ref_delta["force"] = max(ref_delta["force"], float(np.max(np.abs(dpa_ref_f - ref_f))))
        if ref_delta["energy"] > 1.0e-7 or ref_delta["force"] > 1.0e-7:
            raise SafetyError(f"DeePMD detail references differ from canonical labels: {ref_delta}")
        return dpa_e, dpa_f
    path = _prediction_file(output, engine, label, system["system_id"])
    with np.load(path) as prediction:
        energy = np.asarray(prediction["energy"], dtype=float) / natoms
        forces = np.asarray(prediction["forces"], dtype=float)
    if energy.shape != ref_e.shape or forces.shape != ref_f.shape:
        raise SafetyError(f"{engine_display(engine)} prediction shape mismatch: {path}")
    if not np.isfinite(energy).all() or not np.isfinite(forces).all():
        raise SafetyError(f"Non-finite {engine_display(engine)} prediction: {path}")
    return energy, forces


def finalize_comparison(
    campaign_root: str | Path,
    *,
    output_root: str | Path | None = None,
    deepmd_eval_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write matched-frame metrics for every prepared engine once all are complete."""

    campaign = Path(campaign_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve() if output_root else campaign / "audit" / "mlip_compare"
    manifest_path = output / "comparison_manifest.json"
    if not manifest_path.is_file():
        raise SafetyError(f"Run mlip-compare prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arch = str(manifest.get("deepmd_architecture", "dpa2"))
    deepmd_display = DEEPMD_DISPLAY.get(arch, arch.upper())
    status = comparison_status(
        campaign, output_root=output, deepmd_eval_root=deepmd_eval_root
    )
    if status["status"] != "READY_TO_FINALIZE":
        hints = "; ".join(status.get("hints", [])) or "see the counts below"
        raise SafetyError(
            f"Comparison is incomplete ({hints}). MACE {status['mace']}, "
            f"DeePMD {status['deepmd']}, NequIP {status['nequip']} of "
            f"{status['expected_systems_per_model']} systems."
        )
    dpa_root = Path(status["deepmd_eval_root"]) if status.get("deepmd_eval_root") else None
    iread, _ = _ase_io()
    systems = manifest["systems"]
    engine_models = _manifest_engines(manifest)
    engines = [engine for engine in ENGINE_ORDER if engine in engine_models]
    display = {engine: engine_display(engine, arch) for engine in engines}
    data: dict[str, list[list[Any]]] = {engine: [[] for _ in engine_models[engine]] for engine in engines}
    system_rows: list[dict[str, Any]] = []
    ref_delta = {"energy": 0.0, "force": 0.0}
    frame_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []

    for system in systems:
        frames = list(iread(system["mace_input"], index=":"))
        natoms = int(system["natoms"])
        ref_e = np.asarray([float(atoms.info[ENERGY_KEY]) / natoms for atoms in frames])
        ref_f = np.asarray([np.asarray(atoms.arrays[FORCES_KEY]) for atoms in frames])
        frame_ids = system.get("frame_ids") or [
            f'{system["relative_leaf"]}:{int(atoms.info["source_frame"])}' for atoms in frames
        ]
        per_engine: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for engine in engines:
            per_engine[engine] = []
            for model_index, model in enumerate(engine_models[engine]):
                label = model["model"]
                pred_e, pred_f = _load_engine_prediction(
                    engine,
                    output=output,
                    dpa_root=dpa_root,
                    label=label,
                    system=system,
                    ref_e=ref_e,
                    ref_f=ref_f,
                    ref_delta=ref_delta,
                )
                entry = (system, ref_e, pred_e, ref_f, pred_f)
                data[engine][model_index].append(entry)
                per_engine[engine].append((pred_e, pred_f))
                system_rows.append(_system_row(engine, label, model["seed"], system, _metrics(*entry[1:])))
                for frame_index in range(len(frames)):
                    member_rows.append(
                        {
                            "engine": engine,
                            "engine_display": display[engine],
                            "model": label,
                            "seed": model["seed"],
                            "system_id": system["system_id"],
                            "relative_leaf": system["relative_leaf"],
                            "frame_index": frame_index,
                            "frame_id": frame_ids[frame_index],
                            "natoms": natoms,
                            "dft_energy_per_atom_ev": ref_e[frame_index],
                            "energy_per_atom_ev": pred_e[frame_index],
                            "energy_error_mev_per_atom": (pred_e[frame_index] - ref_e[frame_index]) * 1000.0,
                            "force_rmse_mev_per_angstrom": float(
                                np.sqrt(np.mean((pred_f[frame_index] - ref_f[frame_index]) ** 2)) * 1000.0
                            ),
                        }
                    )
        for frame_index in range(len(frames)):
            row: dict[str, Any] = {
                "system_id": system["system_id"],
                "relative_leaf": system["relative_leaf"],
                "frame_index": frame_index,
                "frame_id": frame_ids[frame_index],
                "source_frame": int(frames[frame_index].info["source_frame"]),
                "natoms": natoms,
                "dft_energy_per_atom_ev": ref_e[frame_index],
            }
            for engine in engines:
                column = ENGINE_COLUMN[engine]
                energies = np.asarray([member[0][frame_index] for member in per_engine[engine]])
                forces = np.asarray([member[1][frame_index] for member in per_engine[engine]])
                mean_f = forces.mean(axis=0)
                row[f"{column}_energy_per_atom_ev"] = float(energies.mean())
                row[f"{column}_energy_error_mev_per_atom"] = float((energies.mean() - ref_e[frame_index]) * 1000.0)
                row[f"{column}_energy_spread_mev_per_atom"] = float(energies.std() * 1000.0)
                row[f"{column}_force_rmse_mev_per_angstrom"] = float(
                    np.sqrt(np.mean((mean_f - ref_f[frame_index]) ** 2)) * 1000.0
                )
                row[f"{column}_force_disagreement_mev_per_angstrom"] = float(
                    np.mean(np.linalg.norm(forces.std(axis=0), axis=1)) * 1000.0
                )
            frame_rows.append(row)

    overall_rows, ensemble_rows, uncertainty_rows = [], [], []
    for engine, members in data.items():
        models = engine_models[engine]
        for model, entries in zip(models, members, strict=True):
            overall_rows.extend(_overall(engine, model["model"], model["seed"], entries))
        ensemble_entries = []
        for system_index, system in enumerate(systems):
            template = members[0][system_index]
            mean_e = np.mean([member[system_index][2] for member in members], axis=0)
            mean_f = np.mean([member[system_index][4] for member in members], axis=0)
            entry = (system, template[1], mean_e, template[3], mean_f)
            ensemble_entries.append(entry)
            row = _system_row(
                engine, "ensemble_mean", "committee", system, _metrics(*entry[1:])
            )
            system_rows.append(row)
            ensemble_rows.append(row)
        overall_rows.extend(
            _overall(engine, "ensemble_mean", "committee", ensemble_entries)
        )
        uncertainty_rows.extend(
            [
                _uncertainty(
                    engine,
                    [entry[1] for entry in ensemble_entries],
                    [[entry[2] for entry in member] for member in members],
                    "energy_per_atom",
                ),
                _uncertainty(
                    engine,
                    [entry[3] for entry in ensemble_entries],
                    [[entry[4] for entry in member] for member in members],
                    "force_component",
                ),
            ]
        )

    group_fields = ["heritage", "temperature", "family", "termination", "oxidation"] + [
        field
        for field in METADATA_GROUP_FIELDS
        if any((system.get("metadata") or {}).get(field) not in (None, "NA") for system in systems)
    ]
    group_rows = []
    for engine in engines:
        members = data[engine]
        engine_rows = [row for row in ensemble_rows if row["engine"] == engine]
        for field in group_fields:
            for group_value in sorted({str(row[field]) for row in engine_rows}, key=_natural_key):
                selected = {row["system_id"] for row in engine_rows if str(row[field]) == group_value}
                entries = []
                for index, system in enumerate(systems):
                    if system["system_id"] not in selected:
                        continue
                    ref_e, ref_f = members[0][index][1], members[0][index][3]
                    pred_e = np.mean([member[index][2] for member in members], axis=0)
                    pred_f = np.mean([member[index][4] for member in members], axis=0)
                    entries.append((system, ref_e, pred_e, ref_f, pred_f))
                for row in _overall(engine, "ensemble_mean", "committee", entries):
                    metrics = {
                        key: value
                        for key, value in row.items()
                        if key not in {"engine", "model", "seed", "averaging", "systems"}
                    }
                    group_rows.append(
                        {
                            "engine": engine,
                            "group_field": field,
                            "group_value": group_value,
                            "averaging": row["averaging"],
                            "systems": row["systems"],
                            **metrics,
                        }
                    )

    _write_csv(output / "metrics_by_system.csv", system_rows)
    _write_csv(output / "metrics_overall.csv", overall_rows)
    _write_csv(output / "metrics_by_group.csv", group_rows)
    _write_csv(output / "uncertainty_calibration.csv", uncertainty_rows)
    _write_csv(output / "matched_frames.csv", frame_rows)
    _write_csv(output / "matched_frames_members.csv", member_rows)
    _write_svg(output / "comparison.svg", overall_rows, deepmd_display=deepmd_display)
    heatmaps = _write_force_heatmaps(output, system_rows, deepmd_display=deepmd_display)

    outputs: dict[str, Any] = {}
    views_skipped: dict[str, str] = {}
    views = (
        ("publication", _publication_summary_rows, "physical_group", PUBLICATION_GROUP_ORDER,
         "publication_rmse_summary", "publication_rmse", 4.0, "publication_rmse_by_group.csv"),
        ("temperature", _temperature_summary_rows, "temperature_group", TEMPERATURE_GROUP_ORDER,
         "temperature_rmse_summary", "temperature_rmse", 2.6, "temperature_rmse_by_group.csv"),
        ("oxidation", _oxidation_summary_rows, "oxidation_group", OXIDATION_GROUP_ORDER,
         "oxidation_rmse_summary", "oxidation_rmse", 3.6, "oxidation_rmse_by_group.csv"),
    )
    uninformative_oxidation = all(
        system["heritage"] == "interface" and system["oxidation"] == "0" for system in systems
    )
    for name, builder, group_key, order, stem, key, height, csv_name in views:
        if name == "oxidation" and uninformative_oxidation:
            views_skipped[name] = "no system carries an oxidation (O_x) or bulk coordinate"
            continue
        try:
            summary_rows = builder(system_rows)
        except SafetyError as exc:
            views_skipped[name] = f"chemistry-specific grouping does not apply: {exc}"
            continue
        _write_csv(output / csv_name, summary_rows)
        outputs[f"{name}_by_group"] = str(output / csv_name)
        figures = _write_publication_rmse_figure(
            output,
            summary_rows,
            group_key=group_key,
            group_order=order,
            path_stem=stem,
            output_key=key,
            figure_height=height,
            deepmd_display=deepmd_display,
        )
        outputs.update({figure: str(path) for figure, path in figures.items()})
    for field in ("ligand", "coverage_pct", "stage"):
        if field not in group_fields:
            continue
        summary_rows = _pooled_summary_rows(
            system_rows,
            group_key=f"{field}_group",
            group_order=("Overall",),
            group_for=lambda row, field=field: f"{field}={row.get(field, 'NA')}",
        )
        _write_csv(output / f"{field}_rmse_by_group.csv", summary_rows)
        outputs[f"{field}_by_group"] = str(output / f"{field}_rmse_by_group.csv")

    headline = [
        row
        for row in overall_rows
        if row["model"] == "ensemble_mean" and row["averaging"] == "micro"
    ]
    member_range: dict[str, tuple[float, float]] = {}
    for engine in engines:
        values = [
            float(row["force_rmse_mev_per_angstrom"])
            for row in overall_rows
            if row["engine"] == engine and row["model"] != "ensemble_mean" and row["averaging"] == "micro"
        ]
        member_range[engine] = (min(values), max(values))
    title_names = [display[engine] for engine in engines]
    lines = [
        f"# Matched-frame {' versus '.join(title_names)} audit",
        "",
        "**Scope:** in-distribution interpolation on identical synchronized test frames.",
        "",
        "| Engine | E RMSE (meV/atom) | Centered E RMSE | F RMSE (meV/A) | Relative F RMSE (%) | Member F RMSE range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in headline:
        low, high = member_range[row["engine"]]
        lines.append(
            f'| {display.get(row["engine"], row["engine"])} | '
            f'{row["energy_rmse_mev_per_atom"]:.4f} | '
            f'{row["energy_centered_rmse_mev_per_atom"]:.4f} | '
            f'{row["force_rmse_mev_per_angstrom"]:.4f} | '
            f'{row["force_relative_rmse_percent"]:.3f} | '
            f"{low:.2f}–{high:.2f} |"
        )
    lines.extend(
        [
            "",
            "Micro metrics weight every observation equally; macro metrics weight every trajectory equally.",
            f"Energy error normalization: {ENERGY_NORMALIZATION}",
            "Forces are raw predictions (constraints never applied) against raw DFT forces, all atoms.",
            "Committee spread remains a heuristic until calibrated; see uncertainty_calibration.csv.",
            f"Stress: {STRESS_POLICY}.",
            "Per-frame values for every engine are in matched_frames.csv (ensemble) and "
            "matched_frames_members.csv (every member).",
            "Use an independent trajectory or physical-regime challenge set for transferability claims.",
        ]
    )
    if views_skipped:
        lines.append("")
        lines.extend(f"View `{name}` skipped: {reason}." for name, reason in views_skipped.items())
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "OK",
        "benchmark_scope": "in-distribution interpolation",
        "mace_inference_dtype": "float32",
        "deepmd_architecture": arch,
        "engines": engines,
        "engine_display": display,
        "validation": manifest["validation"],
        "deepmd_reference_max_absolute_delta": ref_delta,
        "energy_normalization": ENERGY_NORMALIZATION,
        "force_convention": "raw predicted vs raw DFT forces; constraints never applied",
        "stress_comparison": STRESS_POLICY,
        "headline": headline,
        "views_skipped": views_skipped,
        "outputs": {
            "by_system": str(output / "metrics_by_system.csv"),
            "overall": str(output / "metrics_overall.csv"),
            "by_group": str(output / "metrics_by_group.csv"),
            "uncertainty": str(output / "uncertainty_calibration.csv"),
            "matched_frames": str(output / "matched_frames.csv"),
            "matched_frames_members": str(output / "matched_frames_members.csv"),
            "markdown": str(output / "comparison.md"),
            "svg": str(output / "comparison.svg"),
            **outputs,
            **{name: str(path) for name, path in heatmaps.items()},
        },
    }
    _write_json(output / "comparison.json", payload)
    return payload


# (csv_name, group_key, group_order, path_stem, output_key, figure_height)
_COMBINE_VIEWS = (
    ("publication_rmse_by_group.csv", "physical_group", PUBLICATION_GROUP_ORDER,
     "publication_rmse_summary", "publication_rmse", 4.0),
    ("temperature_rmse_by_group.csv", "temperature_group", TEMPERATURE_GROUP_ORDER,
     "temperature_rmse_summary", "temperature_rmse", 2.6),
    ("oxidation_rmse_by_group.csv", "oxidation_group", OXIDATION_GROUP_ORDER,
     "oxidation_rmse_summary", "oxidation_rmse", 3.6),
)


def _infer_engine(label: str) -> str:
    """A run label naming a MACE (or NequIP) committee contributes those rows, else DPA2."""

    lowered = label.lower().replace("-", "_")
    if lowered.startswith("mace"):
        return "MACE"
    if lowered.startswith("nequip"):
        return "NEQUIP"
    return "DPA2"


def parse_combine_entry(item: str) -> tuple[str, str, str]:
    """``LABEL=DIR`` or ``LABEL:ENGINE=DIR`` -> (label, engine, directory)."""

    spec, sep, directory = item.partition("=")
    if not sep or not spec.strip() or not directory.strip():
        raise SafetyError(f"--run expects LABEL=DIR (or LABEL:ENGINE=DIR); got {item!r}")
    label, _, engine_raw = spec.partition(":")
    label = label.strip()
    engine = engine_raw.strip().upper().replace("-", "").replace("_", "") or _infer_engine(label)
    engine = {
        "MACE": "MACE",
        "DPA2": "DPA2",
        "DPA": "DPA2",
        "DEEPMD": "DPA2",
        "NEQUIP": "NEQUIP",
    }.get(engine, engine)
    if engine not in {"MACE", "DPA2", "NEQUIP"}:
        raise SafetyError(f"--run engine must be MACE, DPA2 or NEQUIP; got {engine_raw!r}")
    return label, engine, directory.strip()


def _read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def combine_comparisons(
    entries: list[tuple[str, str, str]],
    output_root: str | Path,
    *,
    members: bool | None = None,
) -> dict[str, Any]:
    """Overlay several finalized ``mlip-compare`` runs into one N-family figure set.

    Each entry is ``(label, engine, finalized_output_dir)``: the pooled
    per-group RMSE CSVs written by ``finalize`` are read, the chosen engine's
    rows are relabelled to the family ``label``, and the publication /
    temperature / oxidation summary figures are re-rendered with every family
    on shared axes. No committee is re-evaluated; this is pure post-processing.
    """

    if len(entries) < 2:
        raise SafetyError("combine needs at least two --run entries")
    labels = [label for label, _, _ in entries]
    if len(set(labels)) != len(labels):
        raise SafetyError(f"Duplicate family label in --run entries: {labels}")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    resolved = [(label, engine, Path(d).expanduser().resolve()) for label, engine, d in entries]
    figures: dict[str, str] = {}
    view_status: dict[str, str] = {}
    for csv_name, group_key, group_order, stem, output_key, height in _COMBINE_VIEWS:
        combined: list[dict[str, Any]] = []
        skip_reason = ""
        for label, engine, directory in resolved:
            source = directory / csv_name
            if not source.is_file():
                skip_reason = f"{label}: no {csv_name} in {directory}"
                break
            picked = [row for row in _read_dicts(source) if row.get("engine") == engine]
            if not picked:
                skip_reason = f"{label}: {csv_name} has no {engine} rows"
                break
            for row in picked:
                combined.append(
                    {
                        "family": label,
                        "source_engine": engine,
                        "source_dir": str(directory),
                        "model": row["model"],
                        group_key: row[group_key],
                        "systems": int(row["systems"]),
                        "frames": int(row["frames"]),
                        "energy_rmse_mev_per_atom": float(row["energy_rmse_mev_per_atom"]),
                        "force_rmse_mev_per_angstrom": float(row["force_rmse_mev_per_angstrom"]),
                    }
                )
        if skip_reason:
            view_status[output_key] = f"skipped: {skip_reason}"
            continue
        _write_csv(output / csv_name, combined)
        rendered = _render_rmse_summary(
            output,
            combined,
            group_key=group_key,
            group_order=group_order,
            path_stem=stem,
            output_key=output_key,
            figure_height=height,
            families=labels,
            family_key="family",
            members=members,
        )
        figures.update({name: str(path) for name, path in rendered.items()})
        view_status[output_key] = "OK"

    if not figures:
        raise SafetyError(
            "combine produced no figures; every run is missing its "
            "*_rmse_by_group.csv -- run 'iface mlip-compare finalize' there first "
            f"({view_status})"
        )
    payload = {
        "schema_version": 1,
        "status": "OK",
        "output_root": str(output),
        "families": [
            {"label": label, "engine": engine, "source_dir": str(directory)}
            for label, engine, directory in resolved
        ],
        "views": view_status,
        "outputs": figures,
    }
    _write_json(output / "combined_manifest.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("campaign_root", nargs="?", default=".")
        command.add_argument("--output-root")
        if name in {"status", "finalize"}:
            command.add_argument("--deepmd-eval-root")
        if name == "prepare":
            command.add_argument("--mace-models-root")
            command.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
            command.add_argument(
                "--deepmd-arch", default="dpa2", choices=sorted(DEEPMD_DISPLAY)
            )
            command.add_argument("--backends", nargs="+", choices=sorted(BACKEND_ENGINE))
            command.add_argument("--nequip-root")
            command.add_argument("--nequip-seeds", nargs="+", type=int)
            command.add_argument("--profile", help="Scheduler profile YAML for the NequIP launcher")
            command.add_argument("--nequip-profile", default="nequip_gpu")
            command.add_argument("--force", action="store_true")
    combine = commands.add_parser(
        "combine",
        help="overlay several finalized runs into one N-family RMSE figure set",
    )
    combine.add_argument("output_root")
    combine.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL[:ENGINE]=DIR",
        help="a finalized mlip-compare output dir and the family label for it; "
        "ENGINE (MACE|DPA2|NEQUIP) defaults from the LABEL prefix (mace*, nequip*, else DPA2)",
    )
    members = combine.add_mutually_exclusive_group()
    members.add_argument("--members", action="store_true", default=None)
    members.add_argument("--no-members", dest="members", action="store_false")
    args = parser.parse_args(argv)
    if args.command == "combine":
        payload = combine_comparisons(
            [parse_combine_entry(item) for item in args.run],
            args.output_root,
            members=args.members,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "prepare":
        payload = prepare_comparison(
            args.campaign_root,
            output_root=args.output_root,
            mace_models_root=args.mace_models_root,
            seeds=tuple(args.seeds),
            deepmd_arch=args.deepmd_arch,
            force=args.force,
            backends=tuple(args.backends) if args.backends else None,
            nequip_models_root=args.nequip_root,
            nequip_seeds=tuple(args.nequip_seeds) if args.nequip_seeds else None,
            profile_path=args.profile,
            nequip_profile=args.nequip_profile,
        )
    elif args.command == "status":
        payload = comparison_status(
            args.campaign_root,
            output_root=args.output_root,
            deepmd_eval_root=args.deepmd_eval_root,
        )
    else:
        payload = finalize_comparison(
            args.campaign_root,
            output_root=args.output_root,
            deepmd_eval_root=args.deepmd_eval_root,
        )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") not in {"INCOMPLETE", "FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
