# ruff: noqa: E501
"""Matched-frame, cross-backend MACE/DeePMD accuracy audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
        forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
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


def _discover_models(root: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for index, seed in enumerate(seeds):
        directory = root / f"seed_{seed}" / "mace_model"
        matches = sorted(directory.glob("*_stagetwo.model"))
        if len(matches) != 1:
            raise SafetyError(
                f"Expected one stage-two model for seed {seed}; found {len(matches)} in {directory}"
            )
        rows.append({"model": f"model_{index:03d}", "seed": seed, "model_path": str(matches[0].resolve())})
    return rows


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


def prepare_comparison(
    campaign_root: str | Path,
    *,
    output_root: str | Path | None = None,
    mace_models_root: str | Path | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    deepmd_arch: str = "dpa2",
    force: bool = False,
) -> dict[str, Any]:
    if deepmd_arch not in DEEPMD_DISPLAY:
        raise SafetyError(
            f"Unknown DeePMD architecture {deepmd_arch!r}; expected one of "
            f"{sorted(DEEPMD_DISPLAY)}"
        )
    campaign = Path(campaign_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve() if output_root else campaign / "audit" / "mlip_compare"
    _prepare_output(output, campaign, force)
    canonical = campaign / "datasets" / "canonical"
    model_root = (
        Path(mace_models_root).expanduser().resolve()
        if mace_models_root
        else campaign / "models" / "mace_committee_520eV" / "mace_committee"
    )
    models = _discover_models(model_root, seeds)
    systems, validation = validate_membership(
        canonical / "test.extxyz", canonical / "deepmd" / "test", output / "inputs"
    )
    (output / "evaluate_mace.py").write_text(MACE_EVALUATOR, encoding="utf-8")
    (output / "evaluate_mace.py").chmod(0o755)
    _write_json(output / "systems.json", systems)
    with (output / "mace_models.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "seed", "model_path"), delimiter="\t")
        writer.writeheader()
        writer.writerows(models)
    launcher = output / "run_mace_evaluate.slurm"
    launcher.write_text(_slurm(output, len(models)), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "READY",
        "benchmark_scope": "in-distribution interpolation",
        "mace_inference_dtype": "float32",
        "deepmd_architecture": deepmd_arch,
        "campaign_root": str(campaign),
        "output_root": str(output),
        "models": models,
        "systems": systems,
        "validation": validation,
        "launcher": str(launcher),
        "next": f"sbatch {launcher}",
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
    systems, models = manifest["systems"], manifest["models"]
    arch = str(manifest.get("deepmd_architecture", "dpa2"))
    dpa_root = (
        Path(deepmd_eval_root).expanduser().resolve()
        if deepmd_eval_root
        else _latest_deepmd_eval(campaign, arch)
    )
    mace_counts, dpa_counts = {}, {}
    for model in models:
        label = model["model"]
        mace_counts[label] = sum(
            (output / "predictions" / "mace" / label / f'{system["system_id"]}.npz').is_file()
            for system in systems
        )
        dpa_counts[label] = 0
        if dpa_root:
            for system in systems:
                prefix = dpa_root / "by_system" / system["system_id"] / f"{label}_detail"
                if Path(str(prefix) + ".e_peratom.out").is_file() and Path(str(prefix) + ".f.out").is_file():
                    dpa_counts[label] += 1
    expected = len(systems)
    ready = all(value == expected for value in mace_counts.values()) and all(
        value == expected for value in dpa_counts.values()
    )
    return {
        "schema_version": 1,
        "status": "READY_TO_FINALIZE" if ready else "INCOMPLETE",
        "expected_systems_per_model": expected,
        "deepmd_architecture": arch,
        "mace": mace_counts,
        "deepmd": dpa_counts,
        "deepmd_eval_root": str(dpa_root) if dpa_root else None,
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
    engines = [name for name in ("MACE", "DPA2") if name in rows]
    labels = {"MACE": "MACE", "DPA2": deepmd_display}
    colors = {"MACE": "#2563eb", "DPA2": "#dc2626"}
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
    """Plot member-by-system force RMSE using one scale for both engines."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "MLIP comparison heatmaps require matplotlib; "
            "install InterfaceForge with interfaceforge[report]"
        ) from exc

    engines = ("MACE", "DPA2")
    rows = [row for row in system_rows if row["model"] != "ensemble_mean"]
    model_names = sorted({str(row["model"]) for row in rows})
    system_ids = sorted(
        {str(row["system_id"]) for row in rows},
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    lookup = {
        (str(row["engine"]), str(row["model"]), str(row["system_id"])): row
        for row in rows
    }
    template = {
        str(row["system_id"]): row
        for row in rows
        if row["engine"] == engines[0] and row["model"] == model_names[0]
    }
    expected = len(engines) * len(model_names) * len(system_ids)
    if len(lookup) != expected or len(template) != len(system_ids):
        raise SafetyError(
            "Cannot plot force RMSE heatmaps from an incomplete system/member matrix"
        )

    matrices: dict[str, np.ndarray] = {}
    for engine in engines:
        matrices[engine] = np.asarray(
            [
                [
                    float(
                        lookup[(engine, model, system_id)][
                            "force_rmse_mev_per_angstrom"
                        ]
                    )
                    / 1000.0
                    for model in model_names
                ]
                for system_id in system_ids
            ],
            dtype=float,
        )
    labels = [_heatmap_label(template[system_id]) for system_id in system_ids]
    shared_min = min(float(np.min(matrix)) for matrix in matrices.values())
    shared_max = max(float(np.max(matrix)) for matrix in matrices.values())
    if not math.isfinite(shared_min) or not math.isfinite(shared_max):
        raise SafetyError("Non-finite force RMSE cannot be plotted")
    if shared_max <= shared_min:
        shared_max = shared_min + 1.0e-12

    def render(path_stem: str, selected: tuple[str, ...]) -> tuple[Path, Path]:
        height = max(10.0, 0.31 * len(system_ids) + 2.0)
        width = 12.0 if len(selected) == 1 else 18.0
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
            image = ax.imshow(
                matrix,
                aspect="auto",
                cmap="viridis",
                vmin=shared_min,
                vmax=shared_max,
            )
            ax.set_title(deepmd_display if engine == "DPA2" else engine, fontsize=14)
            ax.set_xticks(range(len(model_names)), labels=model_names, rotation=28, ha="right")
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
            for column_index in range(len(model_names) + 1):
                ax.axvline(column_index - 0.5, color="white", linewidth=0.25, alpha=0.45)

        assert image is not None
        colorbar = fig.colorbar(image, ax=list(axes[0]), shrink=0.78, pad=0.02)
        colorbar.set_label("Force RMSE (eV/Å)")
        title = "Per-system force RMSE"
        if len(selected) == 2:
            title += f" — matched MACE and {deepmd_display} committees (shared scale)"
        fig.suptitle(title, fontsize=16)
        png = output / f"{path_stem}.png"
        svg = output / f"{path_stem}.svg"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        return png, svg

    mace_png, mace_svg = render("force_rmse_heatmap_mace", ("MACE",))
    dpa_png, dpa_svg = render("force_rmse_heatmap_dpa2", ("DPA2",))
    combined_png, combined_svg = render(
        "force_rmse_heatmaps", ("MACE", "DPA2")
    )
    return {
        "force_heatmap_mace_png": mace_png,
        "force_heatmap_mace_svg": mace_svg,
        "force_heatmap_dpa2_png": dpa_png,
        "force_heatmap_dpa2_svg": dpa_svg,
        "force_heatmaps_png": combined_png,
        "force_heatmaps_svg": combined_svg,
    }


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

TEMPERATURE_GROUP_ORDER = ("Overall", "300 K", "450 K")

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
    ]
    rows: list[dict[str, Any]] = []
    engines = [
        engine
        for engine in ("MACE", "DPA2")
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
    """Plot compact, publication-oriented energy and force RMSE panels."""

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

    engines = ("MACE", "DPA2")
    colors = {"MACE": "#0072B2", "DPA2": "#D55E00"}
    offsets = {"MACE": -0.13, "DPA2": 0.13}
    groups = [
        group
        for group in group_order
        if any(row[group_key] == group for row in summary_rows)
    ]
    if not groups:
        raise SafetyError("No groups available for publication figure")
    group_counts = {
        group: int(
            next(
                row["systems"]
                for row in summary_rows
                if row[group_key] == group
            )
        )
        for group in groups
    }
    y_positions = np.arange(len(groups), dtype=float)
    metrics = (
        ("(a) Energy", "energy_rmse_mev_per_atom", r"RMSE (meV atom$^{-1}$)"),
        ("(b) Force", "force_rmse_mev_per_angstrom", r"RMSE (meV $\AA^{-1}$)"),
    )

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
            figsize=(7.2, figure_height),
            sharey=True,
            layout="constrained",
        )
        for ax, (title, metric, xlabel) in zip(axes, metrics, strict=True):
            maximum = 0.0
            for engine in engines:
                for group_index, group in enumerate(groups):
                    selected = [
                        row
                        for row in summary_rows
                        if row["engine"] == engine
                        and row[group_key] == group
                    ]
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
                    if len(member_values) != 4 or len(ensemble_values) != 1:
                        raise SafetyError(
                            "Publication figure requires four members and one "
                            f"ensemble mean for {engine} / {group}"
                        )
                    y = y_positions[group_index] + offsets[engine]
                    maximum = max(maximum, *member_values, ensemble_values[0])
                    ax.plot(
                        [min(member_values), max(member_values)],
                        [y, y],
                        color=colors[engine],
                        linewidth=1.0,
                        zorder=2,
                    )
                    ax.scatter(
                        member_values,
                        [y] * len(member_values),
                        marker="o",
                        s=13,
                        facecolors="white",
                        edgecolors=colors[engine],
                        linewidths=0.75,
                        zorder=3,
                    )
                    ax.scatter(
                        [ensemble_values[0]],
                        [y],
                        marker="D",
                        s=22,
                        color=colors[engine],
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
            Line2D([0], [0], color=colors["MACE"], linewidth=1.5, label="MACE"),
            Line2D([0], [0], color=colors["DPA2"], linewidth=1.5, label=deepmd_display),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#4B5563",
                markerfacecolor="white",
                linewidth=0,
                markersize=4.2,
                label="committee member",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#4B5563",
                markerfacecolor="#4B5563",
                linewidth=0,
                markersize=4.6,
                label="committee-averaged prediction",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncols=4,
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


def finalize_comparison(
    campaign_root: str | Path,
    *,
    output_root: str | Path | None = None,
    deepmd_eval_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write matched MACE/DPA-2 metrics only after both committees are complete."""

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
        raise SafetyError(f"Comparison is incomplete: {status}")
    dpa_root = Path(status["deepmd_eval_root"])
    iread, _ = _ase_io()
    systems, models = manifest["systems"], manifest["models"]
    data: dict[str, list[list[Any]]] = {
        "MACE": [[] for _ in models],
        "DPA2": [[] for _ in models],
    }
    system_rows: list[dict[str, Any]] = []
    ref_delta = {"energy": 0.0, "force": 0.0}

    for system in systems:
        frames = list(iread(system["mace_input"], index=":"))
        natoms = int(system["natoms"])
        ref_e = np.asarray([float(atoms.info[ENERGY_KEY]) / natoms for atoms in frames])
        ref_f = np.asarray([np.asarray(atoms.arrays[FORCES_KEY]) for atoms in frames])
        for model_index, model in enumerate(models):
            label = model["model"]
            mace_file = output / "predictions" / "mace" / label / f'{system["system_id"]}.npz'
            with np.load(mace_file) as prediction:
                mace_e = np.asarray(prediction["energy"], dtype=float) / natoms
                mace_f = np.asarray(prediction["forces"], dtype=float)
            if mace_e.shape != ref_e.shape or mace_f.shape != ref_f.shape:
                raise SafetyError(f"MACE prediction shape mismatch: {mace_file}")
            entry = (system, ref_e, mace_e, ref_f, mace_f)
            data["MACE"][model_index].append(entry)
            system_rows.append(
                _system_row("MACE", label, model["seed"], system, _metrics(*entry[1:]))
            )

            prefix = dpa_root / "by_system" / system["system_id"] / f"{label}_detail"
            e_detail = _numeric(Path(str(prefix) + ".e_peratom.out"))
            f_detail = _numeric(Path(str(prefix) + ".f.out"))
            if e_detail.shape != (len(frames), 2):
                raise SafetyError(f"Unexpected DeePMD energy detail shape: {e_detail.shape}")
            if f_detail.shape != (len(frames) * natoms, 6):
                raise SafetyError(f"Unexpected DeePMD force detail shape: {f_detail.shape}")
            dpa_ref_e, dpa_e = e_detail[:, 0], e_detail[:, 1]
            dpa_ref_f = f_detail[:, :3].reshape(ref_f.shape)
            dpa_f = f_detail[:, 3:].reshape(ref_f.shape)
            ref_delta["energy"] = max(
                ref_delta["energy"], float(np.max(np.abs(dpa_ref_e - ref_e)))
            )
            ref_delta["force"] = max(
                ref_delta["force"], float(np.max(np.abs(dpa_ref_f - ref_f)))
            )
            if ref_delta["energy"] > 1.0e-7 or ref_delta["force"] > 1.0e-7:
                raise SafetyError(
                    f"DeePMD detail references differ from canonical labels: {ref_delta}"
                )
            entry = (system, ref_e, dpa_e, ref_f, dpa_f)
            data["DPA2"][model_index].append(entry)
            system_rows.append(
                _system_row("DPA2", label, model["seed"], system, _metrics(*entry[1:]))
            )

    overall_rows, ensemble_rows, uncertainty_rows = [], [], []
    for engine, members in data.items():
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

    group_rows = []
    for engine in ("MACE", "DPA2"):
        members = data[engine]
        engine_rows = [row for row in ensemble_rows if row["engine"] == engine]
        for field in ("heritage", "temperature", "family", "termination", "oxidation"):
            for group_value in sorted({row[field] for row in engine_rows}):
                selected = {row["system_id"] for row in engine_rows if row[field] == group_value}
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
    _write_svg(output / "comparison.svg", overall_rows, deepmd_display=deepmd_display)
    heatmaps = _write_force_heatmaps(output, system_rows, deepmd_display=deepmd_display)
    publication_rows = _publication_summary_rows(system_rows)
    _write_csv(output / "publication_rmse_by_group.csv", publication_rows)
    publication_figures = _write_publication_rmse_figure(
        output, publication_rows, deepmd_display=deepmd_display
    )
    temperature_rows = _temperature_summary_rows(system_rows)
    _write_csv(output / "temperature_rmse_by_group.csv", temperature_rows)
    temperature_figures = _write_publication_rmse_figure(
        output,
        temperature_rows,
        group_key="temperature_group",
        group_order=TEMPERATURE_GROUP_ORDER,
        path_stem="temperature_rmse_summary",
        output_key="temperature_rmse",
        figure_height=2.6,
        deepmd_display=deepmd_display,
    )
    oxidation_rows = _oxidation_summary_rows(system_rows)
    _write_csv(output / "oxidation_rmse_by_group.csv", oxidation_rows)
    oxidation_figures = _write_publication_rmse_figure(
        output,
        oxidation_rows,
        group_key="oxidation_group",
        group_order=OXIDATION_GROUP_ORDER,
        path_stem="oxidation_rmse_summary",
        output_key="oxidation_rmse",
        figure_height=3.6,
        deepmd_display=deepmd_display,
    )
    headline = [
        row
        for row in overall_rows
        if row["model"] == "ensemble_mean" and row["averaging"] == "micro"
    ]
    engine_label = {"MACE": "MACE", "DPA2": deepmd_display}
    lines = [
        f"# Matched-frame MACE versus {deepmd_display} audit",
        "",
        "**Scope:** in-distribution interpolation on identical synchronized test frames.",
        "",
        "| Engine | E RMSE (meV/atom) | Centered E RMSE | F RMSE (meV/A) | Relative F RMSE (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f'| {engine_label.get(row["engine"], row["engine"])} | '
            f'{row["energy_rmse_mev_per_atom"]:.4f} | '
            f'{row["energy_centered_rmse_mev_per_atom"]:.4f} | '
            f'{row["force_rmse_mev_per_angstrom"]:.4f} | '
            f'{row["force_relative_rmse_percent"]:.3f} |'
        )
    lines.extend(
        [
            "",
            "Micro metrics weight every observation equally; macro metrics weight every trajectory equally.",
            "Committee spread remains a heuristic until calibrated; see uncertainty_calibration.csv.",
            "Virials are excluded because this MACE committee was not trained on virials.",
            "Use an independent trajectory or physical-regime challenge set for transferability claims.",
        ]
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "OK",
        "benchmark_scope": "in-distribution interpolation",
        "mace_inference_dtype": "float32",
        "deepmd_architecture": arch,
        "validation": manifest["validation"],
        "deepmd_reference_max_absolute_delta": ref_delta,
        "headline": headline,
        "outputs": {
            "by_system": str(output / "metrics_by_system.csv"),
            "overall": str(output / "metrics_overall.csv"),
            "by_group": str(output / "metrics_by_group.csv"),
            "uncertainty": str(output / "uncertainty_calibration.csv"),
            "publication_by_group": str(output / "publication_rmse_by_group.csv"),
            "temperature_by_group": str(output / "temperature_rmse_by_group.csv"),
            "oxidation_by_group": str(output / "oxidation_rmse_by_group.csv"),
            "markdown": str(output / "comparison.md"),
            "svg": str(output / "comparison.svg"),
            **{name: str(path) for name, path in heatmaps.items()},
            **{name: str(path) for name, path in publication_figures.items()},
            **{name: str(path) for name, path in temperature_figures.items()},
            **{name: str(path) for name, path in oxidation_figures.items()},
        },
    }
    _write_json(output / "comparison.json", payload)
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
            command.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_comparison(
            args.campaign_root,
            output_root=args.output_root,
            mace_models_root=args.mace_models_root,
            seeds=tuple(args.seeds),
            deepmd_arch=args.deepmd_arch,
            force=args.force,
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
