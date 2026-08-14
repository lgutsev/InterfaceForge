"""Optional Allegro training and pair_allegro LAMMPS deployment support."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .config import Campaign, load_campaign, load_profile
from .errors import InterfaceForgeError, SafetyError
from .scheduler import render_job, write_job
from .state import StateStore

MIN_LAMMPS_DATE = date(2025, 9, 10)
DEFAULT_LAMMPS_REF = "patch_10Sep2025"
PAIR_ALLEGRO_COMMIT = "2e19360b2639d960fb59223c5260eb87d0fbf273"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _prepare_root(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise SafetyError(f"Allegro output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o750)


def _type_names(campaign: Campaign, settings: Mapping[str, Any]) -> list[str]:
    raw = settings.get("type_names") or campaign.dataset.get("type_map", [])
    if not isinstance(raw, list):
        raise SafetyError("models.allegro.type_names must be a list")
    names = [str(item).strip() for item in raw if str(item).strip()]
    if not names:
        raise SafetyError(
            "Allegro needs an explicit atom-type order. Set dataset.type_map or "
            "models.allegro.type_names; InterfaceForge will not guess LAMMPS type mapping."
        )
    if len(set(names)) != len(names):
        raise SafetyError("Allegro type names contain duplicates")
    return names


def _positive_int(settings: Mapping[str, Any], key: str, default: int) -> int:
    value = int(settings.get(key, default))
    if value < 1:
        raise SafetyError(f"models.allegro.{key} must be positive")
    return value


def _positive_float(settings: Mapping[str, Any], key: str, default: float) -> float:
    value = float(settings.get(key, default))
    if value <= 0:
        raise SafetyError(f"models.allegro.{key} must be positive")
    return value


def _training_config(
    campaign: Campaign,
    settings: Mapping[str, Any],
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    type_names: list[str],
) -> dict[str, Any]:
    r_max = _positive_float(settings, "r_max", 5.0)
    batch_size = _positive_int(settings, "batch_size", 4)
    num_workers = int(settings.get("num_workers", 4))
    if num_workers < 0:
        raise SafetyError("models.allegro.num_workers cannot be negative")
    max_epochs = _positive_int(settings, "max_epochs", 200)
    scalar = _positive_int(settings, "num_scalar_features", 64)
    tensor = _positive_int(settings, "num_tensor_features", 32)
    layers = _positive_int(settings, "num_layers", 2)
    l_max = int(settings.get("l_max", 1))
    if l_max < 0:
        raise SafetyError("models.allegro.l_max cannot be negative")
    seed = int(settings.get("seed", 2026))

    loader = {
        "_target_": "torch.utils.data.DataLoader",
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    model: dict[str, Any] = {
        "_target_": "allegro.model.AllegroModel",
        "seed": seed,
        "model_dtype": str(settings.get("model_dtype", "float32")),
        "type_names": "${model_type_names}",
        "r_max": "${cutoff_radius}",
        "radial_chemical_embed": {
            "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
            "num_bessels": _positive_int(settings, "num_bessels", 8),
            "bessel_trainable": bool(settings.get("bessel_trainable", False)),
            "polynomial_cutoff_p": _positive_int(settings, "polynomial_cutoff_p", 6),
        },
        "radial_chemical_embed_dim": scalar,
        "scalar_embed_mlp_hidden_layers_depth": 1,
        "scalar_embed_mlp_hidden_layers_width": scalar,
        "scalar_embed_mlp_nonlinearity": "silu",
        "l_max": l_max,
        "num_layers": layers,
        "num_scalar_features": scalar,
        "num_tensor_features": tensor,
        "allegro_mlp_hidden_layers_depth": 1,
        "allegro_mlp_hidden_layers_width": scalar,
        "allegro_mlp_nonlinearity": "silu",
        "parity": bool(settings.get("parity", True)),
        "tp_path_channel_coupling": bool(settings.get("tp_path_channel_coupling", True)),
        "readout_mlp_hidden_layers_depth": 1,
        "readout_mlp_hidden_layers_width": scalar,
        "readout_mlp_nonlinearity": "silu",
        "avg_num_neighbors": "${training_data_stats:per_type_num_neighbors_mean}",
        "per_type_energy_shifts": "${training_data_stats:per_atom_energy_mean}",
        "per_type_energy_scales": "${training_data_stats:forces_rms}",
        "per_type_energy_scales_trainable": False,
        "per_type_energy_shifts_trainable": False,
    }
    if bool(settings.get("compile_training", True)):
        model["compile_mode"] = "compile"

    return {
        "run": ["train", "test"],
        "cutoff_radius": r_max,
        "model_type_names": type_names,
        "chemical_species": "${model_type_names}",
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "seed": seed,
            "train_file_path": str(train_file),
            "val_file_path": str(valid_file),
            "test_file_path": str(test_file),
            "ase_args": {"format": "extxyz"},
            "include_keys": ["REF_energy", "REF_forces"],
            "key_mapping": {"REF_energy": "total_energy", "REF_forces": "forces"},
            "transforms": [
                {
                    "_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                    "model_type_names": "${model_type_names}",
                },
                {
                    "_target_": "nequip.data.transforms.NeighborListTransform",
                    "r_max": "${cutoff_radius}",
                },
            ],
            "train_dataloader": loader,
            "val_dataloader": loader,
            "test_dataloader": "${data.val_dataloader}",
            "stats_manager": {
                "_target_": "nequip.data.CommonDataStatisticsManager",
                "type_names": "${model_type_names}",
            },
        },
        "trainer": {
            "_target_": "lightning.Trainer",
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": max_epochs,
            "log_every_n_steps": _positive_int(settings, "log_every_n_steps", 10),
            "callbacks": [
                {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": "${hydra:runtime.output_dir}",
                    "save_last": True,
                }
            ],
        },
        "num_scalar_features": scalar,
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "loss": {
                "_target_": "nequip.train.EnergyForceLoss",
                "per_atom_energy": True,
                "coeffs": {
                    "total_energy": _positive_float(settings, "energy_weight", 1.0),
                    "forces": _positive_float(settings, "forces_weight", 1.0),
                },
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceMetrics",
                "coeffs": {"per_atom_energy_mae": 1.0, "forces_mae": 1.0},
            },
            "test_metrics": "${training_module.val_metrics}",
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": _positive_float(settings, "learning_rate", 1.0e-3),
            },
            "model": model,
        },
    }


def _build_lammps_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail

LAMMPS_REF="${LAMMPS_REF:-patch_10Sep2025}"
PAIR_ALLEGRO_REF="${PAIR_ALLEGRO_REF:-2e19360b2639d960fb59223c5260eb87d0fbf273}"
BUILD_ROOT="${ALLEGRO_LAMMPS_BUILD_ROOT:-$PWD/_allegro_lammps}"
BUILD_JOBS="${BUILD_JOBS:-8}"
KOKKOS_ARCH="${KOKKOS_ARCH:-}"

for cmd in git cmake python; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: missing $cmd" >&2; exit 2; }
done

python - <<'PY'
import torch
parts = torch.__version__.split('+', 1)[0].split('.')
version = (int(parts[0]), int(parts[1]))
print(f"PyTorch={torch.__version__} CUDA={torch.version.cuda}")
print(f"CXX11_ABI={bool(torch._C._GLIBCXX_USE_CXX11_ABI)}")
if version < (2, 6):
    raise SystemExit("ERROR: AOTInductor pair_allegro requires PyTorch >= 2.6")
if not bool(torch._C._GLIBCXX_USE_CXX11_ABI):
    raise SystemExit(
        "ERROR: PyTorch uses the old C++ ABI. Use an ABI11 PyTorch/libtorch build "
        "for the Kokkos+AOTI pair_allegro path."
    )
PY

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"
rm -rf lammps pair_nequip_allegro

git clone --depth=1 --branch "$LAMMPS_REF" https://github.com/lammps/lammps.git lammps
git clone --depth=1 https://github.com/mir-group/pair_nequip_allegro.git pair_nequip_allegro
git -C pair_nequip_allegro fetch --depth=1 origin "$PAIR_ALLEGRO_REF"
git -C pair_nequip_allegro checkout --detach FETCH_HEAD
./pair_nequip_allegro/patch_lammps.sh "$BUILD_ROOT/lammps"

TORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
CMAKE_ARGS=(
  -S "$BUILD_ROOT/lammps/cmake"
  -B "$BUILD_ROOT/lammps/build"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_MPI=ON
  -DPKG_KOKKOS=ON
  -DKokkos_ENABLE_CUDA=ON
  -DNEQUIP_AOT_COMPILE=ON
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
  -DMKL_INCLUDE_DIR="${MKL_INCLUDE_DIR:-/tmp}"
)
if [[ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ]]; then
  CMAKE_ARGS+=("-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR")
fi
if [[ -n "$KOKKOS_ARCH" ]]; then
  CMAKE_ARGS+=("-DKokkos_ARCH_${KOKKOS_ARCH}=ON")
fi
if [[ -n "${CMAKE_CXX_COMPILER:-}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER")
elif [[ -x "$BUILD_ROOT/lammps/lib/kokkos/bin/nvcc_wrapper" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$BUILD_ROOT/lammps/lib/kokkos/bin/nvcc_wrapper")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_ROOT/lammps/build" --parallel "$BUILD_JOBS"
LMP="$BUILD_ROOT/lammps/build/lmp"
[[ -x "$LMP" ]] || { echo "ERROR: LAMMPS executable was not built" >&2; exit 2; }
HELP="$($LMP -h 2>&1 || true)"
grep -qi 'KOKKOS' <<<"$HELP" || { echo "ERROR: KOKKOS not visible in lmp -h" >&2; exit 2; }
grep -qi 'allegro' <<<"$HELP" || { echo "ERROR: pair_style allegro not visible in lmp -h" >&2; exit 2; }
echo "pair_allegro LAMMPS built successfully: $LMP"
echo "Next: iface-allegro lammps-preflight --lammps $LMP"
'''


def _select_profile(profile: Mapping[str, Any], preferred: str, fallback: str) -> str:
    jobs = profile.get("jobs", {})
    if preferred in jobs:
        return preferred
    if fallback in jobs:
        return fallback
    raise SafetyError(f"Profile has neither {preferred!r} nor fallback {fallback!r}")


def generate_allegro_training(campaign: Campaign, *, force: bool = False) -> dict[str, Any]:
    """Generate Allegro training, AOTI compilation, and pair_allegro assets."""

    settings = dict(campaign.models.get("allegro", {}))
    if not settings.get("enabled", False):
        raise SafetyError("models.allegro.enabled is false")
    root = campaign.root / "models" / "allegro"
    _prepare_root(root, force=force)

    names = _type_names(campaign, settings)
    train_file = _resolve(campaign.root, settings.get("train_file", "datasets/canonical/train.extxyz"))
    valid_file = _resolve(campaign.root, settings.get("valid_file", "datasets/canonical/valid.extxyz"))
    test_file = _resolve(campaign.root, settings.get("test_file", "datasets/canonical/test.extxyz"))
    for split, path in (("train", train_file), ("valid", valid_file), ("test", test_file)):
        if not path.is_file():
            raise SafetyError(f"Missing Allegro {split} dataset: {path}")

    config_path = root / "config.yaml"
    config = _training_config(campaign, settings, train_file, valid_file, test_file, names)
    _write(config_path, yaml.safe_dump(config, sort_keys=False))

    profile = load_profile(campaign.profile_path)
    train_profile = _select_profile(profile, str(settings.get("profile", "allegro_gpu")), "mace_gpu")
    lammps_profile = _select_profile(
        profile, str(settings.get("lammps_profile", "allegro_lammps")), train_profile
    )

    train_command = f"nequip-train -cp {shlex.quote(str(root))} -cn {shlex.quote(config_path.name)}"
    train_launcher = root / "run_train.slurm"
    write_job(
        train_launcher,
        render_job(
            profile,
            train_profile,
            command=train_command,
            job_name=f"{campaign.name}_allegro_train",
            working_directory=str(root),
        ),
        force=force,
    )

    compiled = root / "compiled" / "model.nequip.pt2"
    compile_device = str(settings.get("compile_device", "cuda")).lower()
    if compile_device not in {"cpu", "cuda"}:
        raise SafetyError("models.allegro.compile_device must be cpu or cuda")
    compile_lines = [
        ': "${ALLEGRO_CHECKPOINT:?Set ALLEGRO_CHECKPOINT to the trained .ckpt file}"',
        f"mkdir -p {shlex.quote(str(compiled.parent))}",
        "nequip-compile \"$ALLEGRO_CHECKPOINT\" "
        f"{shlex.quote(str(compiled))} --device {compile_device} "
        "--mode aotinductor --target pair_allegro",
    ]
    compile_launcher = root / "run_compile.slurm"
    write_job(
        compile_launcher,
        render_job(
            profile,
            train_profile,
            command="\n".join(compile_lines),
            job_name=f"{campaign.name}_allegro_compile",
            working_directory=str(root),
        ),
        force=force,
    )

    build_script = root / "lammps" / "build_pair_allegro.sh"
    _write(build_script, _build_lammps_script(), executable=True)

    scheduler = str(profile.get("scheduler", "")).lower()
    job = dict(profile.get("jobs", {}).get(lammps_profile, {}))
    gpus = int(job.get("gpus", 0) or 0)
    command = [
        ': "${ALLEGRO_LAMMPS:?Set ALLEGRO_LAMMPS to pair_allegro-enabled lmp}"',
        ': "${ALLEGRO_MODEL:?Set ALLEGRO_MODEL to compiled .nequip.pt2 model}"',
        ': "${LAMMPS_INPUT:?Set LAMMPS_INPUT to the LAMMPS input script}"',
        'iface-allegro lammps-preflight --lammps "$ALLEGRO_LAMMPS" --model "$ALLEGRO_MODEL"',
    ]
    if scheduler == "slurm" and gpus > 0:
        command.append(
            'srun -n {ntasks} "$ALLEGRO_LAMMPS" -sf kk -k on g {gpus} '
            '-pk kokkos newton on neigh half -in "$LAMMPS_INPUT"'
        )
    elif scheduler == "slurm":
        command.append('srun -n {ntasks} "$ALLEGRO_LAMMPS" -in "$LAMMPS_INPUT"')
    else:
        command.append('"$ALLEGRO_LAMMPS" -in "$LAMMPS_INPUT"')
    lammps_launcher = root / "lammps" / "run_lammps.slurm"
    write_job(
        lammps_launcher,
        render_job(
            profile,
            lammps_profile,
            command="\n".join(command),
            job_name=f"{campaign.name}_allegro_md",
            working_directory=str(root),
        ),
        force=force,
    )

    manifest = {
        "schema_version": 1,
        "engine": "allegro",
        "campaign": campaign.name,
        "type_names": names,
        "config": str(config_path),
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "test_file": str(test_file),
        "training_profile": train_profile,
        "lammps_profile": lammps_profile,
        "launchers": {
            "train": str(train_launcher),
            "compile": str(compile_launcher),
            "lammps": str(lammps_launcher),
        },
        "compiled_model": str(compiled),
        "lammps_build_script": str(build_script),
        "lammps_baseline": {
            "minimum_release": "10 Sep 2025",
            "default_ref": DEFAULT_LAMMPS_REF,
            "pair_nequip_allegro_commit": PAIR_ALLEGRO_COMMIT,
            "kokkos": "GPU launcher requires Kokkos in default double-double precision",
            "aoti": "compile on the same GPU type used for inference; PyTorch >= 2.6",
        },
        "safety": {
            "type_mapping": "explicit only; never inferred",
            "deployment_preflight": "mandatory in generated LAMMPS launcher",
            "checkpoint_selection": "explicit ALLEGRO_CHECKPOINT; no newest-file guessing",
        },
    }
    manifest_path = root / "training_manifest.json"
    _write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    StateStore(campaign.root).artifact("allegro_training_manifest", manifest_path)
    return manifest


_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_lammps_date(text: str) -> date | None:
    match = re.search(r"LAMMPS\s*\((\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", text)
    if not match:
        return None
    day, month, year = match.groups()
    if month not in _MONTHS:
        return None
    return date(int(year), _MONTHS[month], int(day))


def _check(name: str, passed: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "required": required, "detail": detail}


def allegro_lammps_preflight(lammps: str = "lmp", model: str | None = None) -> dict[str, Any]:
    """Check observable pair_allegro runtime requirements before launching MD."""

    candidate = Path(lammps).expanduser()
    executable = str(candidate.resolve()) if candidate.exists() else shutil.which(lammps)
    if executable is None:
        return {
            "passed": False,
            "lammps": lammps,
            "checks": [_check("lammps_executable", False, f"Could not find {lammps!r}")],
        }
    try:
        result = subprocess.run(
            [executable, "-h"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "lammps": executable,
            "checks": [_check("lammps_help", False, str(exc))],
        }
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    release = _parse_lammps_date(output)
    checks = [
        _check("lammps_help", bool(output.strip()), "lmp -h produced output"),
        _check(
            "lammps_release",
            release is not None and release >= MIN_LAMMPS_DATE,
            f"detected {release}; need >= {MIN_LAMMPS_DATE}" if release else "release date not parsed",
        ),
        _check("kokkos_package", "KOKKOS" in output.upper(), "KOKKOS must appear in lmp -h"),
        _check("pair_allegro", "allegro" in output.lower(), "allegro must appear in lmp -h"),
    ]
    if model is not None:
        path = Path(model).expanduser().resolve()
        checks.extend(
            [
                _check("compiled_model_exists", path.is_file(), str(path)),
                _check(
                    "compiled_model_suffix",
                    path.name.endswith((".nequip.pt2", ".nequip.pth")),
                    "expected .nequip.pt2 or .nequip.pth",
                ),
            ]
        )
    checks.append(
        _check(
            "kokkos_precision",
            True,
            "lmp -h cannot prove double-double Kokkos precision; verify build configuration",
            required=False,
        )
    )
    return {
        "passed": all(item["passed"] for item in checks if item["required"]),
        "lammps": executable,
        "release": release.isoformat() if release else None,
        "checks": checks,
    }


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iface-allegro")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate Allegro training/deployment assets")
    generate.add_argument("-c", "--campaign", default="campaign.yaml")
    generate.add_argument("--force", action="store_true")
    preflight = commands.add_parser("lammps-preflight", help="Check pair_allegro LAMMPS runtime")
    preflight.add_argument("--lammps", default="lmp")
    preflight.add_argument("--model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            _json(generate_allegro_training(load_campaign(args.campaign), force=args.force))
            return 0
        payload = allegro_lammps_preflight(args.lammps, model=args.model)
        _json(payload)
        return 0 if payload["passed"] else 2
    except (InterfaceForgeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
