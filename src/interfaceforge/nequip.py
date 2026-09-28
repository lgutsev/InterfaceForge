# ruff: noqa: E501
"""NequIP committee training backend (``iface train nequip``).

NequIP (the message-passing E(3)-equivariant GNN, ``nequip.model.NequIPGNNModel``)
and Allegro (the strictly local ``allegro.model.AllegroModel``) share the NequIP
framework -- ``nequip-train`` / ``nequip-package`` / ``nequip-compile`` -- but are
different architectures with different hyperparameters and deployment paths.
This module configures NequIP only; Allegro stays in :mod:`interfaceforge.allegro`.

Generated layout (``<campaign>/models/nequip``)::

    training_manifest.json        provenance: commit, dataset/split hashes, defaults applied
    seed_<seed>/config.yaml       one immutable NequIP config per committee member
    seed_<seed>/config.sha256     guard: the member driver refuses a changed config
    seed_<seed>/train_member.sh   train -> package -> compile, with safe ++ckpt_path resume
    run_preflight.slurm           environment + dataset hash check (no training)
    run_smoke.slurm               tiny training/compile/evaluation of the first seed
    run_committee.slurm           Slurm array, one task per seed
    run_finalize.slurm            re-package/compile from best.ckpt (retry only)
    run_evaluate.slurm            Slurm array: canonical test-set predictions per member
    evaluate_nequip.py            standalone ASE evaluator (needs nequip on the cluster)

At runtime each member gains ``outputs/{best,last}.ckpt``, ``logs/csv/version_*/metrics.csv``,
``status.json``, ``versions.json``, ``final/model.nequip.{zip,pt2}`` and
``evaluation/predictions.npz``. Nothing is ever submitted by generation;
``iface nequip submit`` is dry-run unless ``--execute``.
"""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import Campaign, load_profile
from .errors import ConfigurationError, SafetyError
from .provenance import interfaceforge_commit
from .scheduler import render_job, write_job
from .state import StateStore, sha256_file, utc_now

ARCHITECTURE = "nequip"
MODEL_TARGET = "nequip.model.NequIPGNNModel"
MONITORED_METRIC = "val0_epoch/weighted_sum"
REQUIRED_SETTINGS = ("r_max",)

# Documented starting points, not NiO-tuned values. Every key that falls back to
# one of these is listed in training_manifest.json["defaults_applied"].
DEFAULTS: dict[str, Any] = {
    "num_layers": 4,
    "l_max": 1,
    "parity": True,
    "num_features": 32,
    "radial_mlp_depth": 2,
    "radial_mlp_width": 64,
    "num_bessels": 8,
    "bessel_trainable": False,
    "polynomial_cutoff_p": 6,
    "batch_size": 4,
    "val_batch_size": 8,
    "num_workers": 4,
    "max_epochs": 500,
    "learning_rate": 0.005,
    "ema_decay": 0.999,
    "loss_energy_weight": 1.0,
    "loss_forces_weight": 1.0,
    "per_atom_energy_loss": True,
    "early_stopping_patience": 50,
    "lr_factor": 0.6,
    "lr_patience": 5,
    "lr_threshold": 0.2,
    "lr_min": 1.0e-6,
    "log_every_n_steps": 20,
    "compile_training": False,
    "zbl": False,
    "tf32": False,
    "compile_mode": "aotinductor",
    "compile_target": "ase",
}
KNOWN_SETTINGS = set(DEFAULTS) | {
    "enabled",
    "profile",
    "cpu_profile",
    "device",
    "dataset",
    "train_file",
    "valid_file",
    "test_file",
    "type_names",
    "seeds",
    "committee",
    "max_concurrent",
    "r_max",
    "model_dtype",
    "trainer_precision",
    "max_time",
    "data_seed",
    "extra_model",
    "smoke",
    "output_dir",
}
SMOKE_DEFAULTS = {"max_epochs": 2, "limit_train_batches": 5, "limit_val_batches": 2, "test_frames": 20}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _positive(settings: Mapping[str, Any], key: str, kind: type = int) -> Any:
    try:
        value = kind(settings[key])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"models.nequip.{key} must be a {kind.__name__}") from exc
    if value <= 0:
        raise ConfigurationError(f"models.nequip.{key} must be positive")
    return value


def resolve_settings(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Merge campaign settings over the documented defaults.

    Returns the resolved mapping and the sorted list of keys that came from
    :data:`DEFAULTS` (reported so nobody mistakes a default for a tuned value).
    """

    unknown = sorted(set(raw) - KNOWN_SETTINGS)
    if unknown:
        raise ConfigurationError(f"Unknown models.nequip keys: {', '.join(unknown)}")
    missing = [key for key in REQUIRED_SETTINGS if raw.get(key) is None]
    if missing:
        raise ConfigurationError(
            f"models.nequip must set {', '.join(missing)} explicitly (no silent default for the cutoff)"
        )
    resolved = dict(DEFAULTS)
    resolved.update({key: value for key, value in raw.items() if value is not None})
    defaults_applied = sorted(key for key in DEFAULTS if raw.get(key) is None)
    for key in (
        "num_layers",
        "num_features",
        "radial_mlp_depth",
        "radial_mlp_width",
        "num_bessels",
        "polynomial_cutoff_p",
        "batch_size",
        "val_batch_size",
        "max_epochs",
        "log_every_n_steps",
    ):
        resolved[key] = _positive(resolved, key, int)
    for key in ("r_max", "learning_rate", "loss_energy_weight", "loss_forces_weight"):
        resolved[key] = _positive(resolved, key, float)
    if int(resolved["l_max"]) < 0:
        raise ConfigurationError("models.nequip.l_max cannot be negative")
    resolved["l_max"] = int(resolved["l_max"])
    if int(resolved["num_workers"]) < 0:
        raise ConfigurationError("models.nequip.num_workers cannot be negative")
    resolved["num_workers"] = int(resolved["num_workers"])
    if resolved.get("early_stopping_patience") is not None:
        resolved["early_stopping_patience"] = int(resolved["early_stopping_patience"])
    if not 0.0 < float(resolved["ema_decay"]) < 1.0:
        raise ConfigurationError("models.nequip.ema_decay must be in (0, 1)")
    if resolved["compile_mode"] not in {"aotinductor", "torchscript"}:
        raise ConfigurationError("models.nequip.compile_mode must be aotinductor or torchscript")
    resolved["device"] = str(resolved.get("device", "cuda")).lower()
    resolved["model_dtype"] = str(resolved.get("model_dtype", "float32")).lower()
    if resolved["device"] not in {"cuda", "cpu"}:
        raise ConfigurationError("models.nequip.device must be cuda or cpu")
    if resolved["model_dtype"] not in {"float32", "float64"}:
        raise ConfigurationError("models.nequip.model_dtype must be float32 or float64")
    if resolved["tf32"] and resolved["model_dtype"] != "float32":
        raise ConfigurationError("models.nequip.tf32 only applies to float32 models")
    if resolved["tf32"] and resolved.get("trainer_precision"):
        raise ConfigurationError("NequIP: Lightning trainer_precision and tf32 are mutually exclusive")
    extra = resolved.get("extra_model", {}) or {}
    if not isinstance(extra, Mapping):
        raise ConfigurationError("models.nequip.extra_model must be a mapping")
    protected = {"_target_", "seed", "type_names", "r_max", "model_dtype"}
    clash = sorted(protected & set(extra))
    if clash:
        raise ConfigurationError(f"models.nequip.extra_model cannot override {clash}")
    resolved["extra_model"] = dict(extra)
    smoke = dict(SMOKE_DEFAULTS)
    smoke.update(dict(resolved.get("smoke") or {}))
    resolved["smoke"] = smoke
    return resolved, defaults_applied


def resolve_dataset(campaign: Campaign, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Locate train/valid/test extxyz files, preferring a canonical dataset manifest.

    With ``models.nequip.dataset`` (or a ``manifest.json`` beside the default
    canonical files) the split files must still hash to the manifest, so a model
    can only be trained on the exact split the other backends see.
    """

    from .nio_dataset import SCHEMA, dataset_identity, load_dataset_manifest

    dataset_value = settings.get("dataset")
    identity: dict[str, Any] | None = None
    if dataset_value:
        root, manifest = load_dataset_manifest(_resolve(campaign.root, dataset_value))
        files = {split: root / f"{split}.extxyz" for split in ("train", "valid", "test")}
        identity = dataset_identity(root)
        hashes = manifest.get("file_hashes", {})
        for path in files.values():
            if not path.is_file():
                raise SafetyError(f"Canonical dataset is missing {path.name}: {root}")
            recorded = hashes.get(path.name)
            if recorded and sha256_file(path) != recorded:
                raise SafetyError(f"{path} no longer matches its dataset manifest; re-export or re-verify")
        source = "canonical-manifest"
    else:
        files = {
            "train": _resolve(campaign.root, settings.get("train_file", "datasets/canonical/train.extxyz")),
            "valid": _resolve(campaign.root, settings.get("valid_file", "datasets/canonical/valid.extxyz")),
            "test": _resolve(campaign.root, settings.get("test_file", "datasets/canonical/test.extxyz")),
        }
        for split, path in files.items():
            if not path.is_file():
                raise SafetyError(f"Missing NequIP {split} dataset: {path}")
        manifest_path = files["train"].parent / "manifest.json"
        source = "explicit-files"
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if payload.get("schema") == SCHEMA:
                identity = dataset_identity(manifest_path.parent)
                source = "canonical-manifest (implicit)"
    return {
        "source": source,
        "files": {split: str(path) for split, path in files.items()},
        "sha256": {split: sha256_file(path) for split, path in files.items()},
        "identity": identity,
    }


def resolve_type_names(campaign: Campaign, settings: Mapping[str, Any], dataset: Mapping[str, Any]) -> list[str]:
    raw = settings.get("type_names") or []
    if not raw and dataset.get("identity"):
        raw = dataset["identity"].get("type_map") or []
    if not raw:
        raw = campaign.dataset.get("type_map", [])
    names = [str(item).strip() for item in raw if str(item).strip()]
    if not names:
        raise SafetyError(
            "NequIP needs an explicit type order: set models.nequip.type_names or dataset.type_map, "
            "or point models.nequip.dataset at a canonical dataset manifest"
        )
    if len(set(names)) != len(names):
        raise SafetyError("NequIP type names contain duplicates")
    identity = dataset.get("identity") or {}
    manifest_types = identity.get("type_map")
    if manifest_types and sorted(manifest_types) != sorted(names):
        raise SafetyError(f"models.nequip.type_names {names} disagree with the dataset type_map {manifest_types}")
    return names


def nequip_config(
    settings: Mapping[str, Any],
    *,
    seed: int,
    type_names: Sequence[str],
    files: Mapping[str, str],
    log_dir: Path,
    data_seed: int,
    max_epochs: int | None = None,
    extra_trainer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one NequIP (>=0.7 Hydra/Lightning) training config."""

    model: dict[str, Any] = {
        "_target_": MODEL_TARGET,
        "seed": int(seed),
        "model_dtype": settings["model_dtype"],
        "type_names": "${model_type_names}",
        "r_max": "${cutoff_radius}",
        "num_bessels": settings["num_bessels"],
        "bessel_trainable": bool(settings["bessel_trainable"]),
        "polynomial_cutoff_p": settings["polynomial_cutoff_p"],
        "num_layers": settings["num_layers"],
        "l_max": settings["l_max"],
        "parity": bool(settings["parity"]),
        "num_features": settings["num_features"],
        "radial_mlp_depth": settings["radial_mlp_depth"],
        "radial_mlp_width": settings["radial_mlp_width"],
        "avg_num_neighbors": "${training_data_stats:num_neighbors_mean}",
        "per_type_energy_scales": "${training_data_stats:per_type_forces_rms}",
        "per_type_energy_shifts": "${training_data_stats:per_atom_energy_mean}",
        "per_type_energy_scales_trainable": False,
        "per_type_energy_shifts_trainable": False,
    }
    if settings["compile_training"]:
        model["compile_mode"] = "compile"
    if settings["zbl"]:
        model["pair_potential"] = {
            "_target_": "nequip.nn.pair_potential.ZBL",
            "units": "metal",
            "chemical_species": "${chemical_species}",
        }
    model.update(settings.get("extra_model") or {})

    loader_common = {"_target_": "torch.utils.data.DataLoader", "num_workers": settings["num_workers"]}
    callbacks: list[dict[str, Any]] = [
        {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "monitor": "${monitored_metric}",
            "dirpath": "${hydra:runtime.output_dir}",
            "filename": "best",
            "save_last": True,
        },
        {"_target_": "lightning.pytorch.callbacks.LearningRateMonitor", "logging_interval": "epoch"},
    ]
    if settings.get("early_stopping_patience"):
        callbacks.insert(
            0,
            {
                "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                "monitor": "${monitored_metric}",
                "min_delta": 1.0e-4,
                "patience": int(settings["early_stopping_patience"]),
            },
        )
    if settings["tf32"]:
        callbacks.append({"_target_": "nequip.train.callbacks.TF32Scheduler", "schedule": {0: True}})
    trainer: dict[str, Any] = {
        "_target_": "lightning.Trainer",
        "accelerator": "gpu" if settings["device"] == "cuda" else "cpu",
        "devices": 1,
        "enable_checkpointing": True,
        "max_epochs": int(max_epochs if max_epochs is not None else settings["max_epochs"]),
        "log_every_n_steps": settings["log_every_n_steps"],
        "logger": {
            "_target_": "lightning.pytorch.loggers.CSVLogger",
            "save_dir": str(log_dir),
            "name": "csv",
        },
        "callbacks": callbacks,
    }
    if settings.get("max_time"):
        trainer["max_time"] = str(settings["max_time"])
    if settings.get("trainer_precision"):
        trainer["precision"] = str(settings["trainer_precision"])
    trainer.update(dict(extra_trainer or {}))

    return {
        "run": ["train", "test"],
        "cutoff_radius": float(settings["r_max"]),
        "model_type_names": list(type_names),
        "chemical_species": "${model_type_names}",
        "monitored_metric": MONITORED_METRIC,
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "seed": int(data_seed),
            "train_file_path": files["train"],
            "val_file_path": files["valid"],
            "test_file_path": files["test"],
            "ase_args": {"format": "extxyz"},
            "include_keys": ["REF_energy", "REF_forces"],
            "key_mapping": {"REF_energy": "total_energy", "REF_forces": "forces"},
            "transforms": [
                {
                    "_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                    "model_type_names": "${model_type_names}",
                },
                {"_target_": "nequip.data.transforms.NeighborListTransform", "r_max": "${cutoff_radius}"},
            ],
            "train_dataloader": {**loader_common, "batch_size": settings["batch_size"], "shuffle": True},
            "val_dataloader": {**loader_common, "batch_size": settings["val_batch_size"]},
            "test_dataloader": "${data.val_dataloader}",
            "stats_manager": {
                "_target_": "nequip.data.CommonDataStatisticsManager",
                "dataloader_kwargs": {"batch_size": settings["val_batch_size"]},
                "type_names": "${model_type_names}",
            },
        },
        "trainer": trainer,
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "ema_decay": float(settings["ema_decay"]),
            "loss": {
                "_target_": "nequip.train.EnergyForceLoss",
                "per_atom_energy": bool(settings["per_atom_energy_loss"]),
                "coeffs": {
                    "total_energy": float(settings["loss_energy_weight"]),
                    "forces": float(settings["loss_forces_weight"]),
                },
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceMetrics",
                "coeffs": {"per_atom_energy_mae": 1.0, "forces_mae": 1.0},
            },
            "train_metrics": "${training_module.val_metrics}",
            "test_metrics": "${training_module.val_metrics}",
            "optimizer": {"_target_": "torch.optim.Adam", "lr": float(settings["learning_rate"])},
            "lr_scheduler": {
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                    "factor": float(settings["lr_factor"]),
                    "patience": int(settings["lr_patience"]),
                    "threshold": float(settings["lr_threshold"]),
                    "min_lr": float(settings["lr_min"]),
                },
                "monitor": "${monitored_metric}",
                "interval": "epoch",
                "frequency": 1,
            },
            "model": model,
        },
    }


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    header = (
        "# Generated by InterfaceForge `iface train nequip`. DO NOT EDIT between restarts:\n"
        "# NequIP restores training state from the checkpoint but takes every other\n"
        "# hyperparameter from this file, and train_member.sh refuses a changed file.\n"
    )
    return header + yaml.safe_dump(dict(payload), sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- #
# generated shell
# --------------------------------------------------------------------------- #
_STATUS_FUNCTION = r"""write_status() {
  # state stage exit_code detail
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema_version": 1, "architecture": "nequip", "seed": %s, "state": "%s", "stage": "%s", "exit_code": %s, "job_id": "%s", "updated": "%s", "restarts": %s, "config_sha256": "%s", "detail": "%s"}\n' \
    "$SEED" "$1" "$2" "$3" "$JOB_ID" "$now" "$RESTARTS" "$EXPECTED_CONFIG_SHA" "$4" > "$STATUS.tmp"
  mv -f "$STATUS.tmp" "$STATUS"
}"""

_VERSIONS_SNIPPET = r"""python - "$1" <<'PY'
import json, platform, sys
out = {"python": sys.version.split()[0], "platform": platform.platform()}
for name in ("nequip", "torch", "lightning", "e3nn", "ase", "numpy"):
    try:
        module = __import__(name)
        out[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:  # recorded, not fatal here
        out[name] = f"unavailable: {exc}"
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    pass
with open(sys.argv[1], "w") as handle:
    json.dump(out, handle, indent=2)
PY"""


def _member_driver(
    *,
    member_dir: Path,
    seed: int,
    device: str,
    config_sha: str,
    compile_mode: str,
    compile_target: str,
) -> str:
    compiled_name = "model.nequip.pt2" if compile_mode == "aotinductor" else "model.nequip.pth"
    compile_flags = f"--device {device} --mode {compile_mode}"
    if compile_mode == "aotinductor":
        compile_flags += f" --target {compile_target}"
    q = shlex.quote
    return f"""#!/usr/bin/env bash
# InterfaceForge NequIP committee member driver (generated by `iface train nequip`).
# Usage: train_member.sh [all|train|finalize]   (default: all)
#   all      = train (or resume from outputs/last.ckpt) -> package -> compile
#   finalize = package + compile an existing outputs/best.ckpt only
# A completed member is never retrained in place: move its directory aside and
# regenerate to start over; a changed config.yaml is refused (restart safety).
set -Eeuo pipefail
MODE="${{1:-all}}"
MEMBER_DIR={q(str(member_dir))}
SEED={int(seed)}
DEVICE={q(device)}
CONFIG="$MEMBER_DIR/config.yaml"
EXPECTED_CONFIG_SHA={q(config_sha)}
JOB_ID="${{SLURM_JOB_ID:-local}}${{SLURM_ARRAY_TASK_ID:+_${{SLURM_ARRAY_TASK_ID}}}}"
STATUS="$MEMBER_DIR/status.json"
BEST="$MEMBER_DIR/outputs/best.ckpt"
LAST="$MEMBER_DIR/outputs/last.ckpt"
FINAL="$MEMBER_DIR/final"
RESTARTS="$(cat "$MEMBER_DIR/restarts.count" 2>/dev/null || echo 0)"
STAGE=preflight
{_STATUS_FUNCTION}
trap 'code=$?; write_status failed "$STAGE" "$code" "command failed; see the Slurm .err file"; exit "$code"' ERR

record_versions() {{
{_VERSIONS_SNIPPET}
}}

actual_sha="$(sha256sum "$CONFIG" | cut -d' ' -f1)"
if [[ "$actual_sha" != "$EXPECTED_CONFIG_SHA" ]]; then
  write_status failed config_check 3 "config.yaml changed since generation; regenerate instead of editing"
  echo "ERROR: $CONFIG changed since generation (sha256 $actual_sha != $EXPECTED_CONFIG_SHA)." >&2
  echo "       NequIP restarts must use the original config; regenerate with 'iface train nequip'." >&2
  exit 3
fi
record_versions "$MEMBER_DIR/versions.json"
if [[ "$DEVICE" == "cuda" ]]; then
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 4)" || {{
    write_status failed preflight 4 "DEVICE=cuda but torch sees no GPU"; echo "ERROR: no CUDA device visible" >&2; exit 4; }}
fi

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  if [[ -s "$FINAL/{compiled_name}" && -s "$FINAL/model.nequip.zip" ]]; then
    echo "Member seed $SEED is already complete ($FINAL); nothing to do." >&2
    echo "To retrain from scratch, move $MEMBER_DIR aside and regenerate." >&2
    write_status complete done 0 "already complete; nothing to do"
    exit 0
  fi
  TRAIN_ARGS=(-cp "$MEMBER_DIR" -cn config.yaml "hydra.run.dir=$MEMBER_DIR/outputs")
  if [[ -s "$LAST" ]]; then
    RESTARTS=$((RESTARTS + 1))
    echo "$RESTARTS" > "$MEMBER_DIR/restarts.count"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) job $JOB_ID resumes from $LAST (restart $RESTARTS)" >> "$MEMBER_DIR/restarts.log"
    TRAIN_ARGS+=("++ckpt_path=$LAST")
  fi
  STAGE=train
  write_status running train 0 "nequip-train"
  nequip-train "${{TRAIN_ARGS[@]}}"
fi

if [[ "$MODE" == "all" || "$MODE" == "finalize" ]]; then
  [[ -s "$BEST" ]] || {{ write_status failed finalize 5 "no outputs/best.ckpt"; echo "ERROR: missing $BEST" >&2; exit 5; }}
  mkdir -p "$FINAL"
  STAGE=package
  write_status running package 0 "nequip-package build"
  nequip-package build "$BEST" "$FINAL/model.nequip.zip.tmp.nequip.zip"
  mv -f "$FINAL/model.nequip.zip.tmp.nequip.zip" "$FINAL/model.nequip.zip"
  STAGE=compile
  write_status running compile 0 "nequip-compile"
  nequip-compile "$BEST" "$FINAL/{compiled_name}" {compile_flags}
  (cd "$FINAL" && sha256sum model.nequip.zip {compiled_name} > checksums.sha256)
  sha256sum "$BEST" | cut -d' ' -f1 > "$FINAL/best_ckpt.sha256"
  write_status complete done 0 "trained, packaged and compiled"
fi
echo "NequIP member seed $SEED: $MODE finished"
"""


NEQUIP_EVALUATOR = r'''#!/usr/bin/env python3
"""InterfaceForge NequIP evaluator (generated; runs where nequip is installed).

Predicts total energy and raw forces for every frame of a canonical extxyz file
with a compiled NequIP model and writes the backend-agnostic prediction format
read by ``iface nequip evaluate`` / ``interfaceforge.committee_eval``.
Constraints stored as move_mask are never applied to predicted forces.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import iread


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calculator(model, device):
    try:
        from nequip.integrations.ase import NequIPCalculator
    except ImportError:  # older layout
        from nequip.ase import NequIPCalculator
    return NequIPCalculator.from_compiled_model(str(model), device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--label", default="model")
    parser.add_argument("--seed")
    args = parser.parse_args(argv)
    if not args.model.is_file() or args.model.stat().st_size == 0:
        raise SystemExit(f"Missing compiled NequIP model: {args.model}")
    frames_sha = sha256(args.frames)
    if args.expected_sha256 and frames_sha != args.expected_sha256:
        raise SystemExit(f"{args.frames} sha256 {frames_sha} != expected {args.expected_sha256}")
    calculator = load_calculator(args.model, args.device)
    frame_ids, natoms, energies, forces = [], [], [], []
    for index, atoms in enumerate(iread(str(args.frames), index=":")):
        if args.max_frames is not None and index >= args.max_frames:
            break
        frame_ids.append(str(atoms.info.get("frame_id") or f"{args.frames.name}:{index}"))
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64))
        natoms.append(len(atoms))
    if not frame_ids:
        raise SystemExit(f"No frames evaluated from {args.frames}")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "predictions.npz"
    temporary = args.output / "predictions.tmp.npz"
    np.savez_compressed(
        temporary,
        frame_ids=np.asarray(frame_ids),
        natoms=np.asarray(natoms, dtype=int),
        energy=np.asarray(energies, dtype=np.float64),
        forces=np.concatenate(forces),
    )
    os.replace(temporary, target)
    versions = {}
    for name in ("nequip", "torch", "ase"):
        try:
            versions[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    meta = {
        "schema_version": 1,
        "architecture": "nequip",
        "label": args.label,
        "seed": args.seed,
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "frames": str(args.frames.resolve()),
        "frames_sha256": frames_sha,
        "n_frames": len(frame_ids),
        "device": args.device,
        "versions": versions,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "constraints_applied_to_predictions": False,
    }
    (args.output / "predictions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{args.label}: {len(frame_ids)} frames -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _select_profile(profile: Mapping[str, Any], settings: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    jobs = profile.get("jobs", {})
    device = settings["device"]
    default = "nequip_gpu" if device == "cuda" else "nequip_cpu"
    name = str(settings.get("profile") or (settings.get("cpu_profile") if device == "cpu" else None) or default)
    if name not in jobs:
        raise SafetyError(
            f"Scheduler profile has no job {name!r} for NequIP ({device}); add it to "
            f"{profile.get('_path', 'the profile')} (see profiles/loni.yaml: nequip_gpu / nequip_cpu)"
        )
    job = dict(jobs[name])
    gpus = int(job.get("gpus", 0) or 0)
    scheduler = str(profile.get("scheduler", "")).lower()
    if scheduler == "slurm" and device == "cuda" and gpus < 1:
        raise SafetyError(f"models.nequip.device is cuda but profile job {name!r} requests no GPU")
    if scheduler == "slurm" and device == "cpu" and gpus > 0:
        raise SafetyError(f"models.nequip.device is cpu but profile job {name!r} requests {gpus} GPU(s)")
    return name, job


def _member_status_ok_for_regeneration(member_dir: Path, config_sha: str) -> None:
    """Refuse to change the config of a member that already has checkpoints."""

    has_state = any((member_dir / "outputs").glob("*.ckpt")) if (member_dir / "outputs").is_dir() else False
    if not has_state:
        return
    recorded = (
        (member_dir / "config.sha256").read_text(encoding="utf-8").strip()
        if (member_dir / "config.sha256").is_file()
        else None
    )
    if recorded != config_sha:
        raise SafetyError(
            f"{member_dir} already has NequIP checkpoints trained with a different config. "
            "Changing the config under an existing checkpoint breaks restarts; move that member "
            "directory aside (or choose models.nequip.output_dir) before regenerating."
        )


def generate_nequip_training(campaign: Campaign, *, force: bool = False) -> dict[str, Any]:
    """Generate a seeded NequIP committee: configs, drivers, Slurm scripts, manifest."""

    raw = dict(campaign.models.get("nequip", {}))
    if not raw.get("enabled", False):
        raise SafetyError("models.nequip.enabled is false")
    settings, defaults_applied = resolve_settings(raw)
    root = _resolve(campaign.root, settings.get("output_dir", "models/nequip"))
    if root.exists() and any(root.iterdir()) and not force:
        raise SafetyError(f"NequIP output is not empty: {root} (use --force to regenerate scripts)")
    root.mkdir(parents=True, exist_ok=True)
    dataset = resolve_dataset(campaign, settings)
    type_names = resolve_type_names(campaign, settings, dataset)
    seeds = [int(seed) for seed in settings["seeds"]][: int(settings.get("committee", len(settings["seeds"])))]
    # Explicit data_seed: one shuffle order for every member. Default: each member
    # shuffles with its own seed (deterministic, and a second source of diversity).
    explicit_data_seed = settings.get("data_seed")
    profile = load_profile(campaign.profile_path)
    profile_name, profile_job = _select_profile(profile, settings)
    scheduler = str(profile.get("scheduler", "")).lower()

    members: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        member_dir = root / f"seed_{seed}"
        config = nequip_config(
            settings,
            seed=seed,
            type_names=type_names,
            files=dataset["files"],
            log_dir=member_dir / "logs",
            data_seed=int(explicit_data_seed) if explicit_data_seed is not None else seed,
        )
        text = _dump_yaml(config)
        member_dir.mkdir(parents=True, exist_ok=True)
        tmp = member_dir / "config.yaml.tmp"
        tmp.write_text(text, encoding="utf-8")
        config_sha = sha256_file(tmp)
        _member_status_ok_for_regeneration(member_dir, config_sha)
        tmp.replace(member_dir / "config.yaml")
        (member_dir / "config.sha256").write_text(config_sha + "\n", encoding="utf-8")
        driver = member_dir / "train_member.sh"
        driver.write_text(
            _member_driver(
                member_dir=member_dir,
                seed=seed,
                device=settings["device"],
                config_sha=config_sha,
                compile_mode=settings["compile_mode"],
                compile_target=settings["compile_target"],
            ),
            encoding="utf-8",
        )
        driver.chmod(0o750)
        member = {
            "index": index,
            "label": f"model_{index:03d}",
            "seed": seed,
            "data_seed": int(explicit_data_seed) if explicit_data_seed is not None else seed,
            "directory": str(member_dir),
            "config": str(member_dir / "config.yaml"),
            "config_sha256": config_sha,
            "driver": str(driver),
        }
        (member_dir / "member.json").write_text(json.dumps(member, indent=2) + "\n", encoding="utf-8")
        members.append(member)

    # Smoke config: first seed, tiny budget, its own output directory per job.
    smoke = settings["smoke"]
    smoke_config = nequip_config(
        settings,
        seed=seeds[0],
        type_names=type_names,
        files=dataset["files"],
        log_dir=root / "smoke" / "logs",
        data_seed=int(explicit_data_seed) if explicit_data_seed is not None else seeds[0],
        max_epochs=int(smoke["max_epochs"]),
        extra_trainer={
            "limit_train_batches": smoke["limit_train_batches"],
            "limit_val_batches": smoke["limit_val_batches"],
            "limit_test_batches": smoke["limit_val_batches"],
        },
    )
    smoke_config["trainer"]["callbacks"] = [
        callback for callback in smoke_config["trainer"]["callbacks"] if "EarlyStopping" not in callback["_target_"]
    ]
    (root / "smoke").mkdir(exist_ok=True)
    (root / "smoke" / "config.yaml").write_text(_dump_yaml(smoke_config), encoding="utf-8")

    evaluator = root / "evaluate_nequip.py"
    evaluator.write_text(NEQUIP_EVALUATOR, encoding="utf-8")
    evaluator.chmod(0o750)

    compiled_name = "model.nequip.pt2" if settings["compile_mode"] == "aotinductor" else "model.nequip.pth"
    seeds_array = " ".join(str(seed) for seed in seeds)
    q = shlex.quote
    device = settings["device"]
    test_file = dataset["files"]["test"]
    test_sha = dataset["sha256"]["test"]
    dataset_checks = "\n".join(
        f'echo "{sha}  {path}" | sha256sum -c - || {{ echo "ERROR: dataset file changed: {path}" >&2; exit 3; }}'
        for path, sha in ((dataset["files"][split], dataset["sha256"][split]) for split in ("train", "valid", "test"))
    )
    gpu_probe = "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" if device == "cuda" else "true"
    cuda_assert = (
        "python -c \"import torch, sys; print('CUDA available:', torch.cuda.is_available()); "
        'sys.exit(0 if torch.cuda.is_available() else 4)"'
        if device == "cuda"
        else "true"
    )
    preflight_command = "\n".join(
        [
            f"ROOT={q(str(root))}",
            'PRE_DIR="$ROOT/preflight/job_${SLURM_JOB_ID:-local_$(date +%s)}"',
            'mkdir -p "$PRE_DIR"',
            gpu_probe,
            "command -v nequip-train >/dev/null || { echo 'ERROR: nequip-train not on PATH (activate the NequIP env)' >&2; exit 2; }",
            "command -v nequip-compile >/dev/null || { echo 'ERROR: nequip-compile not on PATH' >&2; exit 2; }",
            "command -v nequip-package >/dev/null || { echo 'ERROR: nequip-package not on PATH' >&2; exit 2; }",
            cuda_assert,
            "python - \"$PRE_DIR/versions.json\" <<'PY'",
            *_VERSIONS_SNIPPET.split("\n")[1:-1],
            "PY",
            'cat "$PRE_DIR/versions.json"',
            dataset_checks,
            "nequip-train --help >/dev/null",
            'echo "NequIP preflight passed (versions in $PRE_DIR/versions.json)."',
        ]
    )
    smoke_command = "\n".join(
        [
            f"ROOT={q(str(root))}",
            'SMOKE_DIR="$ROOT/smoke/job_${SLURM_JOB_ID:-local_$(date +%s)}"',
            '[[ ! -e "$SMOKE_DIR" ]] || { echo "ERROR: refusing to overwrite $SMOKE_DIR" >&2; exit 2; }',
            'mkdir -p "$SMOKE_DIR"',
            'cp "$ROOT/smoke/config.yaml" "$SMOKE_DIR/config.yaml"',
            gpu_probe,
            'nequip-train -cp "$SMOKE_DIR" -cn config.yaml "hydra.run.dir=$SMOKE_DIR/outputs" '
            '"trainer.logger.save_dir=$SMOKE_DIR/logs"',
            '[[ -s "$SMOKE_DIR/outputs/best.ckpt" ]] || { echo "ERROR: smoke produced no best.ckpt" >&2; exit 3; }',
            f'nequip-compile "$SMOKE_DIR/outputs/best.ckpt" "$SMOKE_DIR/{compiled_name}" --device {device} '
            f"--mode {settings['compile_mode']}"
            + (f" --target {settings['compile_target']}" if settings["compile_mode"] == "aotinductor" else ""),
            f'python "$ROOT/evaluate_nequip.py" --model "$SMOKE_DIR/{compiled_name}" --frames {q(test_file)} '
            f"--expected-sha256 {test_sha} --max-frames {int(smoke['test_frames'])} --device {device} "
            f'--label smoke --seed {seeds[0]} --output "$SMOKE_DIR/evaluation"',
            'echo "NequIP smoke test passed: $SMOKE_DIR (NOT a trained model)"',
        ]
    )
    array_prelude = [
        f"ROOT={q(str(root))}",
        f"SEEDS=({seeds_array})",
        'TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch (array job)}"',
        'SEED="${SEEDS[$TASK_ID]:?no seed for array index $TASK_ID}"',
        'MEMBER_DIR="$ROOT/seed_${SEED}"',
    ]
    committee_command = "\n".join([*array_prelude, gpu_probe, 'bash "$MEMBER_DIR/train_member.sh" all'])
    finalize_command = "\n".join([*array_prelude, gpu_probe, 'bash "$MEMBER_DIR/train_member.sh" finalize'])
    evaluate_command = "\n".join(
        [
            *array_prelude,
            f'MODEL="$MEMBER_DIR/final/{compiled_name}"',
            '[[ -s "$MODEL" ]] || { echo "ERROR: missing compiled model $MODEL (train/finalize first)" >&2; exit 2; }',
            f'python "$ROOT/evaluate_nequip.py" --model "$MODEL" --frames {q(test_file)} '
            f"--expected-sha256 {test_sha} --device {device} "
            '--label "$(printf \'model_%03d\' "$TASK_ID")" --seed "$SEED" --output "$MEMBER_DIR/evaluation"',
            'echo "NequIP evaluation written for seed $SEED; summarize with: iface nequip evaluate"',
        ]
    )
    launchers: dict[str, str] = {}
    array_spec = f"0-{len(seeds) - 1}%{int(settings.get('max_concurrent', 2))}"
    single_jobs = {"preflight": preflight_command, "smoke": smoke_command}
    array_jobs = {"committee": committee_command, "finalize": finalize_command, "evaluate": evaluate_command}
    for name, command in single_jobs.items():
        path = root / f"run_{name}.slurm"
        write_job(
            path,
            render_job(
                profile,
                profile_name,
                command=command,
                job_name=f"{campaign.name}_nequip_{name}",
                working_directory=str(root),
            ),
            force=True,
        )
        launchers[name] = str(path)
    for name, command in array_jobs.items():
        path = root / f"run_{name}.slurm"
        if scheduler == "slurm":
            content = render_job(
                profile,
                profile_name,
                command=command,
                job_name=f"{campaign.name}_nequip_{name}",
                array=array_spec,
                working_directory=str(root),
            )
        else:
            # A local profile has no array index: run members sequentially.
            body = command.replace(
                'TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch (array job)}"', 'for TASK_ID in "${!SEEDS[@]}"; do'
            )
            body += "\ndone"
            content = render_job(
                profile,
                profile_name,
                command=body,
                job_name=f"{campaign.name}_nequip_{name}",
                working_directory=str(root),
            )
        write_job(path, content, force=True)
        launchers[name] = str(path)

    commit = interfaceforge_commit()
    manifest = {
        "schema_version": 1,
        "engine": ARCHITECTURE,
        "architecture": ARCHITECTURE,
        "model_target": MODEL_TARGET,
        "not_allegro": "NequIP message-passing GNN; Allegro is configured separately under models.allegro",
        "campaign": campaign.name,
        "created_at": utc_now(),
        "interfaceforge_version": __version__,
        "interfaceforge_commit": commit,
        "root": str(root),
        "dataset": dataset,
        "type_names": type_names,
        "seeds": seeds,
        "data_seed": int(explicit_data_seed) if explicit_data_seed is not None else "per-member (= member seed)",
        "device": settings["device"],
        "model_dtype": settings["model_dtype"],
        "trainer_precision": settings.get("trainer_precision"),
        "tf32": bool(settings["tf32"]),
        "hyperparameters": {key: settings[key] for key in sorted(DEFAULTS) if key in settings}
        | {"r_max": settings["r_max"]},
        "extra_model": settings.get("extra_model", {}),
        "defaults_applied": defaults_applied,
        "defaults_note": (
            "Keys in defaults_applied use InterfaceForge starting-point values from the NequIP "
            "tutorial/documentation, not values tuned or validated for this chemistry."
        ),
        "runtime": {
            "profile": profile_name,
            "scheduler": scheduler,
            "modules": [str(value) for value in profile_job.get("modules", [])],
            "preamble": [str(value) for value in profile_job.get("preamble", [])],
            "gpus": int(profile_job.get("gpus", 0) or 0),
            "max_concurrent": int(settings.get("max_concurrent", 2)),
        },
        "compile": {
            "mode": settings["compile_mode"],
            "target": settings["compile_target"],
            "compiled_name": compiled_name,
        },
        "members": members,
        "launchers": launchers,
        "execution_order": ["run_preflight.slurm", "run_smoke.slurm", "run_committee.slurm", "run_evaluate.slurm"],
        "restart_policy": (
            "Resubmitting run_committee.slurm resumes each unfinished member from outputs/last.ckpt "
            "via ++ckpt_path; completed members exit immediately; a changed config.yaml is refused."
        ),
        "evaluation": {
            "evaluator": str(evaluator),
            "test_file": test_file,
            "test_sha256": test_sha,
            "summarize": "iface nequip evaluate",
        },
        "verification_status": "automated-test-only: no NequIP model has been trained or validated by InterfaceForge yet",
    }
    manifest_path = root / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    StateStore(campaign.root).artifact("nequip_training_manifest", manifest_path)
    return manifest


# --------------------------------------------------------------------------- #
# discovery, progress and evaluation (read-only)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def parse_lightning_metrics(log_root: Path) -> dict[str, Any]:
    """Latest epoch/step and validation metrics from Lightning CSVLogger output.

    Reads every ``csv/version_*/metrics.csv`` (a restart opens a new version)
    and reports the furthest epoch reached. Column names follow NequIP's
    ``val0_epoch/<metric>`` convention; anything else is ignored rather than
    guessed.
    """

    result: dict[str, Any] = {"epoch": None, "step": None, "val": {}, "test": {}, "files": 0}
    files = sorted(log_root.glob("csv/version_*/metrics.csv")) if log_root.is_dir() else []
    for path in files:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            continue
        result["files"] += 1
        for row in rows:
            epoch = _float(row.get("epoch"))
            step = _float(row.get("step"))
            if epoch is not None and (result["epoch"] is None or epoch >= result["epoch"]):
                result["epoch"] = int(epoch)
            if step is not None and (result["step"] is None or step >= result["step"]):
                result["step"] = int(step)
            for key, value in row.items():
                if not key:
                    continue
                number = _float(value)
                if number is None:
                    continue
                if key.startswith("val") and "_epoch/" in key:
                    result["val"][key.split("/", 1)[1]] = number
                elif key.startswith("test") and "/" in key:
                    result["test"][key.split("/", 1)[1]] = number
    return result


def _age_hours(path: Path) -> float | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (datetime.now(timezone.utc).timestamp() - mtime) / 3600.0


def member_state(member_dir: Path, *, compiled_name: str | None = None, stale_hours: float = 6.0) -> dict[str, Any]:
    """Read-only state of one NequIP member directory."""

    status = _read_json(member_dir / "status.json") or {}
    outputs = member_dir / "outputs"
    final = member_dir / "final"
    compiled = None
    for name in ([compiled_name] if compiled_name else []) + ["model.nequip.pt2", "model.nequip.pth"]:
        if name and (final / name).is_file() and (final / name).stat().st_size > 0:
            compiled = final / name
            break
    package = final / "model.nequip.zip"
    metrics = parse_lightning_metrics(member_dir / "logs")
    evaluation = member_dir / "evaluation" / "predictions.npz"
    best = outputs / "best.ckpt"
    last = outputs / "last.ckpt"
    recorded_state = str(status.get("state", "")) or None
    config_ok = None
    if (member_dir / "config.yaml").is_file() and (member_dir / "config.sha256").is_file():
        config_ok = (
            sha256_file(member_dir / "config.yaml")
            == (member_dir / "config.sha256").read_text(encoding="utf-8").strip()
        )

    if compiled is not None and package.is_file() and recorded_state in {None, "complete"}:
        state = "complete"
    elif recorded_state == "failed":
        state = "failed"
    elif recorded_state == "running":
        age = _age_hours(member_dir / "status.json")
        log_ages = [
            age
            for age in (_age_hours(path) for path in (member_dir / "logs").glob("csv/version_*/metrics.csv"))
            if age is not None
        ]
        freshest = min([value for value in [age, *log_ages] if value is not None], default=None)
        state = "stalled?" if freshest is not None and freshest > stale_hours else "running"
    elif best.is_file() or last.is_file():
        state = "incomplete"
    elif (member_dir / "config.yaml").is_file():
        state = "not-started"
    else:
        state = "missing"
    return {
        "seed": status.get("seed") if status.get("seed") is not None else _seed_from_name(member_dir.name),
        "member": member_dir.name,
        "state": state,
        "stage": status.get("stage"),
        "detail": status.get("detail"),
        "job_id": status.get("job_id"),
        "restarts": status.get("restarts"),
        "exit_code": status.get("exit_code"),
        "epoch": metrics["epoch"],
        "step": metrics["step"],
        "val_metrics": metrics["val"],
        "test_metrics": metrics["test"],
        "best_checkpoint": best.is_file() and best.stat().st_size > 0,
        "last_checkpoint": last.is_file() and last.stat().st_size > 0,
        "package": package.is_file() and package.stat().st_size > 0,
        "compiled_model": str(compiled) if compiled else None,
        "evaluation": evaluation.is_file(),
        "config_unchanged": config_ok,
        "updated": status.get("updated"),
    }


def _seed_from_name(name: str) -> int | None:
    suffix = name.removeprefix("seed_")
    return int(suffix) if suffix.isdigit() else None


def discover_members(root: str | Path) -> dict[str, Any]:
    """All ``seed_*`` members below a NequIP root plus the training manifest, if any."""

    base = Path(root).expanduser().resolve()
    manifest = _read_json(base / "training_manifest.json") or {}
    compiled_name = (manifest.get("compile") or {}).get("compiled_name")
    member_dirs = sorted(
        (path for path in base.glob("seed_*") if path.is_dir()),
        key=lambda path: (_seed_from_name(path.name) is None, _seed_from_name(path.name) or 0, path.name),
    )
    expected_seeds = [int(seed) for seed in manifest.get("seeds", [])]
    members = [member_state(path, compiled_name=compiled_name) for path in member_dirs]
    present = {member["seed"] for member in members}
    missing = [seed for seed in expected_seeds if seed not in present]
    max_epochs = (manifest.get("hyperparameters") or {}).get("max_epochs")
    return {
        "root": str(base),
        "architecture": ARCHITECTURE,
        "manifest": bool(manifest),
        "expected_seeds": expected_seeds,
        "missing_seeds": missing,
        "max_epochs": max_epochs,
        "members": members,
        "complete": bool(members) and not missing and all(member["state"] == "complete" for member in members),
        "evaluated": bool(members) and all(member["evaluation"] for member in members),
    }


def evaluate_nequip_committee(
    root: str | Path, *, output: str | Path | None = None, reference: str | Path | None = None
) -> dict[str, Any]:
    """Summarize member predictions on the canonical test split (local, no nequip needed)."""

    from .committee_eval import evaluate_committee

    base = Path(root).expanduser().resolve()
    manifest = _read_json(base / "training_manifest.json")
    if not manifest:
        raise SafetyError(f"No NequIP training_manifest.json in {base}")
    test_file = Path(reference) if reference else Path(manifest["evaluation"]["test_file"])
    expected = manifest["evaluation"].get("test_sha256")
    if reference is None and expected and sha256_file(test_file) != expected:
        raise SafetyError(f"{test_file} changed since training generation; evaluation would not be matched")
    members = []
    missing = []
    for member in manifest.get("members", []):
        predictions = Path(member["directory"]) / "evaluation" / "predictions.npz"
        if not predictions.is_file():
            missing.append(member["seed"])
            continue
        members.append({"label": member["label"], "seed": member["seed"], "predictions": predictions})
    if missing:
        raise SafetyError(f"NequIP evaluation incomplete: no predictions for seed(s) {missing}; run run_evaluate.slurm")
    target = Path(output).expanduser().resolve() if output else base / "evaluation"
    summary = evaluate_committee(test_file, members, target, backend="NequIP")
    summary["dataset_identity"] = (manifest.get("dataset") or {}).get("identity")
    (target / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# guarded submission
# --------------------------------------------------------------------------- #
SUBMIT_STAGES = ("preflight", "smoke", "committee", "finalize", "evaluate")


def submit_nequip(root: str | Path, stage: str, *, execute: bool = False, sbatch: str = "sbatch") -> dict[str, Any]:
    """Dry-run by default. With ``execute``, ``sbatch`` one generated launcher and log the job."""

    base = Path(root).expanduser().resolve()
    if stage not in SUBMIT_STAGES:
        raise ConfigurationError(f"stage must be one of {SUBMIT_STAGES}")
    manifest = _read_json(base / "training_manifest.json")
    if not manifest:
        raise SafetyError(f"No NequIP training_manifest.json in {base}; run 'iface train nequip' first")
    launcher = Path(manifest["launchers"][stage])
    if not launcher.is_file():
        raise SafetyError(f"Missing launcher {launcher}")
    if manifest.get("runtime", {}).get("scheduler") != "slurm":
        raise SafetyError("Guarded submission requires a Slurm profile; run the local script directly instead")
    for member in manifest.get("members", []):
        config = Path(member["config"])
        if not config.is_file() or sha256_file(config) != member["config_sha256"]:
            raise SafetyError(f"{config} differs from the generated config; regenerate before submitting")
    if stage in {"evaluate"}:
        state = discover_members(base)
        unfinished = [member["seed"] for member in state["members"] if not member["compiled_model"]]
        if unfinished:
            raise SafetyError(f"Cannot evaluate: seed(s) {unfinished} have no compiled model yet")
    command = [sbatch, "--chdir", str(base), str(launcher)]
    payload: dict[str, Any] = {
        "stage": stage,
        "launcher": str(launcher),
        "launcher_sha256": sha256_file(launcher),
        "command": " ".join(shlex.quote(part) for part in command),
        "executed": False,
    }
    if not execute:
        payload["note"] = "dry run: review the launcher, then re-run with --execute to submit"
        return payload
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    payload.update(
        {
            "executed": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    )
    log = base / "submissions.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**payload, "submitted_at": utc_now()}) + "\n")
    if result.returncode != 0:
        raise SafetyError(f"sbatch failed ({result.returncode}): {result.stderr.strip()}")
    return payload


def _metric_text(values: Mapping[str, float]) -> str:
    for key in ("forces_mae", "forces_rmse", "weighted_sum"):
        if key in values:
            scale = 1000.0 if key.startswith("forces") else 1.0
            unit = " meV/A" if key.startswith("forces") else ""
            return f"val {key}={values[key] * scale:.1f}{unit}"
    return "val –"


def render_status(payload: Mapping[str, Any]) -> str:
    """Compact text table for ``iface nequip status`` / ``iface mlip-progress``."""

    flag = "OK" if payload.get("complete") else ".."
    lines = [f"  [{flag}] nequip  {payload['root']}  target {payload.get('max_epochs') or '?'} epochs"]
    if payload.get("missing_seeds"):
        lines.append(f"      missing member directories for seeds {payload['missing_seeds']}")
    if not payload.get("members"):
        lines.append("      (no seed_* members yet)")
    for member in payload.get("members", []):
        flags = (
            ("C" if member["last_checkpoint"] or member["best_checkpoint"] else "-")
            + ("P" if member["package"] else "-")
            + ("M" if member["compiled_model"] else "-")
            + ("E" if member["evaluation"] else "-")
        )
        epoch = member["epoch"] if member["epoch"] is not None else "-"
        lines.append(
            f"      seed {str(member['seed']):<6} {member['state']:<11} epoch {epoch!s:>5}  "
            f"{_metric_text(member['val_metrics']):<28} [{flags}]  {member.get('updated') or ''}"
        )
    lines.append("      flags: C=checkpoint P=package M=compiled model E=test-set predictions")
    return "\n".join(lines)
