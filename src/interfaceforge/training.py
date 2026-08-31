"""Generate restartable MACE and DeePMD training campaigns."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import Campaign, load_profile
from .errors import SafetyError
from .scheduler import render_job, write_job
from .state import StateStore, sha256_file


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _prepare_root(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise SafetyError(f"Training output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mace_roi_dataset(
    campaign: Campaign, settings: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    roi = dict(settings.get("roi", {}))
    dataset_root = _resolve(campaign.root, roi.get("output_dir", "datasets/mace_roi"))
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise SafetyError(
            f"MACE-ROI dataset is not prepared: {manifest_path}. "
            "Run 'iface mace-roi prepare' first."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SafetyError(f"Could not read MACE-ROI manifest {manifest_path}: {exc}") from exc
    if manifest.get("method") != "mace-roi":
        raise SafetyError(f"Not an InterfaceForge MACE-ROI manifest: {manifest_path}")
    recorded_output = Path(str(manifest.get("output_root", ""))).expanduser().resolve()
    if recorded_output != dataset_root:
        raise SafetyError(
            f"MACE-ROI manifest records output {recorded_output}, not {dataset_root}"
        )
    source_root = Path(str(manifest.get("source_root", ""))).expanduser().resolve()
    source_hashes = manifest.get("source_hashes", {})
    expected_hashes = manifest.get("output_hashes", {})
    for split in ("train", "valid", "test"):
        source_path = source_root / f"{split}.extxyz"
        source_hash = source_hashes.get(split)
        if not source_path.is_file() or not source_hash or sha256_file(source_path) != source_hash:
            raise SafetyError(
                f"Canonical source changed after MACE-ROI preparation: {source_path}. "
                "Re-run 'iface mace-roi prepare --force'."
            )
        path = dataset_root / f"{split}.extxyz"
        if not path.is_file():
            raise SafetyError(f"Missing prepared MACE-ROI split: {path}")
        expected = expected_hashes.get(split)
        if not expected or sha256_file(path) != expected:
            raise SafetyError(
                f"Prepared MACE-ROI split does not match its manifest: {path}. "
                "Re-run 'iface mace-roi prepare --force'."
            )
    return dataset_root, manifest


def generate_mace_training(campaign: Campaign, *, force: bool = False) -> dict[str, Any]:
    """Generate two-stage MACE refinement launchers from the shared extxyz data."""

    settings = dict(campaign.models.get("mace", {}))
    if not settings.get("enabled", False):
        raise SafetyError("models.mace.enabled is false")
    root = campaign.root / "models" / "mace"
    _prepare_root(root, force=force)
    profile = load_profile(campaign.profile_path)
    profile_name = str(settings.get("profile", "mace_gpu"))
    roi = dict(settings.get("roi", {}))
    roi_enabled = bool(roi.get("enabled", False))
    roi_manifest: dict[str, Any] | None = None
    if roi_enabled:
        dataset_root, roi_manifest = _mace_roi_dataset(campaign, settings)
        train_file = dataset_root / "train.extxyz"
        valid_file = dataset_root / "valid.extxyz"
        test_file = dataset_root / "test.extxyz"
    else:
        train_file = _resolve(
            campaign.root, settings.get("train_file", "datasets/canonical/train.extxyz")
        )
        valid_file = _resolve(
            campaign.root, settings.get("valid_file", "datasets/canonical/valid.extxyz")
        )
        test_file = _resolve(
            campaign.root, settings.get("test_file", "datasets/canonical/test.extxyz")
        )
    e0s = settings.get("e0s", "average")
    common = [
        "iface-mace-roi" if roi_enabled else "mace_run_train",
        "--name=interfaceforge_mace",
        f"--train_file={shlex.quote(str(train_file))}",
        f"--valid_file={shlex.quote(str(valid_file))}",
        f"--test_file={shlex.quote(str(test_file))}",
        f"--model_dir={shlex.quote(str(root / 'artifacts'))}",
        "--device=cuda",
        f"--batch_size={int(settings.get('batch_size', 16))}",
        f"--num_workers={int(settings.get('num_workers', 8))}",
        f"--E0s={shlex.quote(str(e0s))}",
        f"--r_max={float(settings.get('r_max', 6.0))}",
        f"--energy_key={settings.get('energy_key', 'REF_energy')}",
        f"--forces_key={settings.get('forces_key', 'REF_forces')}",
        f"--seed={int(settings.get('seed', 2026))}",
        "--restart_latest",
    ]
    if roi_enabled:
        common.extend(
            [
                "--loss=weighted",
                "--if-roi-weight-key=IF_roi_weight",
                "--if-cycle-id-key=IF_cycle_id",
                "--if-cycle-coefficient-key=IF_cycle_coefficient",
                "--if-cycle-scale-key=IF_cycle_scale_ev",
                "--if-cycle-size-key=IF_cycle_size",
            ]
        )
    stage1_cycle_weight = float(
        roi.get("stage1_cycle_weight", roi.get("cycle_weight", 0.0))
    )
    stage2_cycle_weight = float(
        roi.get("stage2_cycle_weight", roi.get("cycle_weight", 0.0))
    )
    if roi_enabled and max(stage1_cycle_weight, stage2_cycle_weight) > 0:
        cycle_groups = int((roi_manifest or {}).get("cycles", {}).get("groups", 0))
        if cycle_groups < 1:
            raise SafetyError(
                "MACE-ROI cycle weight is positive, but the prepared dataset has no cycles"
            )
    stage1_epochs = int(settings.get("max_num_epochs", 200))
    stage2_epochs = int(settings.get("stage2_max_num_epochs", 100))
    if stage1_epochs < 1 or stage2_epochs < 1:
        raise SafetyError("MACE stage epoch counts must be positive")
    stage1 = [
        *common,
        f"--max_num_epochs={stage1_epochs}",
        f"--energy_weight={float(settings.get('stage1_energy_weight', 1.0))}",
        f"--forces_weight={float(settings.get('stage1_forces_weight', 100.0))}",
        f"--patience={int(settings.get('patience', 30))}",
    ]
    if roi_enabled:
        stage1.append(f"--if-cycle-weight={stage1_cycle_weight}")
    stage2 = [
        *common,
        # MACE interprets max_num_epochs as an absolute stopping epoch when
        # --restart_latest is used. Stage 2 therefore has to stop after the
        # stage-1 budget plus its own additional refinement budget; using the
        # stage-2 value alone can make the restarted training loop run zero
        # epochs (for example, restart at epoch 200 with a limit of 100).
        f"--max_num_epochs={stage1_epochs + stage2_epochs}",
        f"--energy_weight={float(settings.get('stage2_energy_weight', 10.0))}",
        f"--forces_weight={float(settings.get('stage2_forces_weight', 50.0))}",
        f"--patience={int(settings.get('stage2_patience', 20))}",
    ]
    if roi_enabled:
        stage2.append(f"--if-cycle-weight={stage2_cycle_weight}")
    stages: list[dict[str, Any]] = []
    model_dir = root / "artifacts"
    for name, command in (("stage1", stage1), ("stage2", stage2)):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        command_text = " \\\n  ".join(command)
        if name == "stage2":
            # stage2 shares --name/--model_dir with stage1 and continues via
            # --restart_latest. If stage1 never produced a model (submitted
            # out of order, or stage1 failed before writing anything),
            # --restart_latest finds nothing and MACE silently starts a
            # brand-new model under stage2's hyperparameters instead of
            # continuing stage1's pretraining. Fail loudly instead.
            quoted_model_dir = shlex.quote(str(model_dir))
            guard = (
                f"if [[ ! -d {quoted_model_dir} ]] || "
                f'[[ -z "$(ls -A {quoted_model_dir} 2>/dev/null)" ]]; then\n'
                f'  echo "ERROR: stage2 expects an existing stage1 model in '
                f'{model_dir}; run stage1 first." >&2\n'
                "  exit 2\n"
                "fi\n"
            )
            command_text = guard + command_text
        launcher = render_job(
            profile,
            profile_name,
            command=command_text,
            job_name=f"{campaign.name}_mace_{name}",
            working_directory=str(root),
        )
        write_job(directory / "run.slurm", launcher, force=force)
        stages.append(
            {
                "name": name,
                "launcher": str(directory / "run.slurm"),
                "command": command_text,
            }
        )

    manifest = {
        "schema_version": 1,
        "engine": "mace",
        "campaign": campaign.name,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "test_file": str(test_file),
        "force_labels": "raw DFT labels; constraints are not applied",
        "method": "mace-roi" if roi_enabled else "mace",
        "roi": (
            {
                "enabled": True,
                "dataset_manifest": str(
                    _resolve(campaign.root, roi.get("output_dir", "datasets/mace_roi"))
                    / "manifest.json"
                ),
                "interface_multiplier": roi.get("interface_multiplier", 4.0),
                "stage1_cycle_weight": stage1_cycle_weight,
                "stage2_cycle_weight": stage2_cycle_weight,
                "cycle_groups": int((roi_manifest or {}).get("cycles", {}).get("groups", 0)),
            }
            if roi_enabled
            else {"enabled": False}
        ),
        "stages": stages,
        "execution_order": ["stage1", "stage2"],
    }
    manifest_path = root / "training_manifest.json"
    _write_json(manifest_path, manifest)
    StateStore(campaign.root).artifact("mace_training_manifest", manifest_path)
    return manifest


def _find_deepmd_systems(split: Path) -> list[Path]:
    return sorted({path.parent for path in split.rglob("type.raw")})


def _validate_deepmd_set(set_dir: Path, natoms: int) -> None:
    """Check array presence, frame-count consistency, shapes, finiteness,
    and nondegenerate cells for one DeePMD `set.*` directory."""

    arrays: dict[str, np.ndarray] = {}
    for name in ("coord.npy", "box.npy", "energy.npy", "force.npy"):
        path = set_dir / name
        if not path.is_file():
            raise SafetyError(f"Missing {name} in {set_dir}")
        try:
            arrays[name] = np.load(path)
        except (OSError, ValueError) as exc:
            raise SafetyError(f"Could not read {path}: {exc}") from exc
    virial_path = set_dir / "virial.npy"
    if virial_path.is_file():
        try:
            arrays["virial.npy"] = np.load(virial_path)
        except (OSError, ValueError) as exc:
            raise SafetyError(f"Could not read {virial_path}: {exc}") from exc

    frame_counts = {name: (array.shape[0] if array.ndim else 0) for name, array in arrays.items()}
    if len(set(frame_counts.values())) > 1:
        raise SafetyError(f"Inconsistent frame counts in {set_dir}: {frame_counts}")
    nframes = next(iter(frame_counts.values()))
    if nframes == 0:
        raise SafetyError(f"{set_dir} has zero frames")

    expected_columns = {
        "coord.npy": natoms * 3,
        "force.npy": natoms * 3,
        "box.npy": 9,
        "energy.npy": 1,
        "virial.npy": 9,
    }
    for name, expected in expected_columns.items():
        if name not in arrays:
            continue
        actual = arrays[name].reshape(nframes, -1).shape[1]
        if actual != expected:
            raise SafetyError(
                f"{name} in {set_dir} has {actual} columns; expected {expected} for {natoms} atoms"
            )

    for name, array in arrays.items():
        try:
            finite = bool(np.isfinite(array).all())
        except TypeError as exc:
            raise SafetyError(f"Non-numeric values in {set_dir / name}") from exc
        if not finite:
            raise SafetyError(f"Non-finite values in {set_dir / name}")

    determinants = np.linalg.det(arrays["box.npy"].reshape(nframes, 3, 3))
    if np.any(np.abs(determinants) < 1e-6):
        bad_frame = int(np.argmin(np.abs(determinants)))
        raise SafetyError(
            f"Degenerate (near-zero-volume) cell at frame {bad_frame} in {set_dir / 'box.npy'}"
        )


def validate_deepmd_dataset(root: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Validate the fixed-shape DeePMD systems generated by InterfaceForge."""

    type_map: list[str] | None = None
    inventory: dict[str, list[str]] = {}
    for split_name in ("train", "valid", "test"):
        split = root / split_name
        systems = _find_deepmd_systems(split) if split.is_dir() else []
        if not systems:
            raise SafetyError(f"No DeePMD systems found below {split}")
        inventory[split_name] = [str(system.resolve()) for system in systems]
        for system in systems:
            current = (system / "type_map.raw").read_text(encoding="utf-8").split()
            if not current:
                raise SafetyError(f"Empty type_map.raw in {system}")
            if len(set(current)) != len(current):
                raise SafetyError(f"Duplicate entries in type_map.raw in {system}")
            if type_map is None:
                type_map = current
            elif current != type_map:
                raise SafetyError(f"Inconsistent type_map.raw in {system}")
            type_tokens = (system / "type.raw").read_text(encoding="utf-8").split()
            natoms = len(type_tokens)
            if natoms == 0:
                raise SafetyError(f"Empty type.raw in {system}")
            try:
                atom_types = [int(value) for value in type_tokens]
            except ValueError as exc:
                raise SafetyError(f"Non-integer atom type in {system / 'type.raw'}") from exc
            invalid_types = sorted({value for value in atom_types if value < 0 or value >= len(current)})
            if invalid_types:
                raise SafetyError(
                    f"Atom type indices {invalid_types} in {system / 'type.raw'} are outside "
                    f"type_map.raw range 0..{len(current) - 1}"
                )
            set_dirs = sorted(path for path in system.glob("set.*") if path.is_dir())
            if not set_dirs:
                raise SafetyError(f"No set.* data directories in {system}")
            for set_dir in set_dirs:
                _validate_deepmd_set(set_dir, natoms)
    return type_map or [], inventory


def _deepmd_descriptor(name: str, backend: str, seed: int) -> dict[str, Any]:
    # dpa2_ft is a fine-tuning run of the dpa2 architecture: identical model
    # structure, only the training command differs.
    if name == "dpa2_ft":
        name = "dpa2"
    if name == "dpa1":
        descriptor_type = "dpa1" if backend == "pt_expt" else "se_atten"
        return {
            "type": descriptor_type,
            "sel": "auto:1.20",
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "resnet_dt": False,
            "attn": 128,
            "attn_layer": 2,
            "attn_mask": False,
            "attn_dotr": True,
            "activation_function": "tanh",
            "precision": "float32",
            "seed": seed,
        }
    if name == "dpa2":
        return {
            "type": "dpa2",
            "repinit": {
                "tebd_dim": 8,
                "rcut": 6.0,
                "rcut_smth": 0.5,
                "nsel": "auto:1.20",
                "neuron": [25, 50, 100],
                "axis_neuron": 12,
            },
            "repformer": {
                "rcut": 4.0,
                "rcut_smth": 3.5,
                "nsel": "auto:1.20",
                "nlayers": 3,
                "g1_dim": 128,
                "g2_dim": 32,
                "axis_neuron": 4,
                "attn1_hidden": 128,
                "attn1_nhead": 4,
                "attn2_hidden": 32,
                "attn2_nhead": 4,
            },
            "precision": "float32",
            "seed": seed,
        }
    if name == "dpa3":
        return {
            "type": "dpa3",
            "repflow": {
                "n_dim": 128,
                "e_dim": 64,
                "a_dim": 32,
                "nlayers": 6,
                "e_rcut": 6.0,
                "e_rcut_smth": 5.3,
                "e_sel": "auto:1.20",
                "a_rcut": 4.0,
                "a_rcut_smth": 3.5,
                "a_sel": "auto:1.20",
                "axis_neuron": 4,
                "fix_stat_std": 0.3,
                "update_angle": True,
            },
            "activation_function": "silu",
            "precision": "float32",
            "concat_output_tebd": False,
            "seed": seed,
        }
    if name == "dpa4":
        return {
            "type": "dpa4",
            "sel": "auto:1.20",
            "rcut": 6.0,
            "channels": 64,
            "n_radial": 16,
            "n_blocks": 3,
            "precision": "float32",
            "seed": seed,
        }
    return {
        "type": "se_e2_a",
        "sel": "auto:1.20",
        "rcut_smth": 0.5,
        "rcut": 6.0,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "seed": seed,
    }


def deepmd_input(
    *,
    dataset_root: Path,
    type_map: Sequence[str],
    architecture: str,
    backend: str,
    seed: int,
    numb_steps: int,
    batch_atoms: int,
    systems: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if architecture == "dpa4":
        model: dict[str, Any] = {
            "type": "dpa4",
            "type_map": list(type_map),
            "descriptor": _deepmd_descriptor(architecture, backend, seed),
            "fitting_net": {
                "type": "dpa4_ener",
                "neuron": [0],
                "activation_function": "silu",
                "precision": "float32",
                "seed": seed + 1000,
            },
        }
    else:
        model = {
            "type_map": list(type_map),
            "type_embedding": {"neuron": [8], "resnet_dt": False, "seed": seed + 500}
            if architecture in {"dpa1", "se_e2_a"}
            else {},
            "descriptor": _deepmd_descriptor(architecture, backend, seed),
            "fitting_net": {
                "type": "ener",
                "neuron": [240, 240, 240],
                "resnet_dt": True,
                "activation_function": "tanh",
                "precision": "float32",
                "seed": seed + 1000,
            },
        }
        if not model["type_embedding"]:
            model.pop("type_embedding")
    modern = architecture in {"dpa2", "dpa2_ft", "dpa3", "dpa4"}
    return {
        "_comment": "InterfaceForge energy/force model; raw DFT force labels.",
        "model": model,
        "learning_rate": {
            "type": "exp",
            "start_lr": 2.0e-4 if modern else 1.0e-3,
            "stop_lr": 1.0e-6,
            "decay_steps": 5000,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_f": 1000.0,
            "limit_pref_f": 1.0,
            "start_pref_v": 0.0,
            "limit_pref_v": 0.0,
        },
        "training": {
            "training_data": {
                "systems": list(systems["train"]),
                "batch_size": f"auto:{batch_atoms}",
                "auto_prob": "prob_uniform",
            },
            "validation_data": {
                "systems": list(systems["valid"]),
                "batch_size": f"auto:{batch_atoms}",
                "auto_prob": "prob_uniform",
                "numb_btch": 28,
            },
            "numb_steps": numb_steps,
            "seed": seed + 2000,
            "disp_file": "lcurve.out",
            "disp_freq": 1000,
            "save_freq": 20000,
            "save_ckpt": "model.ckpt",
            "max_ckpt_keep": 5,
        },
    }


def _backend_flag(backend: str) -> tuple[str, str]:
    if backend == "tensorflow":
        return "--tf", "frozen_model.pb"
    # `pt_expt` names the experimental PyTorch implementation selected in the
    # campaign, but DeePMD's public CLI still uses `--pt`.
    return "--pt", "frozen_model.pth"


def _deepmd_shell_prefix(settings: Mapping[str, Any], backend: str, *, scheduler: str) -> str:
    """Return portable direct/container helpers for generated jobs.

    The LONI DeePMD module is itself a container/MPI wrapper.  Starting that
    wrapper inside a second ``srun`` step can leave PMI uninitialised during
    follow-up commands such as ``dp freeze``.  A batch allocation already
    exposes the requested GPU, so generated jobs invoke DeePMD directly for
    both Slurm and local profiles.
    """

    image = str(settings.get("container_image", "")).strip()
    python_check = (
        "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); "
        "print('TensorFlow-visible GPUs:',g); assert g"
        if backend == "tensorflow"
        else "import torch; print('CUDA available:',torch.cuda.is_available()); "
        "assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    )
    if not image:
        return "\n".join(
            [
                'dp_exec() { dp "$@"; }',
                'container_python() { python "$@"; }',
                f"container_python -c {shlex.quote(python_check)}",
            ]
        )
    return "\n".join(
        [
            f'DEEPMD_IMAGE="${{DEEPMD_IMAGE:-{image}}}"',
            'CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-$(command -v apptainer || command -v singularity || true)}"',
            '[[ -n "$CONTAINER_RUNTIME" ]] || { echo "ERROR: apptainer/singularity not found"; exit 2; }',
            '[[ -s "$DEEPMD_IMAGE" ]] || { echo "ERROR: container image not found: $DEEPMD_IMAGE"; exit 2; }',
            "DEEPMD_BIND_ARGS=()",
            "for bind_dir in /ddnB /project; do",
            '  [[ -d "$bind_dir" ]] && DEEPMD_BIND_ARGS+=(--bind "$bind_dir:$bind_dir")',
            "done",
            'dp_exec() { "$CONTAINER_RUNTIME" exec --nv '
            '"${DEEPMD_BIND_ARGS[@]}" "$DEEPMD_IMAGE" dp "$@"; }',
            "container_python() {",
            '  "$CONTAINER_RUNTIME" exec --nv '
            '"${DEEPMD_BIND_ARGS[@]}" "$DEEPMD_IMAGE" python "$@"',
            "}",
            f"container_python -c {shlex.quote(python_check)}",
        ]
    )


def generate_deepmd_training(campaign: Campaign, *, force: bool = False) -> dict[str, Any]:
    """Generate a seeded DeePMD committee and one Slurm array launcher."""

    settings = dict(campaign.models.get("deepmd", {}))
    if not settings.get("enabled", False):
        raise SafetyError("models.deepmd.enabled is false")
    root = campaign.root / "models" / "deepmd"
    _prepare_root(root, force=force)
    dataset_root = _resolve(
        campaign.root,
        settings.get("dataset_root", "datasets/canonical/deepmd"),
    )
    type_map, split_systems = validate_deepmd_dataset(dataset_root)
    committee = int(settings.get("committee", 4))
    seeds = [int(value) for value in settings.get("seeds", [11, 23, 37, 53])][:committee]
    backend = str(settings.get("backend", "tensorflow"))
    architectures = [str(value) for value in settings.get("architectures", [settings.get("descriptor", "dpa1")])]
    models: list[dict[str, Any]] = []
    for architecture_index, architecture in enumerate(architectures):
        for index, seed in enumerate(seeds):
            run = root / architecture / f"model_{index:03d}"
            if run.exists() and any(run.iterdir()) and not force:
                raise SafetyError(f"Refusing to reuse nonempty model directory: {run}")
            run.mkdir(parents=True, exist_ok=True)
            payload = deepmd_input(
                dataset_root=dataset_root,
                type_map=type_map,
                architecture=architecture,
                backend=backend,
                seed=seed,
                numb_steps=int(settings.get("numb_steps", 500000)),
                batch_atoms=int(settings.get("batch_atoms", 1024)),
                systems=split_systems,
            )
            _write_json(run / "input.json", payload)
            models.append(
                {
                    "task_id": architecture_index * committee + index,
                    "architecture": architecture,
                    "index": index,
                    "seed": seed,
                    "directory": str(run),
                }
            )

    profile = load_profile(campaign.profile_path)
    profile_name = str(settings.get("profile", "deepmd_gpu"))
    profile_job = dict(profile.get("jobs", {}).get(profile_name, {}))
    backend_flag, frozen_name = _backend_flag(backend)
    architectures_array = " ".join(shlex.quote(value) for value in architectures)
    prefix = _deepmd_shell_prefix(settings, backend, scheduler=str(profile.get("scheduler", "")).lower())
    checkpoint = "model.ckpt" if backend == "tensorflow" else "model.ckpt.pt"
    restart_marker = "model.ckpt.index" if backend == "tensorflow" else checkpoint
    freeze_checkpoint = "." if backend == "tensorflow" else checkpoint
    evaluation_model = frozen_name if backend == "tensorflow" else checkpoint
    if backend == "tensorflow":
        freeze_command = f"dp_exec {backend_flag} freeze -c {freeze_checkpoint} -o {frozen_name}"
    else:
        freeze_command = (
            f'if [[ "$ARCH" == "dpa4" ]]; then '
            f"dp_exec {backend_flag} freeze -c {freeze_checkpoint} -o {frozen_name} "
            f'|| {{ echo "ERROR: DPA-4 freeze failed; deployment is not approved."; exit 3; }}; '
            f"else dp_exec {backend_flag} freeze -c {freeze_checkpoint} -o {frozen_name} "
            f'|| echo "WARNING: freeze failed; {checkpoint} remains valid for auditing, '
            'but deployment is not approved."; fi'
        )
    finetune = dict(settings.get("finetune", {}))
    if "dpa2_ft" in architectures:
        ft_pretrained = shlex.quote(str(finetune["pretrained"]))
        ft_branch = shlex.quote(str(finetune.get("model_branch", "RANDOM")))
        train_dispatch = (
            f'if [[ -s {restart_marker} ]]; then TRAIN_ARGS+=(--restart {checkpoint}); '
            f'elif [[ "$ARCH" == "dpa2_ft" ]]; then '
            f"TRAIN_ARGS+=(--finetune {ft_pretrained} --model-branch {ft_branch}); fi"
        )
        smoke_train_line = (
            f'if [[ "$ARCH" == "dpa2_ft" ]]; then dp_exec {backend_flag} train input.json '
            f"--finetune {ft_pretrained} --model-branch {ft_branch}; "
            f"else dp_exec {backend_flag} train input.json; fi"
        )
    else:
        train_dispatch = (
            f'if [[ -s {restart_marker} ]]; then TRAIN_ARGS+=(--restart {checkpoint}); fi'
        )
        smoke_train_line = f"dp_exec {backend_flag} train input.json"
    command = "\n".join(
        [
            prefix,
            f"ARCHITECTURES=({architectures_array})",
            f"NMODELS={committee}",
            'TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"',
            "ARCH_INDEX=$(( TASK_ID / NMODELS ))",
            "MODEL_INDEX=$(( TASK_ID % NMODELS ))",
            'ARCH="${ARCHITECTURES[$ARCH_INDEX]}"',
            'MODEL_ID="$(printf \'%03d\' "${MODEL_INDEX}")"',
            f'RUN_DIR={shlex.quote(str(root))}/${{ARCH}}/model_${{MODEL_ID}}',
            'cd "${RUN_DIR}"',
            f"TRAIN_ARGS=({backend_flag} train input.json)",
            train_dispatch,
            'dp_exec "${TRAIN_ARGS[@]}"',
            freeze_command,
            "mkdir -p test_results",
            f"MODEL_FILE={shlex.quote(evaluation_model)}",
            '[[ -s "$MODEL_FILE" ]] || { echo "ERROR: missing evaluation model $MODEL_FILE"; exit 3; }',
            f"mapfile -t TEST_SYSTEMS < {shlex.quote(str(root / 'test_systems.txt'))}",
            'for TEST_INDEX in "${!TEST_SYSTEMS[@]}"; do',
            "  printf -v TEST_LABEL 'system_%03d' \"$TEST_INDEX\"",
            '  mkdir -p "test_results/${TEST_LABEL}"',
            f'  dp_exec {backend_flag} test -m "$MODEL_FILE" '
            '-s "${TEST_SYSTEMS[$TEST_INDEX]}" -n 0 '
            '-d "test_results/${TEST_LABEL}/detail" '
            '> "test_results/${TEST_LABEL}.log" 2>&1',
            "done",
        ]
    )
    launcher = render_job(
        profile,
        profile_name,
        command=command,
        job_name=f"{campaign.name}_deepmd",
        array=f"0-{len(architectures) * committee - 1}%{int(settings.get('max_concurrent', 2))}",
        working_directory=str(root),
    )
    write_job(root / "run_ensemble.slurm", launcher, force=force)
    preflight_command = "\n".join(
        [
            prefix,
            "nvidia-smi",
            "dp_exec --version",
            f"dp_exec {backend_flag} train --help >/dev/null",
            "echo 'DeepMD GPU preflight passed.'",
        ]
    )
    preflight = render_job(
        profile,
        profile_name,
        command=preflight_command,
        job_name=f"{campaign.name}_deepmd_preflight",
        working_directory=str(root),
    )
    write_job(root / "run_preflight.slurm", preflight, force=force)

    smoke_command = "\n".join(
        [
            prefix,
            f"ARCHITECTURES=({architectures_array})",
            'TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"',
            'ARCH="${ARCHITECTURES[$TASK_ID]}"',
            f'SOURCE_DIR={shlex.quote(str(root))}/${{ARCH}}/model_000',
            f'SMOKE_DIR={shlex.quote(str(root))}/smoke/'
            'job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/${ARCH}',
            '[[ -s "$SOURCE_DIR/input.json" ]] || { echo "ERROR: missing input.json"; exit 2; }',
            '[[ ! -e "$SMOKE_DIR" ]] || { echo "ERROR: refusing to overwrite $SMOKE_DIR"; exit 2; }',
            'mkdir -p "$SMOKE_DIR"',
            'container_python - "$SOURCE_DIR/input.json" "$SMOKE_DIR/input.json" '
            '"${SMOKE_STEPS:-20}" <<\'PY\'',
            "import json",
            "import sys",
            "from pathlib import Path",
            "source, target, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])",
            "data = json.loads(source.read_text())",
            'data["training"]["numb_steps"] = steps',
            'data["training"]["disp_freq"] = 1',
            'data["training"]["save_freq"] = max(1, steps)',
            'target.write_text(json.dumps(data, indent=2) + "\\n")',
            "PY",
            'cd "$SMOKE_DIR"',
            smoke_train_line,
            freeze_command,
            "mkdir -p test_results",
            f"MODEL_FILE={shlex.quote(evaluation_model)}",
            '[[ -s "$MODEL_FILE" ]] || { echo "ERROR: missing smoke evaluation model $MODEL_FILE"; exit 3; }',
            f"mapfile -t TEST_SYSTEMS < {shlex.quote(str(root / 'test_systems.txt'))}",
            'for TEST_INDEX in "${!TEST_SYSTEMS[@]}"; do',
            "  printf -v TEST_LABEL 'system_%03d' \"$TEST_INDEX\"",
            '  mkdir -p "test_results/${TEST_LABEL}"',
            f'  dp_exec {backend_flag} test -m "$MODEL_FILE" '
            '-s "${TEST_SYSTEMS[$TEST_INDEX]}" -n 100 '
            '-d "test_results/${TEST_LABEL}/detail" '
            '> "test_results/${TEST_LABEL}.log" 2>&1',
            "done",
            'echo "DeepMD smoke test passed for $ARCH"',
        ]
    )
    smoke = render_job(
        profile,
        profile_name,
        command=smoke_command,
        job_name=f"{campaign.name}_deepmd_smoke",
        array=f"0-{len(architectures) - 1}%1",
        working_directory=str(root),
    )
    write_job(root / "run_smoke.slurm", smoke, force=force)

    audit_script = root / "summarize_deepmd.py"
    audit_script.write_text(
        Path(__file__).with_name("deepmd_audit.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    evaluation_command = "\n".join(
        [
            prefix,
            f"ARCHITECTURES=({architectures_array})",
            f"NMODELS={committee}",
            "SEEDS=(" + " ".join(str(value) for value in seeds) + ")",
            "MODEL_DEVIATION_FAILED=0",
            'ARCH="${ARCHITECTURES[${SLURM_ARRAY_TASK_ID:?Submit with sbatch}]}"',
            f'EVAL_ROOT={shlex.quote(str(root))}/evaluation/${{ARCH}}/job_${{SLURM_JOB_ID}}',
            'mkdir -p "$EVAL_ROOT/by_system"',
            "MODELS=()",
            'for MODEL_INDEX in $(seq 0 $((NMODELS - 1))); do',
            "  printf -v MODEL_ID '%03d' \"$MODEL_INDEX\"",
            f'  MODEL={shlex.quote(str(root))}/${{ARCH}}/model_${{MODEL_ID}}/{evaluation_model}',
            '  [[ -s "$MODEL" ]] || { echo "ERROR: missing $MODEL"; exit 2; }',
            '  MODELS+=("$MODEL")',
            "done",
            f"mapfile -t TEST_SYSTEMS < {shlex.quote(str(root / 'test_systems.txt'))}",
            'for TEST_INDEX in "${!TEST_SYSTEMS[@]}"; do',
            "  printf -v TEST_LABEL 'system_%03d' \"$TEST_INDEX\"",
            '  SYSTEM_ROOT="$EVAL_ROOT/by_system/$TEST_LABEL"',
            '  mkdir -p "$SYSTEM_ROOT"',
            '  for MODEL_INDEX in "${!MODELS[@]}"; do',
            "    printf -v MODEL_LABEL 'model_%03d' \"$MODEL_INDEX\"",
            f'    dp_exec {backend_flag} test -m "${{MODELS[$MODEL_INDEX]}}" '
            '-s "${TEST_SYSTEMS[$TEST_INDEX]}" -n 0 '
            '-d "$SYSTEM_ROOT/${MODEL_LABEL}_detail" '
            '> "$SYSTEM_ROOT/${MODEL_LABEL}.log" 2>&1',
            "  done",
            f'  if ! dp_exec {backend_flag} model-devi -m "${{MODELS[@]}}" '
            '-s "${TEST_SYSTEMS[$TEST_INDEX]}" '
            '-o "$SYSTEM_ROOT/model_devi.out" --real_error '
            '> "$SYSTEM_ROOT/model_devi.log" 2>&1; then',
            '    echo "WARNING: model deviation failed for $TEST_LABEL" >&2',
            "    MODEL_DEVIATION_FAILED=1",
            "  fi",
            "done",
            f"container_python {shlex.quote(str(audit_script))} "
            f"--eval-root \"$EVAL_ROOT\" --systems {shlex.quote(str(root / 'test_systems.txt'))} "
            '--architecture "$ARCH" --seeds "${SEEDS[@]}"',
            '[[ "$MODEL_DEVIATION_FAILED" -eq 0 ]] || { '
            'echo "ERROR: RMSE reports were written, but committee deviation failed." >&2; exit 4; }',
            'echo "DeepMD committee evaluation completed for $ARCH"',
        ]
    )
    evaluation = render_job(
        profile,
        profile_name,
        command=evaluation_command,
        job_name=f"{campaign.name}_deepmd_evaluate",
        array=f"0-{len(architectures) - 1}%1",
        working_directory=str(root),
    )
    write_job(root / "run_evaluate.slurm", evaluation, force=force)

    test_systems_path = root / "test_systems.txt"
    test_systems_path.write_text(
        "".join(f"{path}\n" for path in split_systems["test"]),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "engine": "deepmd",
        "campaign": campaign.name,
        "backend": backend,
        "backend_flag": backend_flag,
        "runtime": {
            "profile": profile_name,
            "modules": [str(value) for value in profile_job.get("modules", [])],
            "container_image": str(settings.get("container_image", "")),
        },
        "architectures": architectures,
        "finetune": (
            {
                "architecture": "dpa2_ft",
                "pretrained": str(finetune["pretrained"]),
                "model_branch": str(finetune.get("model_branch", "RANDOM")),
            }
            if "dpa2_ft" in architectures
            else None
        ),
        "dataset_root": str(dataset_root),
        "type_map": type_map,
        "split_systems": {name: len(values) for name, values in split_systems.items()},
        "models": models,
        "launcher": str(root / "run_ensemble.slurm"),
        "preflight_launcher": str(root / "run_preflight.slurm"),
        "smoke_launcher": str(root / "run_smoke.slurm"),
        "evaluation_launcher": str(root / "run_evaluate.slurm"),
        "execution_order": [
            "run_preflight.slurm",
            "run_smoke.slurm",
            "run_ensemble.slurm",
            "run_evaluate.slurm",
        ],
        "test_systems": str(test_systems_path),
        "frozen_model_name": frozen_name,
        "evaluation_model_name": evaluation_model,
        "evaluation_reports": ["rmse_by_system.csv", "rmse_overall.csv", "rmse_audit.json"],
        "evaluation_audit_script": str(audit_script),
        "gpu_memory_controls": {
            "DP_INFER_BATCH_SIZE": "profile-controlled; default 4096",
            "TF_FORCE_GPU_ALLOW_GROWTH": "profile-controlled; default true",
        },
        "dpa4_status": (
            "experimental; verify freeze and LAMMPS deployment before production"
            if "dpa4" in architectures
            else "not requested"
        ),
    }
    manifest_path = root / "ensemble_manifest.json"
    _write_json(manifest_path, manifest)
    StateStore(campaign.root).artifact("deepmd_training_manifest", manifest_path)
    return manifest
