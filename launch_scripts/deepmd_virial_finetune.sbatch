#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=deepmd.virial
#SBATCH --output=deepmd_virial.%A_%a.out
#SBATCH --error=deepmd_virial.%A_%a.err

# Submit from Periodic_MLIPs, one architecture at a time:
#   sbatch --array=0-3%2 --export=ALL,DEEPMD_ARCH=dpa2 \
#     /path/to/InterfaceForge/launch_scripts/deepmd_virial_finetune.sbatch
# A 20-step smoke for member 0 uses:
#   sbatch --export=ALL,DEEPMD_ARCH=dpa2,DEEPMD_INDEX=0,DEEPMD_STEPS=20,DEEPMD_RUN_TAG=virial_smoke \
#     /path/to/InterfaceForge/launch_scripts/deepmd_virial_finetune.sbatch
# The default A1 run uses 20,000 steps and writes models/deepmd_virial_a1/.

set -euo pipefail

CAMP="${DEEPMD_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the campaign root}}"
ARCH="${DEEPMD_ARCH:?Set DEEPMD_ARCH to dpa2, dpa2_ft, dpa3, dpa3_ft, or dpa4}"
INDEX="${SLURM_ARRAY_TASK_ID:-${DEEPMD_INDEX:?Set DEEPMD_INDEX or submit an array}}"
STEPS="${DEEPMD_STEPS:-20000}"
LR="${DEEPMD_LR:-1e-5}"
RUN_TAG="${DEEPMD_RUN_TAG:-virial_a1}"
MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"

[[ "$ARCH" =~ ^(dpa2|dpa2_ft|dpa3|dpa3_ft|dpa4)$ ]] || { echo "Invalid architecture: $ARCH" >&2; exit 2; }
[[ "$INDEX" =~ ^[0-9]+$ ]] || { echo "Invalid model index: $INDEX" >&2; exit 2; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "DEEPMD_STEPS must be positive" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run tag" >&2; exit 2; }
printf -v MODEL_ID '%03d' "$((10#$INDEX))"
CAMP="$(cd "$CAMP" && pwd -P)"
SOURCE="$CAMP/models/deepmd/$ARCH/model_$MODEL_ID"
CHECKPOINT="$SOURCE/model.ckpt.pt"
TARGET="$CAMP/models/deepmd_$RUN_TAG/$ARCH/model_$MODEL_ID"
[[ -s "$SOURCE/input.json" && -s "$CHECKPOINT" ]] || {
    echo "Missing source input or checkpoint in $SOURCE" >&2
    exit 2
}
[[ "$TARGET" != "$SOURCE" ]] || { echo "Refusing to train in the source directory" >&2; exit 2; }

module purge
module load "$MODULE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

mkdir -p "$TARGET"
exec 9>"$TARGET/.train.lock"
flock -n 9 || { echo "Another training job is active in $TARGET" >&2; exit 2; }

# Construct a fresh input from the E/F run, preserving its systems and model
# settings. Check every train/validation/test set before touching the trainer.
python - "$SOURCE" "$TARGET" "$CAMP" "$STEPS" "$LR" "$CHECKPOINT" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

source, target, camp = map(Path, sys.argv[1:4])
steps, lr, checkpoint = int(sys.argv[4]), float(sys.argv[5]), Path(sys.argv[6])
if not math.isfinite(lr) or lr <= 0:
    raise SystemExit("DEEPMD_LR must be finite and positive")
config = json.loads((source / "input.json").read_text())
for split in ("training_data", "validation_data"):
    systems = config["training"][split]["systems"]
    if not systems:
        raise SystemExit(f"No {split} systems")
    resolved = []
    for name in systems:
        path = Path(name)
        path = path if path.is_absolute() else (source / path).resolve()
        resolved.append(str(path))
    config["training"][split]["systems"] = resolved

test_root = camp / "datasets/canonical/deepmd/test"
if not test_root.is_dir():
    raise SystemExit(f"Missing held-out DeePMD split: {test_root}")
systems_to_check = [
    Path(s)
    for split in ("training_data", "validation_data")
    for s in config["training"][split]["systems"]
]
test_systems = sorted(p.parent for p in test_root.rglob("type.raw"))
systems_to_check.extend(test_systems)
if not test_systems:
    raise SystemExit(f"No test systems in {test_root}")
for system in systems_to_check:
    sets = sorted(system.glob("set.*"))
    if not sets:
        raise SystemExit(f"No set.* data in {system}")
    for item in sets:
        virial = item / "virial.npy"
        energy = item / "energy.npy"
        if not virial.is_file() or not energy.is_file():
            raise SystemExit(f"Missing virial.npy or energy.npy in {item}")
        v = np.load(virial, mmap_mode="r")
        e = np.load(energy, mmap_mode="r")
        if v.ndim != 2 or v.shape[1] != 9 or len(v) != len(e):
            raise SystemExit(f"Bad virial shape/frame count in {virial}: {v.shape}")

config["loss"]["start_pref_v"] = 1.0
config["loss"]["limit_pref_v"] = 1.0
config["learning_rate"]["start_lr"] = lr
config["training"]["numb_steps"] = steps
config["training"]["save_freq"] = min(2000, steps)
config["training"]["disp_freq"] = min(100, steps)
payload = json.dumps(config, indent=2) + "\n"
input_file = target / "input.json"
if input_file.exists():
    if input_file.read_text() != payload:
        raise SystemExit(f"Existing A1 input differs: {input_file}")
else:
    input_file.write_text(payload)

origin = target / "origin.json"
record = {"source_checkpoint": str(checkpoint), "source_bytes": checkpoint.stat().st_size}
if origin.exists():
    if json.loads(origin.read_text()) != record:
        raise SystemExit(f"Existing A1 origin differs: {origin}")
else:
    origin.write_text(json.dumps(record, indent=2) + "\n")
(target / "test_system.txt").write_text(str(test_systems[0]) + "\n")
print(f"Checked {len(systems_to_check)} systems; A1 input: {input_file}")
PY

cd "$TARGET"
echo "Source checkpoint: $CHECKPOINT"
echo "New run: $TARGET"
dp --version
if [[ -s model.ckpt.pt ]]; then
    echo "Resuming this A1 run from its own checkpoint"
    dp --pt train input.json --restart model.ckpt.pt
else
    echo "Fine-tuning all weights from the corresponding original member"
    dp --pt train input.json --finetune "$CHECKPOINT" --use-pretrain-script
fi
[[ -s model.ckpt.pt ]] || { echo "Training ended without a checkpoint" >&2; exit 3; }
TEST_SYSTEM="$(cat test_system.txt)"
dp --pt test -m model.ckpt.pt -s "$TEST_SYSTEM" -n 100 > test.log 2>&1 || {
    echo "Training succeeded, but held-out dp test failed; inspect $TARGET/test.log" >&2
    exit 4
}
echo "Completed $ARCH model_$MODEL_ID: $TARGET/model.ckpt.pt"
