#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -t 72:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J mace.committee
#SBATCH -o mace.committee.%j.out
#SBATCH -e mace.committee.%j.err

# Train one additional member of the TiN/SiN MACE committee.
#
# The original model is deliberately left untouched at:
#   mace_model/TiN_SiN_mace_stagetwo.model
#
# Submit three new seeds as separate Slurm jobs:
#
#   for seed in 211 307 419; do
#       sbatch --export=ALL,MACE_SEED="$seed" mace_train_committee.sh
#   done

# Conda activation hooks are not always compatible with nounset.
set -eo pipefail

BASE_DIR="${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch}"
SEED="${MACE_SEED:?Set MACE_SEED when submitting this job}"

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MACE_SEED must be a non-negative integer."
    echo "Received: $SEED"
    exit 1
fi

MODEL_NAME="TiN_SiN_mace_seed${SEED}"
RUN_DIR="$BASE_DIR/mace_committee/seed_${SEED}"
MODEL_DIR="$RUN_DIR/mace_model"
CHECKPOINTS_DIR="$RUN_DIR/checkpoints"
RESULTS_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"

mkdir -p \
    "$MODEL_DIR" \
    "$CHECKPOINTS_DIR" \
    "$RESULTS_DIR" \
    "$LOG_DIR"

# Prevent two simultaneous jobs from writing to the same seed directory.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$RUN_DIR/.train.lock"

    if ! flock -n 9; then
        echo "ERROR: another training job is already active for seed $SEED"
        echo "Run directory: $RUN_DIR"
        exit 1
    fi
fi

cd "$RUN_DIR"

module purge

# Disable nounset while Conda runs its activation/deactivation hooks.
set +u
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate /project/lgutsev/env/mace_env
set -u

echo
echo "Activated Conda environment:"
echo "  $CONDA_PREFIX"
echo

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF

export MACE_NUM_WORKERS="${MACE_NUM_WORKERS:-8}"

# Prevent an inherited value from changing PyTorch checkpoint loading.
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/cuequivariance_ops/lib:$LD_LIBRARY_PATH"

MACE_TRAIN_BIN="$CONDA_PREFIX/bin/mace_run_train"
MACE_EVAL_BIN="$CONDA_PREFIX/bin/mace_eval_configs"

if [[ ! -x "$MACE_TRAIN_BIN" ]]; then
    echo "ERROR: mace_run_train not found or not executable:"
    echo "  $MACE_TRAIN_BIN"
    exit 1
fi

# All committee members use the same fixed data split.
TRAIN_FILE="$BASE_DIR/train.extxyz"
VALID_FILE="$BASE_DIR/valid.extxyz"
TEST_FILE="$BASE_DIR/test.extxyz"

for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing or empty file: $f"
        exit 1
    fi
done

echo
echo "Committee member:"
echo "  seed:       $SEED"
echo "  model name: $MODEL_NAME"
echo "  run dir:    $RUN_DIR"
echo
echo "Training files:"
ls -lh "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"
echo

ENERGY_KEY="energy"
FORCES_KEY="forces"

echo "Checking first-frame keys..."

python - <<PY
from ase.io import read

for fname in ["$TRAIN_FILE", "$VALID_FILE", "$TEST_FILE"]:
    atoms = read(fname, index=0)

    print()
    print(fname)
    print("  info keys:  ", sorted(atoms.info.keys()))
    print("  array keys: ", sorted(atoms.arrays.keys()))

    if atoms.calc is not None:
        print("  calc keys:  ", sorted(atoms.calc.results.keys()))
    else:
        print("  calc keys:   None")

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print("  energy:     ", energy)
    print("  forces:     ", forces.shape)
PY

BATCH_SIZE=16
VALID_BATCH_SIZE=32
MAX_EPOCHS=20
PATIENCE=10
START_STAGE_TWO=16

HELP_TXT="$("$MACE_TRAIN_BIN" --help 2>&1 || true)"

EXTRA_ARGS=()

if grep -q -- "--save_cpu" <<< "$HELP_TXT"; then
    EXTRA_ARGS+=(--save_cpu)
fi

if grep -q -- "--keep_checkpoints" <<< "$HELP_TXT"; then
    EXTRA_ARGS+=(--keep_checkpoints)
fi

# Support both current and older MACE names for the second stage.
STAGE_TWO_ARGS=()

if grep -q -- "--stage_two" <<< "$HELP_TXT"; then
    STAGE_TWO_ARGS+=(
        --stage_two
        --start_stage_two "$START_STAGE_TWO"
    )
elif grep -q -- "--swa" <<< "$HELP_TXT"; then
    STAGE_TWO_ARGS+=(
        --swa
        --start_swa "$START_STAGE_TWO"
    )
fi

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"

export MASTER_PORT
MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

echo
echo "Distributed settings:"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "  MACE_NUM_WORKERS=$MACE_NUM_WORKERS"
echo

echo "GPU inventory:"
nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
echo

GPU_LOG="$RUN_DIR/gpu_usage_${SLURM_JOB_ID}.log"

nvidia-smi dmon -s pucm -d 5 > "$GPU_LOG" &
GPU_MON_PID=$!

cleanup() {
    local rc=$?

    if [[ -n "${GPU_MON_PID:-}" ]]; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
    fi

    return "$rc"
}

trap cleanup EXIT

echo "Starting committee training for seed $SEED..."
echo

srun --ntasks=2 --kill-on-bad-exit=1 \
    "$MACE_TRAIN_BIN" \
    --name "$MODEL_NAME" \
    --seed "$SEED" \
    --model "MACE" \
    --train_file "$TRAIN_FILE" \
    --valid_file "$VALID_FILE" \
    --test_file "$TEST_FILE" \
    --energy_key "$ENERGY_KEY" \
    --forces_key "$FORCES_KEY" \
    --model_dir "$MODEL_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --log_dir "$LOG_DIR" \
    --E0s "average" \
    --r_max 5.0 \
    --num_interactions 2 \
    --num_channels 128 \
    --max_L 2 \
    --correlation 3 \
    --batch_size "$BATCH_SIZE" \
    --valid_batch_size "$VALID_BATCH_SIZE" \
    --max_num_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --loss "weighted" \
    --error_table "PerAtomRMSE" \
    --default_dtype "float32" \
    --ema \
    --ema_decay 0.99 \
    --amsgrad \
    --device cuda \
    --distributed \
    --restart_latest \
    "${STAGE_TWO_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo
echo "Training finished for seed $SEED."
echo "MACE evaluated test.extxyz through --test_file."
echo

if [[ -x "$MACE_EVAL_BIN" ]]; then
    MODEL_PATH=""

    # Prefer the uncompiled second-stage model, then the base model.
    for candidate in \
        "$MODEL_DIR/${MODEL_NAME}_stagetwo.model" \
        "$MODEL_DIR/${MODEL_NAME}.model"
    do
        if [[ -f "$candidate" ]]; then
            MODEL_PATH="$candidate"
            break
        fi
    done

    # Compatibility fallback for versions with different final naming.
    if [[ -z "$MODEL_PATH" ]]; then
        MODEL_PATH="$(
            find "$MODEL_DIR" "$CHECKPOINTS_DIR" \
                -type f \
                -name "${MODEL_NAME}*.model" \
                ! -name "*_compiled.model" \
                -printf "%T@ %p\n" 2>/dev/null \
            | sort -nr \
            | sed -n '1p' \
            | cut -d' ' -f2-
        )"
    fi

    if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]; then
        PREDICTIONS_FILE="$RUN_DIR/test_predictions_seed${SEED}.extxyz"

        echo "Writing explicit test prediction file:"
        echo "  model:  $MODEL_PATH"
        echo "  test:   $TEST_FILE"
        echo "  output: $PREDICTIONS_FILE"
        echo

        "$MACE_EVAL_BIN" \
            --configs "$TEST_FILE" \
            --model "$MODEL_PATH" \
            --output "$PREDICTIONS_FILE" \
            --energy_key "$ENERGY_KEY" \
            --forces_key "$FORCES_KEY"

        echo
        echo "Wrote $PREDICTIONS_FILE"
    else
        echo "Could not locate the final .model file for seed $SEED."
        echo "Check these directories manually:"
        echo "  $MODEL_DIR"
        echo "  $CHECKPOINTS_DIR"
    fi
else
    echo "mace_eval_configs not found; skipping explicit prediction generation."
fi

