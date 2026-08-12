#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -t 72:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J mace.train
#SBATCH -o mace.%j.out
#SBATCH -e mace.%j.err

# Conda activation scripts are not always compatible with nounset.
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge

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

TRAIN_FILE="train.extxyz"
VALID_FILE="valid.extxyz"
TEST_FILE="test.extxyz"

for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing or empty file: $f"
        exit 1
    fi
done

echo
echo "Training files:"
ls -lh "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"
echo

# ASE/VASP extxyz property names.
ENERGY_KEY="energy"
FORCES_KEY="forces"

echo "Checking first frame keys..."

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

MODEL_NAME="TiN_SiN_mace"
MODEL_DIR="mace_model"

mkdir -p "$MODEL_DIR"

BATCH_SIZE=16
VALID_BATCH_SIZE=32
MAX_EPOCHS=20
PATIENCE=10
START_SWA=16

HELP_TXT="$("$MACE_TRAIN_BIN" --help 2>&1 || true)"

EXTRA_ARGS=()

if grep -q -- "--save_cpu" <<< "$HELP_TXT"; then
    EXTRA_ARGS+=(--save_cpu)
fi

if grep -q -- "--keep_checkpoints" <<< "$HELP_TXT"; then
    EXTRA_ARGS+=(--keep_checkpoints)
fi

SWA_ARGS=()

if grep -q -- "--swa" <<< "$HELP_TXT"; then
    SWA_ARGS+=(--swa --start_swa "$START_SWA")
fi

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

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
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
echo

GPU_LOG="gpu_usage_${SLURM_JOB_ID}.log"

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

trap cleanup EXIT INT TERM

echo "Starting distributed MACE training..."
echo

srun --ntasks=2 --kill-on-bad-exit=1 \
    "$MACE_TRAIN_BIN" \
    --name "$MODEL_NAME" \
    --model "MACE" \
    --train_file "$TRAIN_FILE" \
    --valid_file "$VALID_FILE" \
    --test_file "$TEST_FILE" \
    --energy_key "$ENERGY_KEY" \
    --forces_key "$FORCES_KEY" \
    --model_dir "$MODEL_DIR" \
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
    "${SWA_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo
echo "Training finished."
echo "MACE evaluated test.extxyz through --test_file."
echo

if [[ -x "$MACE_EVAL_BIN" ]]; then
    SEARCH_DIRS=()

    [[ -d "$MODEL_DIR" ]] && SEARCH_DIRS+=("$MODEL_DIR")
    [[ -d "checkpoints" ]] && SEARCH_DIRS+=("checkpoints")

    MODEL_PATH=""

    if ((${#SEARCH_DIRS[@]} > 0)); then
        MODEL_PATH="$(
            find "${SEARCH_DIRS[@]}" \
                -type f \
                \( -name "${MODEL_NAME}*.model" -o -name "*.model" \) \
                -printf "%T@ %p\n" 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d' ' -f2-
        )"
    fi

    if [[ -n "${MODEL_PATH:-}" && -f "$MODEL_PATH" ]]; then
        echo "Writing explicit test prediction file:"
        echo "  model: $MODEL_PATH"
        echo "  test:  $TEST_FILE"
        echo

        "$MACE_EVAL_BIN" \
            --configs "$TEST_FILE" \
            --model "$MODEL_PATH" \
            --output test_predictions.extxyz \
            --energy_key "$ENERGY_KEY" \
            --forces_key "$FORCES_KEY"

        echo
        echo "Wrote test_predictions.extxyz"
    else
        echo "Could not locate the final .model file for mace_eval_configs."
        echo "Check $MODEL_DIR and checkpoints/ manually."
    fi
else
    echo "mace_eval_configs not found; skipping explicit prediction generation."
fi

