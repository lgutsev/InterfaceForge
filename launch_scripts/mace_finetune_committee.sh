#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -t 72:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J mace.ft.committee
#SBATCH -o mace.ft.committee.%j.out
#SBATCH -e mace.ft.committee.%j.err

# Fine-tune one member of a MACE committee from a foundation model, on the same
# fixed canonical split used by mace_train_committee.sh. Submit four seeds:
#
#   FM=/project/lgutsev/foundational_models/mace/mace-mpa-0-medium.model
#   for seed in 11 23 37 53; do
#       sbatch --export=ALL,MACE_SEED="$seed",MACE_FOUNDATION_MODEL="$FM" \
#           mace_finetune_committee.sh
#   done
#
# The architecture (r_max, channels, max_L, correlation, interactions) is
# inherited from the foundation model and cannot be set here. Output lands in
# mace_finetune_committee/seed_<seed>/ so it never collides with the
# from-scratch committee in mace_committee/.

set -eo pipefail

BASE_DIR="${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch}"
SEED="${MACE_SEED:?Set MACE_SEED when submitting this job}"
FOUNDATION_MODEL="${MACE_FOUNDATION_MODEL:?Set MACE_FOUNDATION_MODEL (a .model path, or small|medium|large)}"

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MACE_SEED must be a non-negative integer. Received: $SEED"
    exit 1
fi

# A path must exist; a bare small|medium|large name is downloaded by MACE and
# only works on a node with outbound network access.
if [[ "$FOUNDATION_MODEL" == */* || "$FOUNDATION_MODEL" == *.model ]]; then
    if [[ ! -s "$FOUNDATION_MODEL" ]]; then
        echo "ERROR: foundation model not found: $FOUNDATION_MODEL"
        exit 1
    fi
fi

MODEL_PREFIX="${MACE_MODEL_PREFIX:-SiN_TiN_TiO_periodic_mace}"
ENERGY_KEY="${MACE_ENERGY_KEY:-REF_energy}"
FORCES_KEY="${MACE_FORCES_KEY:-REF_forces}"
# "foundation" reuses the foundation model's atomic reference energies and is
# correct only for MP-compatible DFT (PBE / PBE+U on the MP settings). Use
# "average" if the reference DFT differs.
E0S="${MACE_E0S:-foundation}"
# Naive fine-tuning by default: specialise to this dataset, converge fast.
# Set MACE_MULTIHEADS=True (and MACE_PT_TRAIN_FILE for a non-MP foundation) to
# keep a replay head against the pretraining data.
MULTIHEADS="${MACE_MULTIHEADS:-False}"
PT_TRAIN_FILE="${MACE_PT_TRAIN_FILE:-}"
# The MP/MPA/OMAT foundation checkpoints are float64. Fine-tuning at float32
# leaves the loaded weights in float64 while the compiled e3nn layers run
# float32 -> "both inputs should have same dtype". Match the checkpoint.
DEFAULT_DTYPE="${MACE_DEFAULT_DTYPE:-float64}"
MAX_EPOCHS="${MACE_MAX_EPOCHS:-20}"
START_STAGE_TWO="${MACE_START_STAGE_TWO:-16}"
PATIENCE="${MACE_PATIENCE:-10}"
BATCH_SIZE="${MACE_BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${MACE_VALID_BATCH_SIZE:-4}"

MODEL_NAME="${MODEL_PREFIX}_ft_seed${SEED}"
RUN_DIR="$BASE_DIR/mace_finetune_committee/seed_${SEED}"
MODEL_DIR="$RUN_DIR/mace_model"
CHECKPOINTS_DIR="$RUN_DIR/checkpoints"
RESULTS_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"

mkdir -p "$MODEL_DIR" "$CHECKPOINTS_DIR" "$RESULTS_DIR" "$LOG_DIR"

if command -v flock >/dev/null 2>&1; then
    exec 9>"$RUN_DIR/.train.lock"
    if ! flock -n 9; then
        echo "ERROR: another training job is already active for seed $SEED ($RUN_DIR)"
        exit 1
    fi
fi

cd "$RUN_DIR"

module purge
set +u
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate /project/lgutsev/env/mace_env
set -u

echo "Activated Conda environment: $CONDA_PREFIX"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MACE_NUM_WORKERS="${MACE_NUM_WORKERS:-8}"
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/cuequivariance_ops/lib:$LD_LIBRARY_PATH"

MACE_TRAIN_BIN="$CONDA_PREFIX/bin/mace_run_train"
MACE_EVAL_BIN="$CONDA_PREFIX/bin/mace_eval_configs"
[[ -x "$MACE_TRAIN_BIN" ]] || { echo "ERROR: mace_run_train not found: $MACE_TRAIN_BIN"; exit 1; }

TRAIN_FILE="$BASE_DIR/train.extxyz"
VALID_FILE="$BASE_DIR/valid.extxyz"
TEST_FILE="$BASE_DIR/test.extxyz"
for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty file: $f"; exit 1; }
done

if [[ "$MULTIHEADS" == "True" && "$FOUNDATION_MODEL" == */* && -z "$PT_TRAIN_FILE" ]]; then
    echo "ERROR: MACE_MULTIHEADS=True with a local foundation model requires"
    echo "       MACE_PT_TRAIN_FILE (the pretraining replay data)."
    exit 1
fi

echo
echo "Fine-tune committee member:"
echo "  seed:             $SEED"
echo "  model name:       $MODEL_NAME"
echo "  foundation model: $FOUNDATION_MODEL"
echo "  default dtype:    $DEFAULT_DTYPE"
echo "  E0s:              $E0S"
echo "  multiheads:       $MULTIHEADS"
echo "  run dir:          $RUN_DIR"
echo

HELP_TXT="$("$MACE_TRAIN_BIN" --help 2>&1 || true)"
EXTRA_ARGS=()
grep -q -- "--save_cpu" <<< "$HELP_TXT" && EXTRA_ARGS+=(--save_cpu)
grep -q -- "--keep_checkpoints" <<< "$HELP_TXT" && EXTRA_ARGS+=(--keep_checkpoints)

STAGE_TWO_ARGS=()
if grep -q -- "--stage_two" <<< "$HELP_TXT"; then
    STAGE_TWO_ARGS+=(--stage_two --start_stage_two "$START_STAGE_TWO")
elif grep -q -- "--swa" <<< "$HELP_TXT"; then
    STAGE_TWO_ARGS+=(--swa --start_swa "$START_STAGE_TWO")
fi

FT_ARGS=(--foundation_model "$FOUNDATION_MODEL")
if grep -q -- "--multiheads_finetuning" <<< "$HELP_TXT"; then
    FT_ARGS+=(--multiheads_finetuning "$MULTIHEADS")
fi
if [[ -n "$PT_TRAIN_FILE" ]] && grep -q -- "--pt_train_file" <<< "$HELP_TXT"; then
    FT_ARGS+=(--pt_train_file "$PT_TRAIN_FILE")
fi

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export MASTER_PORT
MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo

GPU_LOG="$RUN_DIR/gpu_usage_${SLURM_JOB_ID}.log"
( echo "# MACE fine-tune GPU monitor job_id=$SLURM_JOB_ID seed=$SEED"; exec nvidia-smi dmon -s pucm -d 5 ) > "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!
cleanup() { local rc=$?; trap - EXIT INT TERM; [[ -n "${GPU_MON_PID:-}" ]] && { kill "$GPU_MON_PID" 2>/dev/null || true; wait "$GPU_MON_PID" 2>/dev/null || true; }; exit "$rc"; }
trap cleanup EXIT INT TERM

echo "Starting fine-tune for seed $SEED from $FOUNDATION_MODEL ..."
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
    --E0s "$E0S" \
    --batch_size "$BATCH_SIZE" \
    --valid_batch_size "$VALID_BATCH_SIZE" \
    --max_num_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --loss "weighted" \
    --error_table "PerAtomRMSE" \
    --default_dtype "$DEFAULT_DTYPE" \
    --ema --ema_decay 0.99 \
    --amsgrad \
    --device cuda \
    --distributed \
    --restart_latest \
    "${FT_ARGS[@]}" \
    "${STAGE_TWO_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo
echo "Fine-tune finished for seed $SEED."

if [[ -x "$MACE_EVAL_BIN" ]]; then
    MODEL_PATH=""
    for candidate in "$MODEL_DIR/${MODEL_NAME}_stagetwo.model" "$MODEL_DIR/${MODEL_NAME}.model"; do
        [[ -f "$candidate" ]] && { MODEL_PATH="$candidate"; break; }
    done
    if [[ -z "$MODEL_PATH" ]]; then
        MODEL_PATH="$(find "$MODEL_DIR" "$CHECKPOINTS_DIR" -type f -name "${MODEL_NAME}*.model" \
            ! -name "*_compiled.model" -printf "%T@ %p\n" 2>/dev/null | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
    fi
    if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]; then
        PREDICTIONS_FILE="$RUN_DIR/test_predictions_seed${SEED}.extxyz"
        "$MACE_EVAL_BIN" --configs "$TEST_FILE" --model "$MODEL_PATH" \
            --output "$PREDICTIONS_FILE" --energy_key "$ENERGY_KEY" --forces_key "$FORCES_KEY"
        echo "Wrote $PREDICTIONS_FILE"
    else
        echo "Could not locate the final .model file for seed $SEED; check $MODEL_DIR"
    fi
fi
