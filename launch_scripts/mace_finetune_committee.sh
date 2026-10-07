#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J mace.ft.committee
#SBATCH -o mace.ft.committee.%j.out
#SBATCH -e mace.ft.committee.%j.err

# Single-GPU by design. Fine-tuning a periodic-table foundation model on a
# 4-element dataset leaves the linear-layer rows for absent elements without a
# gradient; DDP (find_unused_parameters=False) then aborts the first step with
# "parameters that were not used in producing loss". MPA-0 medium fine-tunes
# comfortably on one A100.

# Fine-tune one member of a MACE committee from a foundation model, on the same
# fixed canonical split used by mace_train_committee.sh. Submit four seeds:
#
#   for seed in 11 23 37 53; do
#       sbatch --export=ALL,MACE_SEED="$seed",MACE_FOUNDATION_MODEL=mace-mpa-0-medium.model \
#           mace_finetune_committee.sh
#   done
#
# Foundation checkpoints live in the shared store
#   $MLIP_FOUNDATION_ROOT (default
#   /ddnB/project/ramu/lgutsev/MLIP_PROJECT_STORAGE/MLIP_Foundational_Models)
# with one subdirectory per model family (mace/, DPA-2.4-7M/, DPA-3.1-3M/, UMA/).
# MACE_FOUNDATION_MODEL may be an absolute .model path, a file name inside
# $MLIP_FOUNDATION_ROOT/mace, a directory holding exactly one .model, or a bare
# small|medium|large name. Unset, it is $MLIP_FOUNDATION_ROOT/mace, which must
# then hold exactly one .model.
#
# The architecture (r_max, channels, max_L, correlation, interactions) is
# inherited from the foundation model and cannot be set here. Output lands in
# mace_finetune_committee/seed_<seed>/ so it never collides with the
# from-scratch committee in mace_committee/.

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch}"
SEED="${MACE_SEED:?Set MACE_SEED when submitting this job}"
FOUNDATION_ROOT="${MLIP_FOUNDATION_ROOT:-/ddnB/project/ramu/lgutsev/MLIP_PROJECT_STORAGE/MLIP_Foundational_Models}"
FOUNDATION_MODEL="${MACE_FOUNDATION_MODEL:-$FOUNDATION_ROOT/mace}"

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MACE_SEED must be a non-negative integer. Received: $SEED"
    exit 1
fi

# A bare small|medium|large name is downloaded by MACE and only works on a node
# with outbound network access; anything else must resolve to a local file.
case "$FOUNDATION_MODEL" in
    small|medium|large) ;;
    *)
        if [[ "$FOUNDATION_MODEL" != */* ]]; then
            FOUNDATION_MODEL="$FOUNDATION_ROOT/mace/$FOUNDATION_MODEL"
        fi
        if [[ -d "$FOUNDATION_MODEL" ]]; then
            mapfile -t FOUNDATION_CANDIDATES < <(find "$FOUNDATION_MODEL" -maxdepth 1 -type f \
                -name '*.model' ! -name '*_compiled.model' | sort)
            if [[ "${#FOUNDATION_CANDIDATES[@]}" -ne 1 ]]; then
                echo "ERROR: $FOUNDATION_MODEL holds ${#FOUNDATION_CANDIDATES[@]} .model files;"
                echo "       set MACE_FOUNDATION_MODEL to one of:"
                printf '         %s\n' "${FOUNDATION_CANDIDATES[@]##*/}"
                exit 1
            fi
            FOUNDATION_MODEL="${FOUNDATION_CANDIDATES[0]}"
        fi
        if [[ ! -s "$FOUNDATION_MODEL" ]]; then
            echo "ERROR: foundation model not found: $FOUNDATION_MODEL"
            exit 1
        fi
        ;;
esac

MODEL_PREFIX="${MACE_MODEL_PREFIX:-SiN_TiN_TiO_periodic_mace}"
ENERGY_KEY="${MACE_ENERGY_KEY:-REF_energy}"
FORCES_KEY="${MACE_FORCES_KEY:-REF_forces}"
LOSS="${MACE_LOSS:-weighted}"
VIRIALS_KEY="${MACE_VIRIALS_KEY:-}"
VIRIALS_WEIGHT="${MACE_VIRIALS_WEIGHT:-1.0}"
STRESS_KEY="${MACE_STRESS_KEY:-}"
STRESS_WEIGHT="${MACE_STRESS_WEIGHT:-1.0}"
ENERGY_WEIGHT="${MACE_ENERGY_WEIGHT:-}"
FORCES_WEIGHT="${MACE_FORCES_WEIGHT:-}"
ERROR_TABLE="${MACE_ERROR_TABLE:-PerAtomRMSE}"

if [[ -n "$VIRIALS_KEY" && -n "$STRESS_KEY" ]]; then
    echo "ERROR: set only one of MACE_VIRIALS_KEY or MACE_STRESS_KEY." >&2
    exit 1
fi
if [[ -n "$VIRIALS_KEY" && "$LOSS" != "virials" ]]; then
    echo "ERROR: MACE_VIRIALS_KEY requires MACE_LOSS=virials so the label enters the loss." >&2
    exit 1
fi
if [[ -n "$STRESS_KEY" && "$LOSS" != "stress" ]]; then
    echo "ERROR: MACE_STRESS_KEY requires MACE_LOSS=stress so the label enters the loss." >&2
    exit 1
fi
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
# Unset -> MACE default (0.01). Lower to 1e-3 if the first epochs make forces
# worse instead of better (the foundation prior being overwritten too fast).
LR="${MACE_LR:-}"
MAX_EPOCHS="${MACE_MAX_EPOCHS:-20}"
START_STAGE_TWO="${MACE_START_STAGE_TWO:-16}"
USE_STAGE_TWO="${MACE_USE_STAGE_TWO:-True}"
PATIENCE="${MACE_PATIENCE:-10}"
BATCH_SIZE="${MACE_BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${MACE_VALID_BATCH_SIZE:-4}"
EMA_DECAY="${MACE_EMA_DECAY:-0.99}"
WEIGHT_DECAY="${MACE_WEIGHT_DECAY:-}"
CLIP_GRAD="${MACE_CLIP_GRAD:-}"

if [[ "$USE_STAGE_TWO" != "True" && "$USE_STAGE_TWO" != "False" ]]; then
    echo "ERROR: MACE_USE_STAGE_TWO must be True or False." >&2
    exit 1
fi

# Resolve both the current campaign layout and the legacy layout used by the
# original standalone committee scripts. Explicit overrides may be absolute or
# relative to the directory from which sbatch was invoked.
dataset_is_complete() {
    local directory="$1"
    [[ -s "$directory/train.extxyz" && \
       -s "$directory/valid.extxyz" && \
       -s "$directory/test.extxyz" ]]
}

if [[ -n "${MACE_DATASET_DIR:-}" ]]; then
    if [[ "$MACE_DATASET_DIR" = /* ]]; then
        DATASET_DIR="$MACE_DATASET_DIR"
    else
        DATASET_DIR="$SUBMIT_DIR/$MACE_DATASET_DIR"
    fi
elif dataset_is_complete "$SUBMIT_DIR/datasets/canonical"; then
    DATASET_DIR="$SUBMIT_DIR/datasets/canonical"
elif dataset_is_complete "$SUBMIT_DIR"; then
    DATASET_DIR="$SUBMIT_DIR"
elif dataset_is_complete "$SUBMIT_DIR/../../datasets/canonical"; then
    DATASET_DIR="$SUBMIT_DIR/../../datasets/canonical"
else
    echo "ERROR: could not locate a complete MACE train/valid/test split." >&2
    echo "Checked:" >&2
    echo "  $SUBMIT_DIR/datasets/canonical" >&2
    echo "  $SUBMIT_DIR" >&2
    echo "  $SUBMIT_DIR/../../datasets/canonical" >&2
    echo "Set MACE_DATASET_DIR to override dataset discovery." >&2
    exit 1
fi

if ! dataset_is_complete "$DATASET_DIR"; then
    echo "ERROR: MACE_DATASET_DIR is not a complete train/valid/test split: $DATASET_DIR" >&2
    exit 1
fi
DATASET_DIR="$(cd "$DATASET_DIR" && pwd -P)"

if [[ -n "${MACE_OUTPUT_ROOT:-}" ]]; then
    if [[ "$MACE_OUTPUT_ROOT" = /* ]]; then
        OUTPUT_ROOT="$MACE_OUTPUT_ROOT"
    else
        OUTPUT_ROOT="$SUBMIT_DIR/$MACE_OUTPUT_ROOT"
    fi
elif [[ -d "$SUBMIT_DIR/datasets/canonical" ]]; then
    # Modern campaign root: keep artifacts where iface mlip-progress expects.
    OUTPUT_ROOT="$SUBMIT_DIR/models/mace_committee_520eV"
else
    # Legacy invocation from the directory containing the split/model roots.
    OUTPUT_ROOT="$SUBMIT_DIR"
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd -P)"

MODEL_NAME="${MODEL_PREFIX}_ft_seed${SEED}"
RUN_DIR="$OUTPUT_ROOT/mace_finetune_committee/seed_${SEED}"
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

TRAIN_FILE="$DATASET_DIR/train.extxyz"
VALID_FILE="$DATASET_DIR/valid.extxyz"
TEST_FILE="$DATASET_DIR/test.extxyz"
for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty file: $f"; exit 1; }
done

if [[ -n "$VIRIALS_KEY" || -n "$STRESS_KEY" ]]; then
    DATASET_FILES="$TRAIN_FILE:$VALID_FILE:$TEST_FILE" \
    TENSOR_KEY="${VIRIALS_KEY:-$STRESS_KEY}" \
    python - <<'PY'
import os

import numpy as np
from ase.io import read

key = os.environ["TENSOR_KEY"]
for fname in os.environ["DATASET_FILES"].split(os.pathsep):
    atoms = read(fname, index=0)
    if key not in atoms.info:
        raise RuntimeError(f"{fname}: missing tensor label {key!r} in atoms.info")
    value = np.asarray(atoms.info[key], dtype=float)
    if value.size not in {6, 9} or not np.isfinite(value).all():
        raise RuntimeError(
            f"{fname}: {key!r} must contain 6 or 9 finite tensor components; "
            f"got shape {value.shape}"
        )
    print(f"{fname}: {key} shape={value.shape}")
PY
fi

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
echo "  loss:             $LOSS"
echo "  virials key:      ${VIRIALS_KEY:-disabled}"
echo "  stress key:       ${STRESS_KEY:-disabled}"
echo "  dataset dir:      $DATASET_DIR"
echo "  output root:      $OUTPUT_ROOT"
echo "  run dir:          $RUN_DIR"
echo

HELP_TXT="$("$MACE_TRAIN_BIN" --help 2>&1 || true)"
EXTRA_ARGS=()
grep -q -- "--save_cpu" <<< "$HELP_TXT" && EXTRA_ARGS+=(--save_cpu)
grep -q -- "--keep_checkpoints" <<< "$HELP_TXT" && EXTRA_ARGS+=(--keep_checkpoints)

STAGE_TWO_ARGS=()
if [[ "$USE_STAGE_TWO" == "True" ]]; then
    if grep -q -- "--stage_two" <<< "$HELP_TXT"; then
        STAGE_TWO_ARGS+=(--stage_two --start_stage_two "$START_STAGE_TWO")
    elif grep -q -- "--swa" <<< "$HELP_TXT"; then
        STAGE_TWO_ARGS+=(--swa --start_swa "$START_STAGE_TWO")
    fi
fi

OPTIM_ARGS=(--ema --ema_decay "$EMA_DECAY")
if [[ -n "$WEIGHT_DECAY" ]]; then
    grep -q -- "--weight_decay" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --weight_decay." >&2
        exit 2
    }
    OPTIM_ARGS+=(--weight_decay "$WEIGHT_DECAY")
fi
if [[ -n "$CLIP_GRAD" ]]; then
    grep -q -- "--clip_grad" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --clip_grad." >&2
        exit 2
    }
    OPTIM_ARGS+=(--clip_grad "$CLIP_GRAD")
fi

LABEL_ARGS=()
if [[ -n "$VIRIALS_KEY" ]]; then
    grep -q -- "--virials_key" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --virials_key; refusing an E/F-only fallback." >&2
        exit 2
    }
    grep -q -- "--virials_weight" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --virials_weight; refusing an E/F-only fallback." >&2
        exit 2
    }
    LABEL_ARGS+=(--virials_key "$VIRIALS_KEY" --virials_weight "$VIRIALS_WEIGHT")
elif [[ -n "$STRESS_KEY" ]]; then
    grep -q -- "--stress_key" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --stress_key; refusing an E/F-only fallback." >&2
        exit 2
    }
    grep -q -- "--stress_weight" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --stress_weight; refusing an E/F-only fallback." >&2
        exit 2
    }
    LABEL_ARGS+=(--stress_key "$STRESS_KEY" --stress_weight "$STRESS_WEIGHT")
fi

WEIGHT_ARGS=()
if [[ -n "$ENERGY_WEIGHT" ]]; then
    grep -q -- "--energy_weight" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --energy_weight." >&2
        exit 2
    }
    WEIGHT_ARGS+=(--energy_weight "$ENERGY_WEIGHT")
fi
if [[ -n "$FORCES_WEIGHT" ]]; then
    grep -q -- "--forces_weight" <<< "$HELP_TXT" || {
        echo "ERROR: installed MACE lacks --forces_weight." >&2
        exit 2
    }
    WEIGHT_ARGS+=(--forces_weight "$FORCES_WEIGHT")
fi

FT_ARGS=(--foundation_model "$FOUNDATION_MODEL")
if grep -q -- "--multiheads_finetuning" <<< "$HELP_TXT"; then
    FT_ARGS+=(--multiheads_finetuning "$MULTIHEADS")
fi
[[ -n "$LR" ]] && FT_ARGS+=(--lr "$LR")
if [[ -n "$PT_TRAIN_FILE" ]] && grep -q -- "--pt_train_file" <<< "$HELP_TXT"; then
    FT_ARGS+=(--pt_train_file "$PT_TRAIN_FILE")
fi

if [[ "${MACE_PREFLIGHT_ONLY:-False}" == "True" ]]; then
    echo "MACE fine-tune preflight succeeded"
    echo "  submit dir:      $SUBMIT_DIR"
    echo "  dataset dir:     $DATASET_DIR"
    echo "  output root:     $OUTPUT_ROOT"
    echo "  run dir:         $RUN_DIR"
    echo "  foundation:      $FOUNDATION_MODEL"
    echo "  loss:            $LOSS"
    echo "  virials key:     ${VIRIALS_KEY:-disabled}"
    echo "  stress key:      ${STRESS_KEY:-disabled}"
    echo "  energy weight:   ${ENERGY_WEIGHT:-MACE default}"
    echo "  forces weight:   ${FORCES_WEIGHT:-MACE default}"
    echo "  virials weight:  $VIRIALS_WEIGHT"
    echo "  stress weight:   $STRESS_WEIGHT"
    echo "  stage two:       $USE_STAGE_TWO"
    echo "  ema decay:       $EMA_DECAY"
    echo "  weight decay:    ${WEIGHT_DECAY:-MACE default}"
    echo "  clip grad:       ${CLIP_GRAD:-MACE default}"
    exit 0
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo

GPU_LOG="$RUN_DIR/gpu_usage_${SLURM_JOB_ID}.log"
( echo "# MACE fine-tune GPU monitor job_id=$SLURM_JOB_ID seed=$SEED"; exec nvidia-smi dmon -s pucm -d 5 ) > "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!
cleanup() { local rc=$?; trap - EXIT INT TERM; [[ -n "${GPU_MON_PID:-}" ]] && { kill "$GPU_MON_PID" 2>/dev/null || true; wait "$GPU_MON_PID" 2>/dev/null || true; }; exit "$rc"; }
trap cleanup EXIT INT TERM

echo "Starting fine-tune for seed $SEED from $FOUNDATION_MODEL ..."
echo

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
    --loss "$LOSS" \
    --error_table "$ERROR_TABLE" \
    --default_dtype "$DEFAULT_DTYPE" \
    --amsgrad \
    --device cuda \
    --restart_latest \
    "${LABEL_ARGS[@]}" \
    "${WEIGHT_ARGS[@]}" \
    "${OPTIM_ARGS[@]}" \
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
