#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=uma.sintin
#SBATCH --output=uma_sintin.%j.out
#SBATCH --error=uma_sintin.%j.err

# Stock UMA/OMat zero-shot audit for the SiN/TiN/TiO Periodic_MLIPs campaign.
#
# Submit from the campaign root:
#   sbatch /project/lgutsev/git_develop/InterfaceForge/launch_scripts/uma_zero_shot_audit.sbatch
#
# Recommended first smoke:
#   sbatch --export=ALL,UMA_MAX_FRAMES=32 \
#     /project/lgutsev/git_develop/InterfaceForge/launch_scripts/uma_zero_shot_audit.sbatch
#
# The default model is uma-s-1p2p1 with task=omat. UMA_MODEL may also be a
# local .pt checkpoint path. Results go to audit/uma_zero_shot/<model>/.

set -euo pipefail

CAMP="${UMA_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the campaign root}}"
IFACE_ROOT="${INTERFACEFORGE_ROOT:-/project/lgutsev/git_develop/InterfaceForge}"
UMA_ENV="${UMA_ENV:-/project/lgutsev/env/lgutsev_dev}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p2p1}"
UMA_TASK="${UMA_TASK:-omat}"
UMA_DEVICE="${UMA_DEVICE:-cuda}"
UMA_INFERENCE_SETTINGS="${UMA_INFERENCE_SETTINGS:-batch}"
UMA_MAX_FRAMES="${UMA_MAX_FRAMES:-0}"
UMA_STRIDE="${UMA_STRIDE:-1}"
UMA_PROGRESS_EVERY="${UMA_PROGRESS_EVERY:-20}"
UMA_RESUME="${UMA_RESUME:-1}"

SCRIPT="$IFACE_ROOT/launch_scripts/uma_zero_shot_audit.py"

[[ -d "$CAMP" ]] || { echo "Missing campaign root: $CAMP" >&2; exit 2; }
CAMP="$(cd "$CAMP" && pwd -P)"
[[ -s "$SCRIPT" ]] || { echo "Missing audit script: $SCRIPT" >&2; exit 2; }
[[ "$UMA_MAX_FRAMES" =~ ^[0-9]+$ ]] || {
    echo "UMA_MAX_FRAMES must be a non-negative integer" >&2
    exit 2
}
[[ "$UMA_STRIDE" =~ ^[1-9][0-9]*$ ]] || {
    echo "UMA_STRIDE must be a positive integer" >&2
    exit 2
}
[[ "$UMA_PROGRESS_EVERY" =~ ^[1-9][0-9]*$ ]] || {
    echo "UMA_PROGRESS_EVERY must be a positive integer" >&2
    exit 2
}

module purge
set +u
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate "$UMA_ENV"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Stock UMA SiN/TiN zero-shot audit"
echo "  campaign:   $CAMP"
echo "  env:        $UMA_ENV"
echo "  model:      $UMA_MODEL"
echo "  task:       $UMA_TASK"
echo "  device:     $UMA_DEVICE"
echo "  inference:  $UMA_INFERENCE_SETTINGS"
echo "  max frames: $UMA_MAX_FRAMES"
echo "  stride:     $UMA_STRIDE"
echo "  HF_HOME:    $HF_HOME"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python - <<'PY'
import sys
import torch
import fairchem

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("fairchem", getattr(fairchem, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the UMA environment")
PY

ARGS=(
    "$CAMP"
    --model "$UMA_MODEL"
    --task "$UMA_TASK"
    --device "$UMA_DEVICE"
    --inference-settings "$UMA_INFERENCE_SETTINGS"
    --max-frames "$UMA_MAX_FRAMES"
    --stride "$UMA_STRIDE"
    --progress-every "$UMA_PROGRESS_EVERY"
)

if [[ -n "${UMA_DATASET:-}" ]]; then
    ARGS+=(--dataset "$UMA_DATASET")
fi
if [[ -n "${UMA_OUTPUT:-}" ]]; then
    ARGS+=(--output "$UMA_OUTPUT")
fi
if [[ "$UMA_RESUME" == "1" ]]; then
    ARGS+=(--resume)
fi

python "$SCRIPT" "${ARGS[@]}"
