#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --job-name=freeze.dpa2
#SBATCH --output=freeze_dpa2.%A_%a.out
#SBATCH --error=freeze_dpa2.%A_%a.err

# Export missing DPA-2 training checkpoints for downstream deployment. Existing
# non-empty frozen models are left untouched.

set -euo pipefail

CAMP="${SEPARATION_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script from Periodic_MLIPs}}"
DEEPMD_MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"
printf -v member '%03d' "$TASK_ID"

model_dir="$CAMP/models/deepmd/dpa2/model_${member}"
checkpoint="$model_dir/model.ckpt.pt"
frozen="$model_dir/frozen_model.pth"

[[ -d "$model_dir" ]] || { echo "ERROR: missing model directory: $model_dir" >&2; exit 2; }
[[ -s "$checkpoint" ]] || { echo "ERROR: missing or empty checkpoint: $checkpoint" >&2; exit 2; }
if [[ -s "$frozen" ]]; then
    echo "Member $member already has a frozen model; leaving it unchanged: $frozen"
    exit 0
fi

module purge
module load "$DEEPMD_MODULE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$model_dir"
echo "Freezing member $member with $(command -v dp)"
module list 2>&1
dp --pt freeze -c "$checkpoint" -o "$frozen"
[[ -s "$frozen" ]] || { echo "ERROR: freeze produced no model: $frozen" >&2; exit 3; }

# Load the artifact through the same inference API used by ASE before declaring
# the export successful.
python - "$frozen" <<'PY'
import sys
from deepmd.infer import DeepPot

model = DeepPot(sys.argv[1])
print("Frozen model type map:", model.get_type_map())
PY
echo "Frozen member $member: $frozen"
