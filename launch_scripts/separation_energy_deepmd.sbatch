#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=sep.deepmd
#SBATCH --output=separation_deepmd.%j.out
#SBATCH --error=separation_deepmd.%j.err

# Evaluate both SiN/TiN terminations with the four-member DPA-2 committee.
# InterfaceForge is imported from this checkout's pure-Python src/ tree; the
# module-provided Python/NumPy/PyTorch/CUDA stack remains otherwise untouched.

set -euo pipefail

CAMP="${SEPARATION_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch}}"
REPO_ROOT="${INTERFACEFORGE_ROOT:?Set INTERFACEFORGE_ROOT to the InterfaceForge checkout}"
OUTPUT="$CAMP/audit/separation/stages/deepmd"
N_TERM="$CAMP/adhesion/N_term_dft"
TI_TERM="$CAMP/adhesion/Ti_term_dft"
DEEPMD_MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"

for required in "$CAMP/campaign.yaml" "$N_TERM/manifest.json" "$TI_TERM/manifest.json"; do
    [[ -s "$required" ]] || { echo "ERROR: missing or empty file: $required" >&2; exit 2; }
done

MODEL_ARGS=()
for member in 000 001 002 003; do
    model_dir="$CAMP/models/deepmd/dpa2/model_${member}"
    frozen="$model_dir/frozen_model.pth"
    checkpoint="$model_dir/model.ckpt.pt"
    if [[ -s "$frozen" ]]; then
        model="$frozen"
    elif [[ -s "$checkpoint" ]]; then
        model="$checkpoint"
        echo "WARNING: member $member has no frozen_model.pth; using its training checkpoint for analysis: $checkpoint" >&2
    else
        echo "ERROR: DeePMD member $member has neither a frozen model nor a training checkpoint" >&2
        exit 2
    fi
    MODEL_ARGS+=(--deepmd-model "$model")
done

module purge
module load "$DEEPMD_MODULE"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DP_INFER_BATCH_SIZE=4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$CAMP"
echo "Python: $(command -v python)"
module list 2>&1
python -c "import interfaceforge, deepmd, ase; print('InterfaceForge:', interfaceforge.__file__); print('DeePMD:', deepmd.__file__); print('ASE:', ase.__file__)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

mkdir -p "$OUTPUT"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    python -u -m interfaceforge.separation_energy "$OUTPUT" \
    "interface/300K/MD_Vac/N_Term/SiN_TiN_N-term=$N_TERM" \
    "interface/300K/MD_Vac/Ti_Term/SiN-TiN-Ti-term=$TI_TERM" \
    "${MODEL_ARGS[@]}" \
    --json-only -c "$CAMP/campaign.yaml"

[[ -s "$OUTPUT/separation_energy.json" ]] || {
    echo "ERROR: DeePMD partial JSON was not created" >&2
    exit 3
}
echo "DeePMD partial: $OUTPUT/separation_energy.json"
