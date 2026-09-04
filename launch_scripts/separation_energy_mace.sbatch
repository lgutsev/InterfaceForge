#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=sep.mace
#SBATCH --output=separation_mace.%j.out
#SBATCH --error=separation_mace.%j.err

# Evaluate both SiN/TiN terminations with the four-member MACE committee.
# Submit from Periodic_MLIPs, or set SEPARATION_CAMPAIGN_ROOT explicitly.

set -eo pipefail

CAMP="${SEPARATION_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch}}"
REPO_ROOT="${INTERFACEFORGE_ROOT:?Set INTERFACEFORGE_ROOT to the InterfaceForge checkout}"
OUTPUT="$CAMP/audit/separation/stages/mace"
N_TERM="$CAMP/adhesion/N_term_dft"
TI_TERM="$CAMP/adhesion/Ti_term_dft"

for required in "$CAMP/campaign.yaml" "$N_TERM/manifest.json" "$TI_TERM/manifest.json"; do
    [[ -s "$required" ]] || { echo "ERROR: missing or empty file: $required" >&2; exit 2; }
done

module purge
set +u
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate /project/lgutsev/env/mace_env
set -u

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true

cd "$CAMP"
python -c "import interfaceforge, mace, ase; print('InterfaceForge:', interfaceforge.__file__); print('MACE:', mace.__file__); print('ASE:', ase.__file__)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

MODEL_ARGS=()
for seed in 11 23 37 53; do
    mapfile -t matches < <(
        find "$CAMP/models/mace_committee/seed_${seed}" -maxdepth 3 -type f \
            -name '*_stagetwo.model' -print 2>/dev/null | sort
    )
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "ERROR: expected one stage-two MACE model for seed $seed; found ${#matches[@]}" >&2
        printf '  %s\n' "${matches[@]}" >&2
        exit 2
    fi
    MODEL_ARGS+=(--mace-model "${matches[0]}")
done

mkdir -p "$OUTPUT"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    python -u -m interfaceforge.separation_energy "$OUTPUT" \
    "interface/300K/MD_Vac/N_Term/SiN_TiN_N-term=$N_TERM" \
    "interface/300K/MD_Vac/Ti_Term/SiN-TiN-Ti-term=$TI_TERM" \
    "${MODEL_ARGS[@]}" \
    --device cuda --json-only -c "$CAMP/campaign.yaml"

[[ -s "$OUTPUT/separation_energy.json" ]] || {
    echo "ERROR: MACE partial JSON was not created" >&2
    exit 3
}
echo "MACE partial: $OUTPUT/separation_energy.json"
