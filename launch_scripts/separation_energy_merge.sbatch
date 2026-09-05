#!/usr/bin/env bash
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=sep.merge
#SBATCH --output=separation_merge.%j.out
#SBATCH --error=separation_merge.%j.err

# Merge the isolated MACE and DeePMD partials and render the final reports.
# Normally submitted by submit_separation_energy.sh with afterok dependencies.

set -euo pipefail

CAMP="${SEPARATION_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch}}"
REPO_ROOT="${INTERFACEFORGE_ROOT:?Set INTERFACEFORGE_ROOT to the InterfaceForge checkout}"
CAMP="$(cd -- "$CAMP" && pwd -P)"
IFACE_PYTHON="${INTERFACEFORGE_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}"
OUTPUT="${SEPARATION_RUN_DIR:-$CAMP/audit/separation}"
MACE_JSON="$OUTPUT/stages/mace/separation_energy.json"
DEEPMD_JSON="$OUTPUT/stages/deepmd/separation_energy.json"

[[ -x "$IFACE_PYTHON" ]] || { echo "ERROR: Python is not executable: $IFACE_PYTHON" >&2; exit 2; }
for required in "$CAMP/campaign.yaml" "$MACE_JSON" "$DEEPMD_JSON"; do
    [[ -s "$required" ]] || { echo "ERROR: missing or empty file: $required" >&2; exit 2; }
done

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$CAMP"
"$IFACE_PYTHON" -c "import interfaceforge, matplotlib; print('InterfaceForge:', interfaceforge.__file__); print('Matplotlib:', matplotlib.__file__)"

srun --ntasks=1 --cpus-per-task=1 \
    "$IFACE_PYTHON" -u -m interfaceforge.separation_energy "$OUTPUT" \
    --merge-json "$MACE_JSON" \
    --merge-json "$DEEPMD_JSON" \
    -c "$CAMP/campaign.yaml"

for artifact in json csv md png svg pdf; do
    [[ -s "$OUTPUT/separation_energy.$artifact" ]] || {
        echo "ERROR: final artifact is missing: $OUTPUT/separation_energy.$artifact" >&2
        exit 3
    }
done
echo "Final separation-energy report: $OUTPUT"
