#!/usr/bin/env bash
#SBATCH --job-name=iface_mapped_collect
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=iface_mapped_collect_%j.out
#SBATCH --error=iface_mapped_collect_%j.err

# Submit with:
#   sbatch launch_scripts/collect_mapped_leaf_single.sbatch /absolute/path/campaign.yaml
#
# To intentionally rebuild existing MACE and DeePMD outputs:
#   sbatch launch_scripts/collect_mapped_leaf_single.sbatch \
#     /absolute/path/campaign.yaml --force-datasets

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 /absolute/path/campaign.yaml [--force-datasets]" >&2
    exit 2
fi

CONFIG=$1
shift

if [[ ! -f "$CONFIG" ]]; then
    echo "Mapped campaign configuration does not exist: $CONFIG" >&2
    exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IFACE_PYTHON="${INTERFACEFORGE_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}"

if [[ ! -x "$IFACE_PYTHON" ]]; then
    echo "InterfaceForge Python is not executable: $IFACE_PYTHON" >&2
    echo "Override it with: sbatch --export=ALL,INTERFACEFORGE_PYTHON=/path/to/python ..." >&2
    exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "SLURM job: ${SLURM_JOB_ID:-not-set}"
echo "Python: $IFACE_PYTHON"
echo "Configuration: $CONFIG"
echo "Additional arguments: $*"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    "$IFACE_PYTHON" -u -m interfaceforge.mapped_collect \
    "$CONFIG" --execute --collect "$@"

echo "Finished: $(date --iso-8601=seconds)"
