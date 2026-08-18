#!/bin/bash -l
#SBATCH --job-name=mace_omm_smoke
#SBATCH --account=loni_perovsk27
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=mace_omm_smoke_%j.out
#SBATCH --error=mace_omm_smoke_%j.err

set -euo pipefail

RUNTIME=${IFACE_MACE_RUNTIME:-/project/lgutsev/env/iface_mace_runtime}
SEED_OR_MODEL=${1:?Usage: sbatch $0 SEED_OR_MODEL STRUCTURE}
STRUCTURE=${2:?Usage: sbatch $0 SEED_OR_MODEL STRUCTURE}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# A Conda PyTorch build already carries its matching CUDA runtime.  Loading a
# different site PyTorch module can shadow it.  A site module is opt-in only.
if [[ -n "${IFACE_GPU_MODULE:-}" ]]; then
    module load "$IFACE_GPU_MODULE"
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset PYTHONPATH PYTHONHOME

nvidia-smi
"$RUNTIME/bin/python" "$SCRIPT_DIR/mace_openmm_smoke.py" "$SEED_OR_MODEL" "$STRUCTURE"
