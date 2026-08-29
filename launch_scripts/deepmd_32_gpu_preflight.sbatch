#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=deepmd.32.preflight
#SBATCH --output=deepmd.32.preflight.%j.out
#SBATCH --error=deepmd.32.preflight.%j.err

# Verify LONI's native DeePMD-kit 3.2 GPU module before generating or launching
# a production committee. This job intentionally makes no filesystem changes.

set -euo pipefail

DEEPMD_MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"

module purge
module load "$DEEPMD_MODULE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loaded module: $DEEPMD_MODULE"
module list 2>&1
echo

DP_BIN="$(command -v dp || true)"
[[ -n "$DP_BIN" ]] || { echo "ERROR: dp is not available after module load"; exit 2; }

echo "dp executable: $DP_BIN"
DP_VERSION="$($DP_BIN --version 2>&1)"
echo "$DP_VERSION"
grep -Eq '3\.2\.0|3\.2' <<<"$DP_VERSION" || {
    echo "ERROR: expected DeePMD-kit 3.2 from $DEEPMD_MODULE"
    exit 2
}
echo

echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo

python - <<'PY'
import deepmd
import torch

print("deepmd.__version__:", deepmd.__version__)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated GPU")
print("torch GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Checking DeePMD 3.2 PyTorch commands..."
"$DP_BIN" --pt train --help >/dev/null
"$DP_BIN" --pt freeze --help >/dev/null
"$DP_BIN" --pt test --help >/dev/null
"$DP_BIN" --pt model-devi --help >/dev/null

echo
echo "Optional LAMMPS/DeePMD inventory:"
LMP_BIN="$(command -v lmp || command -v lmp_mpi || command -v lmp_serial || true)"
if [[ -n "$LMP_BIN" ]]; then
    echo "LAMMPS executable: $LMP_BIN"
    if "$LMP_BIN" -h 2>&1 | grep -qi deepmd; then
        echo "LAMMPS reports DeePMD pair-style support."
    else
        echo "WARNING: LAMMPS was found but its help does not report DeePMD support."
    fi
else
    echo "LAMMPS executable not supplied by this training module (not a preflight failure)."
fi

echo
echo "DeepMD-kit 3.2 PyTorch GPU preflight passed."
