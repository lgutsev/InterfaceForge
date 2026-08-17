#!/bin/bash -l
#SBATCH --job-name=setup_ai2kit
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=setup_ai2kit_%j.out
#SBATCH --error=setup_ai2kit_%j.err

set -euo pipefail

export CONDA_ALWAYS_YES=true
export CONDA_OFFLINE=false
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

TARGET=/project/lgutsev/env/iface_ai2kit_controller
AI2KIT_VERSION=1.0.9
OMB_VERSION=0.7.5

source "$(conda info --base)/etc/profile.d/conda.sh"

check_environment() {
    [[ -x "$TARGET/bin/python" ]] || return 1

    "$TARGET/bin/python" - <<PY
from importlib.metadata import version

actual_ai2kit = version("ai2-kit")
actual_omb = version("oh-my-batch")
print("ai2-kit:", actual_ai2kit)
print("oh-my-batch:", actual_omb)
if actual_ai2kit != "$AI2KIT_VERSION" or actual_omb != "$OMB_VERSION":
    raise SystemExit(1)
PY

    "$TARGET/bin/ai2-kit" --help >/dev/null
    "$TARGET/bin/omb" --help >/dev/null
    "$TARGET/bin/python" -m pip check
}

echo "Target environment: $TARGET"
echo "Conda offline setting with override:"
CONDA_OFFLINE=false conda config --show offline

if check_environment; then
    echo "Existing AI2-Kit controller environment is healthy; nothing to rebuild."
    exit 0
fi

if [[ -e "$TARGET" ]]; then
    BACKUP="${TARGET}.incomplete-$(date +%Y%m%d-%H%M%S)"
    echo "Existing environment failed validation; moving it to $BACKUP"
    mv "$TARGET" "$BACKUP"
fi

echo "Creating clean controller environment..."
CONDA_OFFLINE=false conda create \
    --prefix "$TARGET" \
    --override-channels \
    --channel conda-forge \
    python=3.11 pip \
    --yes

CONDA_OFFLINE=false "$TARGET/bin/python" -m pip install \
    --no-input \
    --upgrade pip setuptools wheel

CONDA_OFFLINE=false "$TARGET/bin/python" -m pip install \
    --no-input \
    "ai2-kit==$AI2KIT_VERSION" \
    "oh-my-batch==$OMB_VERSION"

echo "Running final validation..."
check_environment

echo "AI2-Kit controller environment setup completed successfully."
