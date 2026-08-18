#!/bin/bash -l
#SBATCH --job-name=setup_mace_rt
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=setup_mace_rt_%j.out
#SBATCH --error=setup_mace_rt_%j.err

set -euo pipefail

export CONDA_ALWAYS_YES=true
export CONDA_OFFLINE=false
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

SOURCE=${IFACE_MACE_SOURCE:-/project/lgutsev/env/mace_env}
TARGET=${IFACE_MACE_RUNTIME:-/project/lgutsev/env/iface_mace_runtime}
OPENMM_VERSION=${IFACE_OPENMM_VERSION:-8.5.2}
OPENMMML_VERSION=${IFACE_OPENMMML_VERSION:-1.7}

source "$(conda info --base)/etc/profile.d/conda.sh"

check_environment() {
    [[ -f "$TARGET/conda-meta/history" && -x "$TARGET/bin/python" ]] || return 1
    "$TARGET/bin/python" - <<PY
from importlib.metadata import version
import ase
import mace
import openmm
from openmmml import MLPotential

expected = {"OpenMM": "$OPENMM_VERSION", "OpenMM-ML": "$OPENMMML_VERSION"}
actual = {"OpenMM": openmm.__version__, "OpenMM-ML": version("openmm-ml")}
print("ASE:", ase.__version__)
print("MACE:", version("mace-torch"))
print("OpenMM:", actual["OpenMM"])
print("OpenMM-ML:", actual["OpenMM-ML"])
print("MLPotential:", MLPotential.__name__)
if actual != expected:
    raise SystemExit(f"version mismatch: expected {expected}, got {actual}")
PY
    "$TARGET/bin/python" -m pip check
}

echo "Source environment: $SOURCE"
echo "Runtime environment: $TARGET"

if check_environment; then
    echo "Existing MACE/OpenMM runtime is healthy; nothing to rebuild."
    exit 0
fi

[[ -f "$SOURCE/conda-meta/history" ]] || {
    echo "Source is not a complete Conda environment: $SOURCE" >&2
    exit 1
}

if [[ -e "$TARGET" ]]; then
    BACKUP="${TARGET}.incomplete-$(date +%Y%m%d-%H%M%S)"
    echo "Moving incomplete runtime to $BACKUP"
    mv "$TARGET" "$BACKUP"
fi

# Cloning can take tens of minutes on a shared filesystem.  Do not cancel it
# merely because conda produces no output while copying many small files.
CONDA_OFFLINE=false conda create --prefix "$TARGET" --clone "$SOURCE" --yes

# Install OpenMM separately, then OpenMM-ML.  Installing openmm and openmmml
# under the same pip command can make pip choose incompatible OpenMM releases.
CONDA_OFFLINE=false conda install --prefix "$TARGET" \
    --channel conda-forge --freeze-installed "openmm=$OPENMM_VERSION" --yes
"$TARGET/bin/python" -m pip install --no-input --upgrade-strategy only-if-needed \
    "openmmml==$OPENMMML_VERSION"

check_environment
echo "MACE/OpenMM runtime setup completed successfully."
