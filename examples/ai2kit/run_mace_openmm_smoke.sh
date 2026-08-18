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
DEV_ENV=${IFACE_DEV_ENV:-/project/lgutsev/env/lgutsev_dev}

find_smoke_script() {
    local candidate env_python package_file parent
    local -a checked=()
    local -a python_candidates=()

    # This permits copying only the launcher into a calculation directory.
    for candidate in \
        "$PWD/mace_openmm_smoke.py" \
        "$SCRIPT_DIR/mace_openmm_smoke.py"; do
        checked+=("$candidate")
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if [[ -n "${IFACE_REPO_ROOT:-}" ]]; then
        candidate="$IFACE_REPO_ROOT/examples/ai2kit/mace_openmm_smoke.py"
        checked+=("$candidate")
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    # An editable InterfaceForge install exposes its checkout through
    # interfaceforge.__file__.  Check the submitting Conda environment first,
    # then the configured development environment.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        python_candidates+=("$CONDA_PREFIX/bin/python")
    fi
    python_candidates+=("$DEV_ENV/bin/python")

    for env_python in "${python_candidates[@]}"; do
        [[ -x "$env_python" ]] || continue
        package_file=$(
            "$env_python" -c \
                'import pathlib, interfaceforge; print(pathlib.Path(interfaceforge.__file__).resolve())' \
                2>/dev/null
        ) || continue
        parent=$(dirname -- "$package_file")
        while [[ "$parent" != "/" ]]; do
            candidate="$parent/examples/ai2kit/mace_openmm_smoke.py"
            checked+=("$candidate")
            if [[ -f "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
            parent=$(dirname -- "$parent")
        done
    done

    printf 'Could not locate mace_openmm_smoke.py. Checked:\n' >&2
    printf '  %s\n' "${checked[@]}" >&2
    printf 'Set IFACE_REPO_ROOT or IFACE_DEV_ENV if the checkout moved.\n' >&2
    return 1
}

SMOKE_SCRIPT=$(find_smoke_script)
echo "Smoke-test driver: $SMOKE_SCRIPT"

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
"$RUNTIME/bin/python" "$SMOKE_SCRIPT" "$SEED_OR_MODEL" "$STRUCTURE"
