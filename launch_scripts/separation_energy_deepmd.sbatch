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
source "$REPO_ROOT/launch_scripts/separation_energy_common.sh"
BOOTSTRAP="$REPO_ROOT/launch_scripts/run_interfaceforge_module.py"
sep_file "$BOOTSTRAP"
sep_campaign
OUTPUT="${SEPARATION_RUN_DIR:-$CAMP/audit/separation}/stages/deepmd"
N_TERM="$CAMP/adhesion/N_term_dft"
TI_TERM="$CAMP/adhesion/Ti_term_dft"
DEEPMD_MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"

if [[ -n "${SEPARATION_RUN_DIR:-}" ]]; then
    sep_read_models "$SEPARATION_RUN_DIR/deepmd_models.txt"
else
    sep_deepmd
    SELECTED_MODELS=("${DEEPMD_MODELS[@]}")
fi
MODEL_ARGS=()
for model in "${SELECTED_MODELS[@]}"; do
    [[ "${model##*/}" == frozen_model.pth ]] || {
        echo "ERROR: pinned DeePMD model is a training checkpoint, not a frozen inference export: $model" >&2
        echo 'Create the frozen models and submit a new separation-energy run.' >&2
        exit 2
    }
    MODEL_ARGS+=(--deepmd-model "$model")
done

module purge
module load "$DEEPMD_MODULE"

# The deepmd-kit module provides Python from inside an Apptainer/Singularity
# image ("Mounting image with FUSE" in the job log). Keep PYTHONPATH propagation
# as a convenience, but do not depend on it: the final command executes the
# repository-local bootstrap by path and that bootstrap inserts src/ itself.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export APPTAINERENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
BIND_ROOTS="$(sep_bind_roots "$REPO_ROOT" "$CAMP" "$OUTPUT" "${SELECTED_MODELS[@]}")"
if [[ -n "$BIND_ROOTS" ]]; then
    export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$BIND_ROOTS"
    export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$BIND_ROOTS"
fi

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

# Optional one-time self-heal: install interfaceforge into the module Python's
# per-user site (~/.local). Off by default -- ~/.local is shared by every Python
# of the same version -- so opt in with SEPARATION_DEEPMD_USER_PIP=1. --no-deps
# keeps it to the pure-Python package; --no-build-isolation uses the image's
# own build backend (compute nodes have no outbound network).
if [[ "${SEPARATION_DEEPMD_USER_PIP:-0}" == 1 ]] && ! python -c 'import interfaceforge' 2>/dev/null; then
    echo "SEPARATION_DEEPMD_USER_PIP=1: pip install --user --no-deps '$REPO_ROOT'" >&2
    python -m pip install --user --no-deps --no-build-isolation "$REPO_ROOT" >&2 || true
fi

if ! python - "$REPO_ROOT/src" <<'PY'
import importlib, os, sys

source_root = sys.argv[1]
sys.path.insert(0, source_root)
missing = []
for name in ("interfaceforge", "numpy", "yaml", "ase", "deepmd"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report every failure, do not raise
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
    else:
        print(f"  {name}: {getattr(module, '__file__', '<built-in>')}")
if missing:
    src = os.path.join(source_root, "interfaceforge")
    print("\n  cannot import inside the deepmd-kit container:", *missing, sep="\n    ", file=sys.stderr)
    print(f"\n  sys.executable                = {sys.executable}", file=sys.stderr)
    print(f"  PYTHONPATH (seen in container) = {os.environ.get('PYTHONPATH')}", file=sys.stderr)
    print(f"  checkout visible in container  = {os.path.isdir(src)}  ({src})", file=sys.stderr)
    sys.exit(4)
PY
then
    cat >&2 <<EOF

ERROR: the deepmd-kit container Python is missing modules the DeePMD
separation-energy evaluation needs (details above).

  * "checkout visible in container = False"
      The InterfaceForge checkout sits on a filesystem the Apptainer image does
      not bind. This launcher already binds: $BIND_ROOTS
      Add whatever else is needed and resubmit, e.g.:
        APPTAINER_BIND=/ddnB,/project,/work sbatch separation_energy_deepmd.sbatch

  * a third-party module (numpy / yaml / ase) missing
      It is genuinely absent from the image. Install it once into the module
      Python's user site and resubmit:
        module load $DEEPMD_MODULE
        python -m pip install --user --no-deps <module>
      or run this job once with SEPARATION_DEEPMD_USER_PIP=1.
EOF
    exit 4
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

mkdir -p "$OUTPUT"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    python -u "$BOOTSTRAP" interfaceforge.separation_energy "$OUTPUT" \
    "interface/300K/MD_Vac/N_Term/SiN_TiN_N-term=$N_TERM" \
    "interface/300K/MD_Vac/Ti_Term/SiN-TiN-Ti-term=$TI_TERM" \
    "${MODEL_ARGS[@]}" \
    --json-only -c "$CAMP/campaign.yaml"

[[ -s "$OUTPUT/separation_energy.json" ]] || {
    echo "ERROR: DeePMD partial JSON was not created" >&2
    exit 3
}
echo "DeePMD partial: $OUTPUT/separation_energy.json"
