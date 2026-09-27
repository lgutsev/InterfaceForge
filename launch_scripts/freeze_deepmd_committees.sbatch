#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=freeze.deepmd
#SBATCH --output=freeze_deepmd.%j.out
#SBATCH --error=freeze_deepmd.%j.err

# Freeze all trained DeePMD PyTorch committees in a campaign. The default
# architecture set covers from-scratch and fine-tuned DPA-2/DPA-3 plus DPA-4.
# Existing non-empty deployment artifacts are left untouched.
#
# Submit from the campaign root, e.g. Periodic_MLIPs:
#   sbatch /path/to/InterfaceForge/launch_scripts/freeze_deepmd_committees.sbatch
#
# Override the architecture list if needed:
#   sbatch --export=ALL,DEEPMD_ARCHS="dpa3 dpa3_ft" \
#     /path/to/InterfaceForge/launch_scripts/freeze_deepmd_committees.sbatch

set -euo pipefail

CAMP="${DEEPMD_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script from the campaign root}}"
DEEPMD_MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"
DEEPMD_ARCHS="${DEEPMD_ARCHS:-dpa2 dpa2_ft dpa3 dpa3_ft dpa4}"

CAMP="$(cd -- "$CAMP" && pwd -P)"

module purge
module load "$DEEPMD_MODULE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Campaign root: $CAMP"
echo "Host: $(hostname)"
echo "DeePMD executable: $(command -v dp)"
module list 2>&1
dp --version
echo

freeze_torchscript() {
    local arch="$1"
    local model_dir="$2"
    local checkpoint="$model_dir/model.ckpt.pt"
    local frozen="$model_dir/frozen_model.pth"

    if [[ -s "$frozen" ]]; then
        echo "SKIP: $arch already frozen: $frozen"
        return 0
    fi
    [[ -r "$checkpoint" && -s "$checkpoint" ]] || {
        echo "ERROR: missing, empty, or unreadable checkpoint: $checkpoint" >&2
        return 2
    }

    (
        set -euo pipefail
        exec 9>"$model_dir/.freeze.lock"
        flock -n 9 || {
            echo "ERROR: another freeze is active in $model_dir" >&2
            exit 2
        }

        temporary_dir="$(mktemp -d "$model_dir/.freeze.XXXXXXXX")"
        trap 'rm -rf -- "$temporary_dir"' EXIT
        temporary_model="$temporary_dir/frozen_model.pth"

        echo "Freezing $arch: $model_dir"
        dp --pt freeze -c "$checkpoint" -o "$temporary_model"
        [[ -s "$temporary_model" ]] || {
            echo "ERROR: freeze produced no model for $model_dir" >&2
            exit 3
        }

        python - "$temporary_model" <<'PY'
import sys
from deepmd.infer import DeepPot

model = DeepPot(sys.argv[1])
print("Frozen model type map:", model.get_type_map())
PY

        mv -T -- "$temporary_model" "$frozen"
        echo "Frozen: $frozen"
        ls -lh "$frozen"
    )
}

freeze_dpa4() {
    local model_dir="$1"
    local checkpoint="$model_dir/model.ckpt.pt"
    local frozen="$model_dir/frozen_model.pt2"

    if [[ -s "$frozen" ]]; then
        echo "SKIP: dpa4 already frozen: $frozen"
        return 0
    fi
    [[ -r "$checkpoint" && -s "$checkpoint" ]] || {
        echo "ERROR: missing, empty, or unreadable checkpoint: $checkpoint" >&2
        return 2
    }

    (
        set -euo pipefail
        exec 9>"$model_dir/.freeze.lock"
        flock -n 9 || {
            echo "ERROR: another freeze is active in $model_dir" >&2
            exit 2
        }

        temporary_dir="$(mktemp -d "$model_dir/.freeze.XXXXXXXX")"
        trap 'rm -rf -- "$temporary_dir"' EXIT

        echo "Freezing dpa4: $model_dir"
        echo "DPA-4 uses DeePMD's AOTInductor .pt2 export path."
        (
            cd "$temporary_dir"
            dp --pt freeze -c "$checkpoint" -o frozen_model
        )

        temporary_model="$temporary_dir/frozen_model.pt2"
        [[ -s "$temporary_model" ]] || {
            echo "ERROR: DPA-4 freeze did not produce frozen_model.pt2 in $temporary_dir" >&2
            find "$temporary_dir" -maxdepth 1 -type f -ls >&2 || true
            exit 3
        }

        mv -T -- "$temporary_model" "$frozen"
        echo "Frozen: $frozen"
        ls -lh "$frozen"
        echo "NOTE: DPA-4 deployment still requires an explicit downstream runtime/LAMMPS audit."
    )
}

for arch in $DEEPMD_ARCHS; do
    arch_root="$CAMP/models/deepmd/$arch"
    if [[ ! -d "$arch_root" ]]; then
        echo "SKIP: architecture directory not present: $arch_root"
        continue
    fi

    found=0
    for model_dir in "$arch_root"/model_*; do
        [[ -d "$model_dir" ]] || continue
        found=1
        if [[ "$arch" == "dpa4" ]]; then
            freeze_dpa4 "$model_dir"
        else
            freeze_torchscript "$arch" "$model_dir"
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "SKIP: no model_NNN directories under $arch_root"
    fi
done

echo
echo "================ FINAL DEEPMD INVENTORY ================"
find "$CAMP/models/deepmd" -type f \
    \( -name 'frozen_model.pth' -o -name 'frozen_model.pt2' -o -name 'frozen_model.pb' \) \
    -print | sort

echo
echo "Freeze job completed."
