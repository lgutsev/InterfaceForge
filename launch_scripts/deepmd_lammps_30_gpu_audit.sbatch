#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=deepmd.lmp.audit
#SBATCH --output=deepmd.lmp.audit.%j.out
#SBATCH --error=deepmd.lmp.audit.%j.err

# Audit LONI's LAMMPS build linked to DeePMD-kit 3.0. With no environment
# inputs, verify only that the module exposes pair_style deepmd. After training,
# set DEEPMD_MODELS_FILE and DEEPMD_SYSTEM to test the actual frozen committee.

set -euo pipefail

LAMMPS_MODULE="${LAMMPS_MODULE:-lammps/29Aug2024-r8.0-deepmd3.0.0-gpu}"
DEEPMD_MODELS_FILE="${DEEPMD_MODELS_FILE:-}"
DEEPMD_SYSTEM="${DEEPMD_SYSTEM:-}"
AUDIT_PYTHON="${AUDIT_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}"

module purge
module load "$LAMMPS_MODULE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

echo "Loaded module: $LAMMPS_MODULE"
module list 2>&1
echo
echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo

AUDIT_TMP="$(mktemp -d "${TMPDIR:-/tmp}/iface-deepmd-lmp.XXXXXX")"
cleanup() {
    rm -rf -- "$AUDIT_TMP"
}
trap cleanup EXIT

# A user-level lmp can precede a module binary in PATH. Inspect every available
# candidate and deliberately select one whose help lists the DeePMD pair style.
LMP_BIN=""
declare -A SEEN_LMP=()
for command_name in lmp lmp_mpi lmp_serial; do
    mapfile -t candidates < <(type -a -p "$command_name" 2>/dev/null || true)
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" && -z "${SEEN_LMP[$candidate]:-}" ]] || continue
        SEEN_LMP["$candidate"]=1
        echo "Inspecting candidate: $candidate"
        if "$candidate" -h >"$AUDIT_TMP/lmp-help.txt" 2>&1 \
            && grep -Eqi '(^|[[:space:]])deepmd([[:space:]]|$)' "$AUDIT_TMP/lmp-help.txt"; then
            LMP_BIN="$candidate"
            break 2
        fi
    done
done

if [[ -z "$LMP_BIN" ]]; then
    echo "ERROR: no LAMMPS executable from the loaded module reports pair_style deepmd"
    echo "All visible candidates:"
    type -a lmp lmp_mpi lmp_serial 2>/dev/null || true
    exit 2
fi

echo
echo "Selected DeePMD-enabled LAMMPS: $LMP_BIN"
"$LMP_BIN" -h | sed -n '1,35p'
echo
echo "Linked DeePMD/Torch/TensorFlow libraries, when dynamically visible:"
ldd "$LMP_BIN" 2>/dev/null \
    | grep -Ei 'deepmd|torch|tensorflow|cuda|cudart' \
    || echo "No matching dynamic libraries reported (the launcher may be a wrapper)."

if [[ -z "$DEEPMD_MODELS_FILE" && -z "$DEEPMD_SYSTEM" ]]; then
    echo
    echo "LAMMPS/DeePMD module audit passed."
    echo "No frozen models were supplied, so DeePMD 3.2 model compatibility was not tested."
    exit 0
fi

[[ -n "$DEEPMD_MODELS_FILE" && -n "$DEEPMD_SYSTEM" ]] || {
    echo "ERROR: set both DEEPMD_MODELS_FILE and DEEPMD_SYSTEM for model testing"
    exit 2
}
[[ -s "$DEEPMD_MODELS_FILE" ]] || {
    echo "ERROR: missing or empty model list: $DEEPMD_MODELS_FILE"
    exit 2
}
[[ -d "$DEEPMD_SYSTEM/set.000" ]] || {
    echo "ERROR: DEEPMD_SYSTEM must contain set.000: $DEEPMD_SYSTEM"
    exit 2
}
[[ -x "$AUDIT_PYTHON" ]] || {
    echo "ERROR: audit Python is not executable: $AUDIT_PYTHON"
    exit 2
}

mapfile -t MODELS < <(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$DEEPMD_MODELS_FILE")
[[ "${#MODELS[@]}" -ge 1 ]] || { echo "ERROR: model list contains no paths"; exit 2; }

for model in "${MODELS[@]}"; do
    [[ -s "$model" ]] || { echo "ERROR: missing frozen model: $model"; exit 2; }
    [[ "$model" != *[[:space:]]* ]] || {
        echo "ERROR: model paths containing whitespace are unsupported: $model"
        exit 2
    }
done

"$AUDIT_PYTHON" - "$DEEPMD_SYSTEM" "$AUDIT_TMP/system.data" \
    "$AUDIT_TMP/type-map.txt" <<'PY'
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

system = Path(sys.argv[1])
data_path = Path(sys.argv[2])
map_path = Path(sys.argv[3])
type_map = [
    line.strip()
    for line in (system / "type_map.raw").read_text().splitlines()
    if line.strip()
]
atom_types = [int(value) for value in (system / "type.raw").read_text().split()]
positions = np.load(system / "set.000" / "coord.npy")[0].reshape(-1, 3)
cell = np.load(system / "set.000" / "box.npy")[0].reshape(3, 3)
symbols = [type_map[value] for value in atom_types]

atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
write(
    data_path,
    atoms,
    format="lammps-data",
    atom_style="atomic",
    specorder=type_map,
    masses=True,
)
map_path.write_text(" ".join(type_map) + "\n", encoding="utf-8")
print(f"Prepared one-frame LAMMPS system: {len(atoms)} atoms; type_map={type_map}")
PY

TYPE_MAP="$(<"$AUDIT_TMP/type-map.txt")"
{
    echo "units metal"
    echo "atom_style atomic"
    echo "boundary p p p"
    echo "read_data $AUDIT_TMP/system.data"
    printf 'pair_style deepmd'
    printf ' %s' "${MODELS[@]}"
    echo " out_file $AUDIT_TMP/model_devi.out out_freq 1"
    echo "pair_coeff * * $TYPE_MAP"
    echo "neighbor 2.0 bin"
    echo "neigh_modify every 1 delay 0 check yes"
    echo "velocity all set 0.0 0.0 0.0"
    echo "fix audit_nve all nve"
    echo "timestep 0.0001"
    echo "thermo 1"
    echo "run 1"
} >"$AUDIT_TMP/in.lammps"

echo
echo "Testing ${#MODELS[@]} frozen model(s) with one canonical frame..."
srun -n 1 "$LMP_BIN" -in "$AUDIT_TMP/in.lammps"

echo
if [[ "${#MODELS[@]}" -gt 1 ]]; then
    [[ -s "$AUDIT_TMP/model_devi.out" ]] || {
        echo "ERROR: committee run produced no model_devi.out"
        exit 3
    }
    echo "Committee model-deviation output:"
    sed -n '1,10p' "$AUDIT_TMP/model_devi.out"
fi
echo "LAMMPS loaded and evaluated all supplied frozen models successfully."
