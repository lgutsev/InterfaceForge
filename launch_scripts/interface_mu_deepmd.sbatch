#!/usr/bin/env bash
#SBATCH --partition=gpu2
#SBATCH --account=loni_perovsk27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=ifacemu.deepmd
#SBATCH --output=interface_mu_deepmd.%j.out
#SBATCH --error=interface_mu_deepmd.%j.err

# gamma(dmu_N) for periodic SiN/TiN interface cells, DFT against a DeePMD
# committee. This is a batch job because deepmd-kit lives inside the module's
# Apptainer image, so `iface validate interface-mu --deepmd-model` fails on a
# login node with "needs deepmd-kit; run this where the committee's environment
# is available". The arithmetic itself is a handful of single-point energies, but
# every deepmd-kit module on this system is a -gpu build
# (29Aug2024-r8.0-deepmd3.0.0-gpu and r9.3-deepmd3.2.0.b.0-gpu), and gpu2 is the
# only partition, so --gres=gpu:1 there is the only route rather than a
# preference. Raise --time only if the cells are very large.
#
# Only the interface cells and the compound references (TiN, Si3N4) are handed
# to the models. The molecular N2 reference is a vacuum structure and stays
# DFT-only; the elemental and auxiliary phases only shape the DFT window, which
# both sides share. So a committee that has never seen a molecule or a silicide
# is still valid here.
#
# Environment:
#   INTERFACEFORGE_ROOT      (required) the InterfaceForge checkout
#   IFACE_MU_ENTRIES_FILE    (required) one LABEL=DIR interface cell per line
#   IFACE_MU_PHASES_FILE     (required) one NAME=DIR --phase per line
#   IFACE_MU_AUX_FILE        (optional) one NAME=DIR --aux-phase per line
#   DEEPMD_COMMITTEE_ROOT    default $CAMP/models/deepmd/dpa2_ft
#   DEEPMD_COMMITTEE_MEMBERS default "000 001 002 003"
#   IFACE_MU_OUTPUT          default $CAMP/audit/interface_mu_deepmd
#   IFACE_MU_ANION           default N
#   IFACE_MU_N_INTERFACES    default 2
#   IFACE_MU_AREA_AXIS       optional, e.g. c
#   IFACE_MU_EXTRA           optional extra flags, word-split

set -euo pipefail

CAMP="${INTERFACE_MU_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch}}"
REPO_ROOT="${INTERFACEFORGE_ROOT:?Set INTERFACEFORGE_ROOT to the InterfaceForge checkout}"
source "$REPO_ROOT/launch_scripts/separation_energy_common.sh"
BOOTSTRAP="$REPO_ROOT/launch_scripts/run_interfaceforge_module.py"
sep_file "$BOOTSTRAP"
CAMP="$(cd -- "$CAMP" && pwd -P)"

OUTPUT="${IFACE_MU_OUTPUT:-$CAMP/audit/interface_mu_deepmd}"
ANION="${IFACE_MU_ANION:-N}"
N_INTERFACES="${IFACE_MU_N_INTERFACES:-2}"
DEEPMD_MODULE="${DEEPMD_MODULE:-deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu}"

# --- read the NAME=DIR / LABEL=DIR lists, resolving and checking every dir ---
# Populates the named array plus PATHS_SEEN, which feeds APPTAINER_BIND.
PATHS_SEEN=()
# Populates the global PAIRS array; callers copy it. Deliberately not a
# nameref (local -n): that needs bash 4.3, and RHEL7 login nodes ship 4.2.
read_pairs() {
    local file="$1" flag="$2" line name dir
    PAIRS=()
    sep_file "$file" || return 2
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -n "${line//[[:space:]]/}" ]] || continue
        [[ "$line" != \#* ]] || continue
        name="${line%%=*}"
        dir="${line#*=}"
        [[ "$name" != "$line" && -n "$name" && -n "$dir" ]] || {
            echo "ERROR: $file: expected NAME=DIR, got: $line" >&2
            return 2
        }
        [[ "$dir" = /* ]] || dir="$CAMP/$dir"
        [[ -d "$dir" ]] || { echo "ERROR: $file: no such directory: $dir" >&2; return 2; }
        dir="$(cd -- "$dir" && pwd -P)"
        sep_file "$dir/OUTCAR" || {
            echo "  ($name has no readable OUTCAR; interface-mu needs finished runs)" >&2
            return 2
        }
        PATHS_SEEN+=("$dir")
        if [[ -n "$flag" ]]; then PAIRS+=("$flag" "$name=$dir"); else PAIRS+=("$name=$dir"); fi
    done < "$file"
    (( ${#PAIRS[@]} )) || { echo "ERROR: $file has no entries" >&2; return 2; }
}

read_pairs "${IFACE_MU_ENTRIES_FILE:?Set IFACE_MU_ENTRIES_FILE}" ""
ENTRY_ARGS=("${PAIRS[@]}")
read_pairs "${IFACE_MU_PHASES_FILE:?Set IFACE_MU_PHASES_FILE}" --phase
PHASE_ARGS=("${PAIRS[@]}")
AUX_ARGS=()
if [[ -n "${IFACE_MU_AUX_FILE:-}" ]]; then
    read_pairs "$IFACE_MU_AUX_FILE" --aux-phase
    AUX_ARGS=("${PAIRS[@]}")
fi

# --- the committee: frozen inference exports only, never checkpoints ---
COMMITTEE_ROOT="${DEEPMD_COMMITTEE_ROOT:-$CAMP/models/deepmd/dpa2_ft}"
[[ "$COMMITTEE_ROOT" = /* ]] || COMMITTEE_ROOT="$CAMP/$COMMITTEE_ROOT"
[[ -d "$COMMITTEE_ROOT" ]] || { echo "ERROR: no committee at $COMMITTEE_ROOT" >&2; exit 2; }
COMMITTEE_ROOT="$(cd -- "$COMMITTEE_ROOT" && pwd -P)"
MODEL_ARGS=()
failed=0
for member in ${DEEPMD_COMMITTEE_MEMBERS:-000 001 002 003}; do
    model="$COMMITTEE_ROOT/model_$member/frozen_model.pth"
    if ! sep_file "$model"; then
        echo "ERROR: member $member is not a frozen inference export: $model" >&2
        failed=1
        continue
    fi
    PATHS_SEEN+=("$model")
    MODEL_ARGS+=(--deepmd-model "$model")
done
(( ! failed )) || {
    echo "Freeze the missing members before auditing; a training checkpoint is not deployable." >&2
    exit 2
}
echo "Committee: ${#MODEL_ARGS[@]} members from $COMMITTEE_ROOT" >&2
(( ${#MODEL_ARGS[@]} >= 2 )) || echo "WARNING: fewer than two members: committee_spread will be 0" >&2

EXTRA=()
[[ -z "${IFACE_MU_AREA_AXIS:-}" ]] || EXTRA+=(--area-axis "$IFACE_MU_AREA_AXIS")
# shellcheck disable=SC2206  # deliberate word-splitting of caller-supplied flags
[[ -z "${IFACE_MU_EXTRA:-}" ]] || EXTRA+=(${IFACE_MU_EXTRA})

module purge
module load "$DEEPMD_MODULE"

# The deepmd-kit module provides Python from inside an Apptainer image, which
# does not see the site filesystems by default and may discard PYTHONPATH. Bind
# every root the job touches and let the bootstrap insert src/ itself.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export APPTAINERENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
BIND_ROOTS="$(sep_bind_roots "$REPO_ROOT" "$CAMP" "$OUTPUT" "${PATHS_SEEN[@]}")"
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

ERROR: the deepmd-kit container Python is missing modules interface-mu needs
(details above).

  * "checkout visible in container = False"
      The checkout sits on a filesystem the image does not bind. This launcher
      already binds: $BIND_ROOTS
      Add whatever else is needed and resubmit, e.g.:
        APPTAINER_BIND=/ddnB,/project,/work sbatch interface_mu_deepmd.sbatch

  * a third-party module (numpy / yaml / ase) missing
      Install it once into the module Python's user site and resubmit:
        module load $DEEPMD_MODULE
        python -m pip install --user --no-deps <module>

  * pymatgen / scipy missing
      Only --window hull needs them, and the MLIP audit does not: gamma0 and
      delta_vs_dft_j_per_m2 are computed from mu0 (the N2 reference), the
      decomposition and the area, none of which the window touches. Verified
      identical to 1e-12 across the two methods on a set where the windows
      themselves differ by 1.2 eV. So add:
        IFACE_MU_EXTRA="--window pairwise"
      and take the real window from the login-node `iface phases hull` run.
      What you lose is only the labelling of the N-poor end of the gamma axis,
      which pairwise puts at the elemental-precipitation bound instead of the
      joint-coexistence one.
EOF
    exit 4
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true

mkdir -p "$OUTPUT"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    python -u "$BOOTSTRAP" interfaceforge validate interface-mu "$OUTPUT" \
    "${ENTRY_ARGS[@]}" \
    "${PHASE_ARGS[@]}" \
    "${AUX_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    --anion "$ANION" \
    --n-interfaces "$N_INTERFACES" \
    "${EXTRA[@]}"

[[ -s "$OUTPUT/interface_mu.json" ]] || {
    echo "ERROR: interface_mu.json was not created" >&2
    exit 3
}
echo "Wrote: $OUTPUT/interface_mu.{json,csv,md} and the gamma(dmu) figure"
python - "$OUTPUT/interface_mu.json" <<'PY'
import json, sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
window = payload["chemical_potential_window"]
print("\nwindow: {:.4f} <= dmu({}) <= {:.4f} eV, bounded by {}".format(
    window["dmu_min_ev"], payload["anion"], window["dmu_max_ev"],
    window.get("binding_compound") or window.get("binding_phase"),
))
for row in payload["interfaces"]:
    print("\n{}  dn={:+.3f}  {}".format(
        row["label"], row["decomposition"]["anion_excess"],
        row["energy_interpretation"],
    ))
    print("  DFT   gamma0 = {:.4f} J/m^2   slope = {:+.4f}".format(
        row["dft"]["gamma0_j_per_m2"], row["dft"]["slope_j_per_m2_per_ev"],
    ))
    for family, block in (row.get("mlip") or {}).items():
        print("  {:5s} gamma0 = {:.4f} +/- {:.4f}   delta vs DFT = {:+.4f} J/m^2".format(
            family, block["gamma0_ensemble_j_per_m2"],
            block["committee_spread_j_per_m2"], block["delta_vs_dft_j_per_m2"],
        ))
        for key, error in (block.get("per_structure_error") or {}).items():
            print("        {:22s} {:+8.2f} meV/atom  (spread {:.2f}, N={})".format(
                key, error["ensemble_error_mev_per_atom"],
                error["member_spread_mev_per_atom"], error["natoms"],
            ))
PY
