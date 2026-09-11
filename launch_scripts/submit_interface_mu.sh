#!/usr/bin/env bash
# Submit the gamma(dmu) MLIP audit. Run with bash from the campaign root
# (the directory holding Wadh/ and Step2_300K/); --dry-run validates without
# submitting.
#
# The checkout is derived from this script's own location, so there is no
# INTERFACEFORGE_ROOT to set or get wrong -- invoke it by path:
#   bash /project/lgutsev/git_develop/InterfaceForge/launch_scripts/submit_interface_mu.sh --dry-run
#
# Inputs default to conventional names in the campaign root and can be
# overridden:
#   IFACE_MU_ENTRIES_FILE   default iface_mu_entries.txt   LABEL=DIR per line
#   IFACE_MU_PHASES_FILE    default iface_mu_phases.txt     NAME=DIR per line
#   IFACE_MU_AUX_FILE       default iface_mu_aux.txt if it exists
#   DEEPMD_COMMITTEE_ROOT   default models/deepmd/dpa2_ft
#   DEEPMD_COMMITTEE_MEMBERS default "000 001 002 003"
#   IFACE_MU_ANION / IFACE_MU_N_INTERFACES / IFACE_MU_AREA_AXIS / IFACE_MU_EXTRA

set -euo pipefail

case "${1:-}" in
    '') DRY_RUN=0;;
    --dry-run) DRY_RUN=1;;
    -h|--help) echo 'Usage: bash submit_interface_mu.sh [--dry-run]'; exit 0;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2;;
esac
[[ "$#" -le 1 ]] || { echo 'ERROR: too many arguments' >&2; exit 2; }

CAMP="${INTERFACE_MU_CAMPAIGN_ROOT:-$(pwd -P)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
source "$SCRIPT_DIR/separation_energy_common.sh"
source "$SCRIPT_DIR/interface_mu_common.sh"

for required in "$REPO_ROOT/src/interfaceforge/interface_mu.py" \
    "$SCRIPT_DIR/run_interfaceforge_module.py" \
    "$SCRIPT_DIR/interface_mu_common.sh" \
    "$SCRIPT_DIR/interface_mu_deepmd.sbatch"; do sep_file "$required"; done

# Default the input lists to conventional names, and resolve to absolute paths
# before they reach the job, which chdirs to the campaign root.
: "${IFACE_MU_ENTRIES_FILE:=$CAMP/iface_mu_entries.txt}"
: "${IFACE_MU_PHASES_FILE:=$CAMP/iface_mu_phases.txt}"
if [[ -z "${IFACE_MU_AUX_FILE:-}" && -f "$CAMP/iface_mu_aux.txt" ]]; then
    IFACE_MU_AUX_FILE="$CAMP/iface_mu_aux.txt"
fi
for var in IFACE_MU_ENTRIES_FILE IFACE_MU_PHASES_FILE IFACE_MU_AUX_FILE; do
    value="${!var:-}"
    [[ -z "$value" || "$value" = /* ]] || printf -v "$var" '%s' "$CAMP/$value"
done
export IFACE_MU_ENTRIES_FILE IFACE_MU_PHASES_FILE
[[ -z "${IFACE_MU_AUX_FILE:-}" ]] || export IFACE_MU_AUX_FILE

if [[ ! -f "$IFACE_MU_ENTRIES_FILE" ]]; then
    cat >&2 <<EOF
ERROR: no interface list at $IFACE_MU_ENTRIES_FILE

Create it with one LABEL=DIR per line, e.g.

  printf '%s\n' \\
    "Ideal-Ti-term=Step2_300K/Ideal/Ti_Term/SiN-TiN-Ti-term" \\
    "Ideal-N-term=Step2_300K/Ideal/N_Term/SiN_TiN_N-term" \\
    > iface_mu_entries.txt

and the phase list with one NAME=DIR per line:

  printf '%s\n' TiN=Wadh/TiN_mp492 Si3N4=Wadh/Si3N4_mp988 N2=Wadh/N2_gas \\
    Ti=Wadh/Ti_mp46 Si=Wadh/Si_mp149 > iface_mu_phases.txt
EOF
    exit 2
fi

# The same preflight the job runs, so --dry-run is not a weaker check.
ifmu_inputs

# The *_ARGS arrays are argv, so flagged ones alternate FLAG VALUE. Print only
# the values, else every second line is a bare "--phase".
show() {
    local label="$1" stride="$2" i
    shift 2
    local -a items=("$@")
    for (( i = stride - 1; i < ${#items[@]}; i += stride )); do
        printf '  %-11s %s\n' "$label" "${items[i]}"
    done
}
echo "Campaign:  $CAMP"
echo "Checkout:  $REPO_ROOT"
echo "Committee: $IFMU_COMMITTEE_ROOT  ($(( ${#MODEL_ARGS[@]} / 2 )) members)"
show interface: 1 "${ENTRY_ARGS[@]}"
show phase: 2 "${PHASE_ARGS[@]}"
(( ! ${#AUX_ARGS[@]} )) || show aux: 2 "${AUX_ARGS[@]}"

# Warn, rather than refuse, about the two things most likely to make the numbers
# hard to read later. Both are the user's call, so they do not block a run.
for entry in "${ENTRY_ARGS[@]}"; do
    dir="${entry#*=}"
    if [[ -f "$dir/INCAR" ]] && grep -Eq '^[[:space:]]*IBRION[[:space:]]*=[[:space:]]*0' "$dir/INCAR"; then
        echo "NOTE: ${entry%%=*} is an MD run. gamma^MLIP - gamma^DFT is evaluated on" >&2
        echo '      the same structure, so the MLIP audit is unaffected; an ABSOLUTE' >&2
        echo '      gamma from this cell still carries the snapshot thermal energy.' >&2
        break
    fi
done
if [[ -z "${IFACE_MU_N_INTERFACES:-}" ]]; then
    echo 'NOTE: --n-interfaces defaults to 2 here. With two inequivalent interfaces' >&2
    echo '      gamma is their average; set IFACE_MU_N_INTERFACES to be explicit.' >&2
fi

if (( DRY_RUN )); then
    echo
    echo 'Preflight passed; no job submitted.'
    echo 'File checks do not verify that a model loads or that a run converged.'
    exit 0
fi

command -v sbatch >/dev/null || { echo 'ERROR: sbatch is unavailable' >&2; exit 2; }
export INTERFACE_MU_CAMPAIGN_ROOT="$CAMP" INTERFACEFORGE_ROOT="$REPO_ROOT"
[[ -z "${DEEPMD_COMMITTEE_ROOT:-}" ]] || export DEEPMD_COMMITTEE_ROOT
[[ -z "${DEEPMD_COMMITTEE_MEMBERS:-}" ]] || export DEEPMD_COMMITTEE_MEMBERS
for var in IFACE_MU_OUTPUT IFACE_MU_ANION IFACE_MU_N_INTERFACES IFACE_MU_AREA_AXIS \
    IFACE_MU_EXTRA DEEPMD_MODULE; do
    [[ -z "${!var:-}" ]] || export "$var"
done

JOB="$(sbatch --parsable "$SCRIPT_DIR/interface_mu_deepmd.sbatch")"
echo "Submitted interface-mu DeePMD audit: job $JOB"
echo "Watch:  tail -f $CAMP/interface_mu_deepmd.$JOB.out"
