#!/usr/bin/env bash
# Bulk-referenced interfacial-excess control (DFT vs MLIP), submitted the same
# way as the free-surface separation-energy workflow but with --reference bulk.
#
#   bash submit_interface_excess.sh [--dry-run] \
#       N_TERM_INTERFACE_SRC TI_TERM_INTERFACE_SRC BULK_A_SRC BULK_B_SRC
#
# *_INTERFACE_SRC : the finished interface run for each termination (a VASP run
#                   directory or an `iface vasp adhesion prepare` tree).
# BULK_A_SRC / BULK_B_SRC : the two finished bulk-cell runs (e.g. TiN and
#                   beta-Si3N4) whose compositions sum to the interface. Shared
#                   by both terminations.
#
# Assembles audit/interface_excess/{N_term,Ti_term}/, then reuses
# submit_separation_energy.sh (same MACE/DeePMD committee discovery, isolated
# GPU jobs, afterok merge). Output: audit/interface_excess/runs/run.XXXXXXXX/.
set -euo pipefail

DRY=()
if [[ "${1:-}" == --dry-run ]]; then DRY=(--dry-run); shift; fi
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
[[ "$#" -eq 4 ]] || { echo "ERROR: expected 4 source directories (see --help)" >&2; exit 2; }

N_TERM_SRC="$1"; TI_TERM_SRC="$2"; BULK_A_SRC="$3"; BULK_B_SRC="$4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CAMP="${SEPARATION_CAMPAIGN_ROOT:-$(pwd -P)}"
CAMP="$(cd -- "$CAMP" && pwd -P)"
BASE="$CAMP/audit/interface_excess"

bash "$SCRIPT_DIR/prepare_interface_excess.sh" --force "$BASE/N_term"  "$N_TERM_SRC"  "$BULK_A_SRC" "$BULK_B_SRC"
bash "$SCRIPT_DIR/prepare_interface_excess.sh" --force "$BASE/Ti_term" "$TI_TERM_SRC" "$BULK_A_SRC" "$BULK_B_SRC"

ENTRIES="$BASE/entries.txt"
{
    echo "interface/bulk-ref/N_term/SiN_TiN_N-term=$BASE/N_term"
    echo "interface/bulk-ref/Ti_term/SiN-TiN-Ti-term=$BASE/Ti_term"
} > "$ENTRIES"
echo "Entries file: $ENTRIES"

export SEPARATION_CAMPAIGN_ROOT="$CAMP"
export SEPARATION_ENTRIES_FILE="$ENTRIES"
export SEPARATION_REFERENCE="bulk"
export SEPARATION_N_INTERFACES="${SEPARATION_N_INTERFACES:-1}"
export SEPARATION_RUNS_DIR="$BASE/runs"

exec bash "$SCRIPT_DIR/submit_separation_energy.sh" "${DRY[@]}"
