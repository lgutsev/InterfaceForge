#!/usr/bin/env bash
# Run with bash from Periodic_MLIPs; --dry-run validates without submitting.
set -euo pipefail
case "${1:-}" in
    '') DRY_RUN=0;;
    --dry-run) DRY_RUN=1;;
    -h|--help) echo 'Usage: bash submit_separation_energy.sh [--dry-run]'; exit 0;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2;;
esac
[[ "$#" -le 1 ]] || { echo 'ERROR: too many arguments' >&2; exit 2; }
CAMP="${SEPARATION_CAMPAIGN_ROOT:-$(pwd -P)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
source "$SCRIPT_DIR/separation_energy_common.sh"
sep_campaign
for required in "$REPO_ROOT/src/interfaceforge/separation_energy.py" \
    "$SCRIPT_DIR/separation_energy_mace.sbatch" "$SCRIPT_DIR/separation_energy_deepmd.sbatch" \
    "$SCRIPT_DIR/separation_energy_merge.sbatch"; do sep_file "$required"; done
sep_mace
sep_deepmd
echo "MACE committee: $MACE_COMMITTEE_ROOT"
printf '  MACE model: %s\n' "${MACE_MODELS[@]}"
printf '  DeePMD model: %s\n' "${DEEPMD_MODELS[@]}"
if (( DRY_RUN )); then
    echo 'Preflight passed; no jobs submitted. File checks do not verify training completion or runtime model loading.'
    exit 0
fi
command -v sbatch >/dev/null || { echo 'ERROR: sbatch is unavailable' >&2; exit 2; }
mkdir -p "$CAMP/audit/separation/runs"
SEPARATION_RUN_DIR="$(mktemp -d "$CAMP/audit/separation/runs/run.XXXXXXXX")"
printf '%s\n' "${MACE_MODELS[@]}" > "$SEPARATION_RUN_DIR/mace_models.txt"
printf '%s\n' "${DEEPMD_MODELS[@]}" > "$SEPARATION_RUN_DIR/deepmd_models.txt"
# Export through the environment rather than a comma-delimited value string,
# which Slurm cannot parse safely when a path itself contains commas.
export SEPARATION_CAMPAIGN_ROOT="$CAMP" INTERFACEFORGE_ROOT="$REPO_ROOT"
export MACE_COMMITTEE_ROOT SEPARATION_RUN_DIR
JOBS=()
submission_exit() {
    local rc=$?
    if (( rc != 0 )); then
        echo "ERROR: workflow submission stopped. Inspect $SEPARATION_RUN_DIR before retrying." >&2
        if (( ${#JOBS[@]} )); then printf 'Already submitted (not cancelled): %s\n' "${JOBS[@]}" >&2; fi
    fi
}
trap submission_exit EXIT
submit_job() {
    local label="$1" line parsed id='' output rc=0
    shift
    output="$(sbatch --parsable --chdir="$CAMP" --export=ALL "$@" 2>&1)" || rc=$?
    printf '%s\n' "$output" > "$SEPARATION_RUN_DIR/$label.sbatch.log"
    printf '%s\n' "$output" >&2
    if (( rc )); then echo "ERROR: $label sbatch failed (exit $rc)" >&2; return 3; fi
    # Accept accounting chatter and duplicate confirmation of the same ID;
    # never silently choose between contradictory job IDs.
    while IFS= read -r line; do
        parsed=''
        if [[ "$line" =~ ^[[:space:]]*([0-9]+)(\;[^[:space:]]+)?[[:space:]]*$ ]]; then
            parsed="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^(sbatch:[[:space:]]+lua:[[:space:]]+)?Submitted[[:space:]]+(batch[[:space:]]+)?job[[:space:]]+([0-9]+)[[:space:]]*$ ]]; then
            parsed="${BASH_REMATCH[3]}"
        fi
        [[ -n "$parsed" ]] || continue
        if [[ -n "$id" && "$id" != "$parsed" ]]; then
            echo "ERROR: conflicting $label job IDs; inspect $SEPARATION_RUN_DIR/$label.sbatch.log and your queue" >&2
            return 3
        fi
        id="$parsed"
    done <<< "$output"
    [[ "$id" =~ ^[0-9]+$ && "$id" != 0 ]] || {
        echo "ERROR: no unambiguous $label job ID; inspect the submission log and your queue" >&2; return 3;
    }
    SUBMITTED_ID="$id"
    JOBS+=("$label=$id")
    printf '%s\t%s\n' "$label" "$id" >> "$SEPARATION_RUN_DIR/jobs.tsv"
    echo "Submitted $label: $id"
}
submit_job MACE "$SCRIPT_DIR/separation_energy_mace.sbatch"
mace_id="$SUBMITTED_ID"
submit_job DeePMD "$SCRIPT_DIR/separation_energy_deepmd.sbatch"
deepmd_id="$SUBMITTED_ID"
submit_job merge --dependency="afterok:${mace_id}:${deepmd_id}" "$SCRIPT_DIR/separation_energy_merge.sbatch"
echo "Submitted separation-energy workflow; output: $SEPARATION_RUN_DIR"
echo "Submission record: $SEPARATION_RUN_DIR/jobs.tsv"
