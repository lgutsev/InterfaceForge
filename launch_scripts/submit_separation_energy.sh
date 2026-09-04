#!/usr/bin/env bash

# Submit isolated MACE and DeePMD evaluations, then merge only after both pass.
# Run from the Periodic_MLIPs campaign root:
#   /path/to/InterfaceForge/launch_scripts/submit_separation_energy.sh

set -euo pipefail

CAMP="${SEPARATION_CAMPAIGN_ROOT:-$(pwd -P)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# Same default InterfaceForge itself uses (`iface mlip-progress`, `iface
# package campaign`): the MACE committee lives under an ENCUT-tagged
# sub-directory, not directly under models/.
MACE_COMMITTEE_ROOT="${MACE_COMMITTEE_ROOT:-$CAMP/models/mace_committee_520eV}"

for required in \
    "$CAMP/campaign.yaml" \
    "$CAMP/adhesion/N_term_dft/manifest.json" \
    "$CAMP/adhesion/Ti_term_dft/manifest.json" \
    "$SCRIPT_DIR/separation_energy_mace.sbatch" \
    "$SCRIPT_DIR/separation_energy_deepmd.sbatch" \
    "$SCRIPT_DIR/separation_energy_merge.sbatch"; do
    [[ -s "$required" ]] || { echo "ERROR: missing or empty file: $required" >&2; exit 2; }
done

# Validate every model before submitting either backend. This avoids leaving a
# half-submitted workflow when one committee is incomplete.
for seed in 11 23 37 53; do
    seed_dir="$MACE_COMMITTEE_ROOT/mace_committee/seed_${seed}"
    # mace_train_committee.sh exports the final model into mace_model/; its own
    # checkpoints/ can independently contain a same-named *_stagetwo.model as
    # training-time bookkeeping. Prefer the canonical export directory, and
    # always exclude checkpoints/ so the two are never ambiguous.
    search_dir="$seed_dir/mace_model"
    [[ -d "$search_dir" ]] || search_dir="$seed_dir"
    # Newest-first (by mtime): if more than one candidate survives the name
    # filters, prefer the most recently written one rather than failing.
    mapfile -t matches < <(
        find "$search_dir" -maxdepth 3 -type f \
            -name '*_stagetwo.model' ! -name '*_compiled.model' \
            ! -path '*/checkpoints/*' -printf '%T@\t%p\n' 2>/dev/null | sort -rn | cut -f2-
    )
    if [[ "${#matches[@]}" -eq 0 ]]; then
        mapfile -t matches < <(
            find "$search_dir" -maxdepth 3 -type f -name '*.model' \
                ! -name '*_compiled.model' ! -path '*/checkpoints/*' \
                -printf '%T@\t%p\n' 2>/dev/null | sort -rn | cut -f2-
        )
    fi
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "ERROR: no usable MACE model found for seed $seed" >&2
        echo "  searched: $search_dir" >&2
        exit 2
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
        echo "WARNING: seed $seed has ${#matches[@]} usable MACE models; using the newest:" >&2
        printf '  %s\n' "${matches[@]}" >&2
    fi
done

for member in 000 001 002 003; do
    model_dir="$CAMP/models/deepmd/dpa2/model_${member}"
    frozen="$model_dir/frozen_model.pth"
    checkpoint="$model_dir/model.ckpt.pt"
    if [[ ! -s "$frozen" && ! -s "$checkpoint" ]]; then
        echo "ERROR: DeePMD member $member has neither a frozen model nor a training checkpoint:" >&2
        echo "  $frozen" >&2
        echo "  $checkpoint" >&2
        exit 2
    fi
done

export_arg="ALL,SEPARATION_CAMPAIGN_ROOT=$CAMP,INTERFACEFORGE_ROOT=$REPO_ROOT,MACE_COMMITTEE_ROOT=$MACE_COMMITTEE_ROOT"
submit_job() {
    local label="$1"
    local output
    local job_id
    shift

    if ! output="$(sbatch --parsable "$@" 2>&1)"; then
        echo "ERROR: $label sbatch failed:" >&2
        printf '%s\n' "$output" >&2
        return 3
    fi
    # LONI writes allocation/SU notices to stdout before the parsable result.
    # Preserve those notices, but extract the final standalone job-id line.
    printf '%s\n' "$output" >&2
    job_id="$(printf '%s\n' "$output" | sed -nE \
        's/^[[:space:]]*([0-9]+)(;[^[:space:]]+)?[[:space:]]*$/\1/p' | tail -n 1)"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: unexpected $label sbatch result" >&2
        return 3
    fi
    printf '%s\n' "$job_id"
}

mace_id="$(submit_job MACE --chdir="$CAMP" --export="$export_arg" \
    "$SCRIPT_DIR/separation_energy_mace.sbatch")"
deepmd_id="$(submit_job DeePMD --chdir="$CAMP" --export="$export_arg" \
    "$SCRIPT_DIR/separation_energy_deepmd.sbatch")"

merge_id="$(submit_job merge --chdir="$CAMP" --export="$export_arg" \
    --dependency="afterok:${mace_id}:${deepmd_id}" \
    "$SCRIPT_DIR/separation_energy_merge.sbatch")"

echo "Submitted separation-energy workflow"
echo "  MACE:   $mace_id"
echo "  DeePMD: $deepmd_id"
echo "  merge:  $merge_id (afterok:$mace_id:$deepmd_id)"
echo "  output: $CAMP/audit/separation"
