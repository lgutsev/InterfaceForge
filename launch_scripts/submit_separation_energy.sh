#!/usr/bin/env bash

# Submit isolated MACE and DeePMD evaluations, then merge only after both pass.
# Run from the Periodic_MLIPs campaign root:
#   /path/to/InterfaceForge/launch_scripts/submit_separation_energy.sh

set -euo pipefail

CAMP="${SEPARATION_CAMPAIGN_ROOT:-$(pwd -P)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

for required in \
    "$CAMP/campaign.yaml" \
    "$CAMP/adhesion/N_term_dft/manifest.json" \
    "$CAMP/adhesion/Ti_term_dft/manifest.json" \
    "$SCRIPT_DIR/separation_energy_mace.sbatch" \
    "$SCRIPT_DIR/separation_energy_deepmd.sbatch" \
    "$SCRIPT_DIR/separation_energy_merge.sbatch"; do
    [[ -s "$required" ]] || { echo "ERROR: missing or empty file: $required" >&2; exit 2; }
done

export_arg="ALL,SEPARATION_CAMPAIGN_ROOT=$CAMP,INTERFACEFORGE_ROOT=$REPO_ROOT"
mace_job="$(sbatch --parsable --chdir="$CAMP" --export="$export_arg" \
    "$SCRIPT_DIR/separation_energy_mace.sbatch")"
deepmd_job="$(sbatch --parsable --chdir="$CAMP" --export="$export_arg" \
    "$SCRIPT_DIR/separation_energy_deepmd.sbatch")"

# Some Slurm installations append a cluster name after a semicolon.
mace_id="${mace_job%%;*}"
deepmd_id="${deepmd_job%%;*}"
[[ "$mace_id" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected MACE sbatch result: $mace_job" >&2; exit 3; }
[[ "$deepmd_id" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected DeePMD sbatch result: $deepmd_job" >&2; exit 3; }

merge_job="$(sbatch --parsable --chdir="$CAMP" --export="$export_arg" \
    --dependency="afterok:${mace_id}:${deepmd_id}" \
    "$SCRIPT_DIR/separation_energy_merge.sbatch")"
merge_id="${merge_job%%;*}"
[[ "$merge_id" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected merge sbatch result: $merge_job" >&2; exit 3; }

echo "Submitted separation-energy workflow"
echo "  MACE:   $mace_id"
echo "  DeePMD: $deepmd_id"
echo "  merge:  $merge_id (afterok:$mace_id:$deepmd_id)"
echo "  output: $CAMP/audit/separation"
