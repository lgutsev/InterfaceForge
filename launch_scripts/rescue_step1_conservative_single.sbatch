#!/usr/bin/env bash
#SBATCH -A loni_perovsk27
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH --mem=2G
#SBATCH -J step1_rescue
#SBATCH -o step1-rescue-%j.out
#SBATCH -e step1-rescue-%j.err

set -euo pipefail

# Run the conservative Step1 rescue off the login node.
# Arguments are forwarded verbatim to rescue_step1_conservative.sh.
#
# Examples:
#   sbatch rescue_step1_conservative_single.sbatch Step1
#   sbatch rescue_step1_conservative_single.sbatch Step1 --execute
#
# Environment overrides such as STALE_HOURS, RAMP_FROM, POTIM and
# USE_LANGEVIN are inherited by Slurm with its normal --export=ALL behavior.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESCUE="$SCRIPT_DIR/rescue_step1_conservative.sh"

if [[ ! -f "$RESCUE" ]]; then
    echo "ERROR: rescue script not found: $RESCUE" >&2
    exit 2
fi

# Most submissions inherit the activated InterfaceForge environment. If not,
# use Lavrenty's standard development environment when it is available.
if ! command -v iface >/dev/null 2>&1; then
    DEFAULT_ENV="/project/lgutsev/env/lgutsev_dev"
    if [[ -f "$DEFAULT_ENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$DEFAULT_ENV/bin/activate"
    fi
fi

if ! command -v iface >/dev/null 2>&1; then
    echo "ERROR: 'iface' is not available on the compute node." >&2
    echo "Activate/export the InterfaceForge environment before sbatch, or edit DEFAULT_ENV." >&2
    exit 2
fi

echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "Rescue command: bash $RESCUE $*"
echo

# Slurm starts the job in the submission directory by default; make that
# explicit because a relative Step1 path should resolve exactly as it did at
# submission time.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

bash "$RESCUE" "$@"
