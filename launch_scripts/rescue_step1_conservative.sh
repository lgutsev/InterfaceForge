#!/usr/bin/env bash
set -euo pipefail

# Rescue unstable Step1 AIMD leaves with InterfaceForge's conservative repair.
# A thin, documented wrapper around
#
#   iface vasp step1-recover "$ROOT" --only repair --scheduler slurm [--execute]
#
# Dry-run by default (prints the recovery plan, changes nothing); pass --execute
# to archive, rewind and prepare each repair and submit exactly those runs.
#
# Recover's repair defaults are the conservative NiO rescue, tuned for
# proton-rich magnetic DFT+U surfaces such as NiO/OH50:
#   POTIM=0.5 fs, ALGO=Normal, EDIFF=1e-5, NELM=120, NELMIN=6,
#   a preconditioning static SCF of the magnetic DFT+U state,
#   then a 100 K -> target-temperature ramp.
# NBLOCK is left unchanged (normally 4).  A run that was already repaired is
# repaired again as the next generation (the accepted prefix accumulates); no
# file ever has to be renamed by hand.
#
# All safety gates live in step1-recover itself:
#   1. only runs step1-status classifies as hard-unstable ("repair") are
#      touched; startup transients and other review-level warnings, done,
#      active and interrupted-mutation runs are left alone;
#   2. --scheduler slurm: squeue decides activity (one query per plan, a fresh
#      re-check immediately before each run is changed and before each sbatch);
#      if squeue cannot be asked, nothing is done;
#   3. each run's file fingerprint must be unchanged since planning, and its
#      state is archived before anything is replaced;
#   4. the first failure stops the batch; <ROOT>/step1_recover.json says which
#      runs were changed, which were submitted and which were not attempted.
#
# Usage:
#   bash rescue_step1_conservative.sh [Step1_root] [--execute]
#
# Optional environment overrides:
#   STALE_HOURS=       file-age window in h (default unset: Slurm is verified,
#                      so recover uses its 6 min settle window; set e.g. 6 to
#                      also leave runs written in the last 6 h alone)
#   RAMP_FROM=100      recovery starting temperature in K
#   POTIM=0.5          recovery timestep in fs
#   ALGO=Normal        VASP electronic minimizer
#   USE_LANGEVIN=0     set 1 to use MDALGO=3 + Langevin friction
#   LANGEVIN_GAMMA=10  ps^-1, used only when USE_LANGEVIN=1

ROOT="Step1"
EXECUTE=""

for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE="--execute" ;;
        -h|--help)
            sed -n '3,44p' "$0"
            exit 0
            ;;
        *) ROOT="$arg" ;;
    esac
done

STALE_HOURS="${STALE_HOURS:-}"
RAMP_FROM="${RAMP_FROM:-100}"
POTIM="${POTIM:-0.5}"
ALGO="${ALGO:-Normal}"
USE_LANGEVIN="${USE_LANGEVIN:-0}"
LANGEVIN_GAMMA="${LANGEVIN_GAMMA:-10}"

for cmd in iface squeue; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' is required; refusing rescue because scheduler state cannot be verified" >&2
        exit 2
    fi
done

if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: Step1 root does not exist: $ROOT" >&2
    exit 2
fi

# Options only (never a list of runs): step1-recover discovers and classifies
# every run under ROOT itself.
OPTIONS=(--only repair --scheduler slurm --potim "$POTIM" --algo "$ALGO" --ramp-from "$RAMP_FROM")
if [[ -n "$STALE_HOURS" ]]; then
    OPTIONS+=(--stale-hours "$STALE_HOURS")
fi
if [[ "$USE_LANGEVIN" == "1" ]]; then
    OPTIONS+=(--langevin --langevin-gamma "$LANGEVIN_GAMMA")
fi
if [[ -n "$EXECUTE" ]]; then
    OPTIONS+=("$EXECUTE")
fi

echo "Running: iface vasp step1-recover $ROOT ${OPTIONS[*]}"
exec iface vasp step1-recover "$ROOT" "${OPTIONS[@]}"
