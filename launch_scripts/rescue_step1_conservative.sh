#!/usr/bin/env bash
set -euo pipefail

# Rescue stopped/unstable Step1 AIMD leaves with InterfaceForge's conservative
# recovery settings. Dry-run by default; pass --execute to mutate + resubmit.
#
# Defaults are tuned for proton-rich magnetic DFT+U surfaces such as NiO/OH50:
#   POTIM=0.5 fs, ALGO=Normal, EDIFF=1e-5, NELM=120, NELMIN=6
#   precondition the magnetic DFT+U state, then ramp 100 K -> target T.
# NBLOCK is deliberately left unchanged (normally 4).
#
# Safety gates:
#   1. run must be diagnosed unstable by `iface vasp step1-status`;
#   2. OSZICAR must be older than STALE_HOURS;
#   3. no active Slurm job may have that run directory as WorkDir.
# The Slurm WorkDir check is repeated immediately before any repair is written,
# so a RUNNING/PENDING/COMPLETING/requeued job is never mutated underneath Slurm.
#
# Usage:
#   bash rescue_step1_conservative.sh [Step1_root] [--execute]
#
# Optional environment overrides:
#   STALE_HOURS=6      minimum OSZICAR age before a leaf is considered stopped
#   RAMP_FROM=100      recovery starting temperature in K
#   POTIM=0.5          recovery timestep in fs
#   ALGO=Normal        VASP electronic minimizer
#   USE_LANGEVIN=0     set 1 to use MDALGO=3 + Langevin friction
#   LANGEVIN_GAMMA=10  ps^-1, used only when USE_LANGEVIN=1

ROOT="Step1"
EXECUTE=0

for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=1 ;;
        -h|--help)
            sed -n '3,31p' "$0"
            exit 0
            ;;
        *) ROOT="$arg" ;;
    esac
done

STALE_HOURS="${STALE_HOURS:-6}"
RAMP_FROM="${RAMP_FROM:-100}"
POTIM="${POTIM:-0.5}"
ALGO="${ALGO:-Normal}"
USE_LANGEVIN="${USE_LANGEVIN:-0}"
LANGEVIN_GAMMA="${LANGEVIN_GAMMA:-10}"

for cmd in iface squeue scontrol realpath; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' is required; refusing rescue because scheduler state cannot be verified" >&2
        exit 2
    fi
done

if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: Step1 root does not exist: $ROOT" >&2
    exit 2
fi

ROOT="$(realpath "$ROOT")"
STATUS_JSON="$(mktemp)"
CANDIDATE_LIST="$(mktemp)"
trap 'rm -f "$STATUS_JSON" "$CANDIDATE_LIST"' EXIT

# Return "JOBID STATE" for an active Slurm job whose WorkDir is exactly the
# supplied run directory. A job present in squeue is considered active for
# rescue purposes regardless of state (PENDING/RUNNING/COMPLETING/etc.).
slurm_active_for_run() {
    local run target job_id state record workdir
    run="$1"
    target="$(realpath "$run")"

    while read -r job_id state; do
        [[ -n "$job_id" ]] || continue
        record="$(scontrol show job -o "$job_id" 2>/dev/null || true)"
        [[ -n "$record" ]] || continue
        workdir="$(printf '%s\n' "$record" | sed -n 's/.* WorkDir=\([^ ]*\).*/\1/p')"
        [[ -n "$workdir" ]] || continue
        if [[ "$(realpath -m "$workdir")" == "$target" ]]; then
            printf '%s %s\n' "$job_id" "$state"
            return 0
        fi
    done < <(squeue -h -u "${USER:?USER is not set}" -o '%i %T')
    return 1
}

iface vasp step1-status "$ROOT" --stale-hours "$STALE_HOURS" --json > "$STATUS_JSON"

python - "$STATUS_JSON" > "$CANDIDATE_LIST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

for row in payload.get("runs", []):
    stability = row.get("stability") or {}
    if stability.get("unstable") and row.get("stale"):
        print(row["path"])
PY

mapfile -t CANDIDATES < "$CANDIDATE_LIST"

if ((${#CANDIDATES[@]} == 0)); then
    echo "No unstable leaves older than ${STALE_HOURS} h were found under $ROOT."
    echo "Running/recent jobs were intentionally left untouched."
    exit 0
fi

RUNS=()
ACTIVE_SKIPPED=()
for run in "${CANDIDATES[@]}"; do
    if info="$(slurm_active_for_run "$run")"; then
        ACTIVE_SKIPPED+=("$run|$info")
    else
        RUNS+=("$run")
    fi
done

if ((${#ACTIVE_SKIPPED[@]})); then
    echo "Skipping ${#ACTIVE_SKIPPED[@]} unstable/stale leaves that are STILL ACTIVE in Slurm:"
    for item in "${ACTIVE_SKIPPED[@]}"; do
        run="${item%%|*}"
        info="${item#*|}"
        echo "  $run  [job $info]"
    done
    echo
fi

if ((${#RUNS[@]} == 0)); then
    echo "No stopped unstable leaves are safe to repair yet."
    echo "Re-run after the active Slurm jobs have exited."
    exit 0
fi

echo "Selected ${#RUNS[@]} stopped/unstable Step1 leaves (not present in squeue):"
printf '  %s\n' "${RUNS[@]}"
echo

REPAIR_ARGS=(
    --potim "$POTIM"
    --algo "$ALGO"
    --precondition
    --ramp-from "$RAMP_FROM"
    --stale-hours "$STALE_HOURS"
)

if [[ "$USE_LANGEVIN" == "1" ]]; then
    REPAIR_ARGS+=(--langevin --langevin-gamma "$LANGEVIN_GAMMA")
fi

if ((EXECUTE)); then
    # Race-condition guard: do a second Slurm WorkDir check immediately before
    # touching any leaf. Abort the whole batch rather than partially mutate it.
    RACE=()
    for run in "${RUNS[@]}"; do
        if info="$(slurm_active_for_run "$run")"; then
            RACE+=("$run|$info")
        fi
    done
    if ((${#RACE[@]})); then
        echo "ERROR: one or more selected jobs became active in Slurm after preflight; nothing was changed:" >&2
        for item in "${RACE[@]}"; do
            run="${item%%|*}"
            info="${item#*|}"
            echo "  $run  [job $info]" >&2
        done
        exit 3
    fi

    echo "Preparing conservative repairs..."
    for run in "${RUNS[@]}"; do
        iface vasp step1-repair "$run" "${REPAIR_ARGS[@]}" --execute
    done

    echo
    echo "Submitting only repaired leaves..."
    iface vasp step1-launch "${RUNS[@]}" --only-repaired --execute
else
    echo "DRY RUN: showing repair plans; nothing will be changed or submitted."
    for run in "${RUNS[@]}"; do
        iface vasp step1-repair "$run" "${REPAIR_ARGS[@]}"
    done
    echo
    echo "If those plans look right, rerun with --execute:"
    printf '  bash %q %q --execute\n' "$0" "$ROOT"
fi
