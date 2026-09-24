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
#   2. OSZICAR must be older than STALE_HOURS (including normally terminated
#      jobs that completed NSW but were diagnosed physically/numerically unstable);
#   3. no active Slurm job may have that run directory as WorkDir.
#
# Scheduler efficiency:
#   Slurm state is read once per safety gate with a single `squeue` call using
#   its WorkDir field (%Z).  No per-run/per-job `scontrol` RPC loop is used.
#   A fresh second snapshot is taken immediately before mutation to close the
#   race window. This makes the script suitable for large job sets and avoids
#   excessive scheduler-controller traffic from login nodes.
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
ROOT_SET=0
EXECUTE=0

for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=1 ;;
        -h|--help)
            sed -n '3,34p' "$0"
            exit 0
            ;;
        -*)
            # A mistyped flag (e.g. --exectue) must never be taken as the root.
            echo "ERROR: unknown option: $arg (see --help)" >&2
            exit 2
            ;;
        *)
            if ((ROOT_SET)); then
                echo "ERROR: more than one Step1 root given: '$ROOT' and '$arg'" >&2
                exit 2
            fi
            ROOT="$arg"
            ROOT_SET=1
            ;;
    esac
done

STALE_HOURS="${STALE_HOURS:-6}"
RAMP_FROM="${RAMP_FROM:-100}"
POTIM="${POTIM:-0.5}"
ALGO="${ALGO:-Normal}"
USE_LANGEVIN="${USE_LANGEVIN:-0}"
LANGEVIN_GAMMA="${LANGEVIN_GAMMA:-10}"

for cmd in iface squeue realpath python; do
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
SLURM_SNAPSHOT="$(mktemp)"
trap 'rm -f "$STATUS_JSON" "$CANDIDATE_LIST" "$SLURM_SNAPSHOT"' EXIT

snapshot_slurm() {
    local destination="$1"
    # %i JobID, %T long state, %Z WorkDir. Any job returned by squeue is active
    # for rescue purposes, including pending/running/completing/requeued jobs.
    if ! squeue -h -u "${USER:?USER is not set}" -o '%i|%T|%Z' > "$destination"; then
        echo "ERROR: could not query Slurm; refusing rescue rather than guessing job state" >&2
        exit 2
    fi
}

# Emit "RUN|JOBID STATE" for candidates whose absolute path is present as an
# active Slurm WorkDir. Matching is done locally from one scheduler snapshot.
active_candidates_from_snapshot() {
    local candidates_file="$1"
    local snapshot_file="$2"
    python - "$candidates_file" "$snapshot_file" <<'PY'
import os
import sys

candidates_path, snapshot_path = sys.argv[1:3]

with open(candidates_path, encoding="utf-8") as handle:
    candidates = [line.strip() for line in handle if line.strip()]

active_by_dir: dict[str, list[tuple[str, str]]] = {}
with open(snapshot_path, encoding="utf-8") as handle:
    for raw in handle:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        fields = raw.split("|", 2)
        if len(fields) != 3:
            continue
        job_id, state, workdir = fields
        if not workdir or workdir in {"N/A", "(null)"}:
            continue
        target = os.path.realpath(workdir)
        active_by_dir.setdefault(target, []).append((job_id, state))

for run in candidates:
    target = os.path.realpath(run)
    for job_id, state in active_by_dir.get(target, []):
        print(f"{run}|{job_id} {state}")
        break
PY
}

iface vasp step1-status "$ROOT" --stale-hours "$STALE_HOURS" --json > "$STATUS_JSON"

python - "$STATUS_JSON" > "$CANDIDATE_LIST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

for row in payload.get("runs", []):
    stability = row.get("stability") or {}
    age_hours = row.get("age_hours")
    inactive_by_age = age_hours is not None and age_hours >= float(payload["stale_hours"])
    if stability.get("unstable") and inactive_by_age:
        print(row["path"])
PY

mapfile -t CANDIDATES < "$CANDIDATE_LIST"

if ((${#CANDIDATES[@]} == 0)); then
    echo "No unstable leaves older than ${STALE_HOURS} h were found under $ROOT."
    echo "Running/recent jobs were intentionally left untouched."
    exit 0
fi

snapshot_slurm "$SLURM_SNAPSHOT"
mapfile -t ACTIVE_ROWS < <(active_candidates_from_snapshot "$CANDIDATE_LIST" "$SLURM_SNAPSHOT")

declare -A ACTIVE_INFO=()
for item in "${ACTIVE_ROWS[@]}"; do
    ACTIVE_INFO["${item%%|*}"]="${item#*|}"
done

RUNS=()
ACTIVE_SKIPPED=()
for run in "${CANDIDATES[@]}"; do
    if [[ -n "${ACTIVE_INFO[$run]:-}" ]]; then
        ACTIVE_SKIPPED+=("$run|${ACTIVE_INFO[$run]}")
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
    # Race-condition guard: take one fresh scheduler snapshot immediately before
    # touching any leaf. Abort the whole batch rather than partially mutate it.
    printf '%s\n' "${RUNS[@]}" > "$CANDIDATE_LIST"
    snapshot_slurm "$SLURM_SNAPSHOT"
    mapfile -t RACE < <(active_candidates_from_snapshot "$CANDIDATE_LIST" "$SLURM_SNAPSHOT")
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
    # step1-repair validates a leaf before mutating it, so a rejected leaf is
    # left untouched. Keep going, then submit exactly the leaves that were
    # repaired: stopping here would strand them rewound but never submitted,
    # and a later rescue could not find them (their OSZICAR is archived).
    REPAIRED=()
    FAILED=()
    for run in "${RUNS[@]}"; do
        if iface vasp step1-repair "$run" "${REPAIR_ARGS[@]}" --execute; then
            REPAIRED+=("$run")
        else
            FAILED+=("$run")
        fi
    done

    if ((${#FAILED[@]})); then
        echo >&2
        echo "ERROR: step1-repair failed for ${#FAILED[@]} leaves; they will not be submitted:" >&2
        printf '  %s\n' "${FAILED[@]}" >&2
    fi

    if ((${#REPAIRED[@]})); then
        echo
        echo "Submitting only repaired leaves..."
        iface vasp step1-launch "${REPAIRED[@]}" --only-repaired --execute
    fi

    if ((${#FAILED[@]})); then
        exit 4
    fi
else
    echo "DRY RUN: showing repair plans; nothing will be changed or submitted."
    for run in "${RUNS[@]}"; do
        iface vasp step1-repair "$run" "${REPAIR_ARGS[@]}"
    done
    echo
    echo "If those plans look right, rerun with --execute:"
    printf '  bash %q %q --execute\n' "$0" "$ROOT"
fi
