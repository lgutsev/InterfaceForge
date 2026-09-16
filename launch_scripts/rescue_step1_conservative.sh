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
            sed -n '3,24p' "$0"
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

if ! command -v iface >/dev/null 2>&1; then
    echo "ERROR: 'iface' is not on PATH" >&2
    exit 2
fi

if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: Step1 root does not exist: $ROOT" >&2
    exit 2
fi

STATUS_JSON="$(mktemp)"
RUN_LIST="$(mktemp)"
trap 'rm -f "$STATUS_JSON" "$RUN_LIST"' EXIT

iface vasp step1-status "$ROOT" --stale-hours "$STALE_HOURS" --json > "$STATUS_JSON"

python - "$STATUS_JSON" > "$RUN_LIST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

for row in payload.get("runs", []):
    stability = row.get("stability") or {}
    if stability.get("unstable") and row.get("stale"):
        print(row["path"])
PY

mapfile -t RUNS < "$RUN_LIST"

if ((${#RUNS[@]} == 0)); then
    echo "No unstable leaves older than ${STALE_HOURS} h were found under $ROOT."
    echo "Running/recent jobs were intentionally left untouched."
    exit 0
fi

echo "Selected ${#RUNS[@]} stopped/unstable Step1 leaves:"
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
