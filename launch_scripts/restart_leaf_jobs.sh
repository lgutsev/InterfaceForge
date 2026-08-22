#!/usr/bin/env bash

# Copy root INCAR + runvasp.sh into every leaf directory, restart each case
# from its CONTCAR, and audit whether Slurm accepted the launch.

set -u

ROOT="$(pwd -P)"
INCAR="$ROOT/INCAR"
RUNVASP="$ROOT/runvasp.sh"
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: restart_leaf_jobs.sh [--dry-run]

Run from the calculation-tree root containing INCAR and runvasp.sh.
For every deepest (leaf) directory below the root, the script:
  1. requires a non-empty CONTCAR,
  2. copies root INCAR and runvasp.sh into the leaf,
  3. runs `Restart <leaf-name>` from the leaf's parent directory,
  4. captures the Slurm job ID when possible, and
  5. writes restart_audit_YYYYmmdd_HHMMSS.tsv in the root.

--dry-run  Discover/audit leaf directories without copying or launching jobs.
USAGE
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -f "$INCAR" ]]; then
    echo "ERROR: No INCAR found in $ROOT" >&2
    exit 1
fi

if [[ ! -f "$RUNVASP" ]]; then
    echo "ERROR: No runvasp.sh found in $ROOT" >&2
    exit 1
fi

AUDIT="$ROOT/restart_audit_$(date +%Y%m%d_%H%M%S).tsv"
printf 'directory\trestart_rc\tjob_id\tscheduler_state\tresult\tdetail\n' > "$AUDIT"

# Snapshot the leaf list before any Restart invocation can create directories.
LEAVES=()
while IFS= read -r -d '' dir; do
    if ! find "$dir" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q .; then
        LEAVES+=("$dir")
    fi
done < <(find "$ROOT" -mindepth 1 -type d \
    ! -path '*/.git/*' ! -name '.git' -print0)

total=${#LEAVES[@]}
accepted=0
failed=0
skipped=0

if (( total == 0 )); then
    echo "No leaf directories found below $ROOT"
    echo "Audit: $AUDIT"
    exit 0
fi

run_restart() {
    local parent="$1"
    local leaf_name="$2"

    # If Restart is an executable or an exported shell function, use it directly.
    if command -v Restart >/dev/null 2>&1; then
        ( cd "$parent" && Restart "$leaf_name" )
        return $?
    fi

    # Otherwise fall back to an interactive Bash so ~/.bashrc aliases/functions
    # such as a personal Restart helper are available.
    ( cd "$parent" && bash -ic 'Restart "$1"' _ "$leaf_name" )
}

scheduler_state() {
    local job_id="$1"
    local state=""

    if command -v squeue >/dev/null 2>&1; then
        state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -n 1 | xargs || true)"
        if [[ -n "$state" ]]; then
            printf '%s' "$state"
            return 0
        fi
    fi

    if command -v sacct >/dev/null 2>&1; then
        state="$(sacct -n -X -j "$job_id" --format=State -P 2>/dev/null \
            | awk -F'|' 'NF && $1 != "" {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1; exit}' || true)"
        if [[ -n "$state" ]]; then
            printf '%s' "$state"
            return 0
        fi
    fi

    printf '%s' "NOT_VISIBLE"
}

for dir in "${LEAVES[@]}"; do
    parent="$(dirname "$dir")"
    leaf_name="$(basename "$dir")"

    echo "============================================================"
    echo "[$((accepted + failed + skipped + 1))/$total] $dir"

    if [[ ! -s "$dir/CONTCAR" ]]; then
        echo "SKIP: missing or empty CONTCAR"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "NA" "" "" "SKIPPED" "missing_or_empty_CONTCAR" >> "$AUDIT"
        ((skipped++))
        continue
    fi

    if (( DRY_RUN )); then
        echo "DRY RUN: would copy INCAR/runvasp.sh and run Restart $leaf_name from $parent"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "NA" "" "" "DRY_RUN" "ready" >> "$AUDIT"
        ((skipped++))
        continue
    fi

    if ! cp -f "$INCAR" "$dir/INCAR"; then
        echo "FAIL: could not copy INCAR" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "NA" "" "" "FAILED" "copy_INCAR_failed" >> "$AUDIT"
        ((failed++))
        continue
    fi

    if ! cp -f "$RUNVASP" "$dir/runvasp.sh"; then
        echo "FAIL: could not copy runvasp.sh" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "NA" "" "" "FAILED" "copy_runvasp_failed" >> "$AUDIT"
        ((failed++))
        continue
    fi
    chmod +x "$dir/runvasp.sh"

    echo "Launching: (cd $parent && Restart $leaf_name)"
    restart_output=""
    if restart_output="$(run_restart "$parent" "$leaf_name" 2>&1)"; then
        rc=0
    else
        rc=$?
    fi

    if [[ -n "$restart_output" ]]; then
        printf '%s\n' "$restart_output"
    fi

    # Standard Slurm sbatch output is: "Submitted batch job 123456".
    job_id="$(printf '%s\n' "$restart_output" \
        | sed -nE 's/.*Submitted batch job[[:space:]]+([0-9]+).*/\1/p' \
        | tail -n 1)"

    if (( rc != 0 )); then
        echo "FAIL: Restart exited with status $rc" >&2
        detail="Restart_failed"
        [[ -n "$restart_output" ]] && detail="Restart_failed:_$(printf '%s' "$restart_output" | tail -n 1 | tr '\t' ' ')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "$rc" "$job_id" "" "FAILED" "$detail" >> "$AUDIT"
        ((failed++))
        continue
    fi

    if [[ -z "$job_id" ]]; then
        echo "WARN: Restart returned success, but no Slurm job ID was found in its output." >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "$rc" "" "UNKNOWN" "UNVERIFIED" "Restart_ok_but_no_sbatch_job_id" >> "$AUDIT"
        ((failed++))
        continue
    fi

    state="$(scheduler_state "$job_id")"
    if [[ "$state" == "NOT_VISIBLE" ]]; then
        echo "ACCEPTED: job $job_id (not yet visible in squeue/sacct)"
        result="ACCEPTED_UNVERIFIED"
        detail="sbatch_job_id_captured_scheduler_lookup_not_yet_visible"
    else
        echo "ACCEPTED: job $job_id state=$state"
        result="ACCEPTED"
        detail="scheduler_confirmed"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dir" "$rc" "$job_id" "$state" "$result" "$detail" >> "$AUDIT"
    ((accepted++))
done

echo "============================================================"
echo "Restart campaign complete"
echo "  leaf directories : $total"
echo "  accepted launches: $accepted"
echo "  failed/unverified: $failed"
echo "  skipped/dry-run  : $skipped"
echo "  audit file       : $AUDIT"

(( failed == 0 ))
