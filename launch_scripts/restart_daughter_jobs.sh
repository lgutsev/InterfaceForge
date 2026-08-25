#!/usr/bin/env bash

# Prepare and restart all immediate daughter calculation directories.
# Modes:
#   restart       -> invoke Restart <daughter>
#   total-restart -> preserve CONTCAR, then invoke TotalRestart <daughter>

set -u
set -o pipefail

ROOT="$(pwd -P)"
MODE=""
DRY_RUN=0
COPY_INPUTS=1
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<'USAGE'
Usage:
  restart_daughter_jobs.sh restart [--dry-run] [--no-copy]
  restart_daughter_jobs.sh total-restart [--dry-run] [--no-copy]

Run from the campaign root. Only immediate daughter directories (one level deep) that contain POSCAR or
CONTCAR are processed. Directories beginning with X and directories whose names
contain "backup" (case-insensitive) are skipped.

By default, root INCAR, KPOINTS, and runvasp.sh are copied into every selected
daughter before launching the restart command.

Modes:
  restart
      Requires a non-empty daughter CONTCAR and runs:
          Restart <daughter-name>
      Afterward, verifies that POSCAR matches the pre-launch CONTCAR.

  total-restart
      Saves a wrapper-level backup named
          CONTCAR.pre_TotalRestart_<timestamp>_<pid>
      when a non-empty CONTCAR exists, then runs:
          TotalRestart <daughter-name>
      Afterward, checks that WAVECAR, CHGCAR, and CONTCAR were removed.

Options:
  --dry-run   Show and audit what would happen; make no changes and launch nothing.
  --no-copy   Do not copy INCAR, KPOINTS, or runvasp.sh from the root.
  -h, --help  Show this help.

A timestamped daughter_restart_audit_*.tsv is written in the campaign root.
USAGE
}

if (($# == 0)); then
    usage >&2
    exit 2
fi

case "$1" in
    restart|Restart)
        MODE="restart"
        shift
        ;;
    total-restart|totalrestart|TotalRestart)
        MODE="total-restart"
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "ERROR: first argument must be 'restart' or 'total-restart' (got: $1)" >&2
        usage >&2
        exit 2
        ;;
esac

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --no-copy)
            COPY_INPUTS=0
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

COPY_FILES=(INCAR KPOINTS runvasp.sh)
if (( COPY_INPUTS )); then
    missing=0
    for file in "${COPY_FILES[@]}"; do
        if [[ ! -f "$ROOT/$file" ]]; then
            echo "ERROR: required root input is missing: $ROOT/$file" >&2
            missing=1
        fi
    done
    (( missing == 0 )) || exit 1
fi

AUDIT="$ROOT/daughter_restart_audit_${TIMESTAMP}.tsv"
printf 'directory\tmode\tcontcar_backup\tcommand_rc\tjob_id\tscheduler_state\tresult\tdetail\n' > "$AUDIT"

DAUGHTERS=()
while IFS= read -r -d '' dir; do
    name="$(basename "$dir")"
    [[ "$name" == X* ]] && continue
    [[ "${name,,}" == *backup* ]] && continue
    # Avoid firing restart helpers at unrelated one-level utility directories.
    [[ -e "$dir/POSCAR" || -e "$dir/CONTCAR" ]] || continue
    DAUGHTERS+=("$dir")
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

total=${#DAUGHTERS[@]}
accepted=0
warned=0
failed=0
skipped=0

run_restart_helper() {
    local daughter_name="$1"

    if [[ "$MODE" == "restart" ]]; then
        if command -v Restart >/dev/null 2>&1; then
            ( cd "$ROOT" && Restart "$daughter_name" )
        else
            # Interactive Bash loads ~/.bashrc so a personal alias/function can be used.
            ( cd "$ROOT" && bash -ic 'Restart "$1"' _ "$daughter_name" )
        fi
    else
        if command -v TotalRestart >/dev/null 2>&1; then
            ( cd "$ROOT" && TotalRestart "$daughter_name" )
        else
            ( cd "$ROOT" && bash -ic 'TotalRestart "$1"' _ "$daughter_name" )
        fi
    fi
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

for dir in "${DAUGHTERS[@]}"; do
    name="$(basename "$dir")"
    backup=""
    detail=""

    echo "============================================================"
    echo "[$((accepted + warned + failed + skipped + 1))/$total] $name"

    if [[ "$MODE" == "restart" && ! -s "$dir/CONTCAR" ]]; then
        echo "SKIP: Restart requires a non-empty CONTCAR"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "$MODE" "" "NA" "" "" "SKIPPED" "missing_or_empty_CONTCAR" >> "$AUDIT"
        ((skipped++))
        continue
    fi

    if (( DRY_RUN )); then
        if (( COPY_INPUTS )); then
            echo "DRY RUN: would copy INCAR KPOINTS runvasp.sh -> $name/"
        fi
        if [[ "$MODE" == "total-restart" && -s "$dir/CONTCAR" ]]; then
            echo "DRY RUN: would preserve CONTCAR before TotalRestart"
        fi
        if [[ "$MODE" == "restart" ]]; then
            echo "DRY RUN: would execute Restart $name"
        else
            echo "DRY RUN: would execute TotalRestart $name"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "$MODE" "" "NA" "" "" "DRY_RUN" "ready" >> "$AUDIT"
        ((skipped++))
        continue
    fi

    if (( COPY_INPUTS )); then
        copy_failed=0
        for file in "${COPY_FILES[@]}"; do
            if ! cp -f "$ROOT/$file" "$dir/$file"; then
                echo "FAIL: could not copy $file into $name" >&2
                detail="copy_${file}_failed"
                copy_failed=1
                break
            fi
        done
        if (( copy_failed )); then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$dir" "$MODE" "" "NA" "" "" "FAILED" "$detail" >> "$AUDIT"
            ((failed++))
            continue
        fi
        chmod +x "$dir/runvasp.sh"
    fi

    # Preserve the exact CONTCAR that Restart should promote to POSCAR, or that
    # TotalRestart is allowed to remove. This also gives us a post-launch verifier.
    verify_contcar=""
    if [[ -s "$dir/CONTCAR" ]]; then
        verify_contcar="$(mktemp "$ROOT/.interfaceforge_contcar_verify.XXXXXX")"
        if ! cp -p "$dir/CONTCAR" "$verify_contcar"; then
            echo "FAIL: could not create temporary CONTCAR verification copy" >&2
            rm -f "$verify_contcar"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$dir" "$MODE" "" "NA" "" "" "FAILED" "verification_copy_failed" >> "$AUDIT"
            ((failed++))
            continue
        fi
    fi

    if [[ "$MODE" == "total-restart" && -n "$verify_contcar" ]]; then
        backup="$dir/CONTCAR.pre_TotalRestart_${TIMESTAMP}_$$"
        if ! cp -p "$verify_contcar" "$backup"; then
            echo "FAIL: could not preserve CONTCAR backup: $backup" >&2
            rm -f "$verify_contcar"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$dir" "$MODE" "$backup" "NA" "" "" "FAILED" "CONTCAR_backup_failed" >> "$AUDIT"
            ((failed++))
            continue
        fi
        echo "Saved: ${backup#$ROOT/}"
    fi

    if [[ "$MODE" == "restart" ]]; then
        echo "Launching: Restart $name"
    else
        echo "Launching: TotalRestart $name"
    fi

    restart_output=""
    if restart_output="$(run_restart_helper "$name" 2>&1)"; then
        rc=0
    else
        rc=$?
    fi

    [[ -z "$restart_output" ]] || printf '%s\n' "$restart_output"

    job_id="$(printf '%s\n' "$restart_output" \
        | sed -nE 's/.*Submitted batch job[[:space:]]+([0-9]+).*/\1/p' \
        | tail -n 1)"

    if (( rc != 0 )); then
        detail="restart_helper_failed"
        [[ -z "$restart_output" ]] || detail="restart_helper_failed:_$(printf '%s' "$restart_output" | tail -n 1 | tr '\t' ' ')"
        echo "FAIL: restart helper exited with status $rc" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dir" "$MODE" "$backup" "$rc" "$job_id" "" "FAILED" "$detail" >> "$AUDIT"
        rm -f "$verify_contcar"
        ((failed++))
        continue
    fi

    verification="ok"
    if [[ "$MODE" == "restart" ]]; then
        if [[ -z "$verify_contcar" || ! -f "$dir/POSCAR" ]] || ! cmp -s "$verify_contcar" "$dir/POSCAR"; then
            verification="Restart_returned_success_but_POSCAR_does_not_match_prelaunch_CONTCAR"
        fi
    else
        leftovers=()
        for file in WAVECAR CHGCAR CONTCAR; do
            [[ -e "$dir/$file" ]] && leftovers+=("$file")
        done
        if ((${#leftovers[@]})); then
            verification="TotalRestart_returned_success_but_files_remain:$(IFS=,; echo "${leftovers[*]}")"
        fi
    fi
    rm -f "$verify_contcar"

    state=""
    if [[ -n "$job_id" ]]; then
        state="$(scheduler_state "$job_id")"
    else
        state="NO_JOB_ID"
    fi

    if [[ "$verification" != "ok" ]]; then
        echo "WARN: $verification" >&2
        result="ACCEPTED_WITH_WARNING"
        detail="$verification"
        ((warned++))
    elif [[ -z "$job_id" ]]; then
        echo "WARN: helper returned success, but no Slurm job ID was found" >&2
        result="ACCEPTED_UNVERIFIED"
        detail="helper_ok_but_no_sbatch_job_id"
        ((warned++))
    elif [[ "$state" == "NOT_VISIBLE" ]]; then
        echo "ACCEPTED: job $job_id (not yet visible in squeue/sacct)"
        result="ACCEPTED_UNVERIFIED"
        detail="sbatch_job_id_captured_scheduler_lookup_not_yet_visible"
        ((accepted++))
    else
        echo "ACCEPTED: job $job_id state=$state"
        result="ACCEPTED"
        detail="helper_and_postcondition_verified"
        ((accepted++))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dir" "$MODE" "$backup" "$rc" "$job_id" "$state" "$result" "$detail" >> "$AUDIT"
done

echo "============================================================"
echo "Daughter restart campaign complete"
echo "  mode               : $MODE"
echo "  daughter directories: $total"
echo "  accepted            : $accepted"
echo "  warnings            : $warned"
echo "  failed              : $failed"
echo "  skipped/dry-run     : $skipped"
echo "  audit file          : $AUDIT"

(( failed == 0 ))