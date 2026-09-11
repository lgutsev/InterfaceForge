#!/usr/bin/env bash
# Shared by submit_interface_mu.sh (login node) and interface_mu_deepmd.sbatch
# (batch job), so the preflight and the job agree on what a valid input is.
# No ML imports here. Requires separation_energy_common.sh to be sourced first
# for sep_file / sep_bind_roots.

# Resolved absolute paths of everything a run touches, for APPTAINER_BIND.
IFMU_PATHS=()

# ifmu_read_pairs FILE [FLAG]
# Read a list of NAME=DIR lines, resolve each DIR against $CAMP, require it to
# hold a readable OUTCAR, and build argv in the global PAIRS array -- with FLAG
# interleaved before each pair when given (--phase / --aux-phase), or bare pairs
# when not (the positional LABEL=DIR interface entries).
#
# Populates the global PAIRS and appends to IFMU_PATHS. Deliberately not a
# nameref (local -n): that needs bash 4.3 and RHEL7 login nodes ship 4.2.
ifmu_read_pairs() {
    local file="$1" flag="${2:-}" line name dir
    PAIRS=()
    sep_file "$file" || return 2
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%$'\r'}"                      # tolerate CRLF
        [[ -n "${line//[[:space:]]/}" ]] || continue
        [[ "$line" != \#* ]] || continue
        name="${line%%=*}"
        dir="${line#*=}"
        [[ "$name" != "$line" && -n "$name" && -n "$dir" ]] || {
            echo "ERROR: $file: expected NAME=DIR, got: $line" >&2
            return 2
        }
        [[ "$name" != *[[:space:]]* ]] || {
            echo "ERROR: $file: name cannot contain whitespace: $name" >&2
            return 2
        }
        [[ "$dir" = /* ]] || dir="$CAMP/$dir"
        [[ -d "$dir" ]] || { echo "ERROR: $file: no such directory: $dir" >&2; return 2; }
        dir="$(cd -- "$dir" && pwd -P)" || return 2
        sep_file "$dir/OUTCAR" || {
            echo "  ($name has no readable OUTCAR; interface-mu needs finished runs)" >&2
            return 2
        }
        IFMU_PATHS+=("$dir")
        if [[ -n "$flag" ]]; then PAIRS+=("$flag" "$name=$dir"); else PAIRS+=("$name=$dir"); fi
    done < "$file"
    (( ${#PAIRS[@]} )) || { echo "ERROR: $file has no entries" >&2; return 2; }
}

# ifmu_committee ROOT [MEMBERS...]
# Resolve a DeePMD committee to frozen inference exports only. A training
# checkpoint is not deployable, and silently evaluating one would put an
# untracked model behind every number in the audit.
#
# Populates the global MODEL_ARGS and appends to IFMU_PATHS.
ifmu_committee() {
    local root="$1" member model failed=0
    shift
    local -a members=("$@")
    (( ${#members[@]} )) || members=(000 001 002 003)
    [[ "$root" = /* ]] || root="$CAMP/$root"
    [[ -d "$root" ]] || { echo "ERROR: no committee directory at $root" >&2; return 2; }
    root="$(cd -- "$root" && pwd -P)" || return 2
    IFMU_COMMITTEE_ROOT="$root"
    MODEL_ARGS=()
    for member in "${members[@]}"; do
        model="$root/model_$member/frozen_model.pth"
        if ! sep_file "$model"; then
            echo "ERROR: member $member is not a frozen inference export: $model" >&2
            failed=1
            continue
        fi
        IFMU_PATHS+=("$model")
        MODEL_ARGS+=(--deepmd-model "$model")
    done
    if (( failed )); then
        echo 'A training checkpoint is not an inference export. Freeze the missing' >&2
        echo 'members first (see launch_scripts/freeze_missing_deepmd_dpa2.sbatch).' >&2
        return 2
    fi
    (( ${#members[@]} >= 2 )) || echo \
        "WARNING: ${#members[@]} member(s): committee_spread_j_per_m2 will be 0, so the audit reports no uncertainty" >&2
}

# ifmu_inputs
# The whole input preflight, shared so --dry-run validates exactly what the job
# will do. Reads IFACE_MU_ENTRIES_FILE / IFACE_MU_PHASES_FILE /
# IFACE_MU_AUX_FILE and DEEPMD_COMMITTEE_ROOT / DEEPMD_COMMITTEE_MEMBERS;
# populates ENTRY_ARGS, PHASE_ARGS, AUX_ARGS, MODEL_ARGS and IFMU_PATHS.
ifmu_inputs() {
    ifmu_read_pairs "${IFACE_MU_ENTRIES_FILE:?Set IFACE_MU_ENTRIES_FILE}" || return 2
    ENTRY_ARGS=("${PAIRS[@]}")
    ifmu_read_pairs "${IFACE_MU_PHASES_FILE:?Set IFACE_MU_PHASES_FILE}" --phase || return 2
    PHASE_ARGS=("${PAIRS[@]}")
    AUX_ARGS=()
    if [[ -n "${IFACE_MU_AUX_FILE:-}" ]]; then
        ifmu_read_pairs "$IFACE_MU_AUX_FILE" --aux-phase || return 2
        AUX_ARGS=("${PAIRS[@]}")
    fi
    # interface-mu needs one compound per cation plus the elemental anion; a
    # phase list that is obviously too short fails here rather than in the job.
    (( ${#PHASE_ARGS[@]} >= 6 )) || {
        echo "ERROR: ${IFACE_MU_PHASES_FILE}: only $(( ${#PHASE_ARGS[@]} / 2 )) phase(s)." >&2
        echo 'Expected at least one compound per cation (TiN, Si3N4), the elemental' >&2
        echo 'anion (N2) and one elemental cation per compound (Ti, Si).' >&2
        return 2
    }
    # shellcheck disable=SC2206  # deliberate word-splitting of the member list
    ifmu_committee "${DEEPMD_COMMITTEE_ROOT:-models/deepmd/dpa2_ft}" \
        ${DEEPMD_COMMITTEE_MEMBERS:-} || return 2
}
