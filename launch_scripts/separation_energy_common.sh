#!/usr/bin/env bash
# Shared by the login-node submitter and the batch jobs. No ML imports here.

sep_file() {
    [[ -f "$1" && -r "$1" && -s "$1" ]] || {
        echo "ERROR: missing, empty, or unreadable file: $1" >&2
        return 2
    }
    [[ "$1" != *$'\n'* && "$1" != *$'\r'* ]] || {
        echo 'ERROR: model paths cannot contain line breaks' >&2
        return 2
    }
}

sep_campaign() {
    CAMP="$(cd -- "$CAMP" && pwd -P)" || return 2
    for required in "$CAMP/campaign.yaml" \
        "$CAMP/adhesion/N_term_dft/manifest.json" \
        "$CAMP/adhesion/Ti_term_dft/manifest.json"; do
        sep_file "$required" || return 2
    done
}

sep_mace_member() {
    local seed_dir="$1" seed="$2" search_dir candidate newest
    local -a stage=() base=() matches=()
    search_dir="$seed_dir/mace_model"
    [[ -d "$search_dir" ]] || search_dir="$seed_dir"
    # Only exports immediately inside mace_model/ (or the legacy seed root).
    # Do not recurse into checkpoints, backups, or archived attempts. Globbing
    # includes file symlinks; -f/-s reject dangling links and empty exports.
    for candidate in "$search_dir"/*.model; do
        [[ -f "$candidate" && -r "$candidate" && -s "$candidate" ]] || continue
        [[ "${candidate##*/}" != *compiled* ]] || continue
        sep_file "$candidate" || return 2
        base+=("$candidate")
        case "${candidate##*/}" in
            *_stagetwo.model|*_stage_two.model|*_stage2.model|*_swa.model) stage+=("$candidate");;
        esac
    done
    if (( ${#stage[@]} )); then matches=("${stage[@]}"); else matches=("${base[@]}"); fi
    if (( ! ${#matches[@]} )); then
        echo "ERROR: no usable MACE export for seed $seed; searched: $search_dir" >&2
        echo 'Expected a nonempty uncompiled .model; checkpoint files are not final exports.' >&2
        return 2
    fi
    newest="${matches[0]}"
    for candidate in "${matches[@]:1}"; do
        [[ "$candidate" -nt "$newest" ]] && newest="$candidate"
    done
    if (( ${#matches[@]} > 1 )); then
        echo "WARNING: seed $seed has ${#matches[@]} usable MACE models; using the newest: $newest" >&2
    fi
    if (( ! ${#stage[@]} )); then
        echo "WARNING: seed $seed has no stage-two-named model; using $newest" >&2
    fi
    SELECTED_MODEL="$newest"
}

sep_mace_at_root() {
    local seed
    MACE_MODELS=()
    for seed in 11 23 37 53; do
        sep_mace_member "$1/seed_$seed" "$seed" || return 2
        MACE_MODELS+=("$SELECTED_MODEL")
    done
}

sep_mace() {
    local root candidate
    local -a candidates=() ready=()
    if [[ -n "${MACE_COMMITTEE_ROOT:-}" ]]; then
        root="$MACE_COMMITTEE_ROOT"
        [[ "$root" = /* ]] || root="$CAMP/$root"
        # Accept either the ENCUT-tagged parent (existing iface convention)
        # or the directory that directly contains seed_*.
        if [[ -d "$root/mace_committee" && ! -d "$root/seed_11" ]]; then
            root="$root/mace_committee"
        fi
    else
        candidates=("$CAMP/models/mace_committee_520eV/mace_committee" "$CAMP/models/mace_committee")
        for candidate in "${candidates[@]}"; do
            if (sep_mace_at_root "$candidate") >/dev/null 2>&1; then ready+=("$candidate"); fi
        done
        if (( ${#ready[@]} != 1 )); then
            echo "ERROR: found ${#ready[@]} complete MACE committees; set MACE_COMMITTEE_ROOT explicitly." >&2
            for candidate in "${candidates[@]}"; do
                echo "  checked: $candidate" >&2
                (sep_mace_at_root "$candidate") >/dev/null || true
            done
            return 2
        fi
        root="${ready[0]}"
    fi
    MACE_COMMITTEE_ROOT="$(cd -- "$root" && pwd -P)" || return 2
    sep_mace_at_root "$MACE_COMMITTEE_ROOT" || return 2
}

sep_deepmd() {
    local member dir model
    DEEPMD_MODELS=()
    for member in 000 001 002 003; do
        dir="$CAMP/models/deepmd/dpa2/model_$member"
        if [[ -f "$dir/frozen_model.pth" && -r "$dir/frozen_model.pth" && -s "$dir/frozen_model.pth" ]]; then
            model="$dir/frozen_model.pth"
        else
            model="$dir/model.ckpt.pt"
            sep_file "$model" || return 2
            echo "WARNING: member $member has no usable frozen export; using checkpoint: $model" >&2
        fi
        sep_file "$model" || return 2
        DEEPMD_MODELS+=("$model")
    done
}

sep_read_models() {
    local model
    sep_file "$1" || return 2
    mapfile -t SELECTED_MODELS < "$1"
    [[ "${#SELECTED_MODELS[@]}" -eq 4 ]] || {
        echo "ERROR: expected four pinned models in $1" >&2; return 2;
    }
    for model in "${SELECTED_MODELS[@]}"; do sep_file "$model" || return 2; done
}
