#!/usr/bin/env bash
# Drive the swap-MC -> VASP ordering benchmark on LONI without retyping paths.
#
#   ordering_dft_workflow.sh prepare EXPORT_DIR VASP_DIR REFERENCE_DIR [STAGE...]
#   ordering_dft_workflow.sh submit  VASP_DIR [STAGE]      # sbatch every prepared run
#   ordering_dft_workflow.sh report  VASP_DIR [STAGE]      # collect + compare + print
#
# EXPORT_DIR     directory written by `iface swap-mc export`
# VASP_DIR       output tree (created by prepare, reused by submit/report)
# REFERENCE_DIR  a finished VASP run for the same interface; supplies INCAR,
#                KPOINTS, POTCAR and the launcher. It is never modified.
# STAGE          static (default) and/or relax
#
# `prepare` always ends with a launch dry run: read it before running `submit`.
# Override REPO to point at a different InterfaceForge checkout.

set -euo pipefail

REPO="${REPO:-/project/lgutsev/git_develop/InterfaceForge}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
IFACE=(python -m interfaceforge.swap_mc)

if [[ "${1:-}" == -h || "${1:-}" == --help || $# -eq 0 ]]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

ACTION="$1"; shift

require_dir() {
    [[ -d "$1" ]] || { echo "ERROR: $2 is not a directory: $1" >&2; exit 2; }
}

case "$ACTION" in
prepare)
    [[ $# -ge 3 ]] || { echo "ERROR: prepare needs EXPORT_DIR VASP_DIR REFERENCE_DIR" >&2; exit 2; }
    EXPORT_DIR="$1"; VASP_DIR="$2"; REFERENCE_DIR="$3"; shift 3
    require_dir "$EXPORT_DIR" "EXPORT_DIR"
    require_dir "$REFERENCE_DIR" "REFERENCE_DIR"
    STAGE_ARGS=()
    for stage in "${@:-static}"; do STAGE_ARGS+=(--stage "$stage"); done
    "${IFACE[@]}" dft-prepare "$EXPORT_DIR" "$VASP_DIR" \
        --reference "$REFERENCE_DIR" "${STAGE_ARGS[@]}"
    for stage in "${@:-static}"; do
        echo "--- launch dry run: $stage ---"
        "${IFACE[@]}" dft-launch "$VASP_DIR" --stage "$stage"
    done
    echo "Review the plan above, then: $0 submit $VASP_DIR ${1:-static}"
    ;;
submit)
    [[ $# -ge 1 ]] || { echo "ERROR: submit needs VASP_DIR" >&2; exit 2; }
    VASP_DIR="$1"; STAGE="${2:-static}"
    require_dir "$VASP_DIR" "VASP_DIR"
    command -v sbatch >/dev/null || { echo "ERROR: no sbatch on this node" >&2; exit 2; }
    "${IFACE[@]}" dft-launch "$VASP_DIR" --stage "$STAGE" --execute
    ;;
report)
    [[ $# -ge 1 ]] || { echo "ERROR: report needs VASP_DIR" >&2; exit 2; }
    VASP_DIR="$1"; STAGE="${2:-static}"
    require_dir "$VASP_DIR" "VASP_DIR"
    "${IFACE[@]}" dft-collect "$VASP_DIR" --stage "$STAGE" >/dev/null
    "${IFACE[@]}" dft-compare "$VASP_DIR" --stage "$STAGE"
    echo "--- $VASP_DIR/$STAGE/ordering_comparison.md ---"
    cat "$VASP_DIR/$STAGE/ordering_comparison.md"
    ;;
*)
    echo "ERROR: unknown action '$ACTION' (prepare | submit | report)" >&2
    exit 2
    ;;
esac
