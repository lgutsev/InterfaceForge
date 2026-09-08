#!/usr/bin/env bash
# Assemble one interface/ slab_a/ slab_b/ set directory for the bulk-referenced
# control:  iface validate separation-energy --reference bulk
#
#   prepare_interface_excess.sh [--force] SET_DIR INTERFACE_SRC BULK_A_SRC BULK_B_SRC
#
# INTERFACE_SRC may be a finished VASP run directory or an
# `iface vasp adhesion prepare` tree (its interface_static/ is used).
# BULK_A_SRC / BULK_B_SRC are the two finished bulk-cell VASP runs whose
# compositions sum to the interface (e.g. TiN and beta-Si3N4). Files are copied,
# not symlinked, so the set is self-contained and container-safe.
#
# Run once per termination, then point SEPARATION_ENTRIES_FILE at a file listing
# the resulting "LABEL=SET_DIR" lines (see submit_interface_excess.sh).

set -euo pipefail

FORCE=0
if [[ "${1:-}" == --force ]]; then FORCE=1; shift; fi
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi
[[ "$#" -eq 4 ]] || { echo "ERROR: expected 4 arguments (see --help)" >&2; exit 2; }

SET_DIR="$1"; INTERFACE_SRC="$2"; BULK_A_SRC="$3"; BULK_B_SRC="$4"
COPY_NAMES=(INCAR OUTCAR OSZICAR CONTCAR POSCAR KPOINTS vasprun.xml)

resolve_run() {
    # Echo the directory that actually holds the OUTCAR for a source argument.
    local src="$1" role="$2"
    [[ -d "$src" ]] || { echo "ERROR: $role source is not a directory: $src" >&2; return 2; }
    src="$(cd -- "$src" && pwd -P)"
    if [[ -f "$src/manifest.json" && -d "$src/interface_static" ]]; then
        echo "$src/interface_static"; return 0
    fi
    echo "$src"
}

stage_part() {
    local src="$1" dest="$2" role="$3" name found=0
    src="$(resolve_run "$src" "$role")" || return 2
    [[ -f "$src/OUTCAR" && -s "$src/OUTCAR" ]] || {
        echo "ERROR: $role has no non-empty OUTCAR: $src" >&2; return 2; }
    [[ -f "$src/CONTCAR" && -s "$src/CONTCAR" ]] || [[ -f "$src/POSCAR" && -s "$src/POSCAR" ]] || {
        echo "ERROR: $role has neither CONTCAR nor POSCAR: $src" >&2; return 2; }
    mkdir -p "$dest"
    for name in "${COPY_NAMES[@]}"; do
        if [[ -f "$src/$name" ]]; then cp -f -- "$src/$name" "$dest/$name"; found=1; fi
    done
    (( found )) || { echo "ERROR: nothing to copy from $src" >&2; return 2; }
    printf '  %-9s <- %s\n' "$role" "$src"
}

if [[ -e "$SET_DIR" ]]; then
    (( FORCE )) || { echo "ERROR: $SET_DIR exists; pass --force to replace it" >&2; exit 2; }
    rm -rf -- "$SET_DIR"
fi
mkdir -p "$SET_DIR"
SET_DIR="$(cd -- "$SET_DIR" && pwd -P)"

echo "Assembling $SET_DIR"
stage_part "$INTERFACE_SRC" "$SET_DIR/interface" interface
stage_part "$BULK_A_SRC"    "$SET_DIR/slab_a"    slab_a
stage_part "$BULK_B_SRC"    "$SET_DIR/slab_b"    slab_b
echo "Done. Use with: iface validate separation-energy <out> \"LABEL=$SET_DIR\" --reference bulk --n-interfaces 1"
