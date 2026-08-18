#!/bin/bash
set -euo pipefail

# Text-only rebuild records: no package caches or copied environments.
OUTPUT_ROOT=${1:-/project/lgutsev/env_specs}
STAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT="$OUTPUT_ROOT/interfaceforge-envs-$STAMP"
mkdir -p "$OUTPUT"

source "$(conda info --base)/etc/profile.d/conda.sh"

ENVIRONMENTS=(
  /project/lgutsev/env/iface_ai2kit_controller
  /project/lgutsev/env/iface_mace_runtime
  /project/lgutsev/env/lgutsev_dev
  /project/lgutsev/env/mace_env
)

for prefix in "${ENVIRONMENTS[@]}"; do
    [[ -f "$prefix/conda-meta/history" ]] || {
        echo "Skipping incomplete/non-Conda prefix: $prefix" >&2
        continue
    }
    name=$(basename "$prefix")
    echo "Exporting $name"
    conda env export --prefix "$prefix" --no-builds > "$OUTPUT/$name.yml"
    conda list --prefix "$prefix" --explicit > "$OUTPUT/$name.conda-explicit.txt"
    "$prefix/bin/python" -m pip freeze > "$OUTPUT/$name.pip-freeze.txt"
    printf '%s\n' "$prefix" > "$OUTPUT/$name.original-prefix.txt"
done

cp "$0" "$OUTPUT/export_environment_manifests.sh"
echo "Environment rebuild manifests: $OUTPUT"
