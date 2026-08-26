#!/usr/bin/env bash

# Custom periodic SiN/TiN/TiO mapped-collection job. The YAML beside the example
# is the reusable scientific mapping; this wrapper supplies the cluster default.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/examples/mapped-leaf-campaign/periodic_nitride.yaml"

export CER_INTERFACE_BASE="${CER_INTERFACE_BASE:-/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m interfaceforge.mapped_collect "$CONFIG" "$@"
