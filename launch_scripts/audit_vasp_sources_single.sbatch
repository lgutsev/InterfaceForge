#!/usr/bin/env bash
#SBATCH --job-name=iface_source_audit
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=iface_source_audit_%j.out
#SBATCH --error=iface_source_audit_%j.err
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 /absolute/policy.yaml /absolute/new_report_dir [--logs-only]" >&2
    exit 2
fi
POLICY=$1
AUDIT_OUTPUT=$2
shift 2
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IFACE_PYTHON="${INTERFACEFORGE_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}"
export CER_INTERFACE_BASE="${CER_INTERFACE_BASE:-/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -x "$IFACE_PYTHON" || ! -f "$POLICY" ]]; then
    echo "Missing Python executable or source-audit policy" >&2
    exit 2
fi
srun --ntasks=1 --cpus-per-task=1 "$IFACE_PYTHON" -m interfaceforge.source_audit \
    "$POLICY" --output "$AUDIT_OUTPUT" "$@"
# Exit 2 with reports means REVIEW/FAILED sources exist; inspect, do not auto-accept.
