#!/bin/bash
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -J slab_align
#SBATCH -o slab_align.%j.out
#SBATCH -e slab_align.%j.err

set -eu
set -o pipefail

alignment_root="${SLURM_SUBMIT_DIR:-$PWD}"
alignment_config="${SLAB_ALIGNMENT_CONFIG:-}"
if [ -z "$alignment_config" ]; then
    if [ -f "$alignment_root/slab_alignment_fapi.json" ]; then
        alignment_config="slab_alignment_fapi.json"
    else
        alignment_config="slab_alignment.json"
    fi
fi

cd "$alignment_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

args=(vasp slab-align . --config "$alignment_config" --no-write-dipole-fixes)
if [ "${SLAB_ALIGNMENT_RUN_SUMO:-0}" = "1" ]; then
    args+=(--run-sumo)
fi

iface "${args[@]}"
