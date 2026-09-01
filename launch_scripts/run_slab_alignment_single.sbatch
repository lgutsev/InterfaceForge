#!/bin/bash
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -J slab_align
#SBATCH -o slab_align.%j.out
#SBATCH -e slab_align.%j.err

set -eu
set -o pipefail

alignment_root="${SLURM_SUBMIT_DIR:-$PWD}"
alignment_config="${SLAB_ALIGNMENT_CONFIG:-slab_alignment.json}"

cd "$alignment_root"
iface vasp slab-align . --config "$alignment_config" --run-sumo
