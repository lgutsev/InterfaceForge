#!/bin/bash
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -J slab_publish
#SBATCH -o slab_publish.%j.out
#SBATCH -e slab_publish.%j.err

set -eu
set -o pipefail

publication_root="${SLURM_SUBMIT_DIR:-$PWD}"
publication_config="${SLAB_PUBLICATION_CONFIG:-slab_publication.json}"
publication_output="${SLAB_PUBLICATION_OUTPUT:-publication_figures}"

cd "$publication_root"
iface vasp slab-publish . \
    --config "$publication_config" \
    --output-dir "$publication_output" \
    --run-sumo
