#!/bin/bash
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -J slab_repair_prep
#SBATCH -o slab_repair_prep.%j.out
#SBATCH -e slab_repair_prep.%j.err

set -eu
set -o pipefail

repair_root="${SLURM_SUBMIT_DIR:-$PWD}"
config="${SLAB_ALIGNMENT_CONFIG:-}"
if [ -z "$config" ]; then
    if [ -f "$repair_root/slab_alignment_fapi.json" ]; then
        config="slab_alignment_fapi.json"
    else
        config="slab_alignment.json"
    fi
fi

cd "$repair_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Always refresh the audit first so preparation never consumes stale tilt/flatness results.
audit_rc=0
iface vasp slab-align . \
    --config "$config" \
    --no-write-dipole-fixes > "slab_alignment_prepare.${SLURM_JOB_ID:-manual}.json" || audit_rc=$?
if [ "$audit_rc" -gt 1 ]; then
    echo "slab-align failed with exit code $audit_rc" >&2
    exit "$audit_rc"
fi

prepare_rc=0
iface vasp slab-tight-scf . \
    --config "$config" \
    --copy runvasp.sh \
    --overwrite > "slab_repair_prepare.${SLURM_JOB_ID:-manual}.json" || prepare_rc=$?
if [ "$prepare_rc" -gt 1 ]; then
    echo "slab-tight-scf preparation failed with exit code $prepare_rc" >&2
    exit "$prepare_rc"
fi

echo "Prepared/updated tight-SCF inputs under: $repair_root/tight_scf"
echo "Prepared/updated geometry restarts under: $repair_root/relax_continue"
echo "Existing destinations containing OUTCAR are protected and reported as BLOCKED."
echo "No VASP calculation was submitted."
