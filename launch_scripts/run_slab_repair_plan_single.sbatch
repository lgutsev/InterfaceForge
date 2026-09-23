#!/bin/bash
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -J slab_repair_plan
#SBATCH -o slab_repair_plan.%j.out
#SBATCH -e slab_repair_plan.%j.err

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

audit_json="slab_alignment_scheduler.${SLURM_JOB_ID:-manual}.json"
plan_json="slab_tight_scf_plan_scheduler.${SLURM_JOB_ID:-manual}.json"

audit_rc=0
iface vasp slab-align . \
    --config "$config" \
    --no-write-dipole-fixes > "$audit_json" || audit_rc=$?
if [ "$audit_rc" -gt 1 ]; then
    echo "slab-align failed with exit code $audit_rc" >&2
    exit "$audit_rc"
fi

plan_rc=0
iface vasp slab-tight-scf . \
    --config "$config" \
    --dry-run > "$plan_json" || plan_rc=$?
if [ "$plan_rc" -gt 1 ]; then
    echo "slab-tight-scf planning failed with exit code $plan_rc" >&2
    exit "$plan_rc"
fi

echo "Fresh slab audit: $audit_json (exit $audit_rc)"
echo "Repair dry-run: $plan_json (exit $plan_rc)"
echo "Human-readable plan: $repair_root/tight_scf_plan.txt"
echo "No VASP calculation was submitted."
