#!/bin/bash
#SBATCH -p bigmem
#SBATCH -N 2
#SBATCH -n 128
#SBATCH -c 1
#SBATCH -t 48:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J "FAPI100_MLFF_bigmem"
#SBATCH -o vasp.cpu.%j.out

module purge
# New work defaults to 6.6.1. For an old/reproduction campaign submit with,
# e.g., sbatch --export=ALL,IFACE_VASP_MODULE=vasp6/6.5.1-cpu runvasp_bigmem.sh
VASP_MODULE="${IFACE_VASP_MODULE:-vasp6/6.6.1-cpu}"
module load "$VASP_MODULE"
SECONDS=0
export SINGULARITYENV_OMP_NUM_THREADS=1

srun -n128 vasp_gam
echo "took $SECONDS sec."

# Work-function plot: only fires for jobs that requested LVHAR = .TRUE.
# (e.g. `iface vasp incar static INCAR --workfunction`); a no-op otherwise.
# See examples/vasp/workfunction/README.md. `|| true` keeps a plotting
# failure from being mistaken for a failed VASP run.
if grep -Eqi '^\s*LVHAR\s*=\s*\.?(TRUE|T)\.?' INCAR 2>/dev/null && [ -s LOCPOT ]; then
    python "$HOME/bin/plot_workfunc.py" --title "$SLURM_JOB_NAME" || true
fi
