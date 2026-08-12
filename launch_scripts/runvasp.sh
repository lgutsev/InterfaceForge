#!/bin/bash
#SBATCH -p workq
#SBATCH -N 2
#SBATCH -n 128
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -A loni_perovsk27
#SBATCH -J "FAPI_001_H2O_1"
#SBATCH -o vasp.cpu.%j.out

module purge
module load vasp6/6.5.1-cpu
SECONDS=0
export SINGULARITYENV_OMP_NUM_THREADS=1
# This command uses 128 MPI processes on two nodes
srun -n128 vasp_gam
echo "took $SECONDS sec."
