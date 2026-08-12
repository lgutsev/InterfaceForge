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
module load vasp6/6.5.1-cpu
SECONDS=0
export SINGULARITYENV_OMP_NUM_THREADS=1

srun -n128 vasp_gam
echo "took $SECONDS sec."
