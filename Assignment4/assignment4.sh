#!/bin/bash
#SBATCH --job-name=bdc_ass4
#SBATCH --account=tswarts
#SBATCH --error=assignment4.err
# time units:  d-hh:mm:ss
#SBATCH --time=0-00:05:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
# SBATCH --partition=workstations
# SBATCH --nodelist=nuc[501-502,505]

#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5  # number of nodes = ntasks/ntasks-per-node

# Optional different implementation:
# SBATCH --nodes=1
# SBATCH --cpus-per-tasks=5  # nodes * cpus-per-task should equal ranks in MPI

export FILE="/commons/Themas/Thema12/HPC/rnaseq.fastq";

source /commons/conda/conda_load.sh;

time mpiexec -np 5 python3 assignment4.py $FILE;  # note that here -np equals --ntasks in slurm