#!/bin/bash
#SBATCH --job-name=assignment4
#SBATCH --account=tswarts
#SBATCH --error=assignment4.err
#SBATCH --time=00:5:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5   # nodes * cpu-per-task = 5, should equal ranks in MPI

export FILE="/commons/Themas/Thema12/HPC/rnaseq.fastq";

echo "start of script, loading conda env";
source /commons/conda/conda_load.sh;



echo "starting mpi call now";
mpiexec -n 5 python3 assignment4.py $FILE