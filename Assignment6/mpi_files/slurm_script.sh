#!/bin/bash
#SBATCH --job-name=bdc_ass6
#SBATCH --account=tswarts
#SBATCH --error=output/assignment6_mpi.err
#SBATCH --output=output/assignment6_mpi.out
# time units:  d-hh:mm:ss
#SBATCH --time=0-03:00:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10  # number of nodes = ntasks/ntasks-per-node


# Make file available to all nodes
export FILE="/students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat"

# Make an output file for the accuracy results, format: slurm_<ntasks>_10_networks

# Activate the master conda environment
source /commons/conda/conda_load.sh;

# Set a name for the output file
mkdir -p ./output/slurm
export OUTPUT_FILE="./output/slurm/${SLURM_NTASKS}tasks_n10s8000b3000e25a03.png"

# Run the Python script with mpiexec, using -np equal to --ntasks in slurm
time mpiexec -np 10 python3 ./mpi_files/assignment6_mpi.py \
    --file $FILE \
    --network_count 10 \
    --data_size 8000 \
    --bag_size 3000 \
    --epochs 25 \
    --output $OUTPUT_FILE;

# time mpiexec -np 10 python3 assignment6_mpi.py \
#     --file $FILE \
#     --network_count 10 \
#     --data_size 100 \
#     --bag_size 40 \
#     --epochs 25 \
#     --output $OUTPUT_FILE;
