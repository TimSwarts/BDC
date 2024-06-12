#!/bin/bash
#SBATCH --job-name=mpi_timings
#SBATCH --output=./output/slurm_logs/mpi_timer_%j.out
#SBATCH --error=./output/slurm_logs/mpi_timer_%j.err
#SBATCH --time=0-08:00:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15


#  Make an output directory for the accuracy results
mkdir -p ./output/

# Initialise the CSV file and write the header
echo "Ranks,Time" > ./output/mpi_timings.csv

# Activate the master conda environment
source /commons/conda/conda_load.sh;

# Loop through the number of tasks
for ranks in {1..64}; do
    if [ $ranks -eq 1 ]; then
        echo "$ranks,NULL" >> ./output/mpi_timings.csv
        continue
    fi
    run_start_time=$(date +"%d-%m-%y %T")
    echo "Started running with $ranks ranks at $run_start_time"
    # Run the Python script with mpiexec, using -np to specify the number of ranks
    start_time=$(date +%s.%N)
    mpiexec -np $ranks python3 ./mpi_files/assignment6_mpi.py -n 15 -s 10000 -b 8000 -e 30
    end_time=$(date +%s.%N)

    # Calculate the elapsed time in seconds
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    # Append the result to the CSV file
    echo "$ranks,$elapsed_time" >> ./output/mpi_timings.csv
done