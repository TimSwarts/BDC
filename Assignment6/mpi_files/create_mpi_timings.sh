#!/bin/bash
#SBATCH --job-name=mpi_timings
#SBATCH --output=./output/mpi_timings.out
#SBATCH --error=./output/mpi_timings.err
#SBATCH --time=0-08:00:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10


#  Make an output directory for the accuracy results
mkdir -p ./output/

# Initialise the CSV file and write the header
echo "Ranks,Time" > ./output/mpi_timings.csv

# Activate the master conda environment
source /commons/conda/conda_load.sh;

# Loop through the number of tasks
for ranks in {1..10}; do
    if [ $ranks -eq 1 ]; then
        echo "$ranks,NULL" >> ./output/mpi_timings.csv
        continue
    fi
    # Run the Python script with mpiexec, using -np to specify the number of ranks
    start_time=$(date +%s.%N)
    mpiexec -np $ranks python3 ./mpi_files/assignment6_mpi.py -n 10 -s 8000 -b 3000 -e 25
    end_time=$(date +%s.%N)

    # Calculate the elapsed time in seconds
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    # Append the result to the CSV file
    echo "$ranks,$elapsed_time" >> ./output/mpi_timings.csv
done
