#!/bin/bash
#SBATCH --job-name=mpi_timings
#SBATCH --output=./output/slurm_logs/mpi_timer_%j.out
#SBATCH --error=./output/slurm_logs/mpi_timer_%j.err
#SBATCH --time=0-09:00:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1


#  Make an output directory for the results
mkdir -p ./output/

# Initialise the CSV file and write the header
echo "Ranks,Time" > ./output/mpi_timings.csv

# Loop through the number of tasks
for ranks in {1..15}; do
    if [ $ranks -eq 1 ]; then
        echo "$ranks,NULL" >> ./output/mpi_timings.csv
        continue
    fi
    
    echo "Started running with $ranks ranks at $(date +"%d-%m-%y %T")"
    # Run the Python script with mpiexec, using -np to specify the number of ranks
    start_time=$(date +%s.%N)
    mpiexec -np $ranks python3 ./mpi_files/assignment6_mpi.py -n 15 -s 15000 -b 12000 -e 30
    end_time=$(date +%s.%N)

    # Calculate the elapsed time in seconds
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    echo "Finished running with $ranks ranks at $(date +"%d-%m-%y %T")"

    # Append the result to the CSV file
    echo "$ranks,$elapsed_time" >> ./output/mpi_timings.csv
done
