#!/bin/bash
#SBATCH --job-name=pool_map_timings
#SBATCH --output=./output/slurm_logs/pool_map_time_%j.out
#SBATCH --error=./output/slurm_logs/pool_map_time_%j.err
#SBATCH --time=0-09:00:00
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
#SBATCH --begin=now


# Create the output directory if it doesn't exist
mkdir -p ./output

# Create the CSV file and write the header
echo "Cores,Time" > ./output/pool_timings.csv

# Loop through the number of cores from 1 to 15
for cores in {1..15}; do
    run_start_time=$(date +"%d-%m-%y %T")
    echo "Started running with $cores ranks at $run_start_time"
    # Run the Python script with the specified number of cores and time the execution
    start_time=$(date +%s.%N)
    python3 ./multiprocessing_files/assignment6.py -n 15 -c $cores -s 15000 -b 12000 -e 30
    end_time=$(date +%s.%N)

    # Calculate the elapsed time in seconds
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    # Append the result to the CSV file
    echo "$cores,$elapsed_time" >> ./output/pool_timings.csv
done
