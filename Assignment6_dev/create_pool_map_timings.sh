#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p ../output

# Create the CSV file and write the header
echo "Cores,Time (seconds)" > ../output/timings.csv

# Loop through the number of cores from 1 to 10
for cores in {1..10}; do
    # Run the Python script with the specified number of cores and time the execution
    start_time=$(date +%s.%N)
    python3 assignment6.py -n 10 -c $cores -s 8000 -b 3000 -e 25 # eventually: -n8 -s 10000 -b 6000 -e 25
    end_time=$(date +%s.%N)

    # Calculate the elapsed time in seconds
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    # Append the result to the CSV file
    echo "$cores,$elapsed_time" >> ../output/timings.csv
done