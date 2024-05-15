#!/bin/bash

# Run the multiprocessing timing script
sbatch ./multiprocessing_files/create_pool_map_timings.sh;

# Run the MPI timing script
sbatch ./mpi_files/create_mpi_timings.sh;
