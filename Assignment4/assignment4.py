#!/usr/bin/env python3

"""
Assignment 4: Big Data Computing, run with mpiexec in assignment4.sh,
don't run this script directly on your own.
Usage: mpiexec -n [number of cores] python3 assignment4.py [ fastq files to process]
"""

import sys
import argparse
from pathlib import Path
from mpi4py import MPI

ASSIGNMENT1_PATH = str(Path(__file__).parent.parent.joinpath("Assignment1"))
sys.path.append(ASSIGNMENT1_PATH)
from assignment1 import get_chunks, post_processing, phred_sum_parser

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
HOST = MPI.Get_processor_name()
print(f"rank: {RANK}/{SIZE}, host: {HOST}")


def argument_parser() -> argparse.Namespace:
    """
    Argument parser for the command line arguments.
    :return parser.parse_args(): An object with the command line arguments.
    Use .[argument] to retrieve the arguments by name.
    """
    # Create an argument parser object
    parser = argparse.ArgumentParser(
        description="Script for Assignment 4 of Big Data Computing"
    )
    # Add arguments for input fastq files
    parser.add_argument(
        "fastq_files",
        action="store",
        type=Path,
        nargs="+",
        help="At least 1 Illumina Fastq Format file to process",
    )
    return parser.parse_args()


def main():
    """
    Main function for the script.
    :return 0: Exit code 0 for success.
    """
    args = argument_parser()
    if RANK == 0:  # Controller/Root/Worker 0
        # Get the chunks
        data = get_chunks(args.fastq_files, SIZE)
    else:  # Normal Workers
        data = None
    # Scatter the chunks and fetch current rank's chunk
    rank_data = COMM.scatter(
        data, root=0
    )  # data is None for normal workers, so they only receive and don't send

    # Process the chunks for each worker
    processed = phred_sum_parser(rank_data)

    # Gather the results
    results = COMM.gather(processed, root=0)

    if RANK == 0:
        # Post process the results
        post_processing(results, args.fastq_files)
    return 0


if __name__ == "__main__":
    main()
