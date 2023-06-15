#!/usr/bin/env python3

"""
Assignment 3: Big Data Computing
"""

__author__ = "Tim Swarts"
__version__ = "0.1"

import sys
import argparse
import numpy as np


def argument_parser() -> argparse.Namespace:
    """create an argument parser object
    return parser.parse_args(): an argument parser object that contains the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Script for Assignment 3 of Big Data Computing. Should be run from GNU parallel command."
    )
    # Create exclusive group for the two modes
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--chunk-parser",
        action="store_true",
        help="Run the program in chunk parsing mode;" \
             "print the sum and count per base postion of the chunk phred scores.",
    )
    mode.add_argument(
        "--combine-chunks",
        action="store_true",
        help="Run the program in combine parsing mode;" \
             "calculate the average phred score per base position of all chunks combined."
    )
    return parser.parse_args()


def quality_line_generator():
    """
    Yield lines of quality scores from stdin
    """
    with sys.stdin.buffer as fastq:
        i = 0
        while i < 2:
            # Skip the first 3 lines
            if not fastq.readline():
                i += 1
            fastq.readline()
            fastq.readline()
            # Yield the quality line
            yield fastq.readline().strip()


def combine_array_list(array_list: list[np.ndarray], is_phred: bool = False) -> np.ndarray:
    """
    Combine a list of numpy arrays of varying lengths into one array (padded with zeros).
    """
    # Get the maximum length of the arrays
    max_row_length = np.array([len(item) for item in array_list]).max() # max([len(array) for array in array_list])
    # Create an array to store the combined arrays
    combined_array = np.zeros((len(array_list), max_row_length))
    # Loop over the arrays and add them to the combined array
    for i, array in enumerate(array_list):
        combined_array[i, : len(array)] = array
    # Subtract 33 from the array if it is a PHRED score array
    if is_phred:
        combined_array[np.nonzero(combined_array)] -= 33
    return combined_array


def phred_sum_parser() -> tuple[np.ndarray, np.ndarray]:
    """
    Parse sum of PHRED scores for a chunk of a FastQ file.
    """
    quality_array_list = [
        np.frombuffer(line, dtype=np.uint8)
        for line in quality_line_generator()
    ]
    quality_array = combine_array_list(quality_array_list, is_phred=True)
    chunk_phred_sum = np.sum(quality_array, axis=0)
    chunk_phred_count = np.count_nonzero(quality_array, axis=0)
    return chunk_phred_sum, chunk_phred_count


def main() -> int:
    """Main function"""
    # Parse arguments
    args = argument_parser()
    if args.chunk_parser:
        # Parse the chunk
        phred_sum_array, phred_count_array = phred_sum_parser()
        # Send the results to stdout so it can be piped to the next step
        print("sum:", list(phred_sum_array))
        print("count:", list(phred_count_array))
    elif args.combine_chunks:
        # Combine the results from the chunks
        sum_array_list = []
        count_array_list = []
        with sys.stdin as input_lines:
            # Loop over the lines and add them to the correct list
            for line in input_lines:
                if line.startswith("sum:"):
                    # Parse the sums to a numpy array and add it to the list
                    sum_array_list.append(np.fromstring(line.strip()[6:-1], sep=", "))
                elif line.startswith("count:"):
                    # Parse the counts to a numpy array and add it to the list
                    count_array_list.append(np.fromstring(line.strip()[8:-1], sep=", "))
        # Combine the arrays and calculate the total sum and count
        total_sum = np.sum(combine_array_list(sum_array_list), axis=0)
        total_count = np.sum(combine_array_list(count_array_list), axis=0)
        # Calculate the average phred score per base position
        averages = np.divide(total_sum, total_count, dtype=np.float64)
        # Print the averages to stdout, the result of the script
        for i, value in enumerate(averages):
            print(f"{i},{value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
