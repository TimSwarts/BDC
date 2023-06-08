#!/usr/bin/env python3

"""
Assignment 1: Big Data Computing
"""


import sys
import csv
import argparse as ap
import multiprocessing as mp
from pathlib import Path
from typing import Generator, List, Tuple
import itertools
import numpy as np


__author__ = "Tim Swarts"
__version__ = "1.0"


def argument_parser() -> ap.Namespace:
    """
    Argument parser for the command line arguments.
    :return ap.Namespace: An object with the command line arguments. Use .[argument] to retrieve the arguments by name.
    """

    argparser = ap.ArgumentParser(
        description="Script for Assignment 1 of Big Data Computing"
    )
    argparser.add_argument(
        "-n",
        action="store",
        dest="n",
        required=True,
        type=int,
        help="Number of cores to use.",
    )
    argparser.add_argument(
        "-o",
        action="store",
        dest="csvfile",
        required=False,
        type=Path,
        help="CSV file to save the output. Default is output to terminal STDOUT",
    )
    argparser.add_argument(
        "fastq_files",
        action="store",
        type=Path,
        nargs="+",
        help="At least 1 Illumina Fastq Format file to process",
    )

    return argparser.parse_args()


def write_output_to_csv(output_file_path: Path, phred_scores: List[float]):
    """
    This function writes the average PHRED scores to an output CSV file.
    :param output_file_path: The output file to write to.
    :param phred_scores: The list of average PHRED scores.
    """
    with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
        # Create a csv writer
        csv_writer = csv.writer(csvfile, delimiter=",")

        # Write the PHRED scores
        for i, phred_score in enumerate(phred_scores):
            row = [i, phred_score]
            csv_writer.writerow(row)


def phred_sum_parser(
    chunk_tuple: Tuple[Path, int, int]
) -> Tuple[Path, np.ndarray, np.ndarray]:
    """
    Parse sum of PHRED scores for a chunk of a FastQ file.
    :param chunk_tuple: A tuple consisting of file path, start and stop.
    :return: A tuple with file path, sum of phred scores and count of scores.
    """
    # Get the file path, start and stop
    filepath, start, stop = chunk_tuple

    # Get iterators of the quality lines
    count_iterator, parsing_iterator = itertools.tee(
        quality_line_generator(start, stop, filepath)
    )

    # Initialize variables for line count and maximum line length
    amount_of_lines = 0
    max_line_length = 0

    # Count lines and find the maximum line length
    for line in count_iterator:
        amount_of_lines += 1
        if len(line) > max_line_length:
            max_line_length = len(line)
    # Create an array to store all the PHRED scores
    all_phred_scores = np.zeros((amount_of_lines, max_line_length))

    # Loop over the quality lines
    for i, line in enumerate(parsing_iterator):
        for j, char in enumerate(line):
            all_phred_scores[i, j] = char
    all_phred_scores -= 33

    # Calculate sum of all columns and the amount of non-zero entries
    chunk_phred_sum = np.sum(all_phred_scores, axis=0)
    chunk_phred_count = np.count_nonzero(all_phred_scores, axis=0)

    # Return the file path, sum and count
    return filepath, chunk_phred_sum, chunk_phred_count


def quality_line_generator(
    start: int, stop: int, filepath: Path
) -> Generator[str, None, None]:
    """
    A generator for FastQ file's quality lines.
    :param start: The start byte of the file chunk.
    :param stop: The stop byte of the file chunk.
    :param filepath: Path of the FastQ file.
    :return: Yields quality lines from the FastQ file.
    """
    # Open file in byte mode
    with open(filepath, "rb") as fastq_file:
        # Seek to start
        fastq_file.seek(start)
        # Until stop is reached:
        while fastq_file.tell() < stop:
            # readline
            line = fastq_file.readline()
            # check if identifier line
            if line.startswith(b"@"):
                # if so, read until quality line and yield
                fastq_file.readline()
                fastq_file.readline()
                yield fastq_file.readline().strip()


def get_chunks(file_paths: List[Path], number_of_chunks: int) -> List[Tuple]:
    """
    This function divides a file into a number of chunks.
    :param file_paths: List of file paths.
    :param number_of_chunks: Number of chunks to divide files into.
    :return: A list of tuples. Each tuple contains a file path and start and stop bytes for a chunk.
    """
    # Calculate number of chunks per file
    chunks_per_file = number_of_chunks // len(file_paths)
    # Initialise empty list
    chunks = []
    # Loop through the file_paths to get file specific chunks
    for file_path in file_paths:
        # Get file size
        file_size = file_path.stat().st_size
        # Derive chunk size from file size
        chunk_size = file_size // chunks_per_file
        # Intitialise first chunk start stop values
        start = 0
        stop = chunk_size
        # Create chunks until the end of the file
        while stop < file_size:
            # Chunk gets file name and start stop postion
            chunks.append((file_path, start, stop))
            start = stop
            stop += chunk_size
    return chunks


def multi_processing(
    data: List[Tuple], processes: int, file_paths: List[Path], output_file_path: Path
) -> None:
    """
    This function distributes the quality lines over the given cores.
    :param data: Quality lines to distribute.
    :param processes: The number of cores to use.
    :param file_paths: Paths of the FastQ files.
    :param output_file_path: Path of the output file.
    """
    # Create pool of processes
    with mp.Pool(processes) as pool:  # pylint: disable=no-member
        # Use pool map to parallelise the calculation of the sums of phred scores per base position
        results = pool.map(phred_sum_parser, data)
    # Fetch the results per file
    for fastq_file in file_paths:
        # Merge all sum and count arrays
        all_sum_arrays = [result[1] for result in results if result[0] == fastq_file]
        all_count_arrays = [result[2] for result in results if result[0] == fastq_file]

        # Calculate a total
        total_phred_sums = array_concatenator(all_sum_arrays)
        total_phred_counts = array_concatenator(all_count_arrays)
        # Finally calculate the average
        average_phred_scores = total_phred_sums / total_phred_counts

        # Read output to either
        if output_file_path is None:
            if len(file_paths) > 1:
                print(f"{fastq_file.name}")
            write_output_to_terminal(average_phred_scores)
        else:
            if len(file_paths) > 1:
                # Create line specific output file path
                full_output_file_path = output_file_path.parent.joinpath(
                    f"{fastq_file.name}.{output_file_path.name}"
                )
            write_output_to_csv(full_output_file_path, average_phred_scores)


def array_concatenator(array_list: List[np.ndarray]) -> np.ndarray:
    """
    This function combines a list of numpy arrays into one numpy array.
    :param array_list: The list of numpy arrays to combine.
    :return: The combined numpy array.
    """
    # Calculate the maximum line length
    max_length = max(len(array) for array in array_list)
    complete_array = np.zeros((len(array_list), max_length))

    # Loop over the arrays
    for i, array in enumerate(array_list):
        for j, item in enumerate(array):
            complete_array[i, j] = item
    return np.sum(complete_array, axis=0)


def write_output_to_terminal(phred_scores: List[float]) -> None:
    """
    This function writes the average PHRED scores to the terminal.
    :param phred_scores: The average PHRED scores to write.
    """
    # Write the PHRED scores
    for i, phred_score in enumerate(phred_scores):
        print(f"{i},{phred_score}")


def main() -> int:
    """
    Main function of the script, here the command line arguments are collected and processed.
    :return int: 0 if the script is executed correctly, otherwise an error code.
    """
    # Collect command line arguments
    args = argument_parser()
    fastq_files = args.fastq_files
    output_file_path = args.csvfile
    use_cores = args.n

    # Read the quality lines from the file
    data_chunks = get_chunks(fastq_files, use_cores)
    # Calculate the average PHRED scores in parallel
    multi_processing(data_chunks, use_cores, fastq_files, output_file_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
