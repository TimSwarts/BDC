#!/usr/bin/env python3


"""
Opdracht 1: Big Data Computing
"""


import sys
import csv
import argparse as ap
import multiprocessing as mp
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Tuple
import itertools
import numpy as np


__author__ = "Tim Swarts"
__version__ = "1.0"


def argument_parser() -> ap.ArgumentParser:
    """
    Argument parser voor de command line arguments.
    :return argparse.parse_args(): Dit is een object met de command line arguments.
    Gebruik .[argument] om de argumenten terug te krijgen op naam.
    """

    argparser = ap.ArgumentParser(
        description="Script voor Opdracht 1 van Big Data Computing"
    )
    argparser.add_argument(
        "-n",
        action="store",
        dest="n",
        required=True,
        type=int,
        help="Aantal cores om te gebruiken.",
    )
    argparser.add_argument(
        "-o",
        action="store",
        dest="csvfile",
        required=False,
        type=Path,
        help="CSV file om de output in op te slaan."
        "Default is output naar terminal STDOUT",
    )
    argparser.add_argument(
        "fastq_files",
        action="store",
        type=Path,
        nargs="+",
        help="Minstens 1 Illumina Fastq Format file om te verwerken",
    )

    return argparser.parse_args()


def fastq_quality_line_generator(filepath):
    with open(filepath, 'rb') as fastq_file:
        # Skip the first 3 lines and then yield every 4th line
        for line in itertools.islice(fastq_file, 3, None, 4):
            yield line.strip()


def chunked_file_iterators(filepath, amount_of_chunks):
    # Get the total number of lines in the file
    total_lines = sum(1 for _ in fastq_quality_line_generator(filepath))

    # Calculate the chunk size
    chunk_size = total_lines // amount_of_chunks
    print(f"Chunk size: {chunk_size}")
    # Create the iterators for each chunk
    quality_line_gen = fastq_quality_line_generator(filepath)
    for i in range(amount_of_chunks):
        if i != amount_of_chunks - 1:
            yield itertools.islice(quality_line_gen, chunk_size)
        else:
            # Make sure the last chunk includes any leftover lines
            yield quality_line_gen


def read_fastq_files_into_chunks(filepaths, amount_of_chunks):
    data = []
    for filepath in filepaths:
        chunks = chunked_file_iterators(filepath, amount_of_chunks)
        for chunk in chunks:
            data.append({"file": filepath, "qual_lines": list(chunk)})

    return data


def write_output_to_csv(output_file_path: Path, phred_scores: list[float]):
    """
    Deze functie schrijft de gemiddelde PHRED scores naar een output csv bestand.
    :param output_file: Het output bestand om naar te schrijven.
    """
    with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
        # Creëer een csv writer
        csv_writer = csv.writer(csvfile, delimiter=",")

        # Schrijf de PHRED scores
        for i, phred_score in enumerate(phred_scores):
            row = [i, phred_score]
            csv_writer.writerow(row)


def parse_average_phred_scores_from_lines(data_chunk: Dict[str, str | list[str]]) -> tuple[Path | np.ndarray]:
    """
    Deze functie parsed de PHRED scores uit meegegeven quality lines
    en berekend de gemiddelde PHRED score per kolom.
    Deze gemiddelde PHRED scores worden teruggegeven met de pool queue.
    :param lines: De quality lines om te parsen. Deze zijn al eerder gelezen uit de input file(s).
    :return phred_scores: Een tuple met numpy twee numpy arrays. De eerste bevat de sum van de PHRED scores per kolom.
                          De tweede bevat het aantal PHRED scores per kolom.
    """
    # Fetch the quality lines from the data chunk
    quality_lines = data_chunk["qual_lines"]
    
    # Calculate the maximum line length
    max_line_length = max(len(line) for line in quality_lines)

    # Get amount of lines in the chunk
    amount_of_lines = sum(1 for _ in quality_lines)

    # Create an array to store all the PHRED scores
    all_phred_scores = np.zeros((amount_of_lines, max_line_length))

    # Loop over the quality lines
    for i, line in enumerate(quality_lines):
        for j, char in enumerate(line):
            all_phred_scores[i, j] = char

    all_phred_scores -= 33

    # Calculate the average PHRED scores
    chunk_phred_sum = np.sum(all_phred_scores, axis=0)
    chunk_phred_count = np.count_nonzero(all_phred_scores, axis=0)

    return data_chunk["file"], chunk_phred_sum, chunk_phred_count


def multi_processing(data, cores: int, file_paths: list[Path], output_file_path) -> np.ndarray:
    """
    Deze functie verdeeld de quality lines over de cores die meegegeven zijn.
    :param quality_lines: De quality lines om te verdelen.
    :param cores: Het aantal cores om te gebruiken.
    :return: Een numpy array met gemiddelde PHRED scores per kolom.
    """

    # Maak een pool aan met n processen
    with mp.Pool(cores) as pool:  # pylint: disable=no-member
        # Gebruik de pool.map functie om de PHRED scores te berekenen voor alle chunks
        results = pool.map(parse_average_phred_scores_from_lines, data)


    for fastq_file in file_paths:
        all_sum_arrays = [result[1] for result in results if result[0] == fastq_file]
        all_count_arrays = [result[2] for result in results if result[0] == fastq_file]

        total_phred_sums = array_concatenator(all_sum_arrays)
        total_phred_counts = array_concatenator(all_count_arrays)
        average_phred_scores = total_phred_sums / total_phred_counts

        if output_file_path is None:
            if len(file_paths) > 1:
                print(f"{fastq_file.name}")
            write_output_to_terminal(average_phred_scores)
        else:
            if len(file_paths) > 1:
                # Voeg de bestandsnaam toe aan de output wanneer er meerdere bestanden zijn
                output_file_path = output_file_path.parent.joinpath(f"{fastq_file.name}.{output_file_path.name}")
            write_output_to_csv(output_file_path, average_phred_scores)

    return total_phred_sums / total_phred_counts


def array_concatenator(array_list):
    """
    Deze functie combineert een lijst met numpy arrays tot 1 numpy array.
    :param array_list: De lijst met numpy arrays om te combineren.
    :return: De gecombineerde numpy array.
    """
    # Calculate the maximum line length
    max_length = max(len(array) for array in array_list)
    complete_array = np.zeros((len(array_list), max_length))


    # Loop over the arrays
    for i, array in enumerate(array_list):
        for j, item in enumerate(array):
            complete_array[i, j] = item

    return np.sum(complete_array, axis=0)


def write_output_to_terminal(phred_scores: list[float]):
    """
    Deze functie schrijft de gemiddelde PHRED scores naar de terminal.
    :param phred_scores: De gemiddelde PHRED scores om te schrijven.
    """
    # Schrijf de PHRED scores
    for i, phred_score in enumerate(phred_scores):
        print(f"{i},{phred_score}")


def main():
    """
    Main functie van het script, hierin worden de command line arguments opgehaald en verwerkt.
    :return int: 0 als het script goed is uitgevoerd, anders een foutcode.
    """
    # Verzamel command line arguments
    args = argument_parser()
    fastq_files = args.fastq_files
    output_file_path = args.csvfile
    use_cores = args.n
    # Loop door de files

    # Lees de quality lines uit de file
    data_chunks = read_fastq_files_into_chunks(fastq_files, use_cores)
    print(f"Found {len(data_chunks)} chunks")
    # Bereken de gemiddelde PHRED scores in parrallel
    multi_processing(data_chunks, use_cores, fastq_files, output_file_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
