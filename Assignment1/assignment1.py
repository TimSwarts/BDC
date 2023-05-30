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


def parse_average_phred_scores_from_lines(quality_lines: list[list[str]]) -> np.ndarray:
    """
    Deze functie parsed de PHRED scores uit meegegeven quality lines
    en berekend de gemiddelde PHRED score per kolom.
    Deze gemiddelde PHRED scores worden teruggegeven met de pool queue.
    :param lines: De quality lines om te parsen. Deze zijn al eerder gelezen uit de input file(s).
    :return phred_scores: Een numpy array met gemiddelde PHRED scores per kolom.
    """

    # Maak een lijst, of in dit geval numpy array, met de totale PHRED scores per kolom, start met 0
    total_phred = np.zeros(len(quality_lines[0]), dtype=int)
    # Sla het aantal quality lines op
    amount_of_lines = len(quality_lines)

    for line in quality_lines:
        # Bereken de PHRED score per base en tel deze op bij de totale PHRED score per kolom
        total_phred += np.array([ord(char) - 33 for char in line])

    # Bereken de gemiddelde PHRED score per kolom en return deze
    average_phred_scores = total_phred / amount_of_lines
    return average_phred_scores


def parse_fastq_file(fastq_file: Path) -> list[str]:
    """
    Deze functie leest een FASTQ-bestand in en slaat de quality lines op in een lijst.
    :param fastq_file: Het FASTQ-bestand om in te lezen
    :return quality_lines: Een lijst met quality lines.
    """
    # Maak een lege lijst aan om de quality lines in op te slaan
    quality_lines = []

    with open(fastq_file, mode="r", encoding="utf-8") as fastq:
        quality_lines = [line.strip() for i, line in enumerate(fastq, start=1) if i % 4 == 0]
    return quality_lines


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


def multi_processing(quality_lines: list[list[str]], cores: int) -> np.ndarray:
    """
    Deze functie verdeeld de quality lines over de cores die meegegeven zijn.
    :param quality_lines: De quality lines om te verdelen.
    :param cores: Het aantal cores om te gebruiken.
    :return phred_scores: Een numpy array met gemiddelde PHRED scores per kolom.
    """
    # Split de quality_lines in n gelijke delen
    split_quality_lines = [quality_lines[i::cores] for i in range(cores)]

    # Maak een pool aan met n processen
    with mp.Pool(cores) as pool:  # pylint: disable=no-member
        # Gebruik de pool.map functie om de PHRED scores te berekenen voor alle chunks
        results = pool.map(parse_average_phred_scores_from_lines, split_quality_lines)
    # Combineer de resultaten van de processen
    total_phred_scores = np.sum(results, axis=0)
    phred_scores = [total / cores for total in total_phred_scores]

    return phred_scores


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
    for fastq_file in fastq_files:
        # Lees de quality lines uit de file
        quality_lines = parse_fastq_file(fastq_file)
        # Bereken de gemiddelde PHRED scores in parrallel
        phred_scores = multi_processing(quality_lines, use_cores)
        # Schrijf de output naar een bestand of naar de terminal
        if output_file_path is None:
            if len(fastq_files) > 1:
                print(f"{fastq_file.name}")
            write_output_to_terminal(phred_scores)
        else:
            if len(fastq_files) > 1:
                # Voeg de bestandsnaam toe aan de output wanneer er meerdere bestanden zijn
                output_file_path = output_file_path.parent.joinpath(f"{fastq_file.name}. \
                                                                    {output_file_path.name}")
            write_output_to_csv(output_file_path, phred_scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
