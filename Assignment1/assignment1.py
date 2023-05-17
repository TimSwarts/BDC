#!/usr/bin/env python3


"""
Opdracht 1: Big Data Computing
"""


import argparse as ap
import multiprocessing as mp
import sys
import csv


__author__ = "Tim Swarts"
__version__ = "1.0"


def argument_parser() -> ap.Namespace:
    """
    Argument parser voor de command line arguments.
    :return argparse.parse_args(): Dit is een object met de command line arguments.
    Gebruik .[argument] om de argumenten terug te krijgen op naam.
    """

    argparser = ap.ArgumentParser(description="Script voor Opdracht 1 van Big Data Computing")
    argparser.add_argument("-n", action="store",
                           dest="n", required=True, type=int,
                           help="Aantal cores om te gebruiken.")
    argparser.add_argument("-o", action="store", dest="csvfile", required=False,
                           type=ap.FileType('w', encoding='UTF-8'),
                           help="CSV file om de output in op te slaan." \
                           "Default is output naar terminal STDOUT")
    argparser.add_argument("fastq_files", action="store", type=ap.FileType('r'), nargs='+',
                           help="Minstens 1 Illumina Fastq Format file om te verwerken")

    return argparser.parse_args()


def parse_average_phred_scores_from_lines(quality_lines: list[str]) -> list[float]:
    """
    Deze functie parsed de PHRED scores uit meegegeven quality lines en bereken de gemiddelde PHRED score per kolom.
    Deze gemiddelde PHRED scores worden teruggegeven met de pool queue.
    :param lines: De quality lines om te parsen. Deze zijn al eerder gelezen uit de input file(s).
    :return phred_scores: Een lijst met gemiddelde PHRED scores per kolom.
    """

    # Maak een lijst met de totale PHRED scores per kolom
    total_phred = [0 for _ in range(len(quality_lines[0]))]
    # Sla het aantal quality lines op
    amount_of_lines = len(quality_lines)

    for line in quality_lines:
        # Loop door de quality line en bereken de gemiddelde PHRED score per kolom
        for i, char in enumerate(line):
            try:
                total_phred[i] += ord(char) - 33
            except IndexError:
                total_phred.append(ord(char) - 33)

    # Bereken de gemiddelde PHRED score per kolom en return deze
    average_phred_scores = [phred / amount_of_lines for phred in total_phred]
    return average_phred_scores



def parse_fastq_file(fastq_file: ap.FileType('r')) -> list[str]:
    """
    Deze functie leest een fastq_bestand in en slaat de quality lines op in een lijst.
    :param fastq_file: Het fastq_bestand om in te lezen
    :return quality_lines: Een lijst met quality lines.
    """
    quality_lines = []

    with open(fastq_file.name, "r", encoding="UTF-8") as fastq:
        count = 1
        for line in fastq:
            if count % 4 == 0:
                # Sla de quality line op
                quality_lines.append(line.strip())
            count += 1

    return quality_lines


def write_output(output_file_name: str, phred_scores: list[float]):
    """
    Deze functie schrijft de gemiddelde PHRED scores naar een output csv bestand.
    :param output_file: Het output bestand om naar te schrijven.
    """
    with open(output_file_name, "w", newline="") as csvfile:
        # Creëer een csv writer
        csv_writer = csv.writer(csvfile, delimiter=",")
        # Schrijf de header
        header = ["Base Position", "Average PHRED Score"]
        csv_writer.writerow(header)

        # Schrijf de PHRED scores
        for i, phred_score in enumerate(phred_scores):
            row = [i + 1, phred_score]
            csv_writer.writerow(row)


def multi_processing(quality_lines: list[str], n: int) -> list[float]:
    """
    Deze functie verdeeld de quality lines over de cores die meegegeven zijn.
    :param quality_lines: De quality lines om te verdelen.
    :param n: Het aantal cores om te gebruiken.
    :return phred_scores: Een lijst met gemiddelde PHRED scores per kolom.
    """
    # Split de quality_lines in n gelijke delen
    split_quality_lines = [quality_lines[i::n] for i in range(n)]

    # Maak een pool aan met n processen
    with mp.Pool(n) as pool:  # pylint: disable=no-member
        # Gebruik de pool.map functie om de PHRED scores te berekenen voor elk deel van de quality lines
        results = pool.map(parse_average_phred_scores_from_lines, split_quality_lines)

    # Combineer de resultaten van de processen
    total_phred_scores = [sum(x) for x in zip(*results)]  # De zip functie combineert de resultaten per kolom
    phred_scores = [total / n for total in total_phred_scores]

    return phred_scores


def main():
    """
    Main functie van het script, hierin worden de command line arguments opgehaald en verwerkt.
    :return int: 0 als het script goed is uitgevoerd, anders een foutcode.
    """
    # Verzamel command line arguments
    args = argument_parser()
    fastq_files = args.fastq_files
    if args.csvfile is None:
        ouput_file_name = None
    else:
        ouput_file_name = args.csvfile.name
    use_cores = args.n
    # Loop door de files
    for fastq_file in fastq_files:
        # Lees de quality lines uit de file
        quality_lines = parse_fastq_file(fastq_file)
        # Bereken de gemiddelde PHRED scores in parrallel
        phred_scores = multi_processing(quality_lines, use_cores)
        # Schrijf de output naar een bestand of naar de terminal
        if ouput_file_name is None:
            print(f"Gemmidelde PHRED scores voor {fastq_file.name}: {phred_scores}")
        else:
            if len(fastq_files) > 1:
                # Voeg de bestandsnaam toe aan de output wanneer er meerdere bestanden zijn
                ouput_file_name = f"{fastq_file.name}.{ouput_file_name}"
            write_output(ouput_file_name, phred_scores)

    return 0


if __name__ == "__main__":
    sys.exit(main())
