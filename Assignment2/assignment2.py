#!/usr/bin/env python3


"""
Assignment 2: Big Data Computing
"""


__author__ = "Tim Swarts"
__version__ = "1.0"


import multiprocessing as mp
from pathlib import Path
import argparse


def argument_parser() -> argparse.ArgumentParser:
    """
    Argument parser for the command line arguments.
    :return argparse.parse_args(): This is an object with the command line arguments.
    Use .[argument] to get the arguments back by name.
    """

    description = "Script for Assignment 2 of Big Data Computing; Calculate PHRED scores over the network."
    argparser = argparse.ArgumentParser(description=description)

    mode = argparser.add_mutually_exclusive_group(required=True)

    server_help = "Run the program in Server mode; see extra options needed below"
    mode.add_argument("-s", action="store_true", help=server_help)

    client_help = "Run the program in Client mode; see extra options needed below"
    mode.add_argument("-c", action="store_true", help=client_help)

    # Arguments when run in server mode
    server_args = argparser.add_argument_group(title="Server mode arguments")

    csvfile_help = "CSV file to save the output in. Default is output to terminal STDOUT"
    server_args.add_argument(
        "-o",
        action="store",
        dest="csvfile",
        type=Path,
        required=False,
        help=csvfile_help,
    )

    fastq_files_help = "At least 1 Illumina Fastq Format file to process"
    server_args.add_argument(
        "fastq_files",
        action="store",
        type=argparse.FileType("r"),
        nargs="*",
        help=fastq_files_help,
    )

    chunks_help = "Number of chunks to split the files into. Default is 1"
    server_args.add_argument(
        "--chunks", action="store", required=False, type=int, help=chunks_help, default=1
    )

    # Arguments when run in client mode
    client_args = argparser.add_argument_group(title="Client mode arguments")

    cores_help = "Number of cores to use per host."
    client_args.add_argument(
        "-n", action="store", dest="n", required=False, type=int, help=cores_help
    )

    # Global arguments
    hostname_help = "The hostname where the Server is listening"
    argparser.add_argument("--host", action="store", type=str, help=hostname_help)

    port_help = "The port on which the Server is listening"
    argparser.add_argument("--port", action="store", type=int, help=port_help)

    return argparser.parse_args()


def main() -> None:
    """
    Main function of assignment 2 for Big Data Computing.
    """
    args = argument_parser()
    print(args)
    return 0


if __name__ == "__main__":
    main()
