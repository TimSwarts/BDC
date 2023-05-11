import argparse as ap
import multiprocessing as mp
import sys


def argument_parser():
    argparser = ap.ArgumentParser(description="Script voor Opdracht 1 van Big Data Computing")
    argparser.add_argument("-n", action="store",
                        dest="n", required=True, type=int,
                        help="Aantal cores om te gebruiken.")
    argparser.add_argument("-o", action="store", dest="csvfile", required=False, type=ap.FileType('w', encoding='UTF-8'),
                        help="CSV file om de output in op te slaan. Default is output naar terminal STDOUT")
    argparser.add_argument("fastq_files", action="store", type=ap.FileType('r'), nargs='+',
                           help="Minstens 1 Illumina Fastq Format file om te verwerken")
    return argparser.parse_args()


def main():
    args = argument_parser()
    print(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())