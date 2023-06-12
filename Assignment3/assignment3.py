#!/usr/bin/env python3

"""	
Assignment 3: Big Data Computing
"""

__author__ = "Tim Swarts"
__version__ = "0.1"

import sys


def quality_line_generator():
    """Yield lines of quality scores from stdin"""
    with sys.stdin as fastq:
        while fastq.readline():
            # Skip the first 3 lines
            fastq.readline()
            fastq.readline()
            # Yield the quality line
            yield fastq.readline().strip()

def main():
    """Main function"""
    print("New python instance started")
    for line in enumerate(quality_line_generator()):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
