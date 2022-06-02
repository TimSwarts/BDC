import multiprocessing
import argparse

base_count = {"A": [0, 0], "T": [0, 0], "C": [0, 0], "G": [0, 0]}

with open("test.fastq") as file:
    for i, line in enumerate(file):
        if i % 4 == 1:
            read = line.strip()
        if i % 4 == 3:
            for x in line.strip():
                print(ord(x) - 33)
