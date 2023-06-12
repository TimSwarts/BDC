#!/bin/bash
#SBATCH --job-name=assignment3
#SBATCH --error=assignment3.err
#SBATCH --time=00:5:00
#SBATCH --cpus-per-task=4


# conda activate /commons/conda/dsls/;

# EXAMPLE: Processing a big file using more CPUs
# https://www.gnu.org/software/parallel/parallel_examples.html#

cat ../Assignment1/data/tiny_rnaseq.fastq | parallel --pipepart --block -1 --jobs 4 --regexp \
    --recend '\n' --recstart '@.*(/1| 1:.*)\n[A-Za-z\n\.~]' \
    python3 assignment3.py; # 2> error.txt > output.txt;


# parallel -a <(parallel --pipepart -a ../Assignment1/tiny_rnaseq.fastq --block -1 --number-of-cpus 4) python3 assignment3.py

