#!/bin/bash
#SBATCH --job-name=assignment3
#SBATCH --account=tswarts
#SBATCH --error=assignment3.err
#SBATCH --time=00:5:00
#SBATCH --nodes=1
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --cpus-per-task=20


# EXAMPLE: Processing a big file using more CPUs
# https://www.gnu.org/software/parallel/parallel_examples.html
source activate /commons/conda/dsls;

export FILE="/commons/Themas/Thema12/HPC/rnaseq.fastq";

parallel --jobs 20 --pipepart --block -1 --regexp \
         --recstart '@.*(/1| 1:.*)\n[A-Za-z\n\.~]' --recend '\n' \
         -a $FILE python3 assignment3.py --chunk-parser | \
         python3 assignment3.py --combine-chunks

