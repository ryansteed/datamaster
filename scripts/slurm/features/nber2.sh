#!/bin/bash
#SBATCH -J features-nber2
#SBATCH -o slurm/nber2.out
#SBATCH -e slurm/nber2.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/nber2.json 5000
