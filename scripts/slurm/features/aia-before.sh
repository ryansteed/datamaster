#!/bin/bash
#SBATCH -J features-aia-before
#SBATCH -o slurm/aia-before.out
#SBATCH -e slurm/aia-before.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/aia-before.json 5000
