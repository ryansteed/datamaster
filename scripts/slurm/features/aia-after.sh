#!/bin/bash
#SBATCH -J features-aia-after
#SBATCH -o slurm/aia-after.out
#SBATCH -e slurm/aia-after.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/aia-after.json 5000
