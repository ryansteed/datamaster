#!/bin/bash
#SBATCH -J features-nber1
#SBATCH -o slurm/nber1.out
#SBATCH -e slurm/nber1.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/nber1.json 5000
