#!/bin/bash
#SBATCH -J features-nber6
#SBATCH -o slurm/nber6.out
#SBATCH -e slurm/nber6.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/nber6.json 5000
