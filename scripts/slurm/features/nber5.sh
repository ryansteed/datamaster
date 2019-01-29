#!/bin/bash
#SBATCH -J features-nber5
#SBATCH -o slurm/nber5.out
#SBATCH -e slurm/nber5.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/nber5.json 5000
