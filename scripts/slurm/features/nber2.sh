#!/bin/bash
#SBATCH -J features
#SBATCH -o slurm/nber2.out
#SBATCH -e slurm/nber2.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/prep.sh
python main.py features scripts/json/nber2.json 10000
