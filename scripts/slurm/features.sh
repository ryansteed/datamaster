#!/bin/bash
#SBATCH -J features
#SBATCH -o slurm/patents_all.out
#SBATCH -e slurm/patents_all.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/prep.sh
python main.py features scripts/json/all.json 3000