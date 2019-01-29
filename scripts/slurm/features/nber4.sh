#!/bin/bash
#SBATCH -J features-nber4
#SBATCH -o slurm/nber4.out
#SBATCH -e slurm/nber4.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/nber4.json 5000
