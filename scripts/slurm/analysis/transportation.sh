#!/bin/bash
#SBATCH -J analysis-transportation
#SBATCH -o slurm/analysis-transportation.out
#SBATCH -e slurm/analysis-transportation.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/transportation.json -w h_index forward_cites -k 7
