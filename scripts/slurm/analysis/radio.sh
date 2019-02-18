#!/bin/bash
#SBATCH -J analysis-radio
#SBATCH -o slurm/analysis-radio.out
#SBATCH -e slurm/analysis-radio.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/radio.json -w h_index forward_cites -k 7
