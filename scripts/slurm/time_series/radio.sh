#!/bin/bash
#SBATCH -J analysis-radio
#SBATCH -o slurm/analysis-radio.out
#SBATCH -e slurm/analysis-radio.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py root_all scripts/json/radio.json -w h_index forward_cites -k 7 -b 20
