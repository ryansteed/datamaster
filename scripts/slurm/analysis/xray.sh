#!/bin/bash
#SBATCH -J analysis-xray
#SBATCH -o slurm/analysis-xray.out
#SBATCH -e slurm/analysis-xray.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/xray.json -w h_index forward_cites -k 7
