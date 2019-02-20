#!/bin/bash
#SBATCH -J analysis-chemistry
#SBATCH -o slurm/analysis-chemistry.out
#SBATCH -e slurm/analysis-chemistry.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/chemistry.json -w h_index forward_cites -k 7
