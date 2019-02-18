#!/bin/bash
#SBATCH -J analysis-engines
#SBATCH -o slurm/analysis-engines.out
#SBATCH -e slurm/analysis-engines.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/engines.json -w h_index forward_cites -k 7
