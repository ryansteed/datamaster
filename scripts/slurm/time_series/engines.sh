#!/bin/bash
#SBATCH -J analysis-engines
#SBATCH -o slurm/analysis-engines.out
#SBATCH -e slurm/analysis-engines.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py root_all scripts/json/engines.json -w h_index forward_cites -k 7 -b 20
