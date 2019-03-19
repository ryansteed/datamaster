#!/bin/bash
#SBATCH -J engines
#SBATCH -o slurm/analysis-engines.out
#SBATCH -e slurm/analysis-engines.err
#SBATCH -p defq
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

python main.py root_all scripts/json/engines.json -d 0.5 -k 5 -b 20
