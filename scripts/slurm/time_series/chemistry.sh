#!/bin/bash
#SBATCH -J chemistry
#SBATCH -o slurm/analysis-chemistry.out
#SBATCH -e slurm/analysis-chemistry.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

python main.py root_all scripts/json/chemistry.json -d 0.5 -k 5 -b 20
