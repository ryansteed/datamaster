#!/bin/bash
#SBATCH -J analysis-chemistry
#SBATCH -o slurm/analysis-chemistry.out
#SBATCH -e slurm/analysis-chemistry.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py root_all scripts/json/chemistry.json -d 0.5 -k 5 -b 20
