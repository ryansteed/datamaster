#!/bin/bash
#SBATCH -J analysis-xray
#SBATCH -o slurm/analysis-xray.out
#SBATCH -e slurm/analysis-xray.err
#SBATCH -p defq
#SBATCH -t 14-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py root_all scripts/json/xray.json -d 0.5 -k 5 -b 20
