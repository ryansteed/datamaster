#!/bin/bash
#SBATCH -J analysis-robots
#SBATCH -o slurm/analysis-robots.out
#SBATCH -e slurm/analysis-robots.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/robots.json -w h_index forward_cites -k 7
