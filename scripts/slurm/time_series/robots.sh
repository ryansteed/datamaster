#!/bin/bash
#SBATCH -J robots
#SBATCH -o slurm/analysis-robots.out
#SBATCH -e slurm/analysis-robots.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

python main.py root_all scripts/json/robots.json -d 0.5 -k 5 -b 20
