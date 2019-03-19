#!/bin/bash
#SBATCH -J coherent_light
#SBATCH -o slurm/analysis-coherent_light.out
#SBATCH -e slurm/analysis-coherent_light.err
#SBATCH -p defq
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

python main.py root_all scripts/json/coherent_light.json -d 0.5 -k 5 -b 20
