#!/bin/bash
#SBATCH -J features-coherent_light
#SBATCH -o slurm/coherent_light.out
#SBATCH -e slurm/coherent_light.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py features scripts/json/coherent_light.json 5000
