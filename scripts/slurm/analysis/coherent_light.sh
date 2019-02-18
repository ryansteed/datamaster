#!/bin/bash
#SBATCH -J analysis-coherent_light
#SBATCH -o slurm/analysis-coherent_light.out
#SBATCH -e slurm/analysis-coherent_light.err
#SBATCH -p short
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/slurm/prep.sh
python main.py query scripts/json/coherent_light.json -w h_index forward_cites -k 7
