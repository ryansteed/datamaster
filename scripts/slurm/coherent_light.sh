#!/bin/bash
#SBATCH -J coherent_light
#SBATCH -o slurm/main.out
#SBATCH -e slurm/main.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

./scripts/prep.sh
source activate datamaster
conda info --envs
python main.py query json/coherent_light.json
