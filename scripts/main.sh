#!/bin/bash
#SBATCH -J main
#SBATCH -o slurm/main.out
#SBATCH -e slurm/main.err
#SBATCH -p defq
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

module load miniconda/miniconda3
source deactivate
conda env update -f env.yml
source activate datamaster
python main.py
