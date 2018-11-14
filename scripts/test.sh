#!/bin/bash
#SBATCH -J test
#SBATCH -o slurm/test.out
#SBATCH -e slurm/test.err
#SBATCH -p debug
#SBATCH -n 16
#SBATCH -t 0-02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryansteed@gwu.edu

module load miniconda/miniconda3
source deactivate
conda env update -f env.yml
source activate datamaster
python main.py
