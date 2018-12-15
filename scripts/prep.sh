#!/bin/bash

module load miniconda/miniconda3
source deactivate
conda env update -f env.yml
source activate datamaster
