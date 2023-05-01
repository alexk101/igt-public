#!/bin/bash

#SBATCH -J plot_baselines
#SBATCH -p general
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH -A alkiefer

#Load any modules that your program needs
module load miniconda/python3.9/4.12.0

#Run your program
srun ./my_program my_program_arguments