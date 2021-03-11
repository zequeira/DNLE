#!/bin/bash

#SBATCH -D . 							    # Working Directory
#SBATCH -J MFCC_calc 					# Job Name
#SBATCH --output=./logs/MFCC_calc.log

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

#SBATCH --time=48:00:00 # expected runtime
#SBATCH --partition=standard

#Job-Status per Mail:
#SBATCH --mail-type=NONE
#SBATCH --mail-user=some.one@tu-berlin.de

source activate $1
srun python -u $2 > ./logs/MFCCs_parallel.log
