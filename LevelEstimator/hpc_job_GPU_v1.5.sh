#!/bin/bash
#SBATCH -D .
#SBATCH -J BiLSTMv5
#SBATCH --output=./logs_1.5/%x-%A_%a.log
 
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=250G
 
source activate $1
srun python -u $2 > ./logs_1.5/BiLSTM$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID.log $SLURM_ARRAY_TASK_ID