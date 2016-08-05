#!/bin/sh

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 0:10:00
#SBATCH -L SCRATCH

module load spearmint
module load deeplearning 

spearmint -c config.json -srun
