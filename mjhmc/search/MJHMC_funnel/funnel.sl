#!/bin/sh

#SBATCH -p regular
#SBATCH -N 10
#SBATCH -t 24:00:00
#SBATCH -L SCRATCH

module load spearmint
module load deeplearning 

spearmint -c config.json -srun
