#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --job-name=meta
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gpus=1

sleep 10m