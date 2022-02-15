#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=meta
#SBATCH --time=12:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=/l/users/shikhar.srivastava/workspace/TANS/scripts/%j.out
#SBATCH -q gpu-single

ulimit -u 10000

/home/shikhar.srivastava/miniconda3/envs/tans_a/bin/python ../main.py --gpu 0 \
                   --mode train \
                   --batch-size 140 \
                   --n-epochs 10000 \
                   --base-path /l/users/shikhar.srivastava/workspace/TANS/outcomes \
                   --data-path /l/users/shikhar.srivastava/data/ofa/data_path \
                   --model-zoo /l/users/shikhar.srivastava/data/ofa/model_zoo/p_mod_zoo.pt \
                   --model-zoo-raw /l/users/shikhar.srivastava/data/ofa/model_zoo_raw \
                   --seed 777