#!/bin/bash
#SBATCH --partition=default-short
#SBATCH --job-name=meta
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --output=/nfs/users/ext_shikhar.srivastava/workspace/TANS/logs/%j.out

/nfs/users/ext_shikhar.srivastava/miniconda3/envs/ofa/bin/python ../main.py --gpu 0 \
                   --mode train \
                   --batch-size 140 \
                   --n-epochs 2500 \
                   --base-path /nfs/users/ext_shikhar.srivastava/workspace/TANS/outcomes/ours\
                   --data-path /nfs/projects/mbzuai/shikhar/datasets/ofa/our_data_path\
                   --model-zoo /nfs/projects/mbzuai/shikhar/datasets/ofa/our_mod_zoo.pt\
                   --model-zoo-raw /nfs/projects/mbzuai/shikhar/datasets/ofa/model_zoo_raw\
                   --seed 777