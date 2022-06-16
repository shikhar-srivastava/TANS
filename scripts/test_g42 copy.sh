#!/bin/bash
#SBATCH --partition=default-long
#SBATCH --job-name=meta
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --output=/nfs/users/ext_shikhar.srivastava/workspace/TANS/logs/%j.out

/nfs/users/ext_shikhar.srivastava/miniconda3/envs/ofa/bin/python ../main.py --gpu 0 \
                   --mode test \
                   --n-retrievals 10\
                   --n-eps-finetuning 50\
                   --batch-size 32\
                   --load-path /nfs/users/ext_shikhar.srivastava/workspace/TANS/outcomes/ours/20220302_0226/\
                   --base-path /nfs/users/ext_shikhar.srivastava/workspace/TANS/outcomes/ours/20220302_0226/\
                   --data-path /nfs/projects/mbzuai/shikhar/datasets/ofa/our_data_path\
                   --model-zoo /nfs/projects/mbzuai/shikhar/datasets/ofa/our_mod_zoo.pt\
                   --model-zoo-raw /nfs/projects/mbzuai/shikhar/datasets/ofa/model_zoo_raw/v14/geon/final_data/trained_ofa_models\
                   --seed 777