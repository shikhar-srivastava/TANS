#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --job-name=meta
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=64
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --output=/nfs/users/ext_shikhar.srivastava/workspace/TANS/logs/%j.out

/nfs/users/ext_shikhar.srivastava/miniconda3/envs/ofa/bin/python ../main.py --gpu 0 \
                   --mode test \
                   --n-retrievals 10\
                   --n-eps-finetuning 50\
                   --batch-size 32\
                   --load-path /nfs/users/ext_shikhar.srivastava/workspace/TANS/outcomes/20220209_1428\
                   --data-path /nfs/projects/mbzuai/shikhar/datasets/ofa/data_path\
                   --model-zoo /nfs/projects/mbzuai/shikhar/datasets/ofa/model_zoo/p_mod_zoo.pt\
                   --model-zoo-raw /nfs/projects/mbzuai/shikhar/datasets/ofa/model_zoo_raw/v14/geon/final_data/trained_ofa_models\
                   --seed 777