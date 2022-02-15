#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=meta
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=32
#SBATCH --mem=75G
#SBATCH --gres=gpu:1
#SBATCH --output=/l/users/shikhar.srivastava/workspace/TANS/scripts/%j.out
#SBATCH -q gpu-single

ulimit -u 10000


/home/shikhar.srivastava/miniconda3/envs/tans_a/bin/python  ../main.py --gpu 0 \
                   --mode test \
                   --n-retrievals 10\
                   --n-eps-finetuning 50\
                   --batch-size 32\
                   --load-path /l/users/shikhar.srivastava/workspace/TANS/outcomes/20220209_0945\
                   --data-path /l/users/shikhar.srivastava/data/ofa/data_path\
                   --model-zoo /l/users/shikhar.srivastava/data/ofa/model_zoo/p_mod_zoo.pt\
                   --model-zoo-raw /l/users/shikhar.srivastava/data/ofa/model_zoo_raw/v14/geon/final_data/trained_ofa_models\
                   --seed 777