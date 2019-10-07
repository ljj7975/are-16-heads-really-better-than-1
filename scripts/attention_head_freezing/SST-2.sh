#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

source ~/ENV/bin/activate



CUDA_VISIBLE_DEVICES=0 bash experiments/BERT/heads_freezing.sh SST-2 &
sleep 1m

CUDA_VISIBLE_DEVICES=1 bash experiments/BERT/heads_freezing.sh SST-2 --reverse_freezing &
sleep 1m

CUDA_VISIBLE_DEVICES=2 bash experiments/BERT/heads_freezing.sh SST-2 --incremental_freezing &


wait

deactivate
