#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

source ~/ENV/bin/activate

TASK='SST-2'

CUDA_VISIBLE_DEVICES=0 bash experiments/BERT/heads_ablation.sh $TASK &
sleep 1m

CUDA_VISIBLE_DEVICES=1 bash experiments/BERT/heads_ablation.sh $TASK --reverse_head_mask &

wait

deactivate


