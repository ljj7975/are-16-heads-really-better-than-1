#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

source ~/ENV/bin/activate

TASK='MRCP'

CUDA_VISIBLE_DEVICES=0 bash experiments/BERT/heads_freezing.sh $TASK &
sleep 1m

CUDA_VISIBLE_DEVICES=1 bash experiments/BERT/heads_freezing.sh $TASK --reverse_freezing &
sleep 1m

CUDA_VISIBLE_DEVICES=2 bash experiments/BERT/heads_freezing.sh $TASK --incremental_freezing &


wait

deactivate
