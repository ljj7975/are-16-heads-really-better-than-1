#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

source ~/ENV/bin/activate

TASK='MNLI'
SCRIPT='intermediate_dense_freezing.sh'


CUDA_VISIBLE_DEVICES=0 bash experiments/BERT/$SCRIPT $TASK &
sleep 1m

CUDA_VISIBLE_DEVICES=1 bash experiments/BERT/$SCRIPT $TASK --reverse_freezing &
sleep 1m

CUDA_VISIBLE_DEVICES=2 bash experiments/BERT/$SCRIPT $TASK --incremental_freezing &

wait

deactivate
