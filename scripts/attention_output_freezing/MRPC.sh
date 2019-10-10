#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

source ~/ENV/bin/activate

TASK='MRPC'
SCRIPT='attention_output_freezing.sh'


CUDA_VISIBLE_DEVICES=0 bash experiments/BERT/$SCRIPT $TASK &
sleep 1m

CUDA_VISIBLE_DEVICES=1 bash experiments/BERT/$SCRIPT $TASK --reverse_freezing &
sleep 1m

CUDA_VISIBLE_DEVICES=2 bash experiments/BERT/$SCRIPT $TASK --incremental_freezing &
sleep 1m

CUDA_VISIBLE_DEVICES=3 bash experiments/BERT/$SCRIPT $TASK --reverse_incremental_freezing &

wait

deactivate
