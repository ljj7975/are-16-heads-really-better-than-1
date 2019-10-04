#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"
LOG_FILE="models/${TASK}/ablation_results.txt"
echo "log file : $LOG_FILE"
echo "task : $TASK"
echo "options : $OPTIONS"
echo "data dir : $DATA_DIR"

echo "=== prepare_task ==="

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh $TASK

echo "base_acc $base_acc"
echo $base_acc >> $LOG_FILE

echo "=== evaluate ablation ==="

exp_start_time=$(date +%s.%N)

for layer in `seq 1 12`
do
    echo "layer - $layer"
    for head in `seq 1 12`
    do
        mask_str="${layer}:${head}"
        echo -e "\t$mask_str $OPTIONS"

        start_time=$(date +%s.%N)
        acc=$(run_eval "--attention_mask_heads $mask_str $OPTIONS" | grep $metric | rev | cut -d" " -f1 | rev)
        end_time=$(date +%s.%N)
        echo -e "\ttime elapsed - " $(echo "$end_time - $start_time" | bc)

        echo -en "\tacc - $acc"
        printf "\t diff - %.5f\n" $(echo "$acc - $base_acc" | bc)

        echo "$acc" >> $LOG_FILE
    done
done

exp_end_time=$(date +%s.%N)

echo "experiment time elapsed - " $(echo "$exp_end_time - $exp_start_time" | bc)



