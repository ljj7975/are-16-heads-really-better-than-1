#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

LOG_FILE="models/${TASK}/ablation_results.txt"
if [[ $OPTIONS == *"--reverse_head_mask"* ]]; then
    LOG_FILE="models/${TASK}/reverse_ablation_results.txt"
fi

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
    for head_1 in `seq 1 12`
    do
        for head_2 in `seq 1 12`
        do

            if [ "${head_1}" == "${head_2}" ]; then
                continue
            fi
            
            mask_str="${layer}:${head_1},${head_2}"
            echo -e "\t$mask_str $OPTIONS"

            start_time=$(date +%s.%N)

            acc=$(run_eval "--attention_mask_heads $mask_str_1 $mask_str_2 $OPTIONS" | grep $metric | rev | cut -d" " -f1 | rev)
            end_time=$(date +%s.%N)
            echo -e "\ttime elapsed - " $(echo "$end_time - $start_time" | bc)

            echo -en "\tacc - $acc"
            printf "\t diff - %.5f\n" $(echo "$acc - $base_acc" | bc)

            echo -e "$mask_str\t$acc" >> $LOG_FILE
        done
    done
done


exp_end_time=$(date +%s.%N)

echo "experiment time elapsed - " $(echo "$exp_end_time - $exp_start_time" | bc)
