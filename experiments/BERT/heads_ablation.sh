#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"
echo "task : $TASK"
echo "options : $OPTIONS"
echo "data dir : $DATA_DIR"

echo "=== prepare_task ==="

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh $TASK

echo "=== evaluate ablation ==="

echo "base_acc $base_acc"
for layer in `seq 1 12`
do
    echo "layer - $layer"
    for head in `seq 1 12`
    do
        mask_str="${layer}:${head}"
        echo "head - $head, mask_str - $mask_str"
        acc=$(run_eval "--attention_mask_heads $mask_str $OPTIONS" | grep $metric | rev | cut -d" " -f1 | rev)
        echo -en "\tacc - $acc"
        printf "\t diff - %.5f\n" $(echo "$acc - $base_acc" | bc)
    done
done

