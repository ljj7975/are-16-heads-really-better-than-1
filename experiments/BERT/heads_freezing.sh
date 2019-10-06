#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

LOG_FILE="models/${TASK}/freezing_results.txt"
if [[ $OPTIONS == *"--reverse_freezing"* ]]; then
    LOG_FILE="models/${TASK}/reverse_freezing_results.txt"
fi
echo "log file : $LOG_FILE"
echo "task : $TASK"
echo "options : $OPTIONS"
echo "data dir : $DATA_DIR"

function run_train () {

    if [ $# -eq 0 ]
    then
        model_dir=models/$TASK
    else
        model_dir=models/$TASK/$1
    fi

    mkdir -p $model_dir

    TRAIN_CMD="python BERT/run_classifier.py \
    --task_name $TASK \
    --do_train \
    --do_lower_case \
    --data_dir $DATA_DIR/glue/$TASK/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $model_dir"

    if [ $# -eq 1 ]
    then
        TRAIN_CMD+=" \
        --freeze_param $1"
    elif [ $# -eq 2 ]
    then
        TRAIN_CMD+=" \
        $2"
    fi

    $(TRAIN_CMD 2>&1)
}

function run_eval () {

    if [ $# -eq 0 ]
    then
        model_dir=models/$TASK
    else
        model_dir=models/$TASK/$1
    fi

    EVAL_CMD="python BERT/run_classifier.py \
    --task_name $TASK \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR/glue/$TASK/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --eval_batch_size 8 \
    --output_dir $model_dir"

    $(EVAL_CMD 2>&1)
}

metric="eval_accuracy"
if [ $TASK == "MRPC" ]
then
    metric="F-1"
elif [ $TASK == "CoLA" ]
then
    metric="Matthew"
fi

exp_start_time=$(date +%s.%N)

# base accuracy

if [ ! -e $model_dir/pytorch_model.bin ]
then
    run_train
else
    echo "trained model exist"
fi

base_acc=$(run_eval | grep $metric | rev | cut -d" " -f1 | rev)
echo "base_acc $base_acc"
echo $base_acc >> $LOG_FILE

# freezing layers
for layer in `seq 1 12`
do

    mask_str="layer.$layer.attention.self"
    echo $mask_str

    start_time=$(date +%s.%N)
    $(run_train $mask_str $OPTIONS)
    acc=$(run_eval $mask_str | grep $metric | rev | cut -d" " -f1 | rev)
    end_time=$(date +%s.%N)
    echo -e "\ttime elapsed - " $(echo "$end_time - $start_time" | bc)

    echo -en "\tacc - $acc"
    printf "\tdiff - %.5f\n" $(echo "$acc : $base_acc" | bc)

    echo -e "$mask_str\t$acc" >> $LOG_FILE
done

exp_end_time=$(date +%s.%N)

echo "experiment time elapsed - " $(echo "$exp_end_time - $exp_start_time" | bc)
