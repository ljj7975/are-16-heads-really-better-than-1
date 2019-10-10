#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

model_dir=freeze_all/$TASK/base
if [[ $OPTIONS == *"--reverse_freezing"* ]]; then
    REVERSE=true
    model_dir=freeze_all/$TASK/reverse
elif [[ $OPTIONS == *"--incremental_freezing"* ]]; then
    INCREMENTAL=true
    model_dir=freeze_all/$TASK/incremental
fi

LOG_FILE="$model_dir/freezing_results.txt"
echo "log file : $LOG_FILE"
echo "task : $TASK"
echo "options : $OPTIONS"
echo "data dir : $DATA_DIR"
mkdir -p $model_dir
echo "model dir : $model_dir"

function run_train () {
    # $1 = output dir
    # $2 = args

    mkdir -p $1

    python BERT/run_classifier.py \
    --task_name $TASK \
    --do_train \
    --do_lower_case \
    --data_dir $DATA_DIR/glue/$TASK/ \
    --bert_model $TRAINED_MODEL_DIR/bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $1 \
    $2 2>&1
}

function run_eval () {
    # $1 = output dir

    python BERT/run_classifier.py \
    --task_name $TASK \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR/glue/$TASK/ \
    --bert_model $TRAINED_MODEL_DIR/bert-base-uncased \
    --max_seq_length 128 \
    --eval_batch_size 16 \
    --output_dir $1 2>&1
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
    result=$(run_train $model_dir)
else
    echo "trained model exist"
fi

base_acc=$(run_eval $model_dir | grep $metric | rev | cut -d" " -f1 | rev)
echo "base_acc : $base_acc"
echo $base_acc >> $LOG_FILE

# freezing layers
for layer in `seq 1 12`
do
    mask_str=""

    if [ "$REVERSE" = true ] ; then
        for layer_to_freeze in `seq 1 12`
        do
            if [ "${layer}" == "${layer_to_freeze}" ]; then
                continue
            fi
            mask_str+="${layer_to_freeze}.attention.self ${layer_to_freeze}.attention.output.dense ${layer_to_freeze}.intermediate.dense ${layer_to_freeze}.output.dense. "
        done
    elif [ "$INCREMENTAL" = true ] ; then
        for layer_to_freeze in `seq ${layer} 12`
        do
            mask_str+="${layer_to_freeze}.attention.self ${layer_to_freeze}.attention.output.dense ${layer_to_freeze}.intermediate.dense ${layer_to_freeze}.output.dense. "
        done
    else
        mask_str="${layer}.attention.self ${layer}.attention.output.dense ${layer}.intermediate.dense ${layer}.output.dense. "
    fi

    echo $mask_str

    store_at=$model_dir/$layer/
    mkdir -p store_at

    start_time=$(date +%s.%N)

    if [ ! -e $store_at/pytorch_model.bin ]
    then
        result=$(run_train $store_at "--freeze_param $mask_str")
    else
        echo "trained model exist"
    fi


    acc=$(run_eval $store_at | grep $metric | rev | cut -d" " -f1 | rev)
    end_time=$(date +%s.%N)
    echo -e "\t time elapsed - " $(echo "$end_time - $start_time" | bc)

    echo -e "\t acc - $acc"
    printf "\t diff - %.5f\n" $(echo "$acc - $base_acc" | bc)

    echo -e "$layer\t$mask_str\t$acc" >> $LOG_FILE
done

exp_end_time=$(date +%s.%N)

echo "experiment time elapsed - " $(echo "$exp_end_time - $exp_start_time" | bc)
