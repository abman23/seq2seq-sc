#!/bin/bash
output_dir='checkpoints/seq2seq-europarl-sc'
trainset_path='data/europarl/processed/train.csv'
devset_path='data/europarl/processed/test.csv'

mkdir -p $output_dir

python train.py \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval \
    --model_name_or_path facebook/bart-base \
    --preprocessing_num_workers 4 \
    --save_total_limit 1 \
    --no_use_fast_tokenizer \
    --num_beams 4 \
    --max_source_length 64 \
    --max_target_length 64 \
    --train_file "$trainset_path" \
    --validation_file "$devset_path" \
    --test_file "$devset_path" \
    --output_dir $output_dir \
    --ebno_db 10 \
    --channel_type AWGN \
    --overwrite_output_dir \
    --tokenizer_name facebook/bart-base \
    --pad_to_max_length \
    --dataset_config 3.0.0