#!/bin/bash
ebno_db="10"
metric="sbert" # bleu, sbert
testset_path='data/flickr/processed/flickr30k.json'
checkpoint_path="checkpoints/seq2seq-allnli-sc"

python eval.py \
    --batch 4 \
    --metric "${metric}" \
    --ebno-db "${ebno_db}" \
    --result-json-path "${checkpoint_path}/flikr_${metric}_ebno_${ebno_db}.json" \
    --prediction-json-path "${checkpoint_path}/flikr_prediction_ebno_${ebno_db}.json" \
    --testset-path "${testset_path}" \
    $checkpoint_path