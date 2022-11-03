#!/bin/bash
data_path=data/allnli
mkdir -p $data_path
wget -P $data_path https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/AllNLI.jsonl.gz
gunzip $data_path/AllNLI.jsonl.gz

allnli_dataset="$data_path/AllNLI.jsonl"
out_dir="$data_path/processed"

mkdir -p $out_dir
python -m preprocess.allnli -o $out_dir $allnli_dataset