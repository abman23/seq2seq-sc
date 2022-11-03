#!/bin/bash
data_path="data/flickr"
dataset_path="${data_path}/flickr30k.tar.gz"
out_dir="$data_path/processed"

mkdir -p $out_dir

tar xzf ${dataset_path} -C $data_path
python -m preprocess.flickr30k \
    -o "$out_dir/flickr30k.json" \
    "${data_path}/results_20130124.token"