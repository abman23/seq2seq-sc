#!/bin/bash
europarl_dataset=data/europarl/txt/en
out_dir=data/europarl/processed
njobs=4

mkdir -p $out_dir
python -m preprocess.europarl -j $njobs -o $out_dir $europarl_dataset
