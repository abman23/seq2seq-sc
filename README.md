# seq2seq-SC

## Citation

```bash
@misc{lee2022seq2seqSC,
    author = {Lee, Ju-Hyung and Lee, Dong-Ho and Sheen, Eunsoo and Choi, Thomas and Pujara, Jay and Kim, Joongheon},
    title = {Seq2Seq-SC: End-to-End Semantic Communication Systems with Pre-trained Language Model},
    journal={arXiv preprint arXiv:2210.15237},
    year = {2022},
}
```

## Setup

1. Setup conda environment and activate

```bash
conda env create -f environment.yml
```

## Data Preprocessing

### Europarl dataset

```bash
data_path=data/europarl
mkdir -p $data_path
cd $data_path
wget -P /tmp http://www.statmt.org/europarl/v7/europarl.tgz
tar zxf /tmp/europarl.tgz

europarl_dataset="$data_path/txt/en"
out_dir="$data_path/processed"
njobs=4

mkdir -p $out_dir
python -m preprocess.europarl -j $njobs -o $out_dir $europarl_dataset
```

### AllNLI

Run `./scripts/preprocess_allnli.sh` or the following commands

```bash
data_path=data/allnli
mkdir -p $data_path
wget -P $data_path https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/AllNLI.jsonl.gz
gunzip $data_path/AllNLI.jsonl.gz

allnli_dataset="$data_path/AllNLI.jsonl"
out_dir="$data_path/processed"

mkdir -p $out_dir
python -m preprocess.allnli -o $out_dir $allnli_dataset
```

### Flickr30K 

To download the dataset, go to [Flickr30K](http://hockenmaier.cs.illinois.edu/DenotationGraph/) and fill out the form to get the downloadable link. 

```bash
data_path="data/flickr"
dataset_path="${data_path}/flickr30k.tar.gz"
out_dir="$data_path/processed"

mkdir -p $out_dir

tar xzf ${dataset_path} -C $data_path
python -m preprocess.flickr30k \
    -o "$out_dir/flickr30k.json" \
    "${data_path}/results_20130124.token"
```

## Train

You can run `scripts/train_europarl.sh` or `scripts/train_allnli.sh`. Otherwise, you can train by running the follwing commands.

```bash
output_dir='checkpoints/seq2seq-sc'
trainset_path='data/allnli/processed/allnli_train.csv'
devset_path='data/allnli/processed/allnli_dev.csv'

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
```

## Evaluation

You can use the script `scripts/eval_flickr.sh` or the following commands:

```bash
# BLEU score
ebno_db="10"
metric="bleu" # bleu, sbert
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
```

```bash
# SBERT
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
```


