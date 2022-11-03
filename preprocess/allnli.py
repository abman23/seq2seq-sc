import argparse
import json
import pathlib
import random

from .hf_data_gen import HFDataGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess Eurlparl data. It generates the same dataset used in DeepSC")
    parser.add_argument(
        '-o', '--out-path', 
        dest='out_path', 
        required=True, 
        type=pathlib.Path,
        help='Required. Path of output directory')
    parser.add_argument(
        '--train-dev-split', 
        dest='train_dev_split', 
        default=0.9, 
        type=float,
        help='Trainset/ Devset split ratio')
    parser.add_argument(
        '--seed', 
        dest='seed', 
        default=1234, 
        type=int,
        help='Random seed')
    parser.add_argument(
        dest='path', 
        type=pathlib.Path,
        help="Path of AllNLI.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.path) as f:
        data = [json.loads(line) for line in f]

    data = map(lambda l: (l[0], l[1]), data)
    sentences = filter(lambda l: 'n/a' not in l, data)
    sentences = list(sentences)

    N = len(sentences)
    devset_size = int(N*(1-args.train_dev_split))
    devset_indices = random.sample(range(N), devset_size)
    devset_indices = set(devset_indices)

    trainset_gen = HFDataGenerator()
    devset_gen = HFDataGenerator()
    for i, (s1, s2) in enumerate(sentences):
        if i in devset_indices:
            devset_gen.add(s1, s2)
        else:
            trainset_gen.add(s1, s2)

    trainset_gen.dump(args.out_path / 'allnli_train.csv')
    devset_gen.dump(args.out_path / 'allnli_dev.csv')