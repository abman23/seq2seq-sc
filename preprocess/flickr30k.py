from collections import defaultdict
import json
import random
import argparse
import pathlib

def parse_line(line: str):
    key = line.split('#')[0]
    caption = line.split('\t')[-1].strip()
    return key, caption

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-path',
        dest='out_path',
        type=pathlib.Path,
        help='output json file path')
    parser.add_argument('-n', '--num-samples',
        dest='N',
        default=1000,
        type=int,
        help='number of samples (default: 1000)')
    parser.add_argument('--send',
        dest='seed',
        default=20221017,
        type=int,
        help='seed for random module')

    parser.add_argument(
        dest='token_path',
        help="path of 'results_20130124.token'")
    args = parser.parse_args()

    # read token file
    data = defaultdict(list)
    with open(args.token_path) as f:
        for k, caption in map(parse_line, f):
            data[k].append(caption)
    data = list(data.values())

    # set seed
    random.seed(args.seed)

    # sample dataset
    samples = random.sample(range(len(data)), k=args.N)
    out_data = []
    for i in samples:
        captions = data[i]
        input_idx = random.sample(range(len(captions)), k=1)[0]
        input_sentence = captions[input_idx]
        ref_sentences = captions[:input_idx] + captions[(input_idx+1):]
        out_data.append({
            'input': input_sentence,
            'refs': ref_sentences,
        })

    with open(args.out_path, 'w') as f:
        json.dump(out_data, f, indent=4)