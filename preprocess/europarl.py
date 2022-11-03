import pathlib
import json
import argparse
from .hf_data_gen import  HFDataGenerator

# The following code is copied (and slightly modified) from DeepSC 
#    (https://github.com/zyy598/DeepSC/blob/master/preprocess_text.py)
import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import os
import json
from tqdm import tqdm

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove the XML-tags
    s = remove_tags(s)
    # add white space before !.?
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    # change to lower letter
    s = s.lower()
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines

def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)

def process_text_file(file_path):
    with open(file_path, 'r') as f:
        raw_data = f.read()
        sentences = raw_data.strip().split('\n')
        raw_data_input = [normalize_string(data) for data in sentences]
        raw_data_input = cutted_data(raw_data_input)
    return raw_data_input

def tokenize(s, delim=' ',  add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, token_to_idx = { }, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, ):
    token_to_count = {}

    for seq in sequences:
      seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                      punct_to_remove=punct_to_remove,
                      add_start_token=False, add_end_token=False)
      for token in seq_tokens:
        if token not in token_to_count:
          token_to_count[token] = 0
        token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
      if count >= min_token_count:
        token_to_idx[token] = len(token_to_idx)

    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
      if token not in token_to_idx:
        if allow_unk:
          token = '<UNK>'
        else:
          raise KeyError('Token "%s" not in vocab' % token)
      seq_idx.append(token_to_idx[token])
    return seq_idx

def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
      tokens.append(idx_to_token[idx])
      if stop_at_end and tokens[-1] == '<END>':
        break
    if delim is None:
      return tokens
    else:
      return delim.join(tokens)

SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def process_europarl(input_data_dir, train_test_split=0.9, njobs=1):
    sentences = []
    print('Preprocess Raw Text')
    from joblib import Parallel, delayed
    sentences = Parallel(n_jobs=njobs, verbose=1)(
        delayed(process_text_file)(fn) 
            for fn in pathlib.Path(input_data_dir).glob('*.txt'))
    sentences = [s for s_list in sentences for s in s_list ]

    # remove the same sentences
    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1
    sentences = list(a.keys())
    print('Number of sentences: {}'.format(len(sentences)))
    
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )

    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))

    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    train_data = results[: round(len(results) * train_test_split)]
    test_data = results[round(len(results) * train_test_split):]

    return train_data, test_data, vocab
# End of the copied code

class Tokenizer:

    TOKENS_FILTERED = set([
        '<START>', '<END>'
    ])

    def __init__(self, vocab):
        idx_to_token = [None for _ in range(1 + max(vocab['token_to_idx'].values()))] 
        for token, idx in vocab['token_to_idx'].items():
            idx_to_token[idx] = token
        self.idx_to_token = idx_to_token
        self.token_to_idx = vocab['token_to_idx']

    def decode(self, token_ids):
        tokens = map(lambda i: self.idx_to_token[i], token_ids)
        tokens = filter(lambda t: t not in self.TOKENS_FILTERED, tokens)
        return ' '.join(tokens)

    def batch_decode(self, token_ids_list):
        return list(map(lambda token_ids: self.decode(token_ids), token_ids_list))

def gen_hf_dataset(path: pathlib.Path, output_path=None, train_test_split=0.9, njobs=1):
    path = pathlib.Path(path)
    if output_path is None:
        output_path = path

    train_data, test_data, vocab = process_europarl(path, train_test_split, njobs)

    # save processed sentences
    with open(output_path / 'train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(output_path / 'test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(output_path / 'vocab.json', 'w') as f:
        json.dump(vocab, f)

    tokenizer = Tokenizer(vocab)
    
    # train set
    train_data = tokenizer.batch_decode(train_data)
    train_gen = HFDataGenerator()
    train_gen.add(train_data, train_data)
    train_gen.dump(output_path / 'train.csv')
    
    # test set
    test_data = tokenizer.batch_decode(test_data)
    test_gen = HFDataGenerator()
    test_gen.add(test_data, test_data)
    test_gen.dump(output_path / 'test.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess Eurlparl data. It generates the same dataset used in DeepSC")
    parser.add_argument(
        '-o', '--out-path', 
        dest='out_path', 
        required=True, 
        type=pathlib.Path,
        help='Required. Path of output files.')
    parser.add_argument(
        '--train-test-split', 
        dest='train_test_split', 
        default=0.9, 
        type=float,
        help='Trainset/ Testset split ratio')
    parser.add_argument(
        '-j'
        '--njobs', 
        dest='njobs', 
        default=1, 
        type=int,
        help='Number of threads to be used for preprocessing')
    parser.add_argument(
        dest='path', 
        type=pathlib.Path,
        help="Path of europarl dataset. It should be '<dataset>/txt/en'")
    args = parser.parse_args()
    gen_hf_dataset(args.path, args.out_path, args.train_test_split, args.njobs)

