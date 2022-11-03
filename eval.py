import pathlib
import json
import argparse
import logging
from transformers import BartTokenizer
import evaluate
from tqdm import tqdm
import warnings

def get_test_data(path):
    with open(path) as f:
        return json.load(f)

def from_pretrained(path, ebno_db):
    from models import TFSeq2SeqSCForConditionalGeneration
    import transformers
    transformers.utils.logging.set_verbosity(logging.INFO)
    return TFSeq2SeqSCForConditionalGeneration.from_pretrained(
        path, ebno_db=ebno_db)

def predict(path, ebno_db, tokenizer, batch_size, test_data_path, num_beams):
    import tensorflow as tf
    max_len = 32

    # load model
    model = from_pretrained(path, ebno_db)

    # # load testset
    test_data = get_test_data(test_data_path)
    input_sentences = [d['input'] for d in test_data]
    input_ids = tokenizer(input_sentences, return_tensors="tf", 
                padding='max_length', truncation=True, max_length=max_len).input_ids
    testset = tf.data.Dataset.from_tensor_slices(input_ids)        

    # inference
    pred_sentences = []
    for input_ids in tqdm(testset.batch(batch_size).prefetch(tf.data.AUTOTUNE)):
        pred_batch = model.generate(input_ids, max_new_tokens=max_len, num_beams=num_beams)
        output_strs = tokenizer.batch_decode(pred_batch,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        pred_sentences.extend(output_strs)

    
    res = {
        'input': input_sentences,
        'pred': pred_sentences,
        'refs': [d['refs'] for d in test_data],
    }
    return res

def get_predictions(path, ebno_db, test_data_path, prediction_json_path, batch_size, tokenizer, num_beams):
    path = pathlib.Path(path)
    if not prediction_json_path.exists():
        print('Missing predictions.json')
        res = predict(
            path=path, 
            ebno_db=ebno_db, 
            tokenizer=tokenizer, 
            batch_size=batch_size, 
            test_data_path=test_data_path, 
            num_beams=num_beams,
        )

        # save result
        with open(prediction_json_path, 'w') as f:
            json.dump(res, f, indent=4)
    else:
        with open(prediction_json_path, 'r') as f:
            res = json.load(f)
    return res  

def calc_bleu(predictions, tokenizer, multi_ref, **kwargs):
    bleu = evaluate.load('bleu')
    if multi_ref:
        warnings.warn('BLEU does not support multiple references')
    tokenize = lambda l: tokenizer(l, add_special_tokens=False).input_ids
    results = bleu.compute(
        references=predictions['input'],
        predictions=predictions['pred'],               
        tokenizer=tokenize,
        max_order=4)
    return results

def calc_sbert(predictions, batch_size, multi_ref, **kwargs):
    from sentence_transformers import SentenceTransformer, util
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
            model_name_or_path='all-MiniLM-L6-v2',
            device=device)

    sentences1 = predictions['pred']
    if not multi_ref:
        refs = [[s] for s in predictions['input']]
    else:
        refs = predictions['refs']

    def calc_cos_score(model, hyp_embedding, ref_sentences):
        hyp = hyp_embedding.reshape((1, -1))
        refs = model.encode(ref_sentences, convert_to_tensor=True)
        scores = util.cos_sim(hyp, refs)
        scores = scores.reshape((-1)).tolist()
        return {
                'scores': scores,
                'max_score': max(scores),
                'mean_score': sum(scores) / len(scores),
                }
        

    # compute embedding
    pred_embed = model.encode(sentences1, batch_size=batch_size, convert_to_tensor=True)
    N = pred_embed.shape[0]
    scores = [
            calc_cos_score(model, pred_embed[i], refs[i]) for i in range(N)
    ]
    max_scores = [s['max_score'] for s in scores]
    mean_score = sum(max_scores)/len(max_scores)
    return {
        'metric': 'sentence textual similarity',
        'mean_score': mean_score,
        'scores': scores,
    }

METRIC_TO_SCORER = {
        'bleu': calc_bleu,
        'sbert': calc_sbert,
}

def calc(args):
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
    
    path = args.path
    metric = args.metric
    batch_size = args.batch_size

    predictions = get_predictions(
            path, 
        ebno_db=args.ebno_db,
        prediction_json_path=args.prediction_json_path,
        test_data_path=args.testset_path, 
        batch_size=batch_size, 
        tokenizer=tokenizer,
        num_beams=args.num_beams)
    scorer = METRIC_TO_SCORER[metric]
    results = scorer(
        predictions=predictions,
        tokenizer=tokenizer,
        batch_size=batch_size,
        multi_ref=args.multi_ref,
    )
    # dump result
    with open(args.result_json_path, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', metavar='checkpoint_path', type=pathlib.Path)
    parser.add_argument('-m', '--metric', choices = list(METRIC_TO_SCORER.keys()), dest='metric')
    parser.add_argument('-b', '--batch-size', default=4, type=int, dest='batch_size')
    parser.add_argument('-e', '--ebno-db', required=True, type=float, dest='ebno_db')
    parser.add_argument('--testset-path', 
            required=True, type=pathlib.Path, dest='testset_path')
    parser.add_argument('--prediction-json-path', 
            required=True, 
            type=pathlib.Path,
            dest='prediction_json_path',
            help='Required. Output path of prediction result cache json file. \
                  If the file exists, the prediction result will be reused')
    parser.add_argument('--result-json-path', 
            default=pathlib.Path('./result.json'),
            type= pathlib.Path,
            dest='result_json_path')
    parser.add_argument('--tokenizer', 
            default='facebook/bart-base', 
            dest='tokenizer')
    parser.add_argument('--num-beams', 
            default=1, 
            type=int,
            dest='num_beams')
    parser.add_argument('--multi-ref', 
            action='store_true', 
            dest='multi_ref')
    args = parser.parse_args()
    calc(args)

if __name__ == '__main__':
    main()
