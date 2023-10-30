from typing import List, Dict, Tuple
import argparse
import os
import json
from tqdm import tqdm

from transformers import AutoTokenizer

from utils.utils import load_json_corpus
from utils.logger import logger

def segment_document(tokenizer, text:str, segment_size:int):
    if tokenizer.__class__ in ['transformers.models.t5.tokenization_t5.T5Tokenizer', 'transformers.models.t5.tokenization_t5_fast.T5TokenizerFast']:
        logger.info(f"tokenizer.__class__={tokenizer.__class__}")
        raise NotImplementedError
    start_indicator_token:str = tokenizer.convert_ids_to_tokens(3) # '▁'

    text = text.strip()
    tokens:List[str] = tokenizer.tokenize(text)
    words:List[List[str]] = []
    for token in tokens:
        if token.startswith(start_indicator_token):
            words.append([])
        words[-1].append(token)
    words_ntokens:List[int] = list(map(len, words))
    seg_n_tokens = 0
    segment_tokens:List[List[str]] = [[]]
    for tokens, n_tokens in zip(words, words_ntokens):
        if seg_n_tokens + n_tokens > segment_size:
            seg_n_tokens = 0
            segment_tokens.append([])
        segment_tokens[-1].extend(tokens)
        seg_n_tokens += n_tokens    
    detokenize = lambda x: ''.join(x).replace('▁', ' ').strip()
    segments:List[str] = list(map(detokenize, segment_tokens))
    _st = 0
    segments_char_range:List[Tuple[int, int]] = []
    for seg in segments:
        segments_char_range.append((_st, _st + len(seg)))
        _st += len(seg) + 1 # +1 for white space
    return " ".join(segments), segments_char_range

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_size', type=int, default=128, help="The size of each segment")
    parser.add_argument("--max_n_segments", type=int, default=4)
    parser.add_argument('--data_path', help="/path/to/BEIR dataset.", required=True)
    parser.add_argument('--output_dir', help="/path/to/corpus.segmented.jsonl. If not given, automatically set to data_path/corpus.segmented.jsonl")
    
    args = parser.parse_args()

    print('\n')

    if not args.output_dir:
        args.output_dir = args.data_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    outfile_path:str = os.path.join(args.output_dir, "corpus.jsonl")
    logger.info(f'#> Segmented documents will be saved at {outfile_path}')

    with open(outfile_path, 'w') as fOut:

        logger.info(f'#> Load tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
        
        corpus:Dict[str, Dict] = load_json_corpus(os.path.join(args.data_path, "corpus.jsonl"))

        dataset:str = os.path.basename(args.data_path)
        for doc_id, doc in tqdm(corpus.items(), total=len(corpus), desc=f"Segment documents (Dataset: {dataset})"):
            text, segments_char_range = segment_document(tokenizer=tokenizer, text=doc["text"], segment_size=args.segment_size)
            json_dict = json.dumps({
                "_id": doc_id,
                'text': text,
                'segments_char_range':segments_char_range[:args.max_n_segments],
                'segment_config': {'tokenizer':tokenizer.name_or_path, 'segment_size': args.segment_size, 'max_n_segments': args.max_n_segments},
                **{kw:arg for kw, arg in doc.items() if kw != 'text'}
            })
            fOut.write(f"{json_dict}\n")
    logger.info(f'Done!')
    logger.info(f'\n\n\t {outfile_path}')
    print('\n')