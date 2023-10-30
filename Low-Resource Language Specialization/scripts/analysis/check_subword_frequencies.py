import os

from transformers import HfArgumentParser, AutoTokenizer, BertTokenizer
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm

def agg_files(glob_pattern):
    text = ""
    for f in sorted(glob.glob(glob_pattern)):
        text += open(f, encoding='utf-8').read()
    return text

@dataclass
class Args:
    lang: str = 'ug'
    count: int = 5000

parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)
args.tokenizer_path = f"specializing-multilingual-data/data/{args.lang}/unlabeled/bert_cleaned/{args.lang}_5000-5-1000-{args.count}"
args.target_text = f"specializing-multilingual-data/data/{args.lang}/unlabeled/bert_cleaned/*.txt"


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
total_text = agg_files(args.target_text)

added_dict = defaultdict(int)
for line in tqdm(total_text.split('\n')):
    tokens = tokenizer(line, add_special_tokens=False)['input_ids']
    for t_id in tokens:
        added_dict[t_id] += 1

import pickle
pickle.dump(added_dict, open("ug_corpus_dict.pkl", 'wb'))
