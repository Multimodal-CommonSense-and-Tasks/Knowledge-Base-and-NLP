import os.path

from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from googletrans import Translator
import json

@dataclass
class Args:
    model_folder: str
    csv: str
    trans_src: str = field(default=None)
    trans_tgt: str = field(default='en')
    trans_dict_path: str = field(default=None)

parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)

tok = AutoTokenizer.from_pretrained(args.model_folder)
lines = open(args.csv, encoding='utf-8').readlines()
with open(args.csv+"_tokenized", 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(' '.join(tok.tokenize(line))+'\n')

if args.trans_src:
    trans_dict_path = args.trans_dict_path
    if trans_dict_path is None:
        trans_dict_path = f"{args.trans_src}_to_{args.trans_tgt}.json"

    lines = [line.strip() for line in lines]
    if os.path.exists(trans_dict_path):
        word_to_trans = json.load(open(trans_dict_path, encoding='utf-8'))
    else:
        words = set(lines)
        words.remove("PAD")
        trans = Translator()
        words = list(words)
        translated = trans.translate(words, src=args.trans_src, dest=args.trans_tgt)

        word_to_trans = {"PAD": "PAD"}
        for word, trans in zip(words, translated):
            word_to_trans[word] = trans.text
        json.dump(word_to_trans, open(trans_dict_path, 'w', encoding='utf-8'))

    suffix = f"_{args.trans_tgt}"# if args.trans_tgt != 'en' else ''
    with open(args.csv + f"_translated{suffix}", 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{word_to_trans[line]}\n")
