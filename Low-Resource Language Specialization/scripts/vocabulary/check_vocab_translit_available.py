from transformers import HfArgumentParser, AutoTokenizer, BertTokenizer
from dataclasses import dataclass, field
from scripts.data.script_checker import detector, most_significant_alphabet, agg_files
from scripts.data.transliterate_nfc_ug_panx import transliterate as transliterate_nfc_ug


@dataclass
class Args:
    tokenizer_path: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_shards")
    input_vocab_file: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_shards/5000-5-1000-vocab_augment.txt")
    output_vocab_file: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_shards/5000-5-1000-vocab_augment_filtered.txt")
    translit: str = field(default='ug')
    script_lang: str = field(default='ARABIC')


parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
vocab = tokenizer.get_vocab()
assert args.translit == 'ug'
translit = transliterate_nfc_ug
lines = open(args.input_vocab_file, encoding='utf-8').readlines()


with open(args.output_vocab_file, 'w', encoding='utf-8') as f_v:
    for line in lines:
        tok = line.strip()
        if tok and detector.real_alphabet_chars(tok, args.script_lang):
            translit_tok = translit(tok).replace("ئ", "")
            if translit_tok in vocab:
                f_v.write(f"{tok}\n")