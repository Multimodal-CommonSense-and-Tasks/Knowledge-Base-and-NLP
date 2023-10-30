from transformers import HfArgumentParser, AutoTokenizer, BertTokenizer
from dataclasses import dataclass, field
from scripts.data.script_checker import detector, most_significant_alphabet, agg_files
from scripts.data.transliterate_nfc_ug_panx import transliterate as transliterate_nfc_ug


@dataclass
class Args:
    tokenizer_path: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_shards")
    init_vocabs: str = field(default="init_vocabs_nolower.txt")
    output_map: str = field(default="vocab_map_list.txt")
    output_map_verbose: str = field(default="vocab_map_list_verbose.txt")
    translit: str = field(default='ug')


parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
vocab = tokenizer.get_vocab()
# reverse_vocab = {v: k for (k, v) in vocab.items()}

assert args.translit == 'ug'
translit = transliterate_nfc_ug

lines = open(args.init_vocabs, encoding='utf-8').readlines()
cnt = 0
with open(args.output_map_verbose, 'w', encoding='utf-8') as f_v:
    with open(args.output_map, 'w', encoding='utf-8') as f:
        for line in lines:
            token_id, real_token = line.strip().split('\t')
            translit_token = translit(real_token.replace("ئ", ""))
            # print(f"{real_token.strip()}\t{translit_token}\n")
            f.write(f"{token_id}")
            translit_tokens = tokenizer(translit_token, add_special_tokens=False)['input_ids']
            for t in translit_tokens:
                f.write(f"\t{t}")
            f.write("\n")
            f_v.write(f'{real_token}\t{translit_token}\n')
print(cnt)
