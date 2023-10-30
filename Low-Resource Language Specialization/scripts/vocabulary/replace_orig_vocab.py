import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--replace_map", default='init_vocabs_nolower_unused_arabic.txt')
parser.add_argument("--orig_vocab", default='bert/scripts/convert_to_hf/mbert_vocab.txt')
parser.add_argument("--selected_vocabs", default="specializing-multilingual-data/data/ug/unlabeled/bert_shards_new_onlytrain/20000-5-1000-translit.txt")
parser.add_argument("--out_vocab", default="specializing-multilingual-data/data/ug/unlabeled/bert_shards_new_onlytrain/20000-5-1000-merged.txt")
args = parser.parse_args()

from util.string_utils import unparse_args_to_cmd
print(unparse_args_to_cmd(args))

from collections import OrderedDict
vocab = OrderedDict()

# read vocab
lines = open(args.orig_vocab, encoding='utf-8').readlines()
for i, line in enumerate(lines):
    line = line.strip()
    vocab[i] = line

# get replace_map indicies
replace_index_lines = open(args.replace_map, encoding='utf-8').readlines()
selected_vocab_lines = open(args.selected_vocabs, encoding='utf-8').readlines()
for replace_index_line, selected_vocab_line in zip(replace_index_lines, selected_vocab_lines):
    rep_index = int(replace_index_line.strip().split('\t')[0])
    vocab[rep_index] = selected_vocab_line.strip()

with open(args.out_vocab, 'w', encoding='utf-8') as f:
    for index, word in vocab.items():
        f.write(f"{word}\n")
