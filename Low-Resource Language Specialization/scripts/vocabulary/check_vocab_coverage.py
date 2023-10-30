import argparse
import bert.tokenization as tokenization
import os
from scripts.data.transliterate_nfc_ug_panx import transliterate as transliterate_nfc_ug
from util.other_utils import load_dict

parser = argparse.ArgumentParser()
parser.add_argument("--orig_file", type=str, default="specializing-multilingual-data/data/ug/ud/train.txt")
parser.add_argument("--dict_file", type=str, default="specializing-multilingual-data/data/dict/uig_tur_lexicon_3586.txt")
args = parser.parse_args()

vocab = load_dict(args.dict_file)
multi_word_vocabs = {}
for tok, trans in vocab.items():
    if len(tok.split()) > 1:
        multi_word_vocabs[tok] = trans
print(len(multi_word_vocabs))

lines = open(args.orig_file, encoding='utf-8').read()
total_words = 0
covered_words = 0
multi_words = 0
multi_words_total_splitted_len = 0
for tok, trans in multi_word_vocabs.items():
    if tok in lines:
        multi_words += 1
        multi_words_total_splitted_len += lines.count(tok) * len(tok.split())
print(f"multi words are contained {multi_words} and {multi_words_total_splitted_len}")

for line in lines.split('\n'):
    words = line.strip().split()
    for word in words:
        total_words += 1
        covered_words += int(word in vocab)

print(f"coverage : {covered_words / total_words} .. Covered words {covered_words} among {total_words}")