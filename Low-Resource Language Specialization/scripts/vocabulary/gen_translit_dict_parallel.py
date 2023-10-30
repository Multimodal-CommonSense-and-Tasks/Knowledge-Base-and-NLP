"""DEPRECATED. see README"""
import argparse
import os

import bert.tokenization as tokenization
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--orig_file", type=str, default="specializing-multilingual-data/data/ug/unlabeled/bert_cleaned/train.txt")
parser.add_argument("--dict_file_prefix", type=str, default="ug_to_uglatinnfc/ug_to_uglatinnfc_tok.txt")
parser.add_argument("--tok", type=str, default="bert-base-multilingual-cased")
parser.add_argument("--orig_lang", type=str, default="ug")
args = parser.parse_args()


def translit(token):
    if args.orig_lang == 'ug':
        from scripts.data.transliterate_nfc_ug_panx import transliterate as transliterate_nfc_ug
        return tokenization.convert_to_unicode(transliterate_nfc_ug(token))
    if args.orig_lang == 'ckb':
        from scripts.data.transliterate_nfc_ckb_panx import transliterate as transliterate_nfc_ckb
        return tokenization.convert_to_unicode(transliterate_nfc_ckb(token))
    if args.orig_lang == 'km':
        from scripts.data.transliterate_km_panx import transliterate as transliterate_km
        return tokenization.convert_to_unicode(transliterate_km(token))
    if args.orig_lang == 'mt':
        from scripts.data.transliterate_mt_panx import transliterate as transliterate_mt
        return tokenization.convert_to_unicode(transliterate_mt(token))
    else:
        from scripts.data.transliterate_wiktra_panx import wiktra_translit
        return tokenization.convert_to_unicode(wiktra_translit(token, args.orig_lang))


def process_build(i, line):
    orig_to_translit = {}

    def add_to_dict(orig_w):
        if orig_w not in orig_to_translit:
            orig_to_translit[orig_w] = translit(orig_w)

    for word in tokenize(line):
        add_to_dict(word)

    with open(f"{args.dict_file_prefix}.{i}", 'w', encoding='utf-8') as f:
        for orig_w, trans_w in orig_to_translit.items():
            f.write(f"{orig_w}\t{trans_w}\n")
    if i % 500 == 0:
        print(f"done building {i}th")


from multiprocessing import Pool

if __name__ == '__main__':
    tok = BertTokenizer.from_pretrained(args.tok)
    def tokenize(line):
        return tok.basic_tokenizer.tokenize(line)

    pool = Pool(40)
    os.makedirs(os.path.dirname(args.dict_file_prefix), exist_ok=True)
    lines = open(args.orig_file, encoding='utf-8').readlines()
    args = list(zip(list(range(len(lines))), lines))
    print(len(lines))
    pool.starmap(process_build, args, chunksize=8)
