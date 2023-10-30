import argparse
import os
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--orig_file", type=str, default="specializing-multilingual-data/data/ckb/unlabeled/bert_cleaned/train.txt")
parser.add_argument("--save_path", type=str, default='/data2/ckb_words')
parser.add_argument("--using_basic_tok", default=False, action='store_true')
args = parser.parse_args()

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

def save_dicts(lines):
    words = {}
    start_i = lines[0][0]

    for i, line in lines:
        for w in split(line):
            words[w] = i
        if i % 500 == 0:
            print(i)

    for i, w in enumerate(words.keys()):
        with open(os.path.join(save_path, f"{start_i:010}_{i:010}"), "w", encoding='utf-8') as f:
            f.write(w)

def py_split(line):
    return line.strip().split()

if __name__ == '__main__':
    num_cores = 32
    words = {}
    lines = list(enumerate(open(args.orig_file).readlines()))
    total_lines = len(lines)
    all_sub_lines = []
    for c_i in range(num_cores):
        start_i = total_lines * c_i // num_cores
        end_i = total_lines * (c_i + 1) // num_cores
        sub_lines = lines[start_i:end_i]
        all_sub_lines.append(sub_lines)

    if args.using_basic_tok:
        from transformers import BertTokenizer
        split = BertTokenizer.from_pretrained('bert-base-multilingual-cased').basic_tokenizer.tokenize
    else:
        split = py_split

    pool = Pool(num_cores)
    pool.map(save_dicts, all_sub_lines)