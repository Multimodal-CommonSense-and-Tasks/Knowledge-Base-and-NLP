import os
import pickle
import random
from dataclasses import dataclass
from typing import Union, Optional

from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from sacremoses import MosesTokenizer

import util.other_utils


def limit_len(cache_dir, prefix="", limit=143):
    if len(cache_dir) > limit:
        from util.log_utils import get_hash
        cache_dir = f"{prefix}{get_hash(cache_dir, 32)}"
        return cache_dir


def get_cache_dir(args, regularlize_on_orig_embeddings):
    cache_dir = f"{args.lang}_iguw{args.ignore_unk_words}_reg{regularlize_on_orig_embeddings}"

    if args.align_tgt_model_tok_cfg is not None:
        cache_dir += args.align_tgt_model_tok_cfg.replace(os.sep, '_')
    if args.use_basic_tok:
        cache_dir += '_usebasictok'
    cache_dir = limit_len(cache_dir, prefix=f"{args.lang}_")
    return cache_dir


class OrigTLDataset(Dataset):
    def split_to_words(self, line, align_tgt=False):
        result = []
        cur_tok = self.align_tgt_tok if align_tgt else self.tok
        for w in self.split_fn(line):
            if w not in self.word_to_tok_dict[align_tgt]:
                self.word_to_tok_dict[align_tgt][w] = cur_tok.basic_tokenizer.tokenize(w)
            result.extend(self.word_to_tok_dict[align_tgt][w])
        return result
        # return line.split()
        # return tok.basic_tokenizer.tokenize(line)

    def get_tlen_and_wordhasunk(self, word, align_tgt=False):
        if word not in self.clean_word_to_tlen_dict[align_tgt]:
            cur_tok = self.align_tgt_tok if align_tgt else self.tok
            toks = cur_tok(word, add_special_tokens=False)['input_ids']
            if cur_tok.unk_token_id in toks:
                self.word_has_unk[align_tgt][word] = True
            else:
                self.word_has_unk[align_tgt][word] = False

            self.clean_word_to_tlen_dict[align_tgt][word] = len(toks)
        return self.clean_word_to_tlen_dict[align_tgt][word], self.word_has_unk[align_tgt][word]

    def __init__(self, corpus, tl_dict, tok: BertTokenizer, align_tgt_tok: BertTokenizer, args, cache_dir=None, max_seq_len=512,
                 regularlize_on_orig_embeddings=None):
        self.args = args

        if args.use_basic_tok:
            self.split_fn = lambda x: tok.basic_tokenizer.tokenize(x)
        else:
            self.split_fn = lambda x: x.split()

        self.tok = tok
        self.align_tgt_tok = align_tgt_tok
        if cache_dir is None:
            cache_dir = get_cache_dir(args, regularlize_on_orig_embeddings)

        tmp_cache = cache_dir + '_tmp'
        while os.path.exists(tmp_cache):
            import time
            time.sleep(10)


        if os.path.exists(cache_dir):
            print(f"LOADED CACHE FROM {cache_dir}!!!!!!!!!!!!!!")
            all_orig_tl_lines, all_orig_tl_indexes = pickle.load(open(cache_dir, 'rb'))
        else:
            open(tmp_cache, 'w')
            self.word_to_tok_dict = {True:{}, False:{}}
            self.clean_word_to_tlen_dict = {True:{}, False:{}}
            self.word_has_unk = {True:{}, False:{}}

            orig_to_tl_dict = util.other_utils.load_dict(tl_dict)
            orig_lines = open(corpus, encoding='utf-8').readlines()
            max_bert_seq_len = max_seq_len - tok.num_special_tokens_to_add()  # number of special chars
            all_lines = list(enumerate(orig_lines))

            all_orig_tl_lines = []
            all_orig_tl_indexes = []
            for i, orig_line in tqdm(all_lines):
                orig_line = orig_line.strip()

                if args.use_basic_tok:
                    orig_words = []
                    tl_words = []
                    try:
                        for w in self.split_to_words(orig_line):
                            orig_words.append(w)
                            tl_words.append(orig_to_tl_dict[w][0])
                    except Exception as e:
                        print(e)
                        continue

                else:
                    try:
                        tl_ws = []
                        for w in self.split_fn(orig_line):
                            tl_ws.append(orig_to_tl_dict[w][0])
                        tl_line = ' '.join(tl_ws)
                    except Exception as e:
                        print(e)
                        continue

                    tl_line = tl_line.strip()
                    orig_words = self.split_to_words(orig_line)
                    tl_words = self.split_to_words(tl_line, align_tgt=True)

                    if not len(orig_words) == len(tl_words):
                        print(f"Something wrong in line {args.start_i + i}")
                        continue

                orig_subword_indicies, tl_subword_indicies = [], []
                cur_orig_end_idx_exclusive, cur_tl_end_idx_exclusive = 0, 0
                orig_cut_words, tl_cut_words = [], []
                for orig_w, tl_w in zip(orig_words, tl_words):
                    orig_len, orig_has_unk = self.get_tlen_and_wordhasunk(orig_w)
                    tl_len, tl_has_unk = self.get_tlen_and_wordhasunk(tl_w, align_tgt=True)

                    cur_orig_end_idx_exclusive += orig_len
                    cur_tl_end_idx_exclusive += tl_len
                    if cur_orig_end_idx_exclusive > max_bert_seq_len or cur_tl_end_idx_exclusive > max_bert_seq_len:
                        break

                    orig_cut_words.append(orig_w)
                    tl_cut_words.append(tl_w)

                    if args.ignore_unk_words and (orig_has_unk or tl_has_unk):
                        continue

                    orig_subword_indicies.append(list(range(cur_orig_end_idx_exclusive - orig_len, cur_orig_end_idx_exclusive)))
                    tl_subword_indicies.append(list(range(cur_tl_end_idx_exclusive - tl_len, cur_tl_end_idx_exclusive)))

                orig_line = ' '.join(orig_cut_words)
                tl_line = ' '.join(tl_cut_words)

                all_orig_tl_lines.append((orig_line, tl_line))
                all_orig_tl_indexes.append((orig_subword_indicies, tl_subword_indicies))

            pickle.dump((all_orig_tl_lines, all_orig_tl_indexes), open(cache_dir, 'wb'))
            os.remove(tmp_cache)

        self.all_orig_tl_lines, self.all_orig_tl_indexes = all_orig_tl_lines[args.start_i:args.end_i], all_orig_tl_indexes[args.start_i:args.end_i]

    def __len__(self):
        return len(self.all_orig_tl_lines)

    def __getitem__(self, idx):
        orig_line, tl_line = self.all_orig_tl_lines[idx]
        orig_indicies, tl_indicies = self.all_orig_tl_indexes[idx]

        return orig_line, tl_line, orig_indicies, tl_indicies


class CLDataset(OrigTLDataset):
    def __init__(self, src_lang, tgt_lang, src_corpus, tgt_corpus, word_alignment, tok: BertTokenizer, align_tgt_tok: BertTokenizer, args, cache_dir=None, max_seq_len=512,
                 regularlize_on_orig_embeddings=None):
        self.args = args
        self.tok = tok
        self.align_tgt_tok = align_tgt_tok

        if cache_dir is None:
            cache_dir = f"{src_lang}{tgt_lang}{os.path.basename(src_corpus)}{os.path.basename(tgt_corpus)}{word_alignment}{tok}{align_tgt_tok}_iguw{args.ignore_unk_words}_reg{regularlize_on_orig_embeddings}"
            if args.use_basic_tok:
                cache_dir += '_usebasictok'
            cache_dir = limit_len(cache_dir)

        tmp_cache = cache_dir + '_tmp'
        while os.path.exists(tmp_cache):
            import time
            time.sleep(10)

        if os.path.exists(cache_dir):
            print(f"LOADED CACHE FROM {cache_dir}!!!!!!!!!!!!!!")
            all_tgt_src_lines, all_tgt_src_indexes = pickle.load(open(cache_dir, 'rb'))
        else:
            open(tmp_cache, 'w')
            self.word_to_tok_dict = {True:{}, False:{}}
            self.clean_word_to_tlen_dict = {True:{}, False:{}}
            self.word_has_unk = {True:{}, False:{}}


            if args.use_basic_tok:
                src_tok = self.tok.basic_tokenizer
                tgt_tok = self.align_tgt_tok.basic_tokenizer
                print("USING BASIC TOK!!")
                print(src_tok)
                print(tgt_tok)
            else:
                src_tok = MosesTokenizer(lang=src_lang)
                tgt_tok = MosesTokenizer(lang=tgt_lang)

            src_lines = [s.strip() for s in open(src_corpus, encoding='utf-8').readlines()]
            tgt_lines = [s.strip() for s in open(tgt_corpus, encoding='utf-8').readlines()]
            aln_lines = [s.strip() for s in open(word_alignment, encoding='utf-8').readlines()]

            max_bert_seq_len = max_seq_len - tok.num_special_tokens_to_add()  # number of special chars

            all_tgt_src_lines = []
            all_tgt_src_indexes = []
            for i, (src_line, tgt_line, aln_line) in tqdm(list(enumerate(zip(src_lines, tgt_lines, aln_lines)))):
                # aln_line : 0-0 1-2 3-4 ... alignment map from src to tgt

                src_words = src_tok.tokenize(src_line)
                tgt_words = tgt_tok.tokenize(tgt_line)

                tgt_subword_indicies, src_subword_indicies = [], []
                tgt_cut_words, src_cut_words = [], []


                # check last supported w_i
                src_w_i_to_tis = {}
                cur_src_end_idx_exclusive = 0
                last_src_w  =  -1
                for w_i, src_w in enumerate(src_words):
                    src_len, _, _ = self.get_tlen_and_wordhasunk(src_w)
                    assert src_len > 0

                    cur_src_end_idx_exclusive += src_len
                    if cur_src_end_idx_exclusive > max_bert_seq_len:
                        break
                    src_w_i_to_tis[w_i] = list(range(cur_src_end_idx_exclusive - src_len, cur_src_end_idx_exclusive))
                    src_cut_words.append(src_w)
                    last_src_w  +=  1
                src_line = ' '.join(src_cut_words)
                # last_src_w = w_i - 1

                tgt_w_i_to_tis = {}
                cur_tgt_end_idx_exclusive = 0
                last_tgt_w = -1
                for w_i, tgt_w in enumerate(tgt_words):
                    tgt_len, _, _ = self.get_tlen_and_wordhasunk(tgt_w)
                    assert tgt_len > 0

                    cur_tgt_end_idx_exclusive += tgt_len
                    if cur_tgt_end_idx_exclusive > max_bert_seq_len:
                        break
                    tgt_w_i_to_tis[w_i] = list(range(cur_tgt_end_idx_exclusive - tgt_len, cur_tgt_end_idx_exclusive))
                    tgt_cut_words.append(tgt_w)
                    last_tgt_w  +=  1
                tgt_line = ' '.join(tgt_cut_words)

                # add subword_tis within that w_i
                for src_to_tgt_wi in aln_line.split():
                    src_w_i, tgt_w_i = [int(w_i) for w_i in src_to_tgt_wi.split('-')]
                    # we align low-re tgt to high-re src... align_tgt maps to src
                    # Thus orig becomes tgt, tl becomes  src
                    if src_w_i > last_src_w or tgt_w_i > last_tgt_w:
                        continue

                    src_w = src_words[src_w_i]
                    tgt_w = tgt_words[tgt_w_i]
                    src_len, src_has_unk = self.get_tlen_and_wordhasunk(src_w, align_tgt=True)
                    tgt_len, tgt_has_unk = self.get_tlen_and_wordhasunk(tgt_w)

                    if args.ignore_unk_words and (tgt_has_unk or src_has_unk):
                        continue

                    tgt_subword_indicies.append(tgt_w_i_to_tis[tgt_w_i])
                    src_subword_indicies.append(src_w_i_to_tis[src_w_i])

                all_tgt_src_lines.append((tgt_line, src_line))
                all_tgt_src_indexes.append((tgt_subword_indicies, src_subword_indicies))

            pickle.dump((all_tgt_src_lines, all_tgt_src_indexes), open(cache_dir, 'wb'))
            os.remove(tmp_cache)

        self.all_orig_tl_lines, self.all_orig_tl_indexes = all_tgt_src_lines[args.start_i:args.end_i], all_tgt_src_indexes[args.start_i:args.end_i]




@dataclass
class Collator:
    tokenizer: BertTokenizer
    align_tgt_tok: BertTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def tokenize_and_pad(self, lines, align_tgt=False):
        cur_tok = self.align_tgt_tok if align_tgt else self.tokenizer
        lines = cur_tok(lines, return_tensors='pt',
                               padding=self.padding,
                               max_length=self.max_length,
                               pad_to_multiple_of=self.pad_to_multiple_of,)
        # lines = self.tokenizer.pad(
        #     lines,
        #     return_tensors="pt",
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        # )
        return lines

    def __call__(
            self, examples
    ):
        orig_line, tl_line, orig_indicies, tl_indicies = zip(*examples)
        orig_line, tl_line = [self.tokenize_and_pad(line, align_tgt) for line, align_tgt in [(orig_line, False), (tl_line, True)]]

        return orig_line, tl_line, orig_indicies, tl_indicies