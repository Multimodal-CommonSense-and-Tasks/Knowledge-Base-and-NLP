# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os

import torch
import argparse
import torch.nn as nn
from transformers import BertTokenizer, RobertaTokenizer, AutoModelForPreTraining, AutoTokenizer
from tokenizers import BertWordPieceTokenizer
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
from util.string_utils import unparse_args_to_cmd

try:
    tf.enable_eager_execution()
except AttributeError:
    pass
tf.config.experimental.set_visible_devices([], 'GPU')


@dataclass
class Args:
    mbert_tf_config: str = field(default="bert/mbert/bert_config.json")
    mbert_config: str = field(default="bert/scripts/convert_to_hf/mbert_config.json")
    mbert_tok_config: str = field(default="bert/scripts/convert_to_hf/tokenizer_config.json")
    mbert_vocab: str = field(default="bert/scripts/convert_to_hf/mbert_vocab.txt")
    new_vocab: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_shards_more_aug/5000-5-1000-1000.txt")
    init_type: str = field(default='trunc_norm', metadata={"choices": ['trunc_norm']})
    # skip_same_translit: bool = field(default=True)
    model_type: str = field(default='bert')

    src_model: str = field(default="bert/mbert/pytorch_model.bin",
                           metadata={"help": "source pre-trained file", }
                           )
    save_path: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_shards_more_aug/5000-5-1000-1000",
                           metadata={"help": "save the target model", }
                           )

    init_std: float = field(default=0.02,
                            metadata={"help": "std for rand init", }
                            )


parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)

print(unparse_args_to_cmd(parser.parse_args()))

if args.model_type == 'bert':
    CLS_TOKEN, CLS_INDEX = "[CLS]", 101
    SEP_TOKEN, SEP_INDEX = "[SEP]", 102
    UNK_TOKEN, UNK_INDEX = "[UNK]", 100
    PAD_TOKEN, PAD_INDEX = "[PAD]", 0
    MASK_TOKEN, MASK_INDEX = "[MASK]", 102

    MAP = {
        'word_embeddings': 'bert.embeddings.word_embeddings.weight',
        'output_weight': 'cls.predictions.decoder.weight',
        'output_bias': 'cls.predictions.bias'
    }
else:
    raise NotImplementedError


def guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=None):
    emb_dim = src_embs.size(1)
    num_tgt = tgt_tokenizer.get_vocab_size()

    # init with zero
    tgt_embs = src_embs.new_empty(num_tgt, emb_dim)
    tgt_bias = src_bias.new_zeros(num_tgt)
    nn.init.normal_(tgt_embs, mean=0, std=args.init_std)

    # copy over embeddings of special words
    for i in range(src_tokenizer.vocab_size):
        tgt_embs[i] = src_embs[i]
        tgt_bias[i] = src_bias[i]


def init_tgt(args: Args, mbert_vocab_size, new_vocab_size):
    """
    Initialize the parameters of the target model
    """
    print(f'| load English pre-trained model: {args.src_model}')
    model = torch.load(args.src_model)

    # get English word-embeddings and bias
    orig_embs = model[MAP['word_embeddings']]
    orig_bias = model[MAP['output_bias']]
    embs = orig_embs.clone()
    bias = orig_bias.clone()

    init_weight = tf.random.truncated_normal(shape=[new_vocab_size, embs.shape[1]], mean=0.0, stddev=args.init_std)
    new_embs = torch.tensor(init_weight.numpy())
    new_embs[:mbert_vocab_size, :] = embs
    del init_weight
    new_bias = torch.zeros([new_vocab_size])
    new_bias[:mbert_vocab_size] = bias

    model[MAP['word_embeddings']] = new_embs
    model[MAP['output_bias']] = new_bias
    model[MAP['output_weight']] = model[MAP['word_embeddings']]

    return model


from collections import OrderedDict


def make_vocab_and_write_init_tid_list(args: Args, init_tid_list):
    lines = open(args.mbert_vocab, encoding='utf-8').readlines()
    tok_to_i = OrderedDict()
    i_to_tok = OrderedDict()
    unused_indicies = []
    for i, tok in enumerate(lines):
        tok = tok.strip()
        if tok:
            assert tok not in tok_to_i
            tok_to_i[tok] = i
            i_to_tok[i] = tok
            if tok.startswith("[unused"):
                unused_indicies.append(i)

    lines = open(args.new_vocab, encoding='utf-8').readlines()

    with open(init_tid_list, 'w', encoding='utf-8') as f:
        for tok in lines:
            tok = tok.strip()
            if tok:
                if tok not in tok_to_i:
                    if unused_indicies:
                        next_unused_index = unused_indicies.pop(0)
                        unused_tok = i_to_tok[next_unused_index]
                        del i_to_tok[next_unused_index]
                        del tok_to_i[unused_tok]
                        i_to_tok[next_unused_index] = tok
                        tok_to_i[tok] = next_unused_index
                        f.write(f"{next_unused_index}\n")
                    else:
                        next_index = len(tok_to_i)
                        tok_to_i[tok] = next_index
                        i_to_tok[next_index] = tok
                        f.write(f"{next_index}\n")

    return OrderedDict(sorted(i_to_tok.items()))


from transformers.models.bert.modeling_bert import BertConfig
from pathlib import Path
import shutil
import json

if __name__ == '__main__':
    os.makedirs(args.save_path, exist_ok=True)

    save_path = Path(args.save_path)
    init_tid_list = save_path / "init_tid_list.txt"
    i_to_tok = make_vocab_and_write_init_tid_list(args, init_tid_list)
    new_vocab_size = len(i_to_tok)
    with open(save_path / "vocab.txt", "w", encoding='utf-8') as f:
        for tok in i_to_tok.values():
            f.write(f"{tok}\n")

    config = BertConfig.from_pretrained(args.mbert_config)
    mbert_vocab_size = config.vocab_size
    config.vocab_size = new_vocab_size
    config.save_pretrained(save_path)

    tf_config = json.load(open(args.mbert_tf_config))
    tf_config['vocab_size'] = new_vocab_size
    json.dump(tf_config, open(save_path / "bert_config.json", 'w'), indent=4)

    shutil.copy(args.mbert_tok_config, save_path / "tokenizer_config.json")

    model = init_tgt(args, mbert_vocab_size, new_vocab_size)
    torch.save(model, save_path / "pytorch_model.bin")
