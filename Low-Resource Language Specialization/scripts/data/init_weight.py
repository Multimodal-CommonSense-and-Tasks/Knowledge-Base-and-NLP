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
    init_vocab: str = field(default="init_vocabs.txt")
    init_map: str = field(default="vocab_map_list.txt",
                           metadata={"help": "path to source vocabulary", }
                           )
    init_type: str = field(default='trunc_norm', metadata={"choices": ['trunc_norm', 'plm']})
    # skip_same_translit: bool = field(default=True)
    model_type: str = field(default='bert')

    src_model: str = field(default="bert/mbert/pytorch_model.bin",
                           metadata={"help": "source pre-trained file", }
                           )

    save_path: str = field(default="specializing-multilingual-data/data/ug_sim_trans/trunc_norm/pytorch_model.bin",
                           metadata={"help": "save the target model", }
                           )
    init_tid_list_save_path: str = field(default="specializing-multilingual-data/data/ug_sim_trans/trunc_norm/init_tid_list.txt",
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


def init_tgt(args: Args):
    """
    Initialize the parameters of the target model
    """
    print(f'| load English pre-trained model: {args.src_model}')
    # model = torch.load(args.src_model)
    # model = AutoModelForPreTraining.from_pretrained(args.src_model)
    model = torch.load(args.src_model)

    # get English word-embeddings and bias
    orig_embs = model[MAP['word_embeddings']]
    orig_bias = model[MAP['output_bias']]
    embs = orig_embs.clone()
    bias = orig_bias.clone()
    # embs = model.get_output_embeddings().weight.clone()
    # bias = model.get_output_embeddings().bias.clone()

    init_targets = []
    lines = open(args.init_vocab, encoding='utf-8').readlines()
    for line in lines:
        t_id, token = line.strip().split('\t')
        init_targets.append(int(t_id))

    tid_for_wordpiece_special = 108 # token id for '#'
    init_map = {}
    lines = open(args.init_map, encoding='utf-8').readlines()
    for line in lines:
        tids = line.strip().split('\t')
        orig_tid = int(tids[0])
        if len(tids) >= 2: # min for tid, translit_tid
            translit_tids = [int(i) for i in tids[1:] if int(i) != tid_for_wordpiece_special]
            print(orig_tid, translit_tids)
            if translit_tids:
                init_map[int(orig_tid)] = translit_tids
            else:
                init_targets.remove(orig_tid)
        else:
            init_targets.remove(orig_tid)

    for tid in init_targets:
        if tid in init_map:
            embs[tid] = torch.mean(orig_embs[init_map[tid]], dim=0)
            bias[tid] = torch.mean(orig_bias[init_map[tid]], dim=0)
        else:
            print(tid)
            raise NotImplementedError
                # f.write(f"{tid}\n")
                # if args.init_type == 'trunc_norm':
                #     e = embs[tid]
                #     init_weight = tf.random.truncated_normal(shape=list(e.shape), mean=0.0, stddev=args.init_std)
                #     embs[tid] = torch.tensor(init_weight.numpy())
                #     del init_weight
                #     bias[tid].zero_()
                # elif args.init_type == 'plm':
                #     pass
                # else:
                #     raise NotImplementedError

    model[MAP['word_embeddings']] = embs
    model[MAP['output_bias']] = bias
    model[MAP['output_weight']] = model[MAP['word_embeddings']]
    # model[MAP['word_embeddings']] = embs
    # model[MAP['output_bias']] = bias
    # model[MAP['output_weight']] = model[MAP['word_embeddings']]

    # save the model
    from util.io_utils import get_folderpath
    os.makedirs(get_folderpath(args.save_path), exist_ok=True)
    torch.save(model, args.save_path)
    # model.save_pretrained(args.save_path)


if __name__ == '__main__':
    init_tgt(args)
