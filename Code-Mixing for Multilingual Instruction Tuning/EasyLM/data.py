import copy
import dataclasses
import pprint
import time
from functools import partial
import json
from multiprocessing import Pool
import unicodedata

import h5py
import mlxu
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

import datasets, random

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset


def _is_skip_target(char):
    return _is_whitespace(char) or _is_punctuation(char)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def load_dict(dict_path, load_reverse=False):
    lines = open(dict_path, 'r', encoding='utf-8').readlines()
    src2tgt = {}
    for line in lines:
        line = line.strip()
        try:
            src, tgt = line.split("\t")
        except:
            src, tgt = line.split(" ")

        if load_reverse:
            src, tgt = tgt, src

        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)
    return src2tgt


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.codemix_processor = CodeMixProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        config.cljson_dataset = CLJsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        if config.type == 'cljson':
            text_processor = CodeMixProcessor(config.codemix_processor, tokenizer)
            return CLJsonDataset(config.cljson_dataset, tokenizer, text_processor, **kwargs)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
lang_to_prompt_dict ={
    'en': PROMPT_DICT,
    "ht": {
        "prompt_input": "Anba a se yon enstriksyon ki dekri yon travay, ansanm ak yon opinyon ki bay plis kont\u00e8ks. Ekri yon repons ki byen ranpli demann lan.\n\n### Enstriksyon:\n{instruction}\n\n### Antre:\n{input}\n\n### Repons:",
        "prompt_no_input": "Anba a se yon enstriksyon ki dekri yon travay. Ekri yon repons ki byen ranpli demann lan.\n\n### Enstriksyon:\n{instruction}\n\n### Repons:"
    },
    "qu": {
        "prompt_input": "Uraypiqa huk kamachiymi kachkan, chaymi huk ruwayta willan, huk yaykusqawan tupachisqa, chaymi aswan contextota qun. Ma\u00f1akusqata allinta hunt\u2019aq kutichiyta qillqay.\n\n### Yachachiy:\n{instruction}.\n\n### Yaykuchiy:\n{input}.\n\n### Kutichiy:",
        "prompt_no_input": "Uraypiqa huk kamachikuymi kachkan, chaypim huk llamkaymanta willakun. Ma\u00f1akusqata allinta hunt\u2019aq kutichiyta qillqay.\n\n### Yachachiy:\n{instruction}.\n\n### Kutichiy:"
    },
    "te": {
        "prompt_input": "\u0c26\u0c3f\u0c17\u0c41\u0c35\u0c28 \u0c12\u0c15 \u0c2a\u0c28\u0c3f\u0c28\u0c3f \u0c35\u0c3f\u0c35\u0c30\u0c3f\u0c02\u0c1a\u0c47 \u0c38\u0c42\u0c1a\u0c28, \u0c24\u0c26\u0c41\u0c2a\u0c30\u0c3f \u0c38\u0c02\u0c26\u0c30\u0c4d\u0c2d\u0c3e\u0c28\u0c4d\u0c28\u0c3f \u0c05\u0c02\u0c26\u0c3f\u0c02\u0c1a\u0c47 \u0c07\u0c28\u0c4d\u200c\u0c2a\u0c41\u0c1f\u0c4d\u200c\u0c24\u0c4b \u0c1c\u0c24 \u0c1a\u0c47\u0c2f\u0c2c\u0c21\u0c3f\u0c02\u0c26\u0c3f. \u0c05\u0c2d\u0c4d\u0c2f\u0c30\u0c4d\u0c25\u0c28\u0c28\u0c41 \u0c38\u0c2e\u0c41\u0c1a\u0c3f\u0c24\u0c02\u0c17\u0c3e \u0c2a\u0c42\u0c30\u0c4d\u0c24\u0c3f \u0c1a\u0c47\u0c38\u0c47 \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28\u0c28\u0c41 \u0c35\u0c4d\u0c30\u0c3e\u0c2f\u0c02\u0c21\u0c3f.\n\n### \u0c38\u0c42\u0c1a\u0c28:\n{instruction}\n\n### \u0c07\u0c28\u0c4d\u200c\u0c2a\u0c41\u0c1f\u0c4d:\n{input}\n\n### \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28:",
        "prompt_no_input": "\u0c15\u0c4d\u0c30\u0c3f\u0c02\u0c26 \u0c12\u0c15 \u0c2a\u0c28\u0c3f\u0c28\u0c3f \u0c35\u0c3f\u0c35\u0c30\u0c3f\u0c02\u0c1a\u0c47 \u0c38\u0c42\u0c1a\u0c28 \u0c09\u0c02\u0c26\u0c3f. \u0c05\u0c2d\u0c4d\u0c2f\u0c30\u0c4d\u0c25\u0c28\u0c28\u0c41 \u0c38\u0c2e\u0c41\u0c1a\u0c3f\u0c24\u0c02\u0c17\u0c3e \u0c2a\u0c42\u0c30\u0c4d\u0c24\u0c3f \u0c1a\u0c47\u0c38\u0c47 \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28\u0c28\u0c41 \u0c35\u0c4d\u0c30\u0c3e\u0c2f\u0c02\u0c21\u0c3f.\n\n### \u0c38\u0c42\u0c1a\u0c28:\n{instruction}\n\n### \u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c38\u0c4d\u0c2a\u0c02\u0c26\u0c28:"
    },
    "sw": {
        "prompt_input": "Ifuatayo ni maagizo ambayo yanaelezea kazi, yakioanishwa na ingizo ambalo hutoa muktadha zaidi. Andika jibu ambalo linakamilisha ombi ipasavyo.\n\n### Maagizo:\n{instruction}\n\n### Ingizo:\n{input}\n\n### Jibu:",
        "prompt_no_input": "Chini ni maagizo ambayo yanaelezea kazi. Andika jibu ambalo linakamilisha ombi ipasavyo.\n\n### Maagizo:\n{instruction}\n\n### Jibu:"
    },
    "ta": {
        "prompt_input": "\u0b92\u0bb0\u0bc1 \u0baa\u0ba3\u0bbf\u0baf\u0bc8 \u0bb5\u0bbf\u0bb5\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc1\u0bae\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd \u0b95\u0bc0\u0bb4\u0bc7 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1, \u0bae\u0bc7\u0bb2\u0bc1\u0bae\u0bcd \u0b9a\u0bc2\u0bb4\u0bb2\u0bc8 \u0bb5\u0bb4\u0b99\u0bcd\u0b95\u0bc1\u0bae\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0bc0\u0b9f\u0bcd\u0b9f\u0bc1\u0b9f\u0ba9\u0bcd \u0b87\u0ba3\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b95\u0bcb\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc8\u0baf\u0bc8 \u0b9a\u0bb0\u0bbf\u0baf\u0bbe\u0ba9 \u0bae\u0bc1\u0bb1\u0bc8\u0baf\u0bbf\u0bb2\u0bcd \u0ba8\u0bbf\u0bb1\u0bc8\u0bb5\u0bc1 \u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0bc1\u0bae\u0bcd \u0baa\u0ba4\u0bbf\u0bb2\u0bc8 \u0b8e\u0bb4\u0bc1\u0ba4\u0bb5\u0bc1\u0bae\u0bcd.\n\n### \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd:\n{instruction}\n\n### \u0b89\u0bb3\u0bcd\u0bb3\u0bc0\u0b9f\u0bc1:\n{input}\n\n### \u0baa\u0ba4\u0bbf\u0bb2\u0bcd:",
        "prompt_no_input": "\u0b92\u0bb0\u0bc1 \u0baa\u0ba3\u0bbf\u0baf\u0bc8 \u0bb5\u0bbf\u0bb5\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc1\u0bae\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd \u0b95\u0bc0\u0bb4\u0bc7 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0b95\u0bcb\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bc8\u0baf\u0bc8 \u0b9a\u0bb0\u0bbf\u0baf\u0bbe\u0ba9 \u0bae\u0bc1\u0bb1\u0bc8\u0baf\u0bbf\u0bb2\u0bcd \u0ba8\u0bbf\u0bb1\u0bc8\u0bb5\u0bc1 \u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0bc1\u0bae\u0bcd \u0baa\u0ba4\u0bbf\u0bb2\u0bc8 \u0b8e\u0bb4\u0bc1\u0ba4\u0bb5\u0bc1\u0bae\u0bcd.\n\n### \u0b85\u0bb1\u0bbf\u0bb5\u0bc1\u0bb1\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bcd:\n{instruction}\n\n### \u0baa\u0ba4\u0bbf\u0bb2\u0bcd:"
    },
    "ur": {
        "prompt_input": "\u0630\u06cc\u0644 \u0645\u06cc\u06ba \u0627\u06cc\u06a9 \u06c1\u062f\u0627\u06cc\u062a \u062f\u06cc \u06af\u0626\u06cc \u06c1\u06d2 \u062c\u0648 \u0627\u06cc\u06a9 \u06a9\u0627\u0645 \u06a9\u06cc \u0648\u0636\u0627\u062d\u062a \u06a9\u0631\u062a\u06cc \u06c1\u06d2\u060c \u0627\u06cc\u06a9 \u0627\u0646 \u067e\u0679 \u06a9\u06d2 \u0633\u0627\u062a\u06be \u062c\u0648\u0691\u0627 \u062c\u0648 \u0645\u0632\u06cc\u062f \u0633\u06cc\u0627\u0642 \u0648 \u0633\u0628\u0627\u0642 \u0641\u0631\u0627\u06c1\u0645 \u06a9\u0631\u062a\u0627 \u06c1\u06d2\u06d4 \u0627\u06cc\u06a9 \u062c\u0648\u0627\u0628 \u0644\u06a9\u06be\u06cc\u06ba \u062c\u0648 \u0645\u0646\u0627\u0633\u0628 \u0637\u0631\u06cc\u0642\u06d2 \u0633\u06d2 \u062f\u0631\u062e\u0648\u0627\u0633\u062a \u06a9\u0648 \u0645\u06a9\u0645\u0644 \u06a9\u0631\u06d2\u06d4\n\n### \u06c1\u062f\u0627\u06cc\u0627\u062a:\n{instruction}\n\n### \u0627\u0646 \u067e\u0679:\n{input}\n\n### \u062c\u0648\u0627\u0628:",
        "prompt_no_input": "\u0630\u06cc\u0644 \u0645\u06cc\u06ba \u0627\u06cc\u06a9 \u06c1\u062f\u0627\u06cc\u062a \u06c1\u06d2 \u062c\u0648 \u0627\u06cc\u06a9 \u06a9\u0627\u0645 \u06a9\u06cc \u0648\u0636\u0627\u062d\u062a \u06a9\u0631\u062a\u06cc \u06c1\u06d2\u06d4 \u0627\u06cc\u06a9 \u062c\u0648\u0627\u0628 \u0644\u06a9\u06be\u06cc\u06ba \u062c\u0648 \u0645\u0646\u0627\u0633\u0628 \u0637\u0631\u06cc\u0642\u06d2 \u0633\u06d2 \u062f\u0631\u062e\u0648\u0627\u0633\u062a \u06a9\u0648 \u0645\u06a9\u0645\u0644 \u06a9\u0631\u06d2\u06d4\n\n### \u06c1\u062f\u0627\u06cc\u0627\u062a:\n{instruction}\n\n### \u062c\u0648\u0627\u0628:"
    },
    "my": {
        "prompt_input": "\u1021\u1031\u102c\u1000\u103a\u1010\u103d\u1004\u103a \u1014\u1031\u102c\u1000\u103a\u1011\u1015\u103a\u1021\u1000\u103c\u1031\u102c\u1004\u103a\u1038\u1021\u101b\u102c\u1010\u1005\u103a\u1001\u102f\u1000\u102d\u102f \u1015\u1036\u1037\u1015\u102d\u102f\u1038\u1015\u1031\u1038\u101e\u100a\u1037\u103a \u1011\u100a\u1037\u103a\u101e\u103d\u1004\u103a\u1038\u1019\u103e\u102f\u1010\u1005\u103a\u1001\u102f\u1014\u103e\u1004\u1037\u103a \u1010\u103d\u1032\u101c\u102f\u1015\u103a\u1011\u102c\u1038\u101e\u100a\u1037\u103a \u1021\u101c\u102f\u1015\u103a\u1010\u1005\u103a\u1001\u102f\u1000\u102d\u102f \u1016\u1031\u102c\u103a\u1015\u103c\u101e\u100a\u1037\u103a \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a\u1010\u1005\u103a\u1001\u102f\u1016\u103c\u1005\u103a\u101e\u100a\u103a\u104b \u1010\u1031\u102c\u1004\u103a\u1038\u1006\u102d\u102f\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101e\u1004\u1037\u103a\u101c\u103b\u1031\u102c\u103a\u1005\u103d\u102c \u1015\u103c\u102e\u1038\u1019\u103c\u1031\u102c\u1000\u103a\u1005\u1031\u101e\u1031\u102c \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101b\u1031\u1038\u1015\u102b\u104b\n\n### \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a-\n{instruction}\n\n### \u1011\u100a\u1037\u103a\u101e\u103d\u1004\u103a\u1038\u1019\u103e\u102f-\n{input}\n\n### \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1019\u103e\u102f-",
        "prompt_no_input": "\u1021\u1031\u102c\u1000\u103a\u1010\u103d\u1004\u103a \u1021\u101c\u102f\u1015\u103a\u1010\u1005\u103a\u1001\u102f\u1000\u102d\u102f \u1016\u1031\u102c\u103a\u1015\u103c\u101e\u100a\u1037\u103a \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a\u1010\u1005\u103a\u1001\u102f\u1016\u103c\u1005\u103a\u101e\u100a\u103a\u104b \u1010\u1031\u102c\u1004\u103a\u1038\u1006\u102d\u102f\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101e\u1004\u1037\u103a\u101c\u103b\u1031\u102c\u103a\u1005\u103d\u102c \u1015\u103c\u102e\u1038\u1019\u103c\u1031\u102c\u1000\u103a\u1005\u1031\u101e\u1031\u102c \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1001\u103b\u1000\u103a\u1000\u102d\u102f \u101b\u1031\u1038\u1015\u102b\u104b\n\n### \u100a\u103d\u103e\u1014\u103a\u1000\u103c\u102c\u1038\u1001\u103b\u1000\u103a-\n{instruction}\n\n### \u1010\u102f\u1036\u1037\u1015\u103c\u1014\u103a\u1019\u103e\u102f-"
    },
    "hi": {
        "prompt_input": "\u0928\u0940\u091a\u0947 \u090f\u0915 \u0928\u093f\u0930\u094d\u0926\u0947\u0936 \u0939\u0948 \u091c\u094b \u0915\u093f\u0938\u0940 \u0915\u093e\u0930\u094d\u092f \u0915\u093e \u0935\u0930\u094d\u0923\u0928 \u0915\u0930\u0924\u093e \u0939\u0948, \u091c\u093f\u0938\u0947 \u090f\u0915 \u0907\u0928\u092a\u0941\u091f \u0915\u0947 \u0938\u093e\u0925 \u091c\u094b\u0921\u093c\u093e \u0917\u092f\u093e \u0939\u0948 \u091c\u094b \u0906\u0917\u0947 \u0915\u093e \u0938\u0902\u0926\u0930\u094d\u092d \u092a\u094d\u0930\u0926\u093e\u0928 \u0915\u0930\u0924\u093e \u0939\u0948\u0964 \u0910\u0938\u093e \u0909\u0924\u094d\u0924\u0930 \u0932\u093f\u0916\u0947\u0902 \u091c\u094b \u0905\u0928\u0941\u0930\u094b\u0927 \u0915\u094b \u0909\u091a\u093f\u0924 \u0930\u0942\u092a \u0938\u0947 \u092a\u0942\u0930\u093e \u0915\u0930\u0924\u093e \u0939\u094b\u0964\n\n### \u0928\u093f\u0930\u094d\u0926\u0947\u0936:\n{instruction}\n\n### \u0907\u0928\u092a\u0941\u091f:\n{input}\n\n### \u092a\u094d\u0930\u0924\u093f\u0915\u094d\u0930\u093f\u092f\u093e:",
        "prompt_no_input": "\u0928\u0940\u091a\u0947 \u090f\u0915 \u0928\u093f\u0930\u094d\u0926\u0947\u0936 \u0939\u0948 \u091c\u094b \u0915\u093f\u0938\u0940 \u0915\u093e\u0930\u094d\u092f \u0915\u093e \u0935\u0930\u094d\u0923\u0928 \u0915\u0930\u0924\u093e \u0939\u0948\u0964 \u0910\u0938\u093e \u0909\u0924\u094d\u0924\u0930 \u0932\u093f\u0916\u0947\u0902 \u091c\u094b \u0905\u0928\u0941\u0930\u094b\u0927 \u0915\u094b \u0909\u091a\u093f\u0924 \u0930\u0942\u092a \u0938\u0947 \u092a\u0942\u0930\u093e \u0915\u0930\u0924\u093e \u0939\u094b\u0964\n\n### \u0928\u093f\u0930\u094d\u0926\u0947\u0936:\n{instruction}\n\n### \u092a\u094d\u0930\u0924\u093f\u0915\u094d\u0930\u093f\u092f\u093e:"
    },
    "th": {
        "prompt_input": "\u0e14\u0e49\u0e32\u0e19\u0e25\u0e48\u0e32\u0e07\u0e19\u0e35\u0e49\u0e04\u0e37\u0e2d\u0e04\u0e33\u0e2a\u0e31\u0e48\u0e07\u0e17\u0e35\u0e48\u0e2d\u0e18\u0e34\u0e1a\u0e32\u0e22\u0e07\u0e32\u0e19 \u0e04\u0e27\u0e1a\u0e04\u0e39\u0e48\u0e44\u0e1b\u0e01\u0e31\u0e1a\u0e2d\u0e34\u0e19\u0e1e\u0e38\u0e15\u0e17\u0e35\u0e48\u0e43\u0e2b\u0e49\u0e1a\u0e23\u0e34\u0e1a\u0e17\u0e40\u0e1e\u0e34\u0e48\u0e21\u0e40\u0e15\u0e34\u0e21 \u0e40\u0e02\u0e35\u0e22\u0e19\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e48\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07\u0e04\u0e33\u0e02\u0e2d\u0e44\u0e14\u0e49\u0e2d\u0e22\u0e48\u0e32\u0e07\u0e40\u0e2b\u0e21\u0e32\u0e30\u0e2a\u0e21\n\n### \u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33:\n{instruction}\n\n### \u0e1b\u0e49\u0e2d\u0e19\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25:\n{input}\n\n### \u0e01\u0e32\u0e23\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07:",
        "prompt_no_input": "\u0e14\u0e49\u0e32\u0e19\u0e25\u0e48\u0e32\u0e07\u0e19\u0e35\u0e49\u0e40\u0e1b\u0e47\u0e19\u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33\u0e17\u0e35\u0e48\u0e2d\u0e18\u0e34\u0e1a\u0e32\u0e22\u0e07\u0e32\u0e19 \u0e40\u0e02\u0e35\u0e22\u0e19\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e48\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07\u0e04\u0e33\u0e02\u0e2d\u0e44\u0e14\u0e49\u0e2d\u0e22\u0e48\u0e32\u0e07\u0e40\u0e2b\u0e21\u0e32\u0e30\u0e2a\u0e21\n\n### \u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33:\n{instruction}\n\n### \u0e01\u0e32\u0e23\u0e15\u0e2d\u0e1a\u0e2a\u0e19\u0e2d\u0e07:"
    },
    "bn": {
        "prompt_input": "\u09a8\u09c0\u099a\u09c7 \u098f\u0995\u099f\u09bf \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6 \u09b0\u09af\u09bc\u09c7\u099b\u09c7 \u09af\u09be \u098f\u0995\u099f\u09bf \u099f\u09be\u09b8\u09cd\u0995 \u09ac\u09b0\u09cd\u09a3\u09a8\u09be \u0995\u09b0\u09c7, \u098f\u0995\u099f\u09bf \u0987\u09a8\u09aa\u09c1\u099f\u09c7\u09b0 \u09b8\u09be\u09a5\u09c7 \u09af\u09c1\u0995\u09cd\u09a4 \u09af\u09be \u0986\u09b0\u0993 \u09aa\u09cd\u09b0\u09b8\u0999\u09cd\u0997 \u09b8\u09b0\u09ac\u09b0\u09be\u09b9 \u0995\u09b0\u09c7\u0964 \u098f\u0995\u099f\u09bf \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be \u09b2\u09bf\u0996\u09c1\u09a8 \u09af\u09be \u09af\u09a5\u09be\u09af\u09a5\u09ad\u09be\u09ac\u09c7 \u0985\u09a8\u09c1\u09b0\u09cb\u09a7\u099f\u09bf \u09b8\u09ae\u09cd\u09aa\u09c2\u09b0\u09cd\u09a3 \u0995\u09b0\u09c7\u0964\n\n### \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6:\n{instruction}\n\n### \u0987\u09a8\u09aa\u09c1\u099f:\n{input}\n\n### \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be:",
        "prompt_no_input": "\u09a8\u09c0\u099a\u09c7 \u098f\u0995\u099f\u09bf \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6 \u09af\u09be \u098f\u0995\u099f\u09bf \u099f\u09be\u09b8\u09cd\u0995 \u09ac\u09b0\u09cd\u09a3\u09a8\u09be \u0995\u09b0\u09c7\u0964 \u098f\u0995\u099f\u09bf \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be \u09b2\u09bf\u0996\u09c1\u09a8 \u09af\u09be \u09af\u09a5\u09be\u09af\u09a5\u09ad\u09be\u09ac\u09c7 \u0985\u09a8\u09c1\u09b0\u09cb\u09a7\u099f\u09bf \u09b8\u09ae\u09cd\u09aa\u09c2\u09b0\u09cd\u09a3 \u0995\u09b0\u09c7\u0964\n\n### \u09a8\u09bf\u09b0\u09cd\u09a6\u09c7\u09b6:\n{instruction}\n\n### \u09aa\u09cd\u09b0\u09a4\u09bf\u0995\u09cd\u09b0\u09bf\u09af\u09bc\u09be:"
    },
    "el": {
        "prompt_input": "\u03a0\u03b1\u03c1\u03b1\u03ba\u03ac\u03c4\u03c9 \u03b5\u03af\u03bd\u03b1\u03b9 \u03bc\u03b9\u03b1 \u03bf\u03b4\u03b7\u03b3\u03af\u03b1 \u03c0\u03bf\u03c5 \u03c0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c6\u03b5\u03b9 \u03bc\u03b9\u03b1 \u03b5\u03c1\u03b3\u03b1\u03c3\u03af\u03b1, \u03c3\u03b5 \u03c3\u03c5\u03bd\u03b4\u03c5\u03b1\u03c3\u03bc\u03cc \u03bc\u03b5 \u03bc\u03b9\u03b1 \u03b5\u03af\u03c3\u03bf\u03b4\u03bf \u03c0\u03bf\u03c5 \u03c0\u03b1\u03c1\u03ad\u03c7\u03b5\u03b9 \u03c0\u03b5\u03c1\u03b1\u03b9\u03c4\u03ad\u03c1\u03c9 \u03c0\u03bb\u03b1\u03af\u03c3\u03b9\u03bf. \u0393\u03c1\u03ac\u03c8\u03c4\u03b5 \u03bc\u03b9\u03b1 \u03b1\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7 \u03c0\u03bf\u03c5 \u03bf\u03bb\u03bf\u03ba\u03bb\u03b7\u03c1\u03ce\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03ac\u03bb\u03bb\u03b7\u03bb\u03b1 \u03c4\u03bf \u03b1\u03af\u03c4\u03b7\u03bc\u03b1.\n\n### \u039f\u03b4\u03b7\u03b3\u03af\u03b1:\n{instruction}\n\n### \u0395\u03b9\u03c3\u03b1\u03b3\u03c9\u03b3\u03ae:\n{input}\n\n### \u0391\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7:",
        "prompt_no_input": "\u03a0\u03b1\u03c1\u03b1\u03ba\u03ac\u03c4\u03c9 \u03b5\u03af\u03bd\u03b1\u03b9 \u03bc\u03b9\u03b1 \u03bf\u03b4\u03b7\u03b3\u03af\u03b1 \u03c0\u03bf\u03c5 \u03c0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c6\u03b5\u03b9 \u03bc\u03b9\u03b1 \u03b5\u03c1\u03b3\u03b1\u03c3\u03af\u03b1. \u0393\u03c1\u03ac\u03c8\u03c4\u03b5 \u03bc\u03b9\u03b1 \u03b1\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7 \u03c0\u03bf\u03c5 \u03bf\u03bb\u03bf\u03ba\u03bb\u03b7\u03c1\u03ce\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03ac\u03bb\u03bb\u03b7\u03bb\u03b1 \u03c4\u03bf \u03b1\u03af\u03c4\u03b7\u03bc\u03b1.\n\n### \u039f\u03b4\u03b7\u03b3\u03af\u03b1:\n{instruction}\n\n### \u0391\u03c0\u03ac\u03bd\u03c4\u03b7\u03c3\u03b7:"
    }
}


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            strings,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        source,
        target,
        tokenizer,
):
    """Preprocess the data by tokenizing."""
    example = source + target
    examples_tokenized, sources_tokenized = [_tokenize_fn(string, tokenizer) for string in (example, source)]
    input_ids = examples_tokenized["input_ids"][0]
    loss_mask_tokens = [0] * sources_tokenized["input_ids_lens"][0] + [1] * (len(input_ids) - sources_tokenized["input_ids_lens"][0])
    return input_ids, loss_mask_tokens


class CodeMixProcessor(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.codemix_dict_path = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        self.codemix_dict = load_dict(self.config.codemix_dict_path)
        self.block_codemix_in_template = True
        self.tokenizer = tokenizer
        self.codemix_ratio = 1.0

    def codemix_sentence(self, sent):
        words = sent.split(' ')
        codemixed_words = [self.codemix_word_considering_punct(w) for w in words]
        return ' '.join(codemixed_words)

    def codemix_word_considering_punct(self, w):
        try:
            start_i = 0
            while _is_skip_target(w[start_i]):
                start_i += 1

            end_i = len(w) - 1
            while _is_skip_target(w[end_i]):
                end_i -= 1

            codemix_target_word = w[start_i:end_i + 1]
            codemixed_word = self.codemix_word(codemix_target_word)
            return w[:start_i] + codemixed_word + w[end_i + 1:]
        except:  # all punct or blank
            return w

    def codemix_word(self, skip_removed_w):
        if skip_removed_w in self.codemix_dict and random.random() < self.codemix_ratio:
            return random.choice(self.codemix_dict[skip_removed_w])
        else:
            return skip_removed_w

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        source = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        # target = f"{example['output']}{self.tokenizer.eos_token}"
        target = f"{example['output']}"

        orig_tokens = _tokenize_fn(source + target, self.tokenizer)['input_ids'][0]

        put_example = copy.deepcopy(example)
        if self.config.codemix_dict_path and self.block_codemix_in_template:
            put_example["instruction"] = self.codemix_sentence(put_example["instruction"])
            if put_example.get("input", ""):
                put_example["input"] = self.codemix_sentence(put_example["input"])

        source = prompt_input.format_map(put_example) if put_example.get("input", "") != "" else prompt_no_input.format_map(put_example)
        # target = f"{put_example['output']}{self.tokenizer.eos_token}"
        if self.config.codemix_dict_path and not self.block_codemix_in_template:
            source = self.codemix_sentence(source)
        target = f"{put_example['output']}"
        if self.config.codemix_dict_path:
            target = self.codemix_sentence(target)

        codemixed_tokens = _tokenize_fn(source + target, self.tokenizer)['input_ids'][0]

        return orig_tokens.tolist(), codemixed_tokens.tolist(), *aux


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.alpaca = False
        config.codemix_dict_path = ''
        config.codemix_ratio = 0.0
        config.block_codemix_in_template = False
        config.alpaca_lang = 'en'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        if self.config.codemix_dict_path:
            self.codemix_dict = load_dict(self.config.codemix_dict_path)
            self.codemix_ratio = self.config.codemix_ratio
            self.block_codemix_in_template = self.config.block_codemix_in_template
        self.tokenizer = tokenizer
        if not self.config.alpaca:
            assert self.config.fields != '' or self.config.fields_from_example != '', (
                'Either fields or fields_from_example must be specified.'
            )
        else:
            assert all([tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, tokenizer.unk_token])

    def default(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                if self.config.codemix_dict_path:
                    text = self.codemix_sentence(text)

                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux

    def codemix_sentence(self, sent):
        words = sent.split(' ')
        codemixed_words = [self.codemix_word_considering_punct(w) for w in words]
        return ' '.join(codemixed_words)

    def codemix_word_considering_punct(self, w):
        try:
            start_i = 0
            while _is_skip_target(w[start_i]):
                start_i += 1

            end_i = len(w) - 1
            while _is_skip_target(w[end_i]):
                end_i -= 1

            codemix_target_word = w[start_i:end_i + 1]
            codemixed_word = self.codemix_word(codemix_target_word)
            return w[:start_i] + codemixed_word + w[end_i + 1:]
        except:  # all punct or blank
            return w

    def codemix_word(self, skip_removed_w):
        if skip_removed_w in self.codemix_dict and random.random() < self.codemix_ratio:
            return random.choice(self.codemix_dict[skip_removed_w])
        else:
            return skip_removed_w

    def alpaca(self, example, has_aux):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()

        put_example = copy.deepcopy(example)
        if self.config.codemix_dict_path and self.block_codemix_in_template:
            put_example["instruction"] = self.codemix_sentence(put_example["instruction"])
            if put_example.get("input", ""):
                put_example["input"] = self.codemix_sentence(put_example["input"])

        PROMPT_DICT = lang_to_prompt_dict[self.config.alpaca_lang]

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        source = prompt_input.format_map(put_example) if put_example.get("input", "") != "" else prompt_no_input.format_map(put_example)
        # target = f"{put_example['output']}{self.tokenizer.eos_token}"
        if self.config.codemix_dict_path and not self.block_codemix_in_template:
            source = self.codemix_sentence(source)
        target = f"{put_example['output']}"
        if self.config.codemix_dict_path:
            target = self.codemix_sentence(target)

        input_ids, loss_masks = preprocess(source, target, self.tokenizer)

        return input_ids.tolist(), loss_masks, *aux

    def __call__(self, example, has_aux=False):
        if self.config.alpaca:
            return self.alpaca(example, has_aux)
        else:
            return self.default(example, has_aux)


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:  # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                        (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class CLJsonDataset(JsonDataset):
    def __iter__(self):
        total_tokens = []
        total_codemixed_tokens = []
        cur_tokens = []
        cur_codemixed_tokens = []

        for tokens, codemixed_tokens, loc, index in self.parallel_example_iterator():
            if len(cur_tokens) + len(tokens) < self.config.seq_length and len(cur_codemixed_tokens) + len(codemixed_tokens) < self.config.seq_length:
                cur_tokens.extend(tokens)
                cur_codemixed_tokens.extend(codemixed_tokens)
            else:
                cur_tokens.extend([self.tokenizer.pad_token_id for _ in range(self.config.seq_length - len(cur_tokens))])
                cur_codemixed_tokens.extend([self.tokenizer.pad_token_id for _ in range(self.config.seq_length - len(cur_codemixed_tokens))])

                total_tokens.append(cur_tokens)
                total_codemixed_tokens.append(cur_codemixed_tokens)
                cur_tokens = tokens[:self.config.seq_length]
                cur_codemixed_tokens = codemixed_tokens[:self.config.seq_length]

                if len(cur_tokens) >= self.config.batch_size:
                    metrics = {
                        'dataset_file_loc': loc,
                        'dataset_example_index': index,
                    }
                    batch = {
                        'orig_tokens': np.array(total_tokens, dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'codemixed_tokens': np.array(total_codemixed_tokens, dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    yield batch, metrics
                    total_tokens = []
                    total_codemixed_tokens = []
