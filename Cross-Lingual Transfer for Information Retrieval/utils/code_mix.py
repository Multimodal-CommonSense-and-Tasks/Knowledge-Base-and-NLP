import random
import os
from functools import partial


def get_dict(codemix_set):
    dict_path = os.path.join(os.path.dirname(__file__), "dict", f"{codemix_set}.txt")
    lines = open(dict_path, "r", encoding="utf-8").readlines()
    src2tgt = {}
    for line in lines:
        line = line.strip()
        try:
            src, tgt = line.split("\t")
        except:
            src, tgt = line.split(" ")

        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)
    return src2tgt


def get_codemixed_ids(
    tokenizer, basic_tokenizer, src2tgt, input, max_length, codemix_ratio=0
):
    if codemix_ratio == 0:
        if isinstance(input, str):
            tokenize = partial(
                tokenizer,
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=False,
                truncation=True,
            )
            return tokenize(input, max_length=max_length)["input_ids"]
        else:
            return input
    
    passage = tokenizer.decode(input, True)
    words = basic_tokenizer.tokenize(passage)
    result_ids = []
    for w in words:
        if (random.random() < codemix_ratio) and w in src2tgt:
            trans_w = src2tgt[w][random.randint(0, len(src2tgt[w]) - 1)]
            result_ids.extend(
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trans_w))
            )
        else:
            result_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)))
    return result_ids[:max_length]
