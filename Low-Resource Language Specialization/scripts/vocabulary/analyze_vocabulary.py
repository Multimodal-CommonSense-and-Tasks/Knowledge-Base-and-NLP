import argparse
import os

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default='ug')
parser.add_argument("--vocab_path", type=str, default="specializing-multilingual-data/data/ug/unlabeled/bert_shards_new_onlytrain/20000-5-1000-merged.txt")
args = parser.parse_args()

BASE_DIR = "specializing-multilingual-data/data"

DATA_PATHS = {
    "unlabeled": {
        "train": "unlabeled/bert_cleaned/train.txt",
        "valid": "unlabeled/bert_cleaned/valid.txt",
        "test": "unlabeled/bert_cleaned/test.txt",
    },
    # "normalized": {
    #     "test_NFC": "unlabeled/bert_cleaned/normalize/test_NFC.txt",
    #     "test_NFD": "unlabeled/bert_cleaned/normalize/test_NFD.txt",
    #     "test_NFKC": "unlabeled/bert_cleaned/normalize/test_NFKC.txt",
    #     "test_NFKD": "unlabeled/bert_cleaned/normalize/test_NFKD.txt",
    # },
    "ud": {
        "train": "ud/train.raw",
        "valid": "ud/dev.raw",
        "test": "ud/test.raw",
    },
    "ner": {
        "train": "panx/train.raw",
        "valid": "panx/dev.raw",
        "test": "panx/test.raw",
    },
}


def load_roberta_tokenizer(lang):
    roberta_dir = os.path.join(
        BASE_DIR, lang, "unlabeled/roberta_shards/bpe-52000-2"
    )
    roberta_tokenizer = ByteLevelBPETokenizer(
        vocab=os.path.join(roberta_dir, "vocab.json"),
        merges=os.path.join(roberta_dir, "merges.txt"),
    )
    return roberta_tokenizer


def load_mbert_tokenizer():
    mbert_vocab = os.path.join(BASE_DIR, "mbert/vocab.txt")
    tokenizer = BertWordPieceTokenizer(
        args.vocab_path,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
    )
    return tokenizer


def load_tva_tokenizer(lang):
    tokenizer = BertWordPieceTokenizer(
        args.vocab_path,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
    )
    return tokenizer


def wordpiece_continuation_indicator(_index, wordpiece):
    return wordpiece.startswith("##")


ROBERTA_SPACE_CHAR = chr(288)


def roberta_continuation_indicator(index, subword):
    """
    roberta denotes continuations as follows:
    * except for the first word in a sentence,
        start-of-word subwords start with u+0120 (chr288), which is some
        corruption of the space character
    * all other subwords are as-is
    """
    return index > 0 and not subword.startswith(ROBERTA_SPACE_CHAR)


tokenizers = {
    # "mbert": (
    #     load_mbert_tokenizer(),
    #     wordpiece_continuation_indicator,
    #     "[UNK]",
    # ),
    "tva": (
        load_tva_tokenizer(args.lang),
        wordpiece_continuation_indicator,
        "[UNK]",
    ),
    # "roberta": (
    #     load_roberta_tokenizer(args.lang),
    #     roberta_continuation_indicator,
    #     "<unk>",
    # ),
}


def get_metrics(tokenizer, file_path, *, continuation_indicator, unk_token):
    """
    Computes the following metrics about `file_path`, based on `tokenizer`:
    * average subword fertility: (# of tokenizer subwords) / (# of tokens prior
        to tokenizer)
    * proportion of continuation subwords: (# of continuation subwords) / (#
        of tokenizer subwords)
    * proportion of unknown subwords: (# of unk subwords) / (# of tokenizer
        subwords)
    Assumes that `file_path` is already basic tokenized.
    """
    global_token_count = 0
    global_subword_count = 0
    global_continuation_subword_count = 0
    global_has_continuation_subword_count = 0
    global_unk_subword_count = 0
    global_token_level_subword_count = 0
    global_token_level_continuation_subword_count = 0
    global_tokens_with_conti_count = 0
    global_token_level_unk_subword_count = 0
    global_tokens_with_unk_count = 0
    with open(file_path) as f:
        for line in tqdm(f, leave=False):
            line = line.strip()
            subwords = tokenizer.encode(line, add_special_tokens=False).tokens

            subword_count = len(subwords)
            global_subword_count += subword_count
            split_line = line.split()
            global_token_count += len(split_line)

            for i, subword in enumerate(subwords):
                if continuation_indicator(i, subword):
                    global_continuation_subword_count += 1
                if subword == unk_token:
                    global_unk_subword_count += 1

            # this is a sanity check for WordPiece models and a necessary check
            # for BBPE RoBERTa models, since there's a difference between the
            # with-spaces and no-spaces tokenization in that case
            token_level_subword_count = 0
            for tok in split_line:
                tok_subwords = tokenizer.encode(
                    tok, add_special_tokens=False
                ).tokens
                token_level_subword_count += len(tok_subwords)
                token_has_unk = False
                token_has_conti = False
                for i, subword in enumerate(tok_subwords):
                    if continuation_indicator(i, subword):
                        global_token_level_continuation_subword_count += 1
                        token_has_conti = True
                    if subword == unk_token:
                        global_token_level_unk_subword_count += 1
                        token_has_unk = True
                global_tokens_with_unk_count += int(token_has_unk)
                global_tokens_with_conti_count += int(token_has_conti)
            global_token_level_subword_count += token_level_subword_count

            # ##### BEGIN SANITY CHECK #####
            # if (
            #     token_level_subword_count != subword_count
            #     and unk_token == "[UNK]"
            # ):
            #     import pdb
            #
            #     pdb.set_trace()
            # ##### END SANITY CHECK   #####
    return {
        "token_count": global_token_count,
        "subword_count": global_subword_count,
        "continuation_subword_count": global_continuation_subword_count,
        "unk_subword_count": global_unk_subword_count,
        "fertility": global_subword_count / global_token_count,
        "continuation_proportion": global_continuation_subword_count
        / global_subword_count,
        "unk_proportion": global_unk_subword_count / global_subword_count,
        "token_level_subword_count": global_token_level_subword_count,
        "token_level_continuation_subword_count": global_token_level_continuation_subword_count,
        "token_level_unk_subword_count": global_token_level_unk_subword_count,
        "token_level_fertility": global_token_level_subword_count
        / global_token_count,
        "token_level_continuation_proportion": global_token_level_continuation_subword_count
        / global_token_level_subword_count,
        "token_level_unk_proportion": global_token_level_unk_subword_count
        / global_token_level_subword_count,
        "tokens_with_unk_count": global_tokens_with_unk_count,
        "tokens_with_unk_fraction": global_tokens_with_unk_count
        / global_token_count,
        "tokens_with_conti_count": global_tokens_with_conti_count,
        "tokens_with_conti_fraction": global_tokens_with_conti_count
                                    / global_token_count,
    }


print(f"BEGIN VOCABULARY REPORT FOR {args.lang}")
for data_name, data_paths in DATA_PATHS.items():
    print(f"\tEvaluating dataset: {data_name}")
    for split, path in data_paths.items():
        file_path = os.path.join(BASE_DIR, args.lang, path)
        if not os.path.exists(file_path):
            print(
                f"\t\tData split {{{split}}} at path {file_path} does not exist"
            )
            continue
        print(f"\t\tFor data split {{{split}}} at path {file_path}")
        print(
            f"\t\t\tTokenizer\tFertility\tContinuation\tUnkProportion\tTFertility\tTContinuation\tTUnkProp\tTokWithUnkProp\tTokWithContiProp"
        )

        for tokenizer_name, tokenizer_args in tokenizers.items():
            tokenizer, indicator, unk = tokenizer_args
            metrics = get_metrics(
                tokenizer,
                file_path,
                continuation_indicator=indicator,
                unk_token=unk,
            )
            print(
                f"\t\t\t\t{tokenizer_name}"
                f"\t\t\t{metrics['fertility']:.5f}"
                f"\t\t\t{metrics['continuation_proportion']:.5f}"
                f"\t\t\t{metrics['unk_proportion']:.5f}"
                f"\t\t\t{metrics['token_level_fertility']:.5f}"
                f"\t\t\t{metrics['token_level_continuation_proportion']:.5f}"
                f"\t\t\t{metrics['token_level_unk_proportion']:.5f}"
                f"\t\t\t{metrics['tokens_with_unk_fraction']:.5f}"
                f"\t\t\t{metrics['tokens_with_conti_fraction']:.5f}"
            )
print(f"END VOCABULARY REPORT FOR {args.lang}")
