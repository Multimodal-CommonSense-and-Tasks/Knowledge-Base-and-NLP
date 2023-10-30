"""
1. Get all the words that yield UNK in the corpus
2. If one is UNK and the other is not, add the unique wordpieces
3. Pad to args.count with the most common wordpieces in the new vocab
"""
import argparse
import os
from collections import Counter, OrderedDict

from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corpus", type=str, required=True, help="Reference corpus file"
)
parser.add_argument(
    "--base-vocab", type=str, required=True, help="Base vocab file"
)
parser.add_argument(
    "--new-vocab", type=str, required=True, help="New vocab file"
)
parser.add_argument(
    "--count", type=int, default=99, help="How many values to return"
)
parser.add_argument(
    "--output-file", type=str, required=True, help="Where to output"
)
parser.add_argument(
    "--lower-case",
    action="store_true",
    help="Whether to lowercase inputs (for uncased models)",
)
# NOTE: this flag would more directly model the language but potentially lose
# transfer (not that big of a deal?) vs. more fully modeling the language and
# helping better with downstream.  Common wordpieces might help with
# data augmentation better, but whole word masking might render that moot.
parser.add_argument(
    "--use-common-wordpieces",
    action="store_true",
    help="NOT working right now..",
)

args = parser.parse_args()


def count(tkn, lst):
    return sum(1 if tkn == item else 0 for item in lst)


# NOTE: we use the `tokenizers` implementation here instead of the old
# transformers one
base_tokenizer = BertWordPieceTokenizer(
    args.base_vocab,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=args.lower_case,
    lowercase=args.lower_case,
)

# words in the base vocab
existing_vocab = set()
with open(args.base_vocab) as base_file:
    for line in base_file:
        entry = line.strip()
        existing_vocab.add(entry)

# words in the new vocab that represent something with UNK in the base vocab
# Question: wouldn't it be more straightforward to derive this from the corpus --
# take a pass through the data, tokenize each line, grab new wordpieces that
# appear in the new tokenization?
# or do we even need to do this?  just use the full vocabulary below and compare
# tokenization directly; could even directly optimize for common wordpieces
# (directly model lang, but this may lose transfer) vs. unks (fully model lang;
# probably helps better with downstream)
# NOTE: this is why we commented out the old lines before -- this doesn't make
# sense!

new_vocab = OrderedDict()
unk = "[UNK]"
with open(args.new_vocab) as eval_file:
    for i, line in enumerate(eval_file):
        entry = line.strip()
        new_vocab[entry] = i

new_tokenizer = BertWordPieceTokenizer(
    new_vocab,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=args.lower_case,
    lowercase=args.lower_case,
)

unk_words = []
new_words = []
with open(args.corpus) as corpus:
    for line in tqdm(corpus):
        orig_tokens = line.strip().split()
        for token in orig_tokens:
            wordpieces = base_tokenizer.encode(
                token, add_special_tokens=False
            ).tokens
            new_wordpieces = new_tokenizer.encode(
                token, add_special_tokens=False
            ).tokens
            new_words.extend(new_wordpieces)
            if count(unk, wordpieces) > count(unk, new_wordpieces):
                unk_words.extend(
                    [nwp for nwp in new_wordpieces if nwp not in wordpieces]
                )

# count of unk word frequency
vocab_counter = Counter(unk_words)
selection = vocab_counter.most_common()
selection = [tup[0] for tup in selection if tup[0] not in existing_vocab]
selection = selection[: args.count]

# count of all word frequency
new_counter = Counter(new_words).most_common()
new_counter = [tup[0] for tup in new_counter if tup[0] not in existing_vocab]

# pad
i = 0
while len(selection) < args.count and i < len(new_counter):
    if new_counter[i] not in selection:
        selection.append(new_counter[i])
    i += 1


os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, "w") as ouf:
    for item in selection:
        ouf.write(item)
        ouf.write("\n")

