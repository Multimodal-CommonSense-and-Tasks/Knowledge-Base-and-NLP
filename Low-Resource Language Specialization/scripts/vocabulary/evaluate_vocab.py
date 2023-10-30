"""
Compare a base and new vocab file
"""
import argparse
import os

from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vocab-file-base", type=str, required=True, help="Vocabulary file"
)
parser.add_argument(
    "--vocab-file-eval",
    type=str,
    required=True,
    help="Vocabulary file to evaluate",
)
parser.add_argument(
    "--lower-case",
    action="store_true",
    help="Whether to lowercase inputs (for uncased models)",
)
parser.add_argument(
    "--input-file",
    type=str,
    action="append",
    default=[],
    help="Files to count wordpieces of",
)

args = parser.parse_args()

base_tokenizer = BertWordPieceTokenizer(
    args.vocab_file_base,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=args.lower_case,
    lowercase=args.lower_case,
)
eval_tokenizer = BertWordPieceTokenizer(
    args.vocab_file_eval,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=args.lower_case,
    lowercase=args.lower_case,
)

unk = "[UNK]"

# there will be an UNK token in the eval file anyways
eval_unk_count = -1
with open(args.vocab_file_eval) as eval_file:
    for line in eval_file:
        entry = line.strip()
        wordpieces = base_tokenizer.encode(
            entry, add_special_tokens=False
        ).tokens
        if unk in wordpieces:
            eval_unk_count += 1

print(f"{eval_unk_count} subwords with UNK in new vocab file")


def compute_statistics(infile, tokenizer):
    token_count = 0
    wordpiece_count = 0
    unk_count = 0
    token_with_unk_count = 0
    with open(infile) as inf:
        for line in tqdm(inf):
            orig_tokens = line.strip().split()
            for token in orig_tokens:
                token_count += 1
                wordpieces = tokenizer.encode(
                    token, add_special_tokens=False
                ).tokens
                wordpiece_count += len(wordpieces)
                local_unk_count = sum(
                    1 if piece == unk else 0 for piece in wordpieces
                )
                unk_count += local_unk_count
                token_with_unk_count += 1 if local_unk_count > 0 else 0

    print(
        f"{infile}: {token_count} tokens, {wordpiece_count} wordpieces, "
        f"{wordpiece_count / token_count} wordpieces/token on average, "
        f"{unk_count} unknown wordpieces, {unk_count / wordpiece_count} unk wordpiece fraction, "
        f"{token_with_unk_count} tokens with unk, {token_with_unk_count / token_count} unk tokens fraction"
    )


for infile in args.input_file:
    if os.path.exists(infile):
        print("base tokenizer")
        compute_statistics(infile, base_tokenizer)
        print("eval tokenizer")
        compute_statistics(infile, eval_tokenizer)
