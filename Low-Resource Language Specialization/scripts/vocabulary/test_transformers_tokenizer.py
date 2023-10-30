import argparse
import sys

sys.path.insert(0, "/homes/gws/echau18/ark/bert/pretraining")
import tokenization  # type: ignore

from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vocab-file", type=str, required=True, help="Vocabulary file"
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

base_tokenizer = tokenization.FullTokenizer(
    vocab_file=args.vocab_file, do_lower_case=args.lower_case
)
torch_tokenizer = BertWordPieceTokenizer(
    args.vocab_file,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=args.lower_case,
    lowercase=args.lower_case,
)


line_level = 0
total_lines = 0
token_level = 0
total_tokens = 0
for infile in args.input_file:
    with open(infile, encoding="utf-8") as inf:
        for line in tqdm(inf):
            orig_tokens = line.strip()
            wordpieces = base_tokenizer.tokenize(orig_tokens)
            torch_wordpieces = torch_tokenizer.encode(
                orig_tokens, add_special_tokens=False
            ).tokens
            total_lines += 1
            if wordpieces != torch_wordpieces:
                line_level += 1
                # print(
                #     f"{orig_tokens} became {wordpieces} and {torch_wordpieces}".encode(
                #         "utf-8"
                #     )
                # )
            for token in orig_tokens.split():
                wordpieces = base_tokenizer.tokenize(token)
                torch_wordpieces = torch_tokenizer.encode(
                    token, add_special_tokens=False
                ).tokens
                total_tokens += 1
                if wordpieces != torch_wordpieces:
                    token_level += 1
                    # print(
                    #     f"{token} became {wordpieces} and {torch_wordpieces}".encode(
                    #         "utf-8"
                    #     )
                    # )
    print(f"For infile {infile}")
    print(f"\tDifference in {line_level} / {total_lines} lines")
    print(f"\tDifference in {token_level} / {total_tokens} tokens")

