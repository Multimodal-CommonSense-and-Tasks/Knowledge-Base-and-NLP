import argparse
import os
import glob

from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, help="Training files", required=True)
parser.add_argument(
    "--vocab-size", type=int, help="How big to make vocab", required=True
)
parser.add_argument("--output-dir", type=str, help="Output dir", required=True)
parser.add_argument(
    "--min-frequency", type=int, help="Min frequency to merge", default=5
)
parser.add_argument(
    "--limit-alphabet", type=int, help="Alphabet max size", default=1000
)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,
    lowercase=False,
)

corpora = list(glob.glob(args.corpus))
print(corpora)
tokenizer.train(
    corpora,
    # args.corpus,
    vocab_size=args.vocab_size,
    min_frequency=args.min_frequency,
    limit_alphabet=args.limit_alphabet,
)

os.makedirs(args.output_dir, exist_ok=True)
tokenizer.save_model(
    args.output_dir,
    f"{args.vocab_size}-{args.min_frequency}-{args.limit_alphabet}",
)
