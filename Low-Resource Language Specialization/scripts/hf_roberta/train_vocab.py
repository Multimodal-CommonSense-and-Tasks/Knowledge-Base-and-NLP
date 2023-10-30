import argparse
import os

from tokenizers import ByteLevelBPETokenizer

# most of these args follow the defaults at https://huggingface.co/blog/how-to-train
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, help="Training files", required=True)
parser.add_argument(
    "--vocab-size", type=int, help="How big to make vocab", default=52000
)
parser.add_argument("--output-dir", type=str, help="Output dir", required=True)
parser.add_argument(
    "--min-frequency", type=int, help="Min frequency to merge", default=2
)

args = parser.parse_args()

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=args.corpus,
    vocab_size=args.vocab_size,
    min_frequency=args.min_frequency,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>",],
)

outdir = os.path.join(
    args.output_dir, f"bpe-{args.vocab_size}-{args.min_frequency}"
)

os.makedirs(outdir, exist_ok=True)
tokenizer.save_model(outdir)

