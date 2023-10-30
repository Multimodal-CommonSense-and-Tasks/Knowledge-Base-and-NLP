import argparse
import unicodedata

# NOTE: this doesn't actually transliterate; it just does NFKC.  It's named
# this way so it groups well with other files in the directory.

parser = argparse.ArgumentParser()
parser.add_argument("--raw-file", type=str)
parser.add_argument("--output-file", type=str)
parser.add_argument(
    "--mode", choices=["NFC", "NFD", "NFKC", "NFKD"], default="NFKC"
)
args = parser.parse_args()

with open(args.raw_file) as rf, open(args.output_file, "w") as of:
    for line in rf:
        of.write(unicodedata.normalize(args.mode, line))
