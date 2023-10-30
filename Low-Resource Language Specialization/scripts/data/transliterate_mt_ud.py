import argparse
import subprocess
import unicodedata

import conllu
import os
from scripts.data.transliterate_mt_panx import transliterate

parser = argparse.ArgumentParser()
parser.add_argument("--conllu-file", type=str)
parser.add_argument("--output-file", type=str)
# parser.add_argument(
#     "--mode", choices=["NFC", "NFD", "NFKC", "NFKD"], default="NFKC"
# )
args = parser.parse_args()



with open(args.conllu_file) as tf, open(args.output_file, "w") as ouf:
    for annotation in conllu.parse_incr(tf):
        annotation.metadata["text"] = transliterate(
            annotation.metadata["text"]
        )
        for entry in annotation:
            entry["form"] = transliterate(entry["form"])
        ouf.write(annotation.serialize())