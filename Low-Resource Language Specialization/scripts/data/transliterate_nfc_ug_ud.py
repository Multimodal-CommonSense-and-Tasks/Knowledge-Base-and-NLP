import argparse
import subprocess
import unicodedata

import conllu

parser = argparse.ArgumentParser()
parser.add_argument("--conllu-file", type=str)
parser.add_argument("--output-file", type=str)
parser.add_argument(
    "--mode", choices=["NFC", "NFD", "NFKC", "NFKD"], default="NFC"
)
args = parser.parse_args()


def transliterate(token):
    token = unicodedata.normalize(args.mode, token)
    translit_output = subprocess.run(
        [
            "perl",
            "alTranscribe.pl",
            "-f",
            "ug",
            "-t",
            "tr",
        ],
        capture_output=True,
        input=token,
        text=True,
        cwd="../mbert-unseen-languages/transfer/transliteration/",
    )
    transliterated = translit_output.stdout
    return transliterated.strip()


with open(args.conllu_file) as tf, open(args.output_file, "w") as ouf:
    for annotation in conllu.parse_incr(tf):
        annotation.metadata["text"] = transliterate(
            annotation.metadata["text"]
        )
        for entry in annotation:
            entry["form"] = transliterate(entry["form"])
        ouf.write(annotation.serialize())
