from conllu.parser import parse_line, DEFAULT_FIELDS
# from allennlp.common.file_utils import cached_path
import argparse
from tqdm import tqdm


def lazy_parse(text, fields=DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [
                parse_line(line, fields)
                for line in sentence.split("\n")
                if line and not line.strip().startswith("#")
            ]


def extract(file_path, output_file_path):
    # if `file_path` is a URL, redirect to the cache
    # file_path = cached_path(file_path)

    with open(file_path, "r") as conllu_file, open(
        output_file_path, "w"
    ) as output_file:
        for annotation in tqdm(lazy_parse(conllu_file.read())):
            words = [x["form"] for x in annotation]
            output_file.write(" ".join(words))
            output_file.write("\n")


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--file-path",
    type=str,
    help="Universal Dependencies-formatted file to process",
    required=True,
)
parser.add_argument(
    "--output-file-path",
    type=str,
    help="File to write plaintext to",
    required=True,
)

args = parser.parse_args()
extract(args.file_path, args.output_file_path)
