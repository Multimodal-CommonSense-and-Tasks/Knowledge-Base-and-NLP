import argparse
import os
import random

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-dir",
    type=str,
    help="Input directory with {train, valid, test}.txt",
    required=True,
)
parser.add_argument(
    "--output-dir", type=str, help="Output directory", required=True
)
parser.add_argument(
    "--file-count",
    type=int,
    help="Number of files to distribute between",
    default=8,
)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def remove_blanks(in_file, out_file):
    with open(in_file) as inf, open(out_file, "w") as outf:
        for line in inf:
            if line.strip():
                outf.write(line)


def distribute(in_file, out_dir, count):
    concat = os.path.join(out_dir, "train.txt")
    with open(in_file) as inf:
        document = []
        for line in tqdm(inf):
            if not line.strip():
                # end of document
                if document:
                    outname = os.path.join(
                        out_dir, f"train_{random.randint(1, count)}.txt"
                    )
                    with open(outname, "a") as outf, open(
                        concat, "a"
                    ) as concatf:
                        for sentence in document:
                            outf.write(sentence)
                            concatf.write(sentence)

                document = []
            else:
                document.append(line)


distribute(
    os.path.join(args.input_dir, "train.txt"), args.output_dir, args.file_count
)
remove_blanks(
    os.path.join(args.input_dir, "valid.txt"),
    os.path.join(args.output_dir, "valid.txt"),
)
remove_blanks(
    os.path.join(args.input_dir, "test.txt"),
    os.path.join(args.output_dir, "test.txt"),
)

