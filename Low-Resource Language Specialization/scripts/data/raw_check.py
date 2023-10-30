import argparse
import os
import re

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--target-dir", type=str)
parser.add_argument("--output-dir", type=str)
# parser.add_argument("--filter-file", type=str, action="append", default=[])
parser.add_argument("--filter-base-dir", type=str, default="./")
parser.add_argument("--filter-file", type=str, nargs="+")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

filters = set()
whitespace_re = re.compile(r"\s")


def process_line(line):
    return whitespace_re.sub("", line)


for filter in args.filter_file:
    fpath = os.path.join(args.filter_base_dir, filter)
    if not os.path.exists(fpath):
        print(f"Warning: {fpath} does not exist")
        continue
    with open(fpath) as ff:
        for line in tqdm(ff):
            filters.add(process_line(line.strip()))

os.makedirs(args.output_dir, exist_ok=True)

for f in os.listdir(args.target_dir):
    target_file = os.path.join(args.target_dir, f)
    output_file = os.path.join(args.output_dir, f)

    with open(target_file, encoding="utf-8") as tf, open(
        output_file, "w"
    ) as of:
        collisions = 0
        try:
            for line in tqdm(tf):
                if process_line(line.strip()) in filters:
                    collisions += 1
                    if args.verbose:
                        print(line)
                else:
                    of.write(f"{line.strip()}\n")
        except:
            import pdb

            pdb.set_trace()

        print(f"File {target_file}: total {collisions} collisions detected")
