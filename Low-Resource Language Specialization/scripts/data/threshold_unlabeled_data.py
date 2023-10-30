"""
Given the following files:
1. A file containing newline-separated unlabeled sentences
2. A file containing losses for each of the unlabeled sentences,
   with primary loss in the first column and other values tab-separated

Outputs:
1. A file with the sentences corresponding to the lowest k% of losses
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sentences", type=str)
parser.add_argument("--losses", type=str)
parser.add_argument("--percent", type=float)
parser.add_argument("--output", type=str)
args = parser.parse_args()

with open(args.sentences) as sfile, open(args.losses) as lfile:
    sentences = [line for line in sfile if line.strip()]
    # we want the bottom k% of losses, so in order to use percentiles,
    # we need to switch to the top k% of scores
    scores = [-float(line.strip().split("\t")[0]) for line in lfile]
    assert len(sentences) == len(scores)
    pairs = zip(sentences, scores)

# top 15% means above the 85th percentile
threshold = np.percentile(scores, 100 - args.percent)

with open(args.output, "w") as outfile:
    for sentence, score in pairs:
        if score >= threshold:
            outfile.write(f"{sentence}")
