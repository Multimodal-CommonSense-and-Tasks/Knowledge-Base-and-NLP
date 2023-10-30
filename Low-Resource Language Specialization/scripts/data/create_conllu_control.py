import argparse
import os

import conllu
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--conllu-folder", type=str)
parser.add_argument("--upos", action="store_true")
args = parser.parse_args()

train_file = os.path.join(args.conllu_folder, "train.conllu")
dev_file = os.path.join(args.conllu_folder, "dev.conllu")
test_file = os.path.join(args.conllu_folder, "test.conllu")

tag_counts = {}
tag_ordering = []

with open(train_file) as tf:
    for annotation in conllu.parse_incr(tf):
        # CoNLLU annotations sometimes add back in words that have been elided
        # in the original sentence; we remove these, as we're just predicting
        # dependencies for the original sentence.
        # We filter by integers here as elided words have a non-integer word id,
        # as parsed by the conllu python library.
        annotation = [x for x in annotation if isinstance(x["id"], int)]

        if args.upos:
            pos_tags = [x["xpostag"] for x in annotation]
        else:
            pos_tags = [x["upostag"] for x in annotation]

        for tag in pos_tags:
            if tag not in tag_counts:
                tag_counts[tag] = 0
                tag_ordering.append(tag)
            tag_counts[tag] += 1

total_tags = sum(tag_counts.values())
tag_probabilities = [tag_counts[tag] / total_tags for tag in tag_ordering]


def sample():
    return np.random.choice(tag_ordering, p=tag_probabilities)


tag_assignments = {}

train_out = os.path.join(args.conllu_folder, "train.conllu.randomized")
dev_out = os.path.join(args.conllu_folder, "dev.conllu.randomized")
test_out = os.path.join(args.conllu_folder, "test.conllu.randomized")


def randomize(in_path, out_path):
    with open(in_path) as inf, open(out_path, "w") as ouf:
        for annotation in conllu.parse_incr(inf):
            for token in annotation:
                if isinstance(token["id"], int):
                    word = token["form"]
                    if word not in tag_assignments:
                        tag_assignments[word] = sample()
                    token["xpos"] = tag_assignments[word]
                    token["upos"] = tag_assignments[word]
                    token["xpostag"] = tag_assignments[word]
                    token["upostag"] = tag_assignments[word]

            ouf.write(annotation.serialize())


randomize(train_file, train_out)
randomize(dev_file, dev_out)
randomize(test_file, test_out)
