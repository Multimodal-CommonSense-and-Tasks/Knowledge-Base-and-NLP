"""
Author: Jinu Lee
Credits:
- Dependency parsing:  https://github.com/Unipisa/diaparser
"""

from argparse import ArgumentParser
import json
from tqdm import tqdm, trange
import os, sys
import logging
import re

import torch
import math

from nltk import Tree
from nltk.parse.dependencygraph import DependencyGraph
from diaparser.parsers import Parser
from tokenizer.tokenizer import Tokenizer # from diaparser
# from stanza.pipeline.core import Pipeline

logger = None
dependency_parser = Parser.load('en_ptb.electra', lang='en')
dependency_tokenizer = Tokenizer('en', verbose=False)
# _pipe = dependency_tokenizer.pipeline.__dict__; _pipe.pop('processors')
# dependency_tokenizer.pipeline = Pipeline(processors='tokenize,pos', **_pipe) # Manually add PoS tagger to diaparser pipe

def diaparser_to_nltk_tree(sentences):
    # sentences: List[List[str]] : batch of tokenized sentences
    result = dependency_parser.predict(sentences, text='en') # Parse sentences
    result = [str(sent) for sent in result.sentences] # Get CoNLL-U (10 cols) representation
    result = ['\n'.join([line for line in conll.split('\n') if not line.startswith('#')]) for conll in result] # remove comment lines since NLTK cannot interpret it

    dep_tree = [
        DependencyGraph(conll, top_relation_label='root').tree() # Convert to nltk.tree.Tree
        for conll in result
    ]

    return dep_tree

class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        hashable_args = tuple([x for x in args if x.__hash__])
        if hashable_args not in self.memo:
            self.memo[hashable_args] = self.fn(*args)
        return self.memo[hashable_args]


def tree_to_edges(tree):
    def get_edges(tree, i):
        from_str = f"{tree.label()}▁{i}"
        children = [f"{child.label()}▁{i+1}" for child in tree if isinstance(child, Tree)]
        children.extend([f"{child}▁{i+1}" for child in tree if isinstance(child, str)])
        return [(from_str, child) for child in children]
    height = 0
    rv = []
    to_check = [tree]
    while to_check:
        tree_to_check = to_check.pop(0)
        rv.extend(get_edges(tree_to_check, height))
        height += 1
        to_check.extend([child for child in tree_to_check if isinstance(child, Tree)])
    return rv


def find_arc_pairs(first_tree, second_tree):
    arc_pairs = set()
    # Retrieve list of arcs
    first_tree_arcs = tree_to_edges(first_tree)
    first_tree_arcs = sorted(first_tree_arcs)
    second_tree_arcs = tree_to_edges(second_tree)
    second_tree_arcs = sorted(second_tree_arcs)

    # Align arcs
    if len(first_tree_arcs) == 0 or len(second_tree_arcs) == 0:
        return set()
    node_1 = first_tree_arcs.pop(0)
    node_2 = second_tree_arcs.pop(0)
    while node_1[0] != None and node_2[0] != None:
        if node_1[0] > node_2[0]:
            if len(second_tree_arcs) > 0:
                node_2 = second_tree_arcs.pop(0)
            else:
                node_2 = [None]
        elif node_1[0] < node_2[0]:
            if len(first_tree_arcs) > 0:
                node_1 = first_tree_arcs.pop(0)
            else:
                node_1 = [None]
        else:
            while node_1[0] == node_2[0]:
                second_tree_arcs_index = 1
                while node_1[0] == node_2[0]:
                    arc_pairs.add((str(node_1[1]), str(node_2[1])))
                    if second_tree_arcs_index < len(second_tree_arcs):
                        node_2 = second_tree_arcs[second_tree_arcs_index]
                        second_tree_arcs_index += 1
                    else:
                        node_2 = [None]
                if len(first_tree_arcs) > 0:
                    node_1 = first_tree_arcs.pop(0)
                else:
                    node_1 = [None]
                if len(second_tree_arcs) > 0:
                    node_2 = second_tree_arcs[0]
                else:
                    node_2 = [None]
                if node_1[0] == None and node_2[0] == None:
                    break
    return arc_pairs

def dep_arc_precision_recall(all_parse_trees, first_tree_index, second_tree_index):
    # P/R of the second tree, the gold being the first tree
    score = 0
    first_tree = all_parse_trees[first_tree_index]
    second_tree = all_parse_trees[second_tree_index]

    # If one tree is empty,
    if len(first_tree.leaves()) == 0 or len(second_tree.leaves()) == 0:
        return 0, 0
    
    # Calculate P/R
    arc_pairs = find_arc_pairs(first_tree, second_tree)
    for arcs in arc_pairs:
        if arcs[0] == arcs[1]:
            score += 1
    return score/len(second_tree.leaves()), score/len(first_tree.leaves())


def main(args):
    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, "eval_deparc.log")):
            os.remove(os.path.join(log_path, "eval_deparc.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, "eval_deparc.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")
    
    # Load generated data
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, "result.json")
    with open(result_store_path, "r", encoding="UTF-8") as file:
        result = json.load(file)

    # Generate parse trees
    logger.info("Generate parse trees... (Electra-DiaParser trained for PTB)")
    all_sentences = [] # Temporary storage for sentences
    reference = [] # stores indices of all_parse_trees
    outputs = []
    for r in result:
        # Append tree for reference
        reference.append(len(all_sentences))
        all_sentences.append(r["input"])
        # Append trees for outputs
        output = []
        for sent in r["paraphrases"]:
            output.append(len(all_sentences))
            if sent.strip():
                all_sentences.append(sent)
            else:
                all_sentences.append("😊") # To prevent ListOutOfRange error if string is empty
        outputs.append(output)

    # Batchified parsing
    dep_parse_trees = []
    for start in trange(0, len(all_sentences), args.batch_size):
        end = min(start + args.batch_size, len(all_sentences))
        tokenized_sents = [
            [token.text for token in dependency_tokenizer.predict(sent)[0].tokens]
        for sent in all_sentences[start:end]] # Tokenize and extract only lexical forms
        tree_batch = diaparser_to_nltk_tree(tokenized_sents)
        dep_parse_trees.extend(tree_batch)
    
    # Calculate tree kernel scores
    logger.info("Calculate Dependency arc F1 scores...")
    precision = []
    recall = []
    for ref, outs in zip(tqdm(reference), outputs):
        p_batch, r_batch = [], []
        for out in outs:
            p, r = dep_arc_precision_recall(dep_parse_trees, ref, out)
            p_batch.append(p)
            r_batch.append(r)
        precision.append(p_batch)
        recall.append(r_batch)
    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    f1 = 2 * precision * recall / (precision + recall + (precision == 0)*(recall==0)) # Last term: prevention for zero division
    # num_beams * total_length

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("")
    logger.info("Dependency Arc P/R/F1 score(lower score -> dissimilar paraphrase)")

    logger.info("")
    logger.info("<Precision>")
    logger.info(f"Total average: {torch.mean(precision)}")
    logger.info(f"kernel_scores_per_beam score per beam:")
    for beam_id, score in enumerate(torch.mean(precision, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    precision_sorted, _ = torch.sort(precision, dim=1)
    logger.info(f"kernel_scores_per_beam score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(precision_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("")
    logger.info("<Recall>")
    logger.info(f"Total average: {torch.mean(recall)}")
    logger.info(f"kernel_scores_per_beam score per beam:")
    for beam_id, score in enumerate(torch.mean(recall, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    recall_sorted, _ = torch.sort(recall, dim=1)
    logger.info(f"kernel_scores_per_beam score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(recall_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
        
    logger.info("")
    logger.info("<F1>")
    logger.info(f"Total average: {torch.mean(f1)}")
    logger.info(f"kernel_scores_per_beam score per beam:")
    for beam_id, score in enumerate(torch.mean(f1, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    f1_sorted, _ = torch.sort(f1, dim=1)
    logger.info(f"kernel_scores_per_beam score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(f1_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    
    parser.add_argument("--batch_size", required=False, type=int, default=16, help="Batch size for dependency parsing")
    parser.add_argument("--gpu", required=False, type=int, default=0, help="GPU Id for dependency parsing. Only considered when `torch.cuda.is_available()` is True.")

    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
