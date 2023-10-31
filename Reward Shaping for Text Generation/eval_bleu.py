from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch

from transformers import AutoTokenizer

from model.metrics import *

def main(args):
    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    try:
        eval_postfix = args.eval_file.replace("result", "")
    except:
        eval_postfix = args.eval_file
    eval_postfix = eval_postfix.replace(".json", "")

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, f"eval_BLEU{eval_postfix}.log")):
            os.remove(os.path.join(log_path, f"eval_BLEU{eval_postfix}.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, f"eval_BLEU{eval_postfix}.log"))
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
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, args.eval_file)
    with open(result_store_path, "r", encoding="UTF-8") as file:
        result = json.load(file)

    reference = []
    outputs = []
    for r in result:
        reference.append(r["target"])
        outputs.append(r["outputs"])
        
    # Obtain scores
    bleu = get_bleu_score(
        None,
        reference,
        outputs,
        eval=True
    )

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("")
    logger.info("BLEU score")
    logger.info(f"Total average: {torch.mean(bleu)}")
    logger.info(f"BLEU score per beam:")
    for beam_id, score in enumerate(torch.mean(bleu, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    bleu_sorted, _ = torch.sort(bleu, dim=-1)
    logger.info(f"BLEU score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(bleu_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--eval_file", required=False, default='result.json', help="Name of the result file(generated by inference.py)")
    
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)