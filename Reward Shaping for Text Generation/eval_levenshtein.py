from argparse import ArgumentParser
import json
import os, sys
import logging

from tqdm import tqdm
import torch

from nltk.metrics.distance import edit_distance

def main(args):
    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, "eval_levenshtein.log")):
            os.remove(os.path.join(log_path, "eval_levenshtein.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, "eval_levenshtein.log"))
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

    NED = []
    for r in tqdm(result):
        input_sent = r["input"]
        dists = []
        for p in r["paraphrases"]:
            dists.append(edit_distance(input_sent, p) / max(len(input_sent), len(p)))
        NED.append(dists)
    NED = torch.tensor(NED)
        
    # Obtain scores

    logger.info("=================================================")
    logger.info("Analysis result")
    logger.info("")
    logger.info("Normalized Levenshtein Edit Distance(NED)")
    logger.info(f"Total average: {torch.mean(NED)}")
    logger.info(f"NED score per beam:")
    for beam_id, score in enumerate(torch.mean(NED, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    NED_sorted, _ = torch.sort(NED, dim=-1)
    logger.info(f"NED score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(NED_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
