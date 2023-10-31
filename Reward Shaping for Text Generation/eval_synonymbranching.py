from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, MarianMTModel, AutoTokenizer

from model.model import ParaphraserBase as Paraphraser
from model.dataset import SynonymBranchingEvalDataset

MODEL_ID = {
    'bart': 'facebook/bart-base',
    't5': 't5-small',
    'marian-ende': "Helsinki-NLP/opus-mt-en-de",
    'marian-enfr': "Helsinki-NLP/opus-mt-en-fr",
    'marian-enro': "Helsinki-NLP/opus-mt-en-ro",
}
MODEL_CLASS = {
    'bart': BartForConditionalGeneration,
    't5': T5ForConditionalGeneration,
    'marian-ende': MarianMTModel,
    'marian-enfr': MarianMTModel,
    'marian-enro': MarianMTModel,
}

def main(args):
    # Set torch
    torch.manual_seed(0)

    # For simplicity, if a directory is given, load the last checkpoint(last name in alphabetical order)
    if args.model_store_path.endswith(".pt"):
        model_store_path = args.model_store_path
    else:
        assert os.path.isdir(args.model_store_path)
        log_path = model_store_path = os.path.join(args.model_store_path, args.model_postfix)
        assert os.path.isdir(model_store_path)
        last_checkpoint = sorted([f for f in os.listdir(model_store_path) if f.endswith(".pt")], reverse=True)[0]
        model_store_path = os.path.join(args.model_store_path, args.model_postfix, last_checkpoint)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, "eval_synonymbranching.log")):
            os.remove(os.path.join(log_path, "eval_synonymbranching.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, "eval_synonymbranching.log"))
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
    
    # Load base model(BART, T5, ...)
    model_id = MODEL_ID[args.base_model]
    model_class = MODEL_CLASS[args.base_model]
    base_model = model_class.from_pretrained(model_id)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    # Load state_dict and recover non-tensor member variables
    model = Paraphraser(
        base_model,
        base_tokenizer,
        num_beams=args.num_beams
    )
    model.load_state_dict(torch.load(model_store_path, map_location=device))
    model.device = device
    model = model.to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    test_dataset = SynonymBranchingEvalDataset(test_data)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    # Eval phase (on dev set)
    model.eval()

    original_prob = []
    synonym_prob = []
    logprob_diff = []
    synonym_rank = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            result = model.synonym_branching_test(*batch)
            original_prob.extend(result["original_prob"].tolist())
            synonym_prob.extend(result["synonym_prob"].tolist())
            logprob_diff.extend(result["logprob_diff"].tolist())
            synonym_rank.extend(result["synonym_rank"].tolist())

    logger.info("=================================================")

    logger.info("")
    logger.info("Original token probability: If an appropriate synonym exists, how does the token probability change?")
    logger.info(f"Total average: {sum(original_prob) / len(original_prob)}")
    original_prob = sorted(original_prob)
    logger.info(f"min: {original_prob[0]}")
    logger.info(f"Q1 : {original_prob[1 * len(original_prob)//4]}")
    logger.info(f"Q2 : {original_prob[2 * len(original_prob)//4]}")
    logger.info(f"Q3 : {original_prob[3 * len(original_prob)//4]}")
    logger.info(f"max: {original_prob[-1]}")
    logger.info("")
    logger.info("Synonym probability: If an appropriate synonym exists, how does the token probability change?")
    logger.info(f"Total average: {sum(synonym_prob) / len(synonym_prob)}")
    synonym_prob = sorted(synonym_prob)
    logger.info(f"min: {synonym_prob[0]}")
    logger.info(f"Q1 : {synonym_prob[1 * len(synonym_prob)//4]}")
    logger.info(f"Q2 : {synonym_prob[2 * len(synonym_prob)//4]}")
    logger.info(f"Q3 : {synonym_prob[3 * len(synonym_prob)//4]}")
    logger.info(f"max: {synonym_prob[-1]}")

    logger.info("")
    logger.info("Synonym branching factor = logp(synonym) - logp(original)")
    logger.info(f"Total average: {sum(logprob_diff) / len(logprob_diff)}")
    logprob_diff = sorted(logprob_diff)
    logger.info(f"min: {logprob_diff[0]}")
    logger.info(f"Q1 : {logprob_diff[1 * len(logprob_diff)//4]}")
    logger.info(f"Q2 : {logprob_diff[2 * len(logprob_diff)//4]}")
    logger.info(f"Q3 : {logprob_diff[3 * len(logprob_diff)//4]}")
    logger.info(f"max: {logprob_diff[-1]}")
    positive_values = sum([(1 if x>0 else 0) for x in logprob_diff])
    logger.info(f"positive_value: {positive_values} ({positive_values / len(logprob_diff) * 100} %)")
    
    logger.info("")
    logger.info("Synonym rank = rank of the synonym token among all vocab(lower the better)")
    logger.info(f"Total average: {sum(synonym_rank) / len(synonym_rank)}")
    synonym_rank = sorted(synonym_rank)
    logger.info(f"min: {synonym_rank[0]}")
    logger.info(f"Q1 : {synonym_rank[1 * len(synonym_rank)//4]}")
    logger.info(f"Q2 : {synonym_rank[2 * len(synonym_rank)//4]}")
    logger.info(f"Q3 : {synonym_rank[3 * len(synonym_rank)//4]}")
    logger.info(f"max: {synonym_rank[-1]}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--base_model", required=False, default="bart", choices=["bart", "t5"], help="Base model to train. If using `from_checkpoint`, you do not need to specify this option.")
    
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
