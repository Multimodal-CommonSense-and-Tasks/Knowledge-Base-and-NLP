from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, AutoTokenizer

from model.model import ParaphraserBase as Paraphraser
from model.dataset import TextGenerationDataset, tg_collate_fn
from model.constrained_decoding import *

MODEL_ID = {
    'bart': 'facebook/bart-base',
    't5': 't5-small',
}
MODEL_CLASS = {
    'bart': BartForConditionalGeneration,
    't5': T5ForConditionalGeneration,
}

model_to_paper_name = {
    "exp_penalty": "Thompson & Post (2020) Paraphrase Generation as Zero-Shot Multilingual Translation: Disentangling Semantic Similarity from Lexical and Syntactic Diversity",
    "dyn_block": "Niu et al. (2021) Unsupervised Paraphrasing with Pretrained Language Models",
    "first3": "Hu et al. (2019) ParaBank: Monolingual Bitext Generation and Sentential Paraphrasing via Lexically-constrained Neural Machine Translation"
}
model_to_logits_processor = {
    "exp_penalty": PenalizeNgramLogitsProcessor(
        penalty_for_ngrams=[0.003, 0.048, 0.243, 0.768], sample_p=1.0
    ),
    "dyn_block": PenalizeNgramLogitsProcessor(
        penalty_for_ngrams=[0, float("inf"), 0, 0], sample_p=0.5
    ),
    "first3": PositionalConstraintLogitsProcessor(
        position_n=3, penalty=float("inf")
    )
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
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(model_store_path, f"inference_{args.constraint_mode}.log")):
            os.remove(os.path.join(model_store_path, f"inference_{args.constraint_mode}.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, f"inference_{args.constraint_mode}.log"))
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
    test_dataset = TextGenerationDataset(test_data, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=tg_collate_fn)

    # Eval phase (on dev set)
    model.eval()

    logger.info("<Constrained Decoding penalizing N-grams>")
    logger.info(f"{args.constraint_mode}")
    logger.info(f"From the paper: {model_to_paper_name[args.constraint_mode]}")
    processor = model_to_logits_processor[args.constraint_mode]

    result = []
    first_batch=True
    for data in tqdm(test_loader):
        inputs, _ = data
        with torch.no_grad():
            outputs = model.generate_ngram_constrained(
                inputs, logits_processor=processor
            )

        for outputs, reference in zip(outputs, inputs):
            result.append({
                "input": reference,
                "paraphrases": outputs
            })

        if first_batch:
            test_input = inputs[0]
            test_outputs = outputs
            first_batch = False
    
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, f"result_{args.constraint_mode}.json")
    with open(result_store_path, "w", encoding="UTF-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    logger.info("=================================================")
    logger.info("Test generation result")
    logger.info(f"input: {test_input}")
    logger.info(f"output:")
    for test_output in test_outputs:
        logger.info(f"  {test_output}")
    logger.info("")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference")
    parser.add_argument("--constraint_mode", choices=["exp_penalty", "dyn_block", "first3"], help="Presets for constrained decoding")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True, help="Name for the model.")

    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)