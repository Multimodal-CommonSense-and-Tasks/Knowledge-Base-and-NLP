from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, MarianMTModel, AutoTokenizer

from model.model import ParaphraserBase as Paraphraser
from model.dataset import TextGenerationDataset
from model.metrics import bert_ibleu
from model.arguments import EvaluationArguments

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

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.result_path, exist_ok=True)  # Python>3.2
    # Init logger
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove better log file
        if os.path.exists(os.path.join(args.result_path, "eval_bestoutput_rank.log")):
            os.remove(os.path.join(args.result_path, "eval_bestoutput_rank.log"))
    file_handler = logging.FileHandler(os.path.join(args.result_path, "eval_bestoutput_rank.log"))
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.device = device
    model.num_beams = args.num_beams
    model = model.to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    
    def tg_collate_fn(batch):
        src, tgt = zip(*batch)
        src = pad_sequence(src, batch_first=True, padding_value=base_tokenizer.pad_token_id)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=base_tokenizer.pad_token_id)
        return src, tgt

    # ls = [x for x in args.model_postfix.split('_') if 'seed' not in x]
    # print("cache dir", ls)
    cpath = '.cache/' + args.base_model + "_" + ("qqp" if "qqp" in args.test_data else "mscoco")

    if 'dev' in args.test_data:
        split = 'dev'
    elif 'train' in args.test_data:
        split = 'train'
    else:
        split = 'test'

    test_dataset = TextGenerationDataset(base_tokenizer, test_data, cpath + f'_{split}.pkl', shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=tg_collate_fn)

    # Eval phase (on dev set)
    model.eval()
    loss = 0
    rank_count = [0 for _ in range(25)]
    inversed_rank_count = [0 for _ in range(25)]
    best_token_rank0_count = [0 for _ in range(25)]
    total_count = [0 for _ in range(25)]

    with torch.inference_mode():        
        for data in tqdm(test_dataloader):
            sources, targets = data
            r, t = model.token_rank(sources, bert_ibleu(EvaluationArguments()))
            for i, (rank, tot) in enumerate(zip(r, t)):
                if i > 25:
                    i = 25
                rank_count[i] += rank
                total_count[i] += tot
        
    # logger.info(f"Average 1/rank: {sum(rank_count) / sum(total_count)}")
    for i in range(len(rank_count)):
        if total_count[i]:
            logger.info(f"{i} avg > {rank_count[i] / total_count[i]}")
    logger.info(f"Length distribution: {total_count}")
    # logger.info(f"Best token on rank0 ratio: {sum(best_token_rank0_count) / sum(total_count)}")
    # for i in range(len(rank_count)):
    #     if total_count[i]:
    #         logger.info(f"{i} avg > {best_token_rank0_count[i] / total_count[i]}")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--test_data", required=False, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference")

    # Checkpoint configs
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--base_model", required=False, default="bart", choices=["bart", "t5"], help="Base model to train. If using `from_checkpoint`, you do not need to specify this option.")
    
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")


    args = parser.parse_args()

    main(args)
