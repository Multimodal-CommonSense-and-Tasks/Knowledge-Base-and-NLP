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
        if os.path.exists(os.path.join(args.result_path, "eval_partial_utility.log")):
            os.remove(os.path.join(args.result_path, "eval_partial_utility.log"))
    file_handler = logging.FileHandler(os.path.join(args.result_path, "eval_partial_utility.log"))
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
    print(base_model)

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

    ls = [x for x in args.model_postfix.split('_') if 'seed' not in x]
    cpath = '.cache/' + '_'.join(ls)

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
    win_count = [0 for _ in range(26)]
    total_count = [0 for _ in range(26)]

    with torch.inference_mode():        
        for data in tqdm(test_dataloader):
            sources, targets = data
            w, t = model.branching_test(sources, bert_ibleu(EvaluationArguments()))
            for i, (win, tot) in enumerate(zip(w, t)):
                if i > 25:
                    i = 25
                win_count[i] += win
                total_count[i] += tot
            
        logger.info(f"Win rate percentage: {sum(win_count) / sum(total_count) * 100}")
        for i in range(len(win_count)):
            if total_count[i]:
                logger.info(f"{i} > {win_count[i]} / {total_count[i]} ({win_count[i] / total_count[i] * 100}%)")

    # better_probs = [[], [], [], []]
    # worse_probs = [[], [], [], []]
    # logprob_diffs = [[], [], [], []]
    # better_ranks = [[], [], [], []]
    # worse_ranks = [[], [], [], []]
    # rank_diffs = [[], [], [], []]
    # for batch in tqdm(test_loader):
    #     with torch.no_grad():
    #         input_sent, prefix, better, worse = batch
    #         results = model.branching_test(input_sent, prefix, better, worse)
    #         # Dict[List] to List[Dict] conversion
    #         results = [{key: value[i] for key, value in results.items()} for i in range(len(results["better_prob"]))]
    #         input_tokens = base_tokenizer(input_sent).input_ids

    #         for input_tok, b, w, result in zip(input_tokens, better, worse, results):
    #             # print(input_tok, b, w)
    #             index = None
    #             if b in input_tok and w in input_tok: # better in inputs / worse in inputs
    #                 index = 0
    #             if b in input_tok and w not in input_tok: # better in inputs / worse not in inputs
    #                 index = 1
    #             if b not in input_tok and w in input_tok: # better not in inputs / worse in inputs
    #                 index = 2
    #             if b not in input_tok and w not in input_tok: # better not in inputs / worse not in inputs
    #                 index = 3

    #             better_probs[index].append(result["better_prob"])
    #             worse_probs[index].append(result["worse_prob"])
    #             logprob_diffs[index].append(result["logprob_diff"])
    #             better_ranks[index].append(result["better_rank"])
    #             worse_ranks[index].append(result["worse_rank"])
    #             rank_diffs[index].append((result["rank_diff"]))

    # logger.info("=================================================")

    # for x in [better_probs, worse_probs, logprob_diffs, better_ranks, worse_ranks, rank_diffs]:
    #     x.append(x[0] + x[1] + x[2] + x[3])

    # for i in range(5):
    #     text = [
    #         "better in inputs / worse in inputs",
    #         "better in inputs / worse not in inputs",
    #         "better not in inputs / worse in inputs",
    #         "better not in inputs / worse not in inputs",
    #         "Total"
    #     ][i]
    #     better_prob = better_probs[i]
    #     worse_prob = worse_probs[i]
    #     logprob_diff = logprob_diffs[i]
    #     better_rank = better_ranks[i]
    #     worse_rank = worse_ranks[i]
    #     rank_diff = rank_diffs[i]

    #     logger.info("")
    #     logger.info(f"<< {text} >>")
    #     if i != 4:
    #         logger.info(f"-> total {len(better_prob)} samples ( total {len(better_probs[-1])}, {len(better_prob)/len(better_probs[-1])*100} % )")
    #     if len(better_prob) == 0:
    #         continue

    #     logger.info("")
    #     logger.info("better token probability: If an appropriate worse token exists, how does the token probability change?")
    #     logger.info(f"Total average: {sum(better_prob) / len(better_prob)}")
    #     better_prob = sorted(better_prob)
    #     logger.info(f"min: {better_prob[0]}")
    #     logger.info(f"Q1 : {better_prob[1 * len(better_prob)//4]}")
    #     logger.info(f"Q2 : {better_prob[2 * len(better_prob)//4]}")
    #     logger.info(f"Q3 : {better_prob[3 * len(better_prob)//4]}")
    #     logger.info(f"max: {better_prob[-1]}")
    #     logger.info("")
    #     logger.info("worse probability: If an appropriate worse token exists, how does the token probability change?")
    #     logger.info(f"Total average: {sum(worse_prob) / len(worse_prob)}")
    #     worse_prob = sorted(worse_prob)
    #     logger.info(f"min: {worse_prob[0]}")
    #     logger.info(f"Q1 : {worse_prob[1 * len(worse_prob)//4]}")
    #     logger.info(f"Q2 : {worse_prob[2 * len(worse_prob)//4]}")
    #     logger.info(f"Q3 : {worse_prob[3 * len(worse_prob)//4]}")
    #     logger.info(f"max: {worse_prob[-1]}")

    #     logger.info("")
    #     logger.info("branching factor = logp(better) - logp(worse)")
    #     logger.info(f"Total average: {sum(logprob_diff) / len(logprob_diff)}")
    #     logprob_diff = sorted(logprob_diff)
    #     logger.info(f"min: {logprob_diff[0]}")
    #     logger.info(f"Q1 : {logprob_diff[1 * len(logprob_diff)//4]}")
    #     logger.info(f"Q2 : {logprob_diff[2 * len(logprob_diff)//4]}")
    #     logger.info(f"Q3 : {logprob_diff[3 * len(logprob_diff)//4]}")
    #     logger.info(f"max: {logprob_diff[-1]}")
    #     positive_values = sum([(1 if x>0 else 0) for x in logprob_diff])
    #     logger.info(f"positive_value: {positive_values} ({positive_values / len(logprob_diff) * 100} %)")
        
    #     logger.info("")
    #     logger.info("better rank = rank of the better token among all vocab(lower number -> higher rank)")
    #     logger.info(f"Total average: {sum(better_rank) / len(better_rank)}")
    #     better_rank = sorted(better_rank)
    #     logger.info(f"min: {better_rank[0]}")
    #     logger.info(f"Q1 : {better_rank[1 * len(better_rank)//4]}")
    #     logger.info(f"Q2 : {better_rank[2 * len(better_rank)//4]}")
    #     logger.info(f"Q3 : {better_rank[3 * len(better_rank)//4]}")
    #     logger.info(f"max: {better_rank[-1]}")
    #     logger.info("")
    #     logger.info("worse rank = rank of the worse token among all vocab (lower number -> higher rank)")
    #     logger.info(f"Total average: {sum(worse_rank) / len(worse_rank)}")
    #     worse_rank = sorted(worse_rank)
    #     logger.info(f"min: {worse_rank[0]}")
    #     logger.info(f"Q1 : {worse_rank[1 * len(worse_rank)//4]}")
    #     logger.info(f"Q2 : {worse_rank[2 * len(worse_rank)//4]}")
    #     logger.info(f"Q3 : {worse_rank[3 * len(worse_rank)//4]}")
    #     logger.info(f"max: {worse_rank[-1]}")
    #     logger.info("")
    #     logger.info("rank difference, negative value: better token ranked higher than worse token")
    #     logger.info(f"Total average: {sum(rank_diff) / len(rank_diff)}")
    #     rank_diff = sorted(rank_diff)
    #     logger.info(f"min: {rank_diff[0]}")
    #     logger.info(f"Q1 : {rank_diff[1 * len(rank_diff)//4]}")
    #     logger.info(f"Q2 : {rank_diff[2 * len(rank_diff)//4]}")
    #     logger.info(f"Q3 : {rank_diff[3 * len(rank_diff)//4]}")
    #     logger.info(f"max: {rank_diff[-1]}")

    #     logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--test_data", required=False, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference")

    # Checkpoint configs
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--base_model", required=False, default="bart", choices=["bart", "t5"], help="Base model to train. If using `from_checkpoint`, you do not need to specify this option.")
    
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")


    args = parser.parse_args()
    
    if args.test_data is None:
        args.test_data = os.path.join(args.model_store_path, args.model_postfix, "partial_utility_estimation.json")

    main(args)
