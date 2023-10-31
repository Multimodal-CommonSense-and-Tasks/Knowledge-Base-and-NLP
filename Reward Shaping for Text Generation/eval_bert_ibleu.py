from argparse import ArgumentParser
import json
import pickle
from tqdm import tqdm
import os, sys
import logging

import torch
from scipy.stats import spearmanr

from transformers import AutoTokenizer

from model.arguments import EvaluationArguments
from model.metrics import bert_ibleu #, bert_ibleu_fluency

def main(args):
    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    eval_postfix = args.eval_postfix
    # try:
    #     eval_postfix = args.eval_file.replace("result", "")
    # except:
    #     eval_postfix = args.eval_file
    # eval_postfix = eval_postfix.replace(".json", "")

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, f"eval_BERT-iBLEU_{eval_postfix}.log")):
            os.remove(os.path.join(log_path, f"eval_BERT-iBLEU_{eval_postfix}.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, f"eval_BERT-iBLEU_{eval_postfix}.log"))
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

    rpath = os.path.join(args.model_store_path, args.model_postfix, f'{args.eval_file[:-5]}.pkl')
    print(rpath)

    reference = []
    outputs = []
    for r in result:
        reference.append(r["source"])
        if args.quick:
            outputs.append([r["outputs"][0]])
        else:
            outputs.append(r["outputs"])
    
    if args.fluency:
        # metric_class = bert_ibleu_fluency
        metric_class = bert_ibleu
    else:
        metric_class = bert_ibleu
    
    eval_args = EvaluationArguments(use_sacre=args.use_sacre, use_smoothing=args.use_smoothing, fluency_hard_threshold=args.fluency_hard_threshold)
    metric = metric_class(eval_args)

    # Obtain scores
    bert_ibleu_score, bert, bleu = metric(reference, None, outputs, eval=True)

    # How frequently does the model copy the input?
    first_beam = [beam[0] for beam in outputs]
    mod_reference = [x.replace("paraphrase: ", "") for x in reference] # remove T5 headers from reference if present
    first_beam_eq_input = sum([(1 if x==y else 0) for x, y in zip(mod_reference, first_beam)])
    first_beam_eq_ratio = first_beam_eq_input / len(reference) * 100

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("BERT-score")
    logger.info(f"Total average: {torch.mean(bert).item()}")
    logger.info(f"BERT-score per beam:")
    for beam_id, score in enumerate(torch.mean(bert, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    bert_sorted, _ = torch.sort(bert, dim=-1)
    logger.info(f"BERT-score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(bert_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")


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

    logger.info("")
    logger.info("BERT-iBLEU score")
    logger.info(f"Total average: {torch.mean(bert_ibleu_score)}")
    logger.info(f"BERT-iBLEU score per beam:")
    for beam_id, score in enumerate(torch.mean(bert_ibleu_score, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    bert_ibleu_sorted, _ = torch.sort(bert_ibleu_score, dim=-1)
    logger.info(f"BERT-iBLEU score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(bert_ibleu_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    
    logger.info("")
    logger.info("Repeated original sentence")
    logger.info("  (in the first beam of model output)")
    logger.info(f"{first_beam_eq_input} / {len(reference)} ({first_beam_eq_ratio} %)")

    logger.info("=================================================")

    with open(rpath, 'wb') as f:
        pickle.dump(bert_ibleu_score.cpu().numpy(), f)
        pickle.dump(bert.cpu().numpy(), f)
        pickle.dump(bleu.cpu().numpy(), f)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--eval_file", required=False, default='result.json', help="Name of the result file(generated by inference.py)")
    parser.add_argument("--eval_postfix", type=str, default=None)

    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")
    parser.add_argument("--quick", action="store_true", help="")
    parser.add_argument("--fluency", action="store_true", help="When set, use bert-ibleu-ppl instead of bert-ibleu")
    parser.add_argument("--use_sacre", action="store_true", help="")
    parser.add_argument("--use_smoothing", action="store_true", help="")
    parser.add_argument("--fluency_hard_threshold", action="store_true", help="")

    args = parser.parse_args()
    main(args)
