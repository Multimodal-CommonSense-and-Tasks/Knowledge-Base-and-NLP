from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, MarianMTModel, AutoTokenizer

from model.model import ParaphraserBase as Paraphraser
from model.dataset import TextGenerationDataset

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
    device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

    # Init logger
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(args.result_path, "inference.log")):
            os.remove(os.path.join(args.result_path, "inference.log"))
    file_handler = logging.FileHandler(os.path.join(args.result_path, "inference.log"))
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
    result = []
    
    with torch.inference_mode():        
        for data in tqdm(test_dataloader):
            sources, targets = data
            outputs = model.generate(sources, sampling=args.sampling, length_penalty=args.lenpen)
            # hypos = [output[0] for output in outputs]
            sources_decode = base_tokenizer.batch_decode(sources, skip_special_tokens=True)
            targets_decode = base_tokenizer.batch_decode(targets, skip_special_tokens=True)

            for output, source, target in zip(outputs, sources_decode, targets_decode):
                result.append({
                    "source": source,
                    "target": target,
                    "outputs": output
                })
    
    if 'dev' in args.test_data:
        result_name = 'result_dev'
    elif 'train' in args.test_data:
        result_name = 'result_train'
    else:
        result_name = 'result'
    suffix = ''
    if args.sampling:
        suffix += '_sampling'
    if args.num_beams != 16:
        suffix += f'_bs{args.num_beams}'
    suffix += '.json'
    result_name = result_name + suffix

    with open(os.path.join(args.result_path, result_name), "w", encoding="UTF-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    import pickle
    with open(os.path.join(args.result_path, result_name[:-4] + 'pkl'), 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference")
    parser.add_argument("--lenpen", type=float, default=1.0, help="length penalty for beam search")

    # Checkpoint configs
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--model_postfix", default=None, help="Name for the model.")
    parser.add_argument("--base_model", required=False, default="bart", choices=["bart", "t5", "marian-ende", "marian-enfr", "marian-enro"], help="Base model to train. If using `from_checkpoint`, you do not need to specify this option.")

    parser.add_argument("--secure", action="store_true", help="")
    parser.add_argument("--sampling", action="store_true", help="")

    args = parser.parse_args()
    main(args)