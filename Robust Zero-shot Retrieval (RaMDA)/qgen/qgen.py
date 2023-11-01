## Codes are adopted from GPL
from typing import Dict, List, Tuple, Set, Any
from beir.datasets.data_loader import GenericDataLoader
from beir.generation.models import QGenModel
from beir.generation import QueryGenerator
import os
import argparse
import copy

import time
import json

from utils.logger import logger
from utils.constant.constant_segmentation import SEGMENT_SEP
from utils.utils import load_json_corpus

import torch
from transformers import AutoTokenizer
from qgen.generator import ResidualT5ForConditionalGeneration
class ResidualQGenModel(QGenModel):
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = ResidualT5ForConditionalGeneration.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)

        #! We only use top-k sampling for generation.
        self.model.config.num_beams = 1

    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None) -> List[str]:
        ## Corpus contains a batch of documents with the format of {'title': '...', 'text': '...', 'splade' (List[Tuple[str, float]])}
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        splade_vocabulary:List[List[Tuple[str, float]]] = [doc["splade"] for doc in corpus]
        lm_head_size:int = self.model.lm_head.weight.shape[0]
        splade_token_logits = torch.zeros(len(splade_vocabulary), lm_head_size, dtype=torch.float32, requires_grad=False)
        for example_index, qtset in enumerate(splade_vocabulary):
            splade_token_ids:List[int] = self.tokenizer.convert_tokens_to_ids([_[0] for _ in qtset])
            splade_token_scores:List[float] = [_[1] for _ in qtset]
            splade_token_logits[example_index, splade_token_ids] = torch.tensor(splade_token_scores)

        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            if not temperature:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage,  # 1

                    #!@ custom: 
                    splade_token_logits=splade_token_logits.to(self.model.device),
                )
            else:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    temperature=temperature,
                    num_return_sequences=ques_per_passage,  # 1
                
                    #!@ custom: 
                    splade_token_logits=splade_token_logits.to(self.model.device),
                )

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)
def qgen(
    corpus_path,
    corpus_segments:Dict[str, str],
    splade_vocabulary:Dict[str, List[Tuple[str, float]]],
    output_dir,
    generator_name_or_path="google/t5-v1_1-base",
    ques_per_passage=3,
    bsz=32,
    qgen_prefix="qgen", # prefix is not used.
):

    ## Adapt format: doc_id -> {"title": "", "text": "..."}
    ##              in addition, we add splade vocabulary to the text
    corpus:Dict[str, Dict[str, str]] = {
        doc_id: {"title": "", "text": doc_concat, "splade": splade_vocabulary[doc_id]}
        for doc_id, doc_concat in corpus_segments.items()
    }

    #### question-generation model loading
    generator = QueryGenerator(model=ResidualQGenModel(generator_name_or_path))

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = qgen_prefix

    #### Generating 3 questions per passage.
    #### Reminder the higher value might produce lots of duplicates
    #### Generate queries per passage from docs in corpus
    try:
        os.makedirs(output_dir, exist_ok=True)
        generator.generate(
            corpus,
            output_dir=output_dir,
            ques_per_passage=ques_per_passage,
            prefix=prefix, # prefix is not used.
            batch_size=bsz,
            # save_after=1, #?@ debugging
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                f"CUDA out of memory during query generation "
                f"(queries_per_passage: {ques_per_passage}, batch_size_generation: {bsz}). "
                f"Please try smaller `queries_per_passage` and/or `batch_size_generation`."
            )
        else:
            print(e)
        exit()

    if not os.path.exists(os.path.join(output_dir, "corpus.jsonl")):
        os.system(f"cp {corpus_path} {output_dir}/")
    
    print(f'\n\t#> generated queries are saved in {output_dir}\n')
    return True

def main(args):

    assert os.path.exists(args.corpus_path)
    from utils.utils import load_json_corpus, get_flat_document_repr
    _corpus:Dict[str, Dict[str, Any]] = load_json_corpus(args.corpus_path)
    corpus_segments:Dict[str, str] = {
        doc_id: get_flat_document_repr(doc)
        for doc_id, doc in _corpus.items()
    }

    ## Load splade vocabulary
    from tools.run_splade.utils.loader import load_splade_vocabulary
    splade_vocabulary: Dict[str, List[Tuple[str, float]]] = load_splade_vocabulary(args.splade_path)
    
    ## Share splade vocabulary across segments
    logger.info(f"Sharing splade vocabulary across segments")
    from utils.constant.constant_segmentation import doc_id_dict
    splade_vocabulary_segments:Dict[str, List[Tuple[str, float]]] = {}
    for segment_id in corpus_segments:
        doc_id:str = doc_id_dict(segment_id=segment_id)
        splade_vocabulary_segments[segment_id] = splade_vocabulary[doc_id]
    logger.info(f"len(corpus_segments)={len(corpus_segments)}, len(splade_vocabulary)={len(splade_vocabulary)}, len(splade_vocabulary_segments)={len(splade_vocabulary_segments)}")
    splade_vocabulary = splade_vocabulary_segments
    
    ## Generate pseudo-queries
    # Refer to https://github.com/beir-cellar/beir/blob/bbcb245a981aa3001ff217d65cecbb27073e13c0/beir/generation/generate.py
    is_successfully_completed = qgen(
        corpus_path=args.corpus_path, 
        output_dir=args.output_dir,
        generator_name_or_path=args.generator_name_or_path,
        ques_per_passage=args.ques_per_passage,
        bsz=args.bsz,
        corpus_segments=corpus_segments,
        splade_vocabulary=splade_vocabulary,
    )
    if is_successfully_completed:
        with open(args.output_dir + '/config.json', 'w') as outfile:
            config = vars(args)
            config['elapsed'] = time.time() - start_time
            outfile.write(json.dumps(config, indent=4)+'\n')
        
        with open(args.output_dir + '/successfully_completed.txt', 'w') as outfile:
            outfile.write('done!\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--elapselog_path", type=str, default="", help="/path/to/elapsed time log")
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--generator_name_or_path", required=True)
    parser.add_argument("--ques_per_passage", type=int, default=50)
    parser.add_argument("--bsz", type=int, default=8)
    
    parser.add_argument('--splade_path', type=str, required=True)

    args = parser.parse_args()

    import time
    start_time = time.time()

    ## Check if output file already exists
    if os.path.exists(args.output_dir + '/successfully_completed.txt'):
        logger.info(f"#> Results already exist at {args.output_dir + '/successfully_completed.txt'}")
        if args.overwrite:
            logger.info(f'We will overwrite results.')
        else:
            print()
            exit()

    ## Save input arguments and python script
    from utils.utils import save_data_to_reproduce_experiments
    import sys
    save_data_to_reproduce_experiments(output_directory=args.output_dir, path_to_python_script=__file__, input_arguments=args, prefix=os.path.basename(__file__), argv=sys.argv)

    ## Run main function
    main(args)

    ## Save elapsed time
    from utils.utils import save_elapsed_time
    if not args.elapselog_path:
        save_elapsed_time(output_directory=args.output_dir, start_time=start_time, outpath_prefix=os.path.basename(__file__)+"__")
    else:
        save_elapsed_time(output_directory=os.path.dirname(args.elapselog_path), start_time=start_time, outpath_prefix=os.path.basename(args.elapselog_path)+"."+os.path.basename(__file__)+"__")


    
    