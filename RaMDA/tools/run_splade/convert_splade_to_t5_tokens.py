from typing import List, Dict, Tuple, Set
import copy
import json
import linecache
import argparse
import math
import random
from tqdm import tqdm
import os
import torch
import numpy as np
from collections import OrderedDict

from utils.logger import logger


def get_mapping_between_vocab_dictionaries(
        splade_vocab_dict, target_vocab_dict, 
        splade_tokenizer_type:str, target_tokenizer_type:str,
        stat_outfile_path:str):
    print()
    ## Get mapping between dictionaries
    num_total = 0
    num_hit = 0
    num_miss = 0
    from tqdm import tqdm
    splade_to_target_vocab:Dict[str, str] = {}
    for splade_token in tqdm(splade_vocab_dict, desc="Mapping from SPLADE vocab to target vocab"):
        num_total += 1
        if not splade_token.startswith("##"): # e.g., donation
            target_token = f"▁{splade_token}" # e.g., ▁donation
            if target_token in target_vocab_dict:
                num_hit += 1
                splade_to_target_vocab[splade_token] = target_token
            else:
                num_miss += 1
        else: # e.g., ##ation
            target_token = f"{splade_token[2:]}" # e.g., ▁ation
            if target_token in target_vocab_dict:
                # print(f'{splade_token} -> {target_token}')
                num_hit += 1
                splade_to_target_vocab[splade_token] = target_token
            else:
                # print(f'{splade_token} -> miss')
                num_miss += 1
    stat = {
        "source_tokenizer": splade_tokenizer_type,
        "target_tokenizer": target_tokenizer_type,
        "num_total": num_total,
        "num_hit": num_hit,
        "num_miss": num_miss,
        "num_hit_percent": 100*num_hit/num_total,
        "num_miss_percent": 100*num_miss/num_total,
    }
    logger.info("SPLADE vocab -> target vocab mapping:")
    logger.info(f"\n{json.dumps(stat, indent=4)}")

    ## Save stat
    with open(stat_outfile_path, "w") as f:
        json.dump(stat, f, indent=4)
    logger.info(f"\t{stat_outfile_path}")

    return splade_to_target_vocab

from transformers.tokenization_utils_base import BatchEncoding
def convert_splade_tokens_to_target_tokens(
    splade_tokens:List[Tuple[str, float]],
    
    splade_encoding:Dict[str, BatchEncoding], 
    target_encoding:Dict[str, BatchEncoding],
    splade_to_target_vocab:Dict[str, str],

    use_dictionary:bool=True,
    use_document:bool=True,
):
    assert use_dictionary or use_document, "Either use_dictionary or use_document must be True"

    splade_token_converted:List[Tuple[str, float]] = []
    unique_target_tokens:Set[str] = set()
    
    ## Get SPLADE/target tokens from the given document
    splade_encoding_tokens:List[str] = splade_encoding.tokens()
    target_encoding_tokens:List[str] = target_encoding.tokens()

    def search_target_token_in_document(
        splade_token:str,
        source_doc_encoding, target_doc_encoding,
        source_doc_tokens:List[str], target_doc_tokens:List[str], 
    ):
        try:
            ## Get index of the SPLADE token in the list of tokens
            source_token_index = source_doc_tokens.index(splade_token)
        
            ## Get Char positions of the SPLADE token
            char_start, char_end = source_doc_encoding.token_to_chars(source_token_index)
            
            ## Get target token corresponding to the char positions
            target_token_index:int = target_doc_encoding.char_to_token(char_start)
            target_token:str = target_doc_tokens[target_token_index]

            return target_token

        except ValueError as e:
            return ""

    ## Prepare stats
    n_original = 0
    n_converted = 0
    n_hits = 0
    n_hits_in_doc = 0
    n_hits_in_vocab = 0
    n_duplicates = 0
    n_misses = 0
    
    for splade_token, splade_score in splade_tokens:
            
        target_token = ""

        ## --------------------------------------------------------------
        ## 1. First, search SPLADE token in the given document
        ## --------------------------------------------------------------
        if use_document:
            target_token = search_target_token_in_document(
                splade_token=splade_token,
                source_doc_encoding=splade_encoding, target_doc_encoding=target_encoding,
                source_doc_tokens=splade_encoding_tokens, target_doc_tokens=target_encoding_tokens, 
            )
            if target_token:
                n_hits_in_doc += 1
        
        if use_dictionary and not target_token:
            ## --------------------------------------------------------------
            ## 2. If the token was found in the given document
            ##    search the token from the vocabulary mapping dictionary
            ## --------------------------------------------------------------    
            target_token = splade_to_target_vocab.get(splade_token, "")
        
            if target_token:
                n_hits_in_vocab += 1
            else:
                ## --------------------------------------------------------------
                ## 3. If the token was found in neither the given document nor vocabulary mapping dictionary
                ##   drop the token
                ## --------------------------------------------------------------    
                pass

        ## Log stats
        n_original += 1
        if target_token:
            n_hits += 1
            if target_token in unique_target_tokens:
                n_duplicates += 1
        else:
            n_misses += 1

        ## Log the result
        if target_token not in unique_target_tokens:
            splade_token_converted.append((target_token, splade_score))
            unique_target_tokens.add(target_token)
            n_converted += 1

    return {
        "target_tokens": splade_token_converted,
        "stats": {
            "n_original": n_original,
            "n_converted": n_converted,
            "n_hits": n_hits,
            "n_hits_in_doc": n_hits_in_doc,
            "n_hits_in_vocab": n_hits_in_vocab,
            "n_misses": n_misses,
            "n_duplicates": n_duplicates,
        }
    }
        

def main(args):

    ## Sanity check: args.outfile_path must end with ".jsonl"
    outfile_path_postfix = ".jsonl"
    assert args.outfile_path.endswith(outfile_path_postfix), f"args.outfile_path must end with '{outfile_path_postfix}'"

    ## Load corpus
    from utils.utils import load_json_corpus
    corpus = load_json_corpus(args.corpus_path)

    ## Instantiate tokenizer
    from transformers import AutoTokenizer
    print()
    logger.info(f"Loading T5 tokenizer ...")
    target_tokenizer_type = "google/t5-v1_1-base" # All T5 variants share the same vocabulary
    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_type) 
    logger.info(f"Loading SPLADE tokenizer ...")
    splade_tokenizer_type = "bert-base-uncased"
    splade_tokenizer = AutoTokenizer.from_pretrained(splade_tokenizer_type)
    target_vocab_dict = target_tokenizer.get_vocab()
    splade_vocab_dict = splade_tokenizer.get_vocab()
    logger.info(f"\t Target tokenizer: {len(target_vocab_dict)} tokens")
    logger.info(f"\t SPLADE tokenizer: {len(splade_vocab_dict)} tokens")

    ## Convert SPLADE tokens to target tokens
    """
    1. First, identify words that correspond to each SPLADE token
    2. Use target token for the word
    3. If the SPLADE token does not correspond to any word, use mapping between vocabulary dictionaries
    """
    ## -- Get mapping between dictionaries
    splade_to_target_vocab:Dict[str, str] = \
        get_mapping_between_vocab_dictionaries(
        splade_vocab_dict, target_vocab_dict, 
        splade_tokenizer_type=splade_tokenizer_type, target_tokenizer_type=target_tokenizer_type,
        stat_outfile_path=args.outfile_path[:-len(outfile_path_postfix)]+".vocab_mapping_stat")
    # splade_to_target_vocab: splade_token -> target_token

    ## Main loop: Convert SPLADE tokens to target tokens
    print()
    stats_all = {
        "n_original": [],
        "n_converted": [],
        "n_hits": [],
        "n_hits_in_doc": [],
        "n_hits_in_vocab": [],
        "n_misses": [],
        "n_duplicates": [],
    }

    ## Load SPLADE vocabulary
    # -- check the number of lines in ``args.splade_path``, using linecache
    import linecache
    num_lines = len(linecache.getlines(args.splade_path))
    from utils.utils import get_flat_document_repr
    with open(args.splade_path) as fIn, open(args.outfile_path, "w") as fOut:
        for line in tqdm(fIn, desc=f"Converting SPLADE tokens to target tokens ...", total=num_lines):
            
            content = json.loads(line)
            doc_id:str = content["_id"]
            splade_tokens: List[Tuple[str, float]] = content["splade"]
            # splade_tokens:List[Tuple[str, float]]

            ## Encode document
            doc_text:str = get_flat_document_repr(corpus[doc_id])
            if args.lower_case:
                doc_text = doc_text.lower()
            # -- using target tokenizer
            target_encoding = target_tokenizer(doc_text)
            # -- using SPLADE tokenizer
            splade_encoding = splade_tokenizer(doc_text)

            ## Convert SPLADE tokens to target tokens
            results = convert_splade_tokens_to_target_tokens(
                splade_tokens, 
                splade_encoding, target_encoding, splade_to_target_vocab,
                use_dictionary=args.use_dictionary, use_document=args.use_document,
            )
            target_tokens:List[Tuple[str, float]] = results["target_tokens"]
            stats:Dict[str, Dict[str, int]] = results["stats"]
            ## Update statistics
            stats_all["n_original"].append(stats["n_original"])
            stats_all["n_converted"].append(stats["n_converted"])
            stats_all["n_hits"].append(stats["n_hits"])
            stats_all["n_hits_in_doc"].append(stats["n_hits_in_doc"])
            stats_all["n_hits_in_vocab"].append(stats["n_hits_in_vocab"])
            stats_all["n_misses"].append(stats["n_misses"])
            stats_all["n_duplicates"].append(stats["n_duplicates"])

            ## Save converted SPLADE vocabulary
            content = {"_id": doc_id, "splade": target_tokens}
            json_line = json.dumps(content)
            fOut.write(f"{json_line}\n")
    
    ## Report statistics
    print()
    import numpy as np
    stats_summary = {
        "size": {
            "n_documents": len(stats_all["n_original"]),
            "original_vocab_size": {
                "min": float(np.min(stats_all["n_original"])),
                "max": float(np.max(stats_all["n_original"])),
                "mean": float(np.mean(stats_all["n_original"])),
                "median": float(np.median(stats_all["n_original"])),
            },
            "converted_vocab_size": {
                "min": float(np.min(stats_all["n_converted"])),
                "max": float(np.max(stats_all["n_converted"])),
                "mean": float(np.mean(stats_all["n_converted"])),
                "median": float(np.median(stats_all["n_converted"])),
            },
            "conversion_ratio": {
                "min": float(np.min([x/y for x, y in zip(stats_all["n_converted"], stats_all["n_original"]) if y > 0])),
                "max": float(np.max([x/y for x, y in zip(stats_all["n_converted"], stats_all["n_original"]) if y > 0])),
                "mean": float(np.mean([x/y for x, y in zip(stats_all["n_converted"], stats_all["n_original"]) if y > 0])),
                "median": float(np.median([x/y for x, y in zip(stats_all["n_converted"], stats_all["n_original"]) if y > 0])),
            },
        },
        "micro-average": {
            "n_hits": sum(stats_all["n_hits"])/sum(stats_all["n_original"]),
            "n_hits_in_doc": sum(stats_all["n_hits_in_doc"])/sum(stats_all["n_original"]),
            "n_hits_in_vocab": sum(stats_all["n_hits_in_vocab"])/sum(stats_all["n_original"]),
            "n_misses": (sum(stats_all["n_misses"]))/sum(stats_all["n_original"]),
            "n_duplicates": (sum(stats_all["n_duplicates"]))/sum(stats_all["n_original"]),
            "n_misses+dup": (sum(stats_all["n_misses"])+sum(stats_all["n_duplicates"]))/sum(stats_all["n_original"]),
        },
        "macro-average": {
            "n_hits": float(np.mean([x/y for x,y in zip(stats_all["n_hits"], stats_all["n_original"]) if y>0])),
            "n_hits_in_doc": float(np.mean([x/y for x,y in zip(stats_all["n_hits_in_doc"], stats_all["n_original"]) if y>0])),
            "n_hits_in_vocab": float(np.mean([x/y for x,y in zip(stats_all["n_hits_in_vocab"], stats_all["n_original"]) if y>0])),
            "n_misses": float(np.mean([x/y for x,y in zip(stats_all["n_misses"], stats_all["n_original"]) if y>0])),
            "n_duplicates": float(np.mean([x/y for x,y in zip(stats_all["n_duplicates"], stats_all["n_original"]) if y>0])),
            "n_misses+dup": float(np.mean([(x+z)/y for x,z,y in zip(stats_all["n_misses"], stats_all["n_duplicates"], stats_all["n_original"]) if y>0])),
        },
    }
    logger.info(f"Statistics:")
    logger.info(f"{json.dumps(stats_summary, indent=4)}")
    ## Save statistics
    stats_outfile_path = args.outfile_path[:-len(outfile_path_postfix)]+".conversion.stats.json"
    logger.info(f"Saving statistics to {stats_outfile_path} ...")
    with open(stats_outfile_path, "w") as f:
        json.dump(stats_summary, f, indent=4)

    logger.info(f"Done.")
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile_path', required=True)
    parser.add_argument('--overwrite', action='store_true')
    ## Data path
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument('--splade_path', required=True)
    ## Instruction
    parser.add_argument('--use_dictionary', action='store_true')
    parser.add_argument('--use_document', action='store_true')
    parser.add_argument('--lower_case', action='store_true')
    args = parser.parse_args()

    import time
    start_time = time.time()

    ## Check if output file already exists
    args.output_directory:str = os.path.dirname(args.outfile_path)
    if os.path.exists(args.outfile_path):
        logger.info(f"#> Results already exist at {args.outfile_path}")
        if args.overwrite:
            logger.info(f'We will overwrite results.')
        else:
            print()
            exit()
    logger.info(f"#> Results will be saved at {args.outfile_path}")

    ## Save input arguments and python script
    from utils.utils import save_data_to_reproduce_experiments
    import sys
    save_data_to_reproduce_experiments(output_directory=args.output_directory, path_to_python_script=__file__, input_arguments=args, prefix=os.path.basename(__file__), argv=sys.argv)

    ## Run main function
    main(args)
    logger.info(f"\t{args.outfile_path}")

    ## Save elapsed time
    from utils.utils import save_elapsed_time
    save_elapsed_time(output_directory=args.output_directory, start_time=start_time, outpath_prefix=os.path.basename(__file__)+"__")



