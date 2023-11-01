from collections import OrderedDict
from tqdm import tqdm
from utils.utils import get_flat_document_repr
import torch
import os

from tools.run_splade.utils.score import run_splade_batch

def main(args):

    ## Run Splade
    from tools.run_splade.utils.score import load_model
    import torch
    model, tokenizer, reverse_voc = load_model('naver/splade-cocondenser-ensembledistil')
    device = torch.device('cuda')
    model.to(device)
    
    ## Load corpus
    from utils.utils import load_json_corpus
    corpus = load_json_corpus(corpus_path=args.corpus_path)

    ## Run Splade
    print()
    corpus_importance = OrderedDict()
    with torch.no_grad():
        doc_ids = list(corpus.keys())
        docs = [corpus[doc_id] for doc_id in doc_ids]
        num_docs = len(docs)
        for i in tqdm(range(0, num_docs, args.batch_size)):
            ids_batch = doc_ids[i:i+args.batch_size]
            docs_batch = docs[i:i+args.batch_size]
            docs_text_batch = [get_flat_document_repr(doc) for doc in docs_batch]
            result_batch = run_splade_batch(model, tokenizer, reverse_voc, doc_batch=docs_text_batch, device=device)
            
            for doc_id, doc, (tokens, scores) in zip(ids_batch, docs_batch, result_batch):
                corpus_importance[doc_id] = doc
                scores = list(map(float, scores))
                corpus_importance[doc_id]["splade"] = list(zip(tokens, scores))

    ## Save results
    save_corpus_splade(corpus=corpus_importance, path=os.path.join(args.output_directory, 'corpus.splade.jsonl'))
    
    return

import json
from typing import List, Tuple
def load_corpus_splade(path:str):
    print()
    logger.info(f'Loading corpus with SPLADE scores from {path}')
    splade_results = OrderedDict()
    with open(path) as fIn:
        for line in fIn:
            content = json.loads(line)
            splade_results[content["_id"]]:List[Tuple[str, float]] = content["splade"]
    return splade_results
def save_corpus_splade(corpus, path:str):
    with open(path, 'w') as fOut:
        for doc_id, doc in corpus.items():
            fOut.write(json.dumps({"_id": doc_id, **doc})+"\n")
    print(f"\n\t{path}\n")
    


from utils.logger import logger
if __name__=='__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_directory", type=str, required=True)
    args = parser.parse_args()

    ## Check whether the result file already exists.
    if os.path.exists(os.path.join(args.output_directory, 'corpus.splade.jsonl')):
        logger.info(f"#> Skip for {args.dataset} dataset. Results already exist: {os.path.join(args.output_directory, 'corpus.splade.jsonl')}")
        exit()

    import time
    start_time = time.time()
    
    ## Save input arguments and python script
    from utils.utils import save_data_to_reproduce_experiments
    import sys
    save_data_to_reproduce_experiments(output_directory=args.output_directory, path_to_python_script=__file__, input_arguments=args, prefix=os.path.basename(__file__), argv=sys.argv)


    ## Run Splade
    print('\n\n')
    logger.info(f'\n#> Results will be saved at {os.path.join(args.output_directory, "corpus.splade.jsonl")}')
    main(args)
    print('\n\n')
    

    ## Save elapsed time
    from utils.utils import save_elapsed_time
    save_elapsed_time(output_directory=args.output_directory, start_time=start_time, outpath_prefix=os.path.basename(__file__)+"__")




    