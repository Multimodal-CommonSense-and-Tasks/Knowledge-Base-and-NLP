from typing import List, Dict, Union, Tuple
from itertools import chain
import torch
from argparse import ArgumentParser
import os
import json
import time

from sentence_transformers import SentenceTransformer, models as s_models

from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader

from utils.logger import logger
from utils.utils import load_pseudo_queries, save_input_arguments, get_flat_document_repr
from utils.utils import pad_and_stack_list_of_2d_embeddings
from utils.constant.constant_segmentation import SEGMENT_SEP
from utils.constant.constant_dataset import BEIR_DATASET

if __name__=='__main__':
    parser = ArgumentParser("")

    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument('--model_name', type=str, default="OpenMatch/cocodr-base-msmarco",
        help="Retriever. e.g., OpenMatch/cocodr-base-msmarco or OpenMatch/cocodr-large-msmarco.")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    parser.add_argument("--max_seq_length", type=int, default=128)
    
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--corpus_path', type=str)
    
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    
    st = time.time()

    ## Create output directory
    output_dir = args.output_dir
    outfile_path = os.path.join(output_dir, "doc_embs.pt") 
    print()
    if os.path.exists(outfile_path):
        logger.info(f"#> Results already exist at {outfile_path}")
        if args.overwrite:
            logger.info(f'We will overwrite results.')
        else:
            print()
            exit()
    logger.info(f"#> Results will be saved at {outfile_path}")
    print()
    os.makedirs(output_dir, exist_ok=True)

    ## Save input arguments into the output directory
    save_input_arguments(args=args, output_dir=output_dir, prefix=os.path.basename(outfile_path))

    ## Load BEIR dataset
    print()
    logger.info(f'#> Load dataset from {args.data_path}')
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split="test")
    except Exception as e:
        from utils.utils import load_json_corpus
        if args.data_path and os.path.exists(args.data_path):
            corpus = GenericDataLoader(data_folder=args.data_path).load_corpus()
        else:
            assert args.corpus_path and os.path.exists(args.corpus_path), f"args.corpus_path={args.corpus_path}"
            corpus = load_json_corpus(args.corpus_path)

    ## Load encoder
    encoder: SentenceTransformer = SentenceTransformer(model_name_or_path=args.model_name)
    dim = encoder[1].word_embedding_dimension
    encoder[1] = s_models.Pooling(dim, args.pooling)
    encoder[0].max_seq_length = args.max_seq_length
    print();logger.info(f'encoder=\n{encoder}');print()

    ## Encode sentences
    print()
    sentence_ids:List[str] = list(corpus.keys())
    sentences:List[str] = [get_flat_document_repr(doc=corpus[doc_id]) for doc_id in sentence_ids]
    logger.info(f"#> Cache embeddings for [title, body]")
    embeddings = encoder.encode(sentences=sentences, batch_size=args.batch_size, convert_to_tensor=False, convert_to_numpy=True)
    logger.info(f'#> Done!')
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    logger.info(f'#> embeddings {embeddings.size()}')

    print()
    cached_embeddings = {idx: embeddings[i] for i, idx in enumerate(sentence_ids)}
    torch.save(cached_embeddings, outfile_path)
    logger.info(f'\n\t{outfile_path}')
    
    ## Save elapsed time
    print()
    elapsed_path:str = os.path.join(output_dir, f"{os.path.basename(outfile_path)}__elapsed.txt")
    with open(elapsed_path, 'w') as fOut:
        ed = time.time()
        fOut.write(f"{ed-st} seconds\n")
        fOut.write(f"{(ed-st)/60} minutes\n")
        fOut.write(f"{(ed-st)/3600} hours\n")
    logger.info(f'#> \t {elapsed_path}')

    print()
