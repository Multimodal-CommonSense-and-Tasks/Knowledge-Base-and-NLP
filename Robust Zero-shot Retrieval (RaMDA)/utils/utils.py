from typing import List, Dict, Tuple, Set, Callable, Union
import json
from collections import OrderedDict

from utils.logger import logger


## Save data to reproduce experiments
import shutil
def save_data_to_reproduce_experiments(output_directory, path_to_python_script, input_arguments, prefix:str="", argv:str=""):
    os.makedirs(output_directory, exist_ok=True)
    # Copy this script to output directory
    shutil.copy(path_to_python_script, output_directory)
    if argv:
        __python_script_file_path = os.path.join(output_directory, os.path.basename(path_to_python_script))
        with open(__python_script_file_path, 'a') as fOut:
            fOut.write("\n\n# Script was called via:\n#python " + " ".join(argv))
        logger.info(f"Write sys.argv into {__python_script_file_path}")
    # Save input arguments to output directory
    save_input_arguments(args=input_arguments, output_dir=output_directory, prefix=prefix)

## Save elapsed time
def save_elapsed_time(output_directory, start_time, outpath_prefix:str=""):
    print()
    elapsed_path:str = os.path.join(output_directory, f"{outpath_prefix}elapsed.txt")
    with open(elapsed_path, 'w') as fOut:
        end_time = time.time()
        fOut.write(f"{end_time-start_time} seconds\n")
        fOut.write(f"{(end_time-start_time)/60} minutes\n")
        fOut.write(f"{(end_time-start_time)/3600} hours\n")
    logger.info(f'#> \t {elapsed_path}')
    print()

## Load SentenceTransformer encoder
from sentence_transformers import models as s_models
from sentence_transformers import SentenceTransformer
def load_encoder(model_name:str, max_seq_length:int, pooling:str):
    model = SentenceTransformer(model_name) 
    print(f'#> Load model from SentenceTransformer')
    model[1] = s_models.Pooling(model[1].word_embedding_dimension, pooling)
    logger.info(f'Set pooling={pooling}')
    model[0].max_seq_length=max_seq_length
    logger.info(f'Set max_seq_length={max_seq_length}')
    logger.info(f'model={model}')
    return model

## Load corpus
def load_json_corpus(corpus_path):
    logger.info(f'#> Load {corpus_path}')
    corpus = OrderedDict()
    with open(corpus_path) as fIn:
        _manual_keys = {"_id", "title", "text", "metadata"}
        for line in fIn:
            content = json.loads(line)
            corpus[content["_id"]] = {
                "title": content.get("title", ""), 
                "text": content.get("text", ""), 
                "metadata": content.get("metadata", {}),
                **{kw:arg for kw, arg in content.items() if kw not in _manual_keys}
            }
    logger.info(f'\t {len(corpus)} unique documents are loaded.')
    return corpus
def save_json_corpus(corpus, outfile_path):
    logger.info(f'#> Save corpus (size:{len(corpus)}) into {outfile_path}')
    with open(outfile_path, 'w') as fOut:
        for doc_id, doc in corpus.items():
            fOut.write(json.dumps({"_id": doc_id, **doc})+"\n")

## Load queries
def load_json_queries(queries_path):
    logger.info(f'#> Load {queries_path}')
    queries = OrderedDict()
    with open(queries_path) as fIn:
        for line in fIn:
            content = json.loads(line)
            queries[content["_id"]] = content["text"].strip()
    logger.info(f'\t {len(queries)} unique queries are loaded.')
    return queries

## Load qrels: qid -> {doc_id -> label(int)}
def load_tsv_qrels(qrels_path):
    print()
    logger.info(f'#> Load {qrels_path}')
    qrels = OrderedDict()
    with open(qrels_path) as fIn:
        fIn.readline() # column names: ["query-id", "corpus-id", "score"]
        for line in fIn:
            qid, doc_id, label = line.strip().split('\t')
            qrels[qid] = qrels.get(qid, {})
            qrels[qid][doc_id] = int(label)
    logger.info(f'\t Annotations for {len(qrels)} unique queries are loaded.')
    return qrels
## Reverse qrels: doc_id -> {qid -> label(int)}
def reverse_qrels(qrels:Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    print()
    logger.info("Reverse qrels: doc_id -> {qid(str) -> label(int)}")
    drels:Dict[str, Dict[str, int]] = OrderedDict()
    for qid, annotations in qrels.items():
        for doc_id, rel in annotations.items():
            drels[doc_id] = drels.get(doc_id, {})
            drels[doc_id][qid] = rel
    logger.info(f"#> The number of annotated documents: {len(drels)}")
    return drels


## Load ranking
import pandas as pd
def load_tsv_ranking(ranking_path, depth:int=1000, return_dataframe:bool=False, include_rank:bool=False):
    print()
    logger.info(f'#> Load {ranking_path}')
    ranking:Dict[str, Dict[str, float]] = OrderedDict()
    with open(ranking_path) as fIn:
        for line in fIn:
            qid, doc_id, rank, score = line.strip().split()
            if int(rank) <= depth:
                ranking[qid] = ranking.get(qid, {})
                ranking[qid][doc_id] = float(score)
    logger.info(f'\t rankings on {len(ranking)} unique queries are loaded.')
    
    n_docs_per_query = [len(ranking[qid]) for qid in ranking]
    logger.info(f'\t the number of ranked documents is [min {min(n_docs_per_query)}, max {max(n_docs_per_query)}, mean {sum(n_docs_per_query)/len(n_docs_per_query):.1f}].')
    
    if return_dataframe:
        if not include_rank:
            qid_list, doc_id_list, score_list = [], [], []
            for qid, topk_docs in ranking.items():
                for doc_id, score in topk_docs.items():
                    qid_list.append(qid)
                    doc_id_list.append(doc_id)
                    score_list.append(score)
            ranking = pd.DataFrame({
                'qid':qid_list,
                'docno':doc_id_list,
                'score':score_list,
            })
        else:
            qid_list, doc_id_list, score_list, rank_list = [], [], [], []
            for qid, topk_docs in ranking.items():
                for rank, (doc_id, score) in enumerate(sorted(topk_docs.items(), key=lambda x: x[1], reverse=True)):
                    qid_list.append(qid)
                    doc_id_list.append(doc_id)
                    score_list.append(score)
                    rank_list.append(rank+1)
            ranking = pd.DataFrame({
                'qid':qid_list,
                'docno':doc_id_list,
                'score':score_list,
                'rank':rank_list,
            })
    return ranking
from collections import OrderedDict
def ranking_truncate(ranking, depth:int=1000):
    ranking_truncated = OrderedDict()
    for qid, topk_docs in ranking.items():
        ranking_truncated[qid] = OrderedDict(sorted(topk_docs.items(), key=lambda x: x[1], reverse=True)[:depth])
    return ranking_truncated

## Load pseudo-queries 
from utils.constant.constant_segmentation import SEGMENT_SEP
def load_pseudo_queries(pseudo_queries_path, max_num_pseudo_queries:int=-1) -> Dict[str, List[str]]:
    print()
    logger.info(f'#> Load {pseudo_queries_path}')
    if max_num_pseudo_queries != -1:
        logger.info(f' ... A list of pseudo-queries (PQs) will be truncated by the first-{max_num_pseudo_queries} PQs.')
    with open(pseudo_queries_path) as fIn:
        pseudo_queries:Dict[str, List[str]] = OrderedDict()
        for line in fIn:
            json_dict = json.loads(line)
            doc_id, list_of_pseudo_queries = iter(json_dict.items()).__next__()
            # if use_doc_id:
            #     doc_id:str = doc_id.split(SEGMENT_SEP)[0]
            if max_num_pseudo_queries != -1:
                list_of_pseudo_queries = list_of_pseudo_queries[:max_num_pseudo_queries]
            pseudo_queries[doc_id] = list_of_pseudo_queries
    logger.info(f'\t Pseudo-queries for {len(pseudo_queries)} documents (or segments) are loaded.')
    return pseudo_queries
def merge_pseudo_queries_over_segments(pseudo_queries):
    print()
    logger.info(f"#> Merge pseudo-queries over segments")
    # Convert segment IDs into document IDs
    corpus_pseudo_queries:Dict[str, List[List[str]]] = {}
    for segment_id, list_of_texts in pseudo_queries.items():
        doc_id:str = segment_id.split(SEGMENT_SEP)[0]
        corpus_pseudo_queries[doc_id] = corpus_pseudo_queries.get(doc_id, [])
        corpus_pseudo_queries[doc_id].append(list_of_texts)
    logger.info(f'\t Pseudo-queries for {len(corpus_pseudo_queries)} documents are loaded.')
    return corpus_pseudo_queries

## Load kNN
def load_knn(knn_path, knn_depth:int=9999) -> Dict[str, List[str]]:
    logger.info(f'#> Load {knn_path}')
    with open(knn_path) as fIn:
        knn:Dict[str, List[str]] = OrderedDict()
        for line in fIn:
            json_dict = json.loads(line)
            knn[json_dict["_id"]] = json_dict["kNN"][:knn_depth]
    logger.info(f'\t kNN for {len(knn)} documents are loaded.')
    return knn

## Get flat document repr
def get_flat_document_repr(doc:Dict, sep=" "):
    plain_text:str = doc["title"].strip() + sep + doc["text"].strip() \
        if "title" in doc and doc["title"].strip() else doc["text"].strip()
    return plain_text

## Save input arguments
def save_input_arguments(args, output_dir, prefix=""):
    if prefix:
        outfile_path = os.path.join(output_dir, f"{prefix}__input_arguments.json")
    else:
        outfile_path = os.path.join(output_dir, "input_arguments.json")
    print()
    logger.info(f'#> Save input arguments into {outfile_path}')
    if not isinstance(args, dict):
        input_arguments = vars(args) 
    else:
        input_arguments = args
    logger.info(f'\n{json.dumps(input_arguments, indent=4)}')
    with open(outfile_path, 'w') as fOut:
        fOut.write(f"{json.dumps(input_arguments, indent=4)}\n")
## Load input arguments
def load_input_arguments(path):
    return json.load(open(path))

## Pad and stack a list of 2d embeddings
import torch
def pad_and_stack_list_of_2d_embeddings(list_of_2d_embeddings:List[torch.Tensor]):
    l1 = len(list_of_2d_embeddings)
    l2 = max([len(embs) for embs in list_of_2d_embeddings])

    _sample_emb = list_of_2d_embeddings[0]
    dim, dtype, device = _sample_emb.size(1), _sample_emb.dtype, _sample_emb.device
    embeddings_3d = torch.zeros(l1, l2, dim, dtype=dtype, device=device)
    masks_3d = torch.zeros(l1, l2, dtype=torch.float, device=device)
    for i, embs in enumerate(list_of_2d_embeddings):
        embeddings_3d[i, :len(embs)] = embs
        masks_3d[i, :len(embs)] = 1.0

    return embeddings_3d, masks_3d

import os
import time
import traceback
class ElapsedTimeLogger():
    def __init__(self, outfile_path:str):
        self.start_time:float = time.time()
        self.outfile_path:str = outfile_path
    def log(self):
        elapsed_time:float = time.time() - self.start_time

        os.makedirs(os.path.dirname(self.outfile_path), exist_ok=True)
        with open(self.outfile_path, 'w') as outfile:
            outfile.write(f'{int(elapsed_time)}\n')
            
            hours = int(elapsed_time // 3600)
            _remaining_time = elapsed_time - hours*3600
            minutes = int(_remaining_time // 60)
            outfile.write(f'{hours} hours {minutes} minutes\n')
        
        print(f'#> elapsed time is logged at {self.outfile_path}')

    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False # uncomment to pass exception through
        self.log()
        return True

