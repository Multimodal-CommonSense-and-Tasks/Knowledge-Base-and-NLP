from typing import Dict, List, Tuple, Set, Union, Optional, Any
from tqdm import tqdm
import linecache
from collections import OrderedDict
import json
import os

from utils.logger import logger

def load_splade_vocabulary(splade_path:str) -> Dict[str, List[Tuple[str, float]]]:
    print()
    predicted_query_term_set:Dict[str, List[Tuple[str, float]]] = OrderedDict()
    nlines:int = len(linecache.getlines(splade_path))
    with open(splade_path) as fIn:
        for line in tqdm(fIn, desc=f"#> Load SPLADE vocab", total=nlines):
            content = json.loads(line)
            predicted_query_term_set[content["_id"]]:List[Tuple[str, float]] = content["splade"]
    logger.info(f'\t {len(predicted_query_term_set)} unique documents are loaded.')
    ## Show examples (only terms not scores)
    for doc_id, qtset in list(predicted_query_term_set.items())[:5]:
        qtset = [term for term, score in qtset]
        logger.info(f'\t {doc_id} -> (lenght={len(qtset)}) {qtset[:10]} ... {qtset[-10:] if len(qtset)>10 else []}')
    ## Show stats: the size of SPLADE vocabulary for each document
    qtset_sizes:List[int] = [len(qtset) for qtset in predicted_query_term_set.values()]
    logger.info(f'\t SPLADE vocabulary size (min, max, avg): {min(qtset_sizes)}, {max(qtset_sizes)}, {sum(qtset_sizes)/len(qtset_sizes):.1f}')
    
    return predicted_query_term_set
