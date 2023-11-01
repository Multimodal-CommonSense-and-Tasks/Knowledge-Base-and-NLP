from typing import List, Dict, Tuple

import argparse
import os
import json
from collections import OrderedDict

from utils.utils import load_json_corpus, load_json_queries, load_tsv_qrels, reverse_qrels

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpl_dataset_directory', required=True)
    parser.add_argument('--outfile_path', required=True)
    parser.add_argument('--queries_path')

    args = parser.parse_args()

    print()

    qrels:Dict[str, List[str]] = load_tsv_qrels(os.path.join(args.gpl_dataset_directory, 'qgen-qrels/train.tsv')) # qid -> list of dids
    if args.queries_path and os.path.exists(args.queries_path):
        print()
        print(f"queries_path is explicitly given: {args.queries_path}")
        queries:Dict[str, str] = load_json_queries(args.queries_path) # qid -> query
        qrels = {qid: annotations for qid, annotations in qrels.items() if qid in queries}
        print()
    else:
        queries:Dict[str, str] = load_json_queries(os.path.join(args.gpl_dataset_directory, 'qgen-queries.jsonl')) # qid -> query
    drels:Dict[str, List[str]] = reverse_qrels(qrels=qrels) # did -> list of qids
    
    print(f'#> Write format-converted pseudo-queries into {args.outfile_path}')
    n_metatexts:List[int] = []
    with open(args.outfile_path, 'w') as outfile:
        for did, qid_list in drels.items():
            # {"11547": ["eponyms are words that you would not be proud of?", "what is the meaning of the idiom an unwanted eponym?", "are eponyms a good thing?", "are unwanted eponyms real words?", "meaning an unwanted eponym?"]}
            outline:str = json.dumps({
                did:[queries[qid].strip() for qid in qid_list],
            })+'\n'
            outfile.write(outline)
            n_metatexts.append(len(qid_list))
    print(f'#> The number of metatexts: \
            Min {min(n_metatexts)}, \
            Max {max(n_metatexts)}, \
            Mean {sum(n_metatexts)/len(n_metatexts)}')
    
    ## Save statistics
    stat_outfile_path = args.outfile_path.replace('.jsonl', '.stat')
    print(f'#> Write statistics into {stat_outfile_path}')
    import numpy as np
    with open(stat_outfile_path, 'w') as outfile:
        outfile.write(json.dumps({
            'n_metatexts': len(n_metatexts),
            'min': min(n_metatexts),
            'max': max(n_metatexts),
            'mean': sum(n_metatexts)/len(n_metatexts),
            'median': float(np.median(n_metatexts)),
            'std': float(np.std(n_metatexts)),
        }, indent=4))
    print(f'#> Done\n')