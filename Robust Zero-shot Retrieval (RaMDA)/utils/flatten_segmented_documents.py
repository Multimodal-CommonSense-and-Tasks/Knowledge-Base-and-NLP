from typing import Dict, List, Tuple
import os
import argparse
import json

from utils.utils import load_json_corpus
from utils.logger import logger
from utils.constant.constant_segmentation import SEGMENT_SEP

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flatten corpus into segments: a document -> multiple documents with segments")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--outfile_path")
    args = parser.parse_args()
    print('\n')

    org_corpus = load_json_corpus(os.path.join(args.data_path, 'corpus.jsonl'))
    
    if not args.outfile_path:
        args.outfile_path = os.path.join(args.data_path, 'corpus.flatten.jsonl')
    
    logger.info(f'#> Flatten documents wil be saved at {args.outfile_path}')
    with open(args.outfile_path, 'w') as fOut:
        for doc_id, doc in org_corpus.items():
            segments_char_range:List[Tuple[int, int]] = doc["segments_char_range"]
            title:str = doc.get("title", "")
            text:str = doc["text"]
            for seg_index, (st, ed) in enumerate(segments_char_range):
                json_dict = json.dumps({
                    "_id": doc_id+f"{SEGMENT_SEP}{seg_index}",
                    "title": title,
                    "text": text[st:ed],
                })
                fOut.write(f"{json_dict}\n")
    
    logger.info(f'Done!')
    logger.info(f'\n\n\t {args.outfile_path}')
    print('\n')
