from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

from argparse import ArgumentParser

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", help="dataset to download", default="scidocs")
    parser.add_argument("--target_directory", dest="target_directory", help="target directory to download the dataset", default="data/beir/")
    args = parser.parse_args()

    #### Download dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    data_path = util.download_and_unzip(url, args.target_directory)

    print(f"Dataset is downloaded at {data_path}")
