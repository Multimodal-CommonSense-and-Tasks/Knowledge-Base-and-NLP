import json
import logging
import os
import os.path

import hydra
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from omegaconf import DictConfig
from tqdm.auto import tqdm

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import BeirDataset
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing, SparseRetrieval
from .utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def retrieve(exp_dict: DictConfig):
    
    import time
    st = time.time()

    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    model = get_model(config, init_dict)

    batch_size_d = config["index_retrieve_batch_size"]
    batch_size_q = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)

    # Just some code to print debug information to stdout
    logging.basicConfig(format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #!@ cusotm
    # Download and unzip the dataset
    # assert os.path.exists(exp_dict["beir"]["dataset_path"]), f'exp_dict["beir"]["dataset_path"]={exp_dict["beir"]["dataset_path"]}'
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
    #     exp_dict["beir"]["dataset"])
    # out_dir = exp_dict["beir"]["dataset_path"]
    # data_path = util.download_and_unzip(url, out_dir)
    # data_path = exp_dict["beir"]["dataset_path"]

    config["index_dir"] = os.path.join(config["index_dir"], "beir", exp_dict["beir"]["dataset"])
    os.makedirs(config["index_dir"], exist_ok=True)

    config["out_dir"] = os.path.join(config["out_dir"], "beir", exp_dict["beir"]["dataset"])
    os.makedirs(config["out_dir"], exist_ok=True)

    # Provide the data path where dataset has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) datapath/corpus.jsonl  (format: jsonlines)
    # (2) datapath/queries.jsonl (format: jsonlines)
    # (3) datapath/qrels/test.tsv (format: tsv ("\t"))
    # corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    corpus = GenericDataLoader(data_folder=os.path.join(exp_dict["beir"]["dataset_path"], exp_dict["beir"]["dataset"])).load_corpus()
    ## Load queries
    from collections import OrderedDict
    queries = OrderedDict()
    with open(exp_dict["beir"]["queries_path"]) as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
            
            #?@ debugging
            # if len(queries) > 5: break

    d_collection = BeirDataset(corpus, information_type="document")
    q_collection = BeirDataset(queries, information_type="query")

    rankingTsv_outfile = os.path.join(config.out_dir, exp_dict["beir"]["rankingTsv_outfile"])
    if not os.path.exists(rankingTsv_outfile):
        # Index BEIR collection
        d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size_d,
                                        shuffle=False, num_workers=4)
        evaluator = SparseIndexing(model=model, config=config, compute_stats=True) #!@ original: force_new=True by default
        # evaluator = SparseIndexing(model=model, config=config, compute_stats=True, force_new=False) #TODO
        # evaluator = SparseIndexing(model=model, config=config, compute_stats=True, force_new=True) #!@ custom
        evaluator.index(d_loader, id_dict=d_collection.idx_to_key)

        # Retrieve from BEIR collection
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size_q,
                                        shuffle=False, num_workers=1)
        evaluator = SparseRetrieval(config=config, model=model, compute_stats=True, dim_voc=model.output_dim, is_beir=True)
        evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"] + 1, id_dict=q_collection.idx_to_key)

        with open(os.path.join(config.out_dir, "run.json")) as reader:
            run = json.load(reader)
        new_run = dict()
        print("Removing query id from document list")
        for query_id, doc_dict in tqdm(run.items()):
            query_dict = dict()
            for doc_id, doc_values in doc_dict.items():
                if query_id != doc_id:
                    query_dict[doc_id] = doc_values
            new_run[query_id] = query_dict

        ### Custom save
        print(f'\n\n\t ranking results will be saved at {rankingTsv_outfile}\n')
        with open(rankingTsv_outfile, 'w') as fOut:
            for qid, rankings in new_run.items():
                for rank, (doc_id, score) in enumerate(sorted(rankings.items(), key=lambda x: x[1], reverse=True)):
                    fOut.write(f'{qid}\t{doc_id}\t{rank+1}\t{score}\n')
    else:
        print(f'\n\n\t Load preivous ranking results from {rankingTsv_outfile}\n')
        new_run = OrderedDict()
        with open(rankingTsv_outfile) as fIn:
            for line in fIn:
                qid, doc_id, rank, score = line.strip().split()
                new_run[qid] = new_run.get(qid, {})
                new_run[qid][doc_id] = float(score)
    
    with open(os.path.join(config.out_dir, f"{os.path.basename(rankingTsv_outfile)[:-len('.tsv')]}.{os.path.basename(__file__)[:-len('.py')]}.elapsed"), 'w') as fOut:
        fOut.write(f'{time.time()-st}\n')

if __name__ == "__main__":
    retrieve()
