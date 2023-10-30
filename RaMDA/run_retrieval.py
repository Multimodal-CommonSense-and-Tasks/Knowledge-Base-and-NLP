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
from utils.utils import load_pseudo_queries, save_input_arguments, save_data_to_reproduce_experiments, save_elapsed_time
from utils.utils import pad_and_stack_list_of_2d_embeddings
from utils.constant.constant_segmentation import SEGMENT_SEP

class CustomSentenceBERTForMultiFields(SentenceBERT):
    def __init__(
        self, 
        query_model_path: str="",
        query_max_len:int=64, 
        body_model_path: str="",
        pq_model_path: str="",

        body_max_len:int=128,
        pq_max_len:int=128,
        
        
        sep: str = " ", 
        pooling:str = "cls",
        
        use_body:bool=True,
        use_pseudo_queries:bool=True,
        field_weights:Dict[str, float] = {"body":0.5, "pseudo_queries":0.5},
        cahced_embedding_dict:Dict[str, Dict[str, torch.Tensor]]={}, # doc_id -> {"title": torch.Tensor, "text": torch.Tensor}
        
        **kwargs):
        self.sep = sep

        self.q_model: SentenceTransformer = SentenceTransformer(query_model_path)
        self.dim = self.q_model[1].word_embedding_dimension
        self.q_model[1] = s_models.Pooling(self.dim, pooling)
        self.q_model[0].max_seq_length = query_max_len
        print();logger.info(f'q_model=\n{self.q_model}');print()
        
        self.use_body = use_body
        if self.use_body:
            if "body" not in cahced_embedding_dict:
                self.body_model: SentenceTransformer = SentenceTransformer(body_model_path)
                self.body_model[1] = s_models.Pooling(self.dim, pooling)
                self.body_model[0].max_seq_length = body_max_len
                print();logger.info(f'body_model=\n{self.body_model}');print()

        self.use_pseudo_queries = use_pseudo_queries
        if self.use_pseudo_queries:
            self.pq_model: SentenceTransformer = SentenceTransformer(pq_model_path)
            self.pq_model[1] = s_models.Pooling(self.dim, pooling)
            self.pq_model[0].max_seq_length = pq_max_len
            print();logger.info(f'pq_model=\n{self.pq_model}');print()
            
        self.cahced_embedding_dict:Dict[str, Dict[str, ]] = cahced_embedding_dict
        logger.info(f'#> Field usage: use_body={use_body}, use_pseudo_queries={use_pseudo_queries}')
        self.field_weights:Dict[str, float] = field_weights
        logger.info(f'#> with weights={self.field_weights}')
        logger.info(f"#> Cached field embeddings: {[field for field, emb_dict in self.cahced_embedding_dict.items() if emb_dict]}")
        print()

    def encode_single_segment(self, encoder, corpus: List[str], batch_size: int, **kwargs):
        ### Each document in the corpus consists of a single section,
        # - corpus[i] corresponds to each document in the corpus
        return encoder.encode(corpus, batch_size=batch_size, **kwargs)
    
    def encode_multi_segments(self, encoder, corpus: List[List[List[str]]], batch_size: int, **kwargs) -> torch.Tensor:
        ### Each document in the corpus consists of multiple sections,
        # and each section also consists of multiple texts.
        # For example, a section can have multiple pseudo-queries.
        # - corpus[i] corresponds to each document in the corpus
        # - corpus[i, j] corresponds to each section in the document
        # - corpus[i, j, k] corresponds to each text in the section
        n_documents:int = len(corpus)
        n_sections:List[int] = [len(document) for document in corpus]
        max_n_sections:int = max(n_sections)
        n_texts_per_sec:List[List[int]] = [[len(section) for section in document] for document in corpus]
            
        # We first flat texts in all sections and documents,
        # and then encode the flatten texts using batch encoding.
        # After the encoding, we undo flattening by referring to "n_sections" and "n_texts_per_sec"
        flatten_sentences:List[str] = list(chain(*chain(*corpus)))
        embs_all:torch.Tensor = encoder.encode(flatten_sentences, batch_size=batch_size, **kwargs)
        
        # We preserve the vector representation for each section in each document.
        # For multiple texts in each section, we average the vector representations for texts,
        # to have a single vector for the section.
        if not isinstance(embs_all, torch.Tensor):
            embs_all = torch.tensor(embs_all)
        dtype, device = embs_all.dtype, embs_all.device
        doc_embeddings:torch.Tensor = torch.zeros(n_documents, max_n_sections, self.dim, dtype=dtype, device=device)
        offset:int = 0
        for i, l1 in enumerate(n_sections):
            for j, l2 in enumerate(n_texts_per_sec[i]):
                embs = embs_all[offset:offset+l2, :]
                doc_embeddings[i, j, :] = embs.mean(dim=0)
                offset += l2
        return doc_embeddings

    def encode_corpus(self, corpus: List[Dict[str, Dict]], batch_size: int, **kwargs) -> torch.Tensor:
        ### Each document in the corpus consists of multiple segments,
        # and each segment consists of multiple texts.
        # For example, a segment can have multiple pseudo-queries.
        # - corpus["pseudo_queries"][i] corresponds to each document in the corpus
        # - corpus["pseudo_queries"][i, j] corresponds to each section in the document
        # - corpus["pseudo_queries"][i, j, k] corresponds to each text in the section
        # A document also has a body ("text") and (optionally) a title ("title"):
        # - corpus["text"] corresponds to the body of the document.
        # - corpus["title"] corresponds to the title of the document.

        if self.use_body:
            field_name = "body"
            if self.cahced_embedding_dict.get(field_name, {}):
                body_cached_embeddings:Dict[str, torch.Tensor] = self.cahced_embedding_dict[field_name]
                body_embeddings = [body_cached_embeddings[doc["_id"]] for doc in corpus]
                body_embeddings:torch.Tensor = torch.stack(body_embeddings, dim=0)
            else:
                sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc and doc["title"].strip() else doc["text"].strip() for doc in corpus]
                body_embeddings:torch.Tensor = self.encode_single_segment(self.body_model, sentences, batch_size=batch_size)
        else:
            body_embeddings = None
        # shape: (n_documents, dim)
        
        if self.use_pseudo_queries:
            field_name = "pseudo_queries"
            if self.cahced_embedding_dict.get(field_name, {}):
                pq_cached_embeddings:Dict[str, torch.Tensor] = self.cahced_embedding_dict[field_name]
                pq_embeddings = [pq_cached_embeddings[doc["_id"]] for doc in corpus]
                pq_embeddings, _ =pad_and_stack_list_of_2d_embeddings(list_of_2d_embeddings=pq_embeddings)
            else:
                pseudo_queries:List[List[List[str]]] = [doc.get("pseudo_queries", "") for doc in corpus]
                pq_embeddings:torch.Tensor = self.encode_multi_segments(self.pq_model, pseudo_queries, batch_size=batch_size)
        else:
            pq_embeddings = None
        # shape: (n_documents, n_segments, dim)
        
        return (body_embeddings, pq_embeddings)

class CustomDenseRetrievalExactSearchForMultiFields(DenseRetrievalExactSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, 
        #!@ custom
        field_weights:Dict[str, float] = {"body":0.5, "pseudo_queries":0.5},
        **kwargs,
        ):
        super().__init__(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)

        self.field_weights:Dict[str, float] = field_weights

    ## For title/body
    def single_segment_matching(self, score_function:str, query_embeddings:torch.Tensor, corpus_embeddings:torch.Tensor):
        cos_scores = self.score_functions[score_function](query_embeddings, corpus_embeddings)
        cos_scores[torch.isnan(cos_scores)] = -1
        return cos_scores
    ## For pseudo-queries over multiple segments
    def multi_segments_matching(self, score_function:str, query_embeddings:torch.Tensor, corpus_embeddings:torch.Tensor):
        n_documents, n_sections, dim = tuple(corpus_embeddings.size())

        # We first compute scores for each segment of a document in sub-corpus,
        # and then, we max-pool scores across segments in the document
        cos_scores = self.score_functions[score_function](query_embeddings, corpus_embeddings.view(-1, dim))
        cos_scores[torch.isnan(cos_scores)] = -1
        cos_scores = cos_scores.view(-1, n_documents, n_sections)
        # shape: (n_queries, n_documents, n_sections)
        cos_scores = cos_scores.max(dim=-1).values
        # shape: (n_queries, n_documents)
        
        return cos_scores
        
    def search(self, 
               corpus: Dict[str, Dict], 
               queries: Dict[str, str], 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False,                
               **kwargs) -> Dict[str, Dict[str, float]]:
        
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        for k, v in score_function.items():
            if v not in self.score_functions:
                raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(v))

            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        if self.model.cahced_embedding_dict.get('queries', {}):
            emb_dict:Dict[str, torch.Tensor] = self.model.cahced_embedding_dict["queries"]
            embs = [emb_dict[qid] for qid in query_ids]
            query_embeddings:torch.Tensor = torch.stack(embs, dim=0)
            query_embeddings = query_embeddings.to(torch.device('cuda'))
        else:
            query_embeddings = self.model.encode_queries(
                queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
            
        corpus_ids:List[str] = list(corpus.keys())
        corpus:List[Dict] = [corpus[cid] for cid in corpus_ids]
        
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function for body : {}".format(self.score_function_desc[score_function["body"]]))
        logger.info("Scoring Function for PQs  : {}".format(self.score_function_desc[score_function["pq"]]))

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            sub_corpus = corpus[corpus_start_idx:corpus_end_idx]

            #Encode chunk of corpus    
            body_embeddings, pq_embeddings = self.model.encode_corpus(
                corpus=sub_corpus,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = False, #!@ custom: to avoid GPU OOM
                convert_to_numpy = True, #!@ custom: to avoid GPU OOM
            )
            if (body_embeddings is not None):
                body_embeddings = torch.tensor(body_embeddings) #!@ custom: to avoid GPU OOM
            if (pq_embeddings is not None):
                if not isinstance(pq_embeddings, torch.Tensor):
                    pq_embeddings = torch.tensor(pq_embeddings) #!@ custom: to avoid GPU OOM
            
            #Compute similarites using either cosine-similarity or dot product
            cos_scores_top_k_values, cos_scores_top_k_idx = [], []
            score_function_batch_size = 128
            for score_function_query_index in range(0, len(query_embeddings), score_function_batch_size):
                st, ed = score_function_query_index, score_function_query_index + score_function_batch_size
                sub_query_embeddings = query_embeddings[st:ed].cpu().data

                #!@ custom: masks on titles
                cos_scores_fields = torch.zeros(len(sub_query_embeddings), len(sub_corpus), 3)
                field_weights = torch.zeros(len(sub_corpus), 3)
                if (body_embeddings is not None):
                    body_scores:torch.Tensor = self.single_segment_matching(score_function=score_function["body"], query_embeddings=sub_query_embeddings, corpus_embeddings=body_embeddings)
                    cos_scores_fields[:, :, 0] = body_scores
                    field_weights[:, 0] = self.field_weights["body"]
                if (pq_embeddings is not None):
                    pq_scores:torch.Tensor = self.multi_segments_matching(score_function=score_function["pq"], query_embeddings=sub_query_embeddings, corpus_embeddings=pq_embeddings)
                    cos_scores_fields[:, :, 2] = pq_scores
                    field_weights[:, 2] = self.field_weights["pseudo_queries"]
                field_weights = field_weights / (field_weights.sum(dim=1, keepdim=True) + 1e-6)
                cos_scores = (cos_scores_fields * field_weights.unsqueeze(0)).sum(-1)

                #Get top-k values
                cos_scores_top_k_values_batch, cos_scores_top_k_idx_batch = torch.topk(cos_scores, min(top_k+1, len(cos_scores[0])), dim=1, largest=True, sorted=return_sorted)
                cos_scores_top_k_values.extend(cos_scores_top_k_values_batch.tolist())
                cos_scores_top_k_idx.extend(cos_scores_top_k_idx_batch.tolist())
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        self.results[query_id][corpus_id] = score
        
        return self.results 


if __name__=='__main__':
    parser = ArgumentParser("")

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_ranking", action="store_true")

    parser.add_argument('--q_model_path', type=str, default="OpenMatch/cocodr-base-msmarco",
        help="Retriever. e.g., OpenMatch/cocodr-base-msmarco or OpenMatch/cocodr-large-msmarco.")
    parser.add_argument("--q_max_len", type=int, default=64)

    parser.add_argument('--use_body', action='store_true')
    parser.add_argument('--body_model_path', type=str, default="OpenMatch/cocodr-base-msmarco",
        help="Retriever. e.g., OpenMatch/cocodr-base-msmarco or OpenMatch/cocodr-large-msmarco.")
    parser.add_argument("--body_max_len", type=int, default=128)
    parser.add_argument('--body_score_function', type=str, choices=['dot', 'cos_sim'],  default='dot')
    parser.add_argument('--body_weight', type=float, default=0.0)

    parser.add_argument('--use_pseudo_queries', action='store_true')
    parser.add_argument('--pseudo_queries_path', type=str)
    parser.add_argument('--pq_model_path', type=str, default="OpenMatch/cocodr-base-msmarco",
        help="Retriever. e.g., OpenMatch/cocodr-base-msmarco or OpenMatch/cocodr-large-msmarco.")
    parser.add_argument("--pq_max_len", type=int, default=64)
    parser.add_argument('--pq_score_function', type=str, choices=['dot', 'cos_sim'],  default='dot')
    parser.add_argument('--pseudo_queries_weight', type=float, default=0.0)

    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    
    parser.add_argument('--data_path', type=str, required=True)

    parser.add_argument('--queries_cached_embedding_path', type=str)
    parser.add_argument('--body_cached_embedding_path', type=str)
    parser.add_argument('--pseudo_queries_cached_embedding_path', type=str)

    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    start_time = time.time()

    assert 0.999 < args.body_weight + args.pseudo_queries_weight < 1.001


    ## Create output directory
    output_dir = args.output_dir
    print()
    outfile_path:str = os.path.join(output_dir, "results.json")
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

    ## Save input arguments and python script
    args.dataset:str = os.path.basename(args.data_path)
    save_data_to_reproduce_experiments(output_directory=output_dir, path_to_python_script=__file__, input_arguments=args, prefix=os.path.basename(__file__))

    ## Load BEIR dataset
    print()
    logger.info(f'#> Load dataset from {args.data_path}')
    corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split="test")
    
    ## Load pseudo-queries
    if args.use_pseudo_queries and not (args.pseudo_queries_cached_embedding_path and os.path.exists(args.pseudo_queries_cached_embedding_path)):
        pseudo_queries:Dict[str, List[str]] = load_pseudo_queries(pseudo_queries_path=args.pseudo_queries_path)
        # Convert segment IDs into document IDs
        corpus_pseudo_queries:Dict[str, List[List[str]]] = {}
        for segment_id, list_of_texts in pseudo_queries.items():
            doc_id:str = segment_id.split(SEGMENT_SEP)[0]
            corpus_pseudo_queries[doc_id] = corpus_pseudo_queries.get(doc_id, [])
            corpus_pseudo_queries[doc_id].append(list_of_texts)
        
    ## Construct single-field corpus: 
    # - Each document consists of multiple fields
    print()
    logger.info(f"Construct multi-field documents")
    multi_field_corpus = {}
    for doc_id, doc in corpus.items():
        multi_field_corpus[doc_id] = {}
        multi_field_corpus[doc_id]["_id"] = doc_id
        multi_field_corpus[doc_id]["text"] = doc["text"]
        multi_field_corpus[doc_id]["title"] = doc.get("title", "")
        if args.use_pseudo_queries and not (args.pseudo_queries_cached_embedding_path and os.path.exists(args.pseudo_queries_cached_embedding_path)):
            multi_field_corpus[doc_id]["pseudo_queries"]:List[List[str]] = corpus_pseudo_queries.get(doc_id, [[""]])
        if len(multi_field_corpus) == 1:
            logger.info(f"Sample document: \n{json.dumps(multi_field_corpus[doc_id], indent=4)}")
    print()

    ## Load encoder
    cahced_embedding_dict = {}
    if args.queries_cached_embedding_path and os.path.exists(args.queries_cached_embedding_path):
        cahced_embedding_dict['queries'] = torch.load(args.queries_cached_embedding_path)
    if args.body_cached_embedding_path and os.path.exists(args.body_cached_embedding_path):
        cahced_embedding_dict['body'] = torch.load(args.body_cached_embedding_path)
    if args.pseudo_queries_cached_embedding_path and os.path.exists(args.pseudo_queries_cached_embedding_path):
        cahced_embedding_dict['pseudo_queries'] = torch.load(args.pseudo_queries_cached_embedding_path)
    field_weights = {"body":args.body_weight, "pseudo_queries":args.pseudo_queries_weight}
    sbert = CustomSentenceBERTForMultiFields(
        query_model_path=args.q_model_path, 
        body_model_path=args.body_model_path, 
        pq_model_path=args.pq_model_path, 

        pooling=args.pooling, 
        
        query_max_len=args.q_max_len, 
        body_max_len=args.body_max_len,
        pq_max_len=args.pq_max_len,

        use_body=args.use_body,
        use_pseudo_queries=args.use_pseudo_queries,
        field_weights = field_weights,
        cahced_embedding_dict=cahced_embedding_dict, # field name -> {doc_id: torch.Tensor}
    )
    model = CustomDenseRetrievalExactSearchForMultiFields(sbert, batch_size=args.batch_size, field_weights=field_weights)

    ## Load retriever and retrieve documents
    retriever = EvaluateRetrieval(model, score_function={"body": args.body_score_function, "pq": args.pq_score_function,})
    results = retriever.retrieve(multi_field_corpus, queries)

    ## Evaluate the ranking performance
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    ## Save ranking performance
    print()
    with open(outfile_path, 'w') as fOut:
        fOut.write(f"{json.dumps({**ndcg, **_map, **recall, **precision,}, indent=4)}\n")
    print(f'\n\t{outfile_path}\n')
    
    ## Save ranking: [qid]\t[doc_id]\t[rank]\t[score]
    if args.save_ranking:
        ranking_path:str = os.path.join(output_dir, "ranking.tsv")
        logger.info(f"#> Save ranking")
        with open(ranking_path, 'w') as fOut:
            for qid, topk_docs in results.items():
                ranking_ordered = list(sorted(topk_docs.items(), key=lambda x: x[1], reverse=True))
                ranking_ordered = ranking_ordered[:1000]
                for rank, (doc_id, score) in enumerate(ranking_ordered):
                    fOut.write(f"{qid}\t{doc_id}\t{rank+1}\t{score}\n")
        print(f'\n\t{ranking_path}\n')

    ## Save elapsed time
    save_elapsed_time(output_directory=output_dir, start_time=start_time, outpath_prefix=os.path.basename(__file__)+"__")

    print()
