BEIR_DATASET={
    'arguana', # ArguAna
    'fiqa', # FiQA
    'nfcorpus', # NFCorpus
    'scidocs', # SCIDOCS
    'scifact', # SciFact
    'trec-covid-v2', # TREC-COVID (v2)
    'webis-touche2020', # Touché-2020
    'quora', # Quora
    'dbpedia-entity', # DBPedia
    'fever', # FEVER
    'climate-fever', # Climate-FEVER
    'hotpotqa', # HotpotQA
    'nq', # NQ

    #TODO: cqadupstack # CQADupStack
    #TODO: bioasq # BioASQ
    #TODO: signal1m # Signal-1M (RT) 
    #TODO: trec-news # TREC-NEWS
    #TODO: robust04 # Robust04
}


DATASET_NAME_DICT = {
    'msmarco_train': 'MS MARCO\n(Source)',
    'msmarco-dev': 'MS MARCO\n(Source)',
    'trec2019': "TREC-DL'19",
    'trec2020': "TREC-DL'20",

    'fiqa': 'FiQA',
    'scifact': 'SciFact', 
    # 'trec-covid-v2': 'TREC-COVID (v2)',
    'trec-covid-v2': 'TREC-C',
    'nfcorpus': 'NFCorpus',
    'scidocs': 'SCIDOCS',
    'arguana': 'ArguAna',
    # 'webis-touche2020': 'Touché-2020',
    'webis-touche2020': 'Touché',

    'hotpotqa': 'HotpotQA',
    'cqadupstack': 'CQADupStack',
    'dbpedia-entity': 'DBPedia', 
    'fever': 'FEVER',
    'climate-fever': 'Climate-FEVER',
    
    'cqadupstack.merged': 'CQADupStack',
}

DATASET_CORPUS_SIZE_DICT = {
    'fiqa': 57638,
    'scifact': 5183, 
    'trec-covid-v2': 129179,
    'nfcorpus': 3633,
    'scidocs': 25657,
    'arguana': 8674,
    'webis-touche2020': 382545,

    # 'hotpotqa': ,
    # 'cqadupstack': ,
    # 'dbpedia-entity': , 
    # 'fever': ,
    # 'climate-fever': ,
}

DATASET_TEST_QUERY_SIZE_DICT = {
    'fiqa': 648,
    'scifact': 300, 
    'trec-covid-v2': 50,
    'nfcorpus': 323,
    'scidocs': 1000,
    'arguana': 1406,
    'webis-touche2020': 49,

    # 'hotpotqa': ,
    # 'cqadupstack': ,
    # 'dbpedia-entity': , 
    # 'fever': ,
    # 'climate-fever': ,
}

DATASET_TITLE_RATIO_DICT = {
    'fiqa': 0.0,
    'scifact': 100.0, 
    'trec-covid-v2': 100.0,
    'nfcorpus': 100.0,
    'scidocs': 100.0,
    'arguana': 31.1,
    'webis-touche2020': 100.0,

    # 'hotpotqa': ,
    # 'cqadupstack': ,
    # 'dbpedia-entity': , 
    # 'fever': ,
    # 'climate-fever': ,
}
"""
from utils.utils import load_json_corpus
for dataset in ['arguana', 'fiqa', 'scifact', 'nfcorpus', 'scidocs', 'trec-covid-v2', 'webis-touche2020']:
    corpus = load_json_corpus(f'experiments/datasets/beir/{dataset}/corpus.jsonl')
    corpus_size = len(corpus)
    corpus_have_title = 0
    for doc_id, doc in corpus.items():
        if "title" in corpus[doc_id] and corpus[doc_id]["title"].strip():
            corpus_have_title += 1

    print(f"{dataset}, {corpus_size}, {corpus_have_title}, {100*corpus_have_title/corpus_size:.1f}% documents have titles.")
    print()
"""