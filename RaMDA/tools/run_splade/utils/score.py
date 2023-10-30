import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from run_splade.splade.splade.models.transformer_rep import Splade
def load_model(model_type_or_dir:str="naver/splade-cocondenser-ensembledistil"):
    print(f"Loading model...: {model_type_or_dir}")
    model = Splade(model_type_or_dir, agg="max")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    print("\n")
    return model, tokenizer, reverse_voc

def run_splade_single(model, tokenizer, reverse_voc, doc):
    # compute the document representation
    with torch.no_grad():
        doc_rep = model(d_kwargs=tokenizer(doc, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

    # get the number of non-zero dimensions in the rep:
    col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

    # now let's inspect the bow representation:
    weights = doc_rep[col].cpu().tolist()
    d = {k: v for k, v in zip(col, weights)}
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    
    tokens = []
    scores = []
    # bow_rep = []
    for k, v in sorted_d.items():
        # bow_rep.append((reverse_voc[k], round(v, 2)))
        tokens.append(reverse_voc[k])
        scores.append(v)
    # print("SPLADE BOW rep:\n", bow_rep)
    assert len(tokens) == len(scores), f'len(tokens)={len(tokens)}, len(scores)={len(scores)}'
    return tokens, scores

def run_splade_batch(model, tokenizer, reverse_voc, doc_batch, device):
    
    # compute the document representation
    with torch.no_grad():
        doc_rep_batch = model(d_kwargs=tokenizer(doc_batch, padding='longest', truncation='longest_first', return_tensors="pt").to(device))["d_rep"]  # (sparse) doc rep in voc space, shape (30522,)

    tokens_and_scores_batch = []
    for doc_rep in doc_rep_batch.cpu():
        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().tolist()
        # print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {k: v for k, v in zip(col, weights)}
        sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

        tokens = []
        scores = []
        # bow_rep = []
        for k, v in sorted_d.items():
            # bow_rep.append((reverse_voc[k], round(v, 2)))
            tokens.append(reverse_voc[k])
            scores.append(v)
        # print("SPLADE BOW rep:\n", bow_rep)
        tokens_and_scores_batch.append((tokens, scores))
    
    return tokens_and_scores_batch