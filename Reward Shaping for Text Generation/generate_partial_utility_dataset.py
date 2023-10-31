import json
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from transformers import AutoTokenizer
from model.metrics import get_bert_ibleu_score, set_gpu

MODEL_ID = {
    'bart': 'facebook/bart-base',
    't5': 't5-small',
}

def _dfs(subtree, rank, curr_seq, results):
    """
    DFS function for Trie traversal.
    """
    # Reached an end
    if len(subtree) == 0:
        return

    # Branching trie
    if len(subtree) > 1:
        tokens = []
        for token, value in subtree.items():
            tokens.append((token, value[0]))
        tokens = sorted(tokens, key=lambda x: rank[x[1]])
        for i in range(len(tokens) - 1):
            results.append((curr_seq[:], tokens[i+1][0], tokens[i][0]))

    for token, value in subtree.items():
        curr_seq.append(token)
        _dfs(value[1], rank, curr_seq, results)
        curr_seq.pop()

def get_prefix(sequences, ranks):
    prefixes = []
    first_diff_tok_idx = []
    for batch, rank in zip(sequences, ranks):
        # Build trie
        trie = {}
        for seq_id, seq in enumerate(batch):
            curr_trie = trie
            not_first_tok = False
            for tok in seq:
                if tok not in curr_trie:
                    curr_trie[tok] = [seq_id, {}]
                # Keep track of beam ID with highest score
                curr_trie[tok][0] = seq_id if rank[seq_id] > rank[curr_trie[tok][0]] else curr_trie[tok][0]
                curr_trie = curr_trie[tok][1] 
                if not_first_tok and tok in [PAD_ID]:
                    break
                not_first_tok = True
        # Extract prefix pairs and the branching token
        prefix_token_pairs = []
        _dfs(trie, rank, [], prefix_token_pairs)

        beam_size = len(rank)
        while len(prefix_token_pairs) < beam_size:
            # Patch for (rare) cases prefix_token_pair size is not consistent
            prefix_token_pairs.append(([PAD_ID], PAD_ID, PAD_ID))
        assert len(prefix_token_pairs) == beam_size

        prefixes.append([pair[0] for pair in prefix_token_pairs])
        first_diff_tok_idx.append(torch.tensor([[pair[1], pair[2]] for pair in prefix_token_pairs]).unsqueeze(0))

    first_diff_tok_idx = torch.cat(first_diff_tok_idx, dim=0)

    # return prefixes, first_diff_tok_idx
    return prefixes, first_diff_tok_idx


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--base_model", choices=["t5", "bart"], help="Dataset to generate partial utility test set.")
    parser.add_argument("--gpu", required=False, type=int, default=0, help="GPU device id")
    args = parser.parse_args()

    model_id = MODEL_ID[args.base_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    PAD_ID = tokenizer.pad_token_id
    set_gpu(args.gpu)
    
    print("==========================================================")
    print("Load data")
    data_file = f"checkpoints/{args.model_postfix}/result.json"
    with open(data_file, "r", encoding="UTF-8") as file:
        data = json.load(file)
        # DEBUG
        # data = data[:100]
    
    target = []
    samples = []
    tok_samples = []
    for d in data:
        target.append(d["source"])
        samples.append(d["outputs"])
        tok_samples.append(tokenizer(d["outputs"]).input_ids)
    
    print("Rank outputs")
    bert_ibleu, bert, bleu = get_bert_ibleu_score(
        target,
        None,
        samples,
        eval=True
    )
    ranks = torch.argsort(bert_ibleu, dim=-1)
    decoder_prefix, first_diff_tok_idx = get_prefix(tok_samples, ranks)
    first_diff_tok_idx = first_diff_tok_idx.tolist()

    print("Generate trie-set from the dataset...")
    results = "[\n"
    for tgt, dp_batch, fdti_batch in zip(target, decoder_prefix, first_diff_tok_idx):
        for dp, fdti in zip(dp_batch, fdti_batch):
            if len(dp) <= 1 or tokenizer.decode(fdti[0]) == "" or tokenizer.decode(fdti[1]) == "":
                continue
            results += '  {{\n    "input": {0},\n    "output_prefix": {1},\n    "better": {2},\n    "worse": {3}\n  }},\n'.format(json.dumps(tgt), dp, fdti[0], fdti[1])
    results = results[:-2] + "\n]"
    
    output_file = f"checkpoints/{args.model_postfix}/partial_utility_estimation.json"
    print("Dump result to:", output_file)
    with open(output_file, "w", encoding="UTF-8") as file:
        file.write(results)
    file.close()