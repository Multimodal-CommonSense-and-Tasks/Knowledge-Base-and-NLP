import os
import pickle
from tqdm import tqdm
from typing import List, Set, Tuple

from scipy.stats import rankdata

import torch
from torch.utils.data import Dataset

from transformers import T5Tokenizer, T5TokenizerFast

class TextGenerationDataset(Dataset):
    def __init__(self, tokenizer, data, cache_path, shuffle=True):
        try:
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
        except:
            # Prefix for T5 models
            if isinstance(tokenizer, T5Tokenizer) or isinstance(tokenizer, T5TokenizerFast):
                prefix = "paraphrase: "
            else:
                prefix = ""
            for _, d in tqdm(enumerate(data)):
                src = tokenizer(prefix + d['source'], return_tensors='pt')['input_ids'][0]
                tgt = [tokenizer(t, return_tensors='pt')['input_ids'][0] for t in d['targets']]
                
                d['source'] = src
                d['targets'] = tgt

            # Create directory if not exist
            if not os.path.exists(os.path.abspath(os.path.join(cache_path, os.pardir))):
                os.makedirs(os.path.abspath(os.path.join(cache_path, os.pardir)))
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.data = data
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        datum = self.data[i]
        if self.shuffle:
            tgt_idx = torch.randint(len(datum["targets"]), (1,)).item()
            return datum["source"], datum["targets"][tgt_idx]
        else:
            return datum["source"], datum["targets"][0]


def _dfs(subtree, rank, curr_seq_len: int, branches: Set[Tuple[int, int, List[int]]]):
    """
    DFS function for Trie traversal.
    """
    # Reached an end
    if len(subtree) == 0:
        return

    # value: [best sequence ID, subtree, num of sequences on this node]
    if len(subtree) > 1: # Branching
        # Find the branch with highest rank
        best_token = None
        not_best_tokens = []
        s = 0
        for token, value in subtree.items():
            if best_token is None:
                best_token = (token, value[0])
            else:
                if rank[value[0]] < rank[best_token[1]]:
                    not_best_tokens.append((rank[best_token[1]], best_token[1]))
                    best_token = (token, value[0])
                else:
                    not_best_tokens.append((rank[value[0]], value[0]))
            s += value[2]
        # for not_best_token in not_best_tokens:
            # results.append((curr_seq[:], best_token[0], not_best_token))
        # if not_best_tokens:
        branches.add((curr_seq_len, best_token[1], tuple(x for _, x in sorted(not_best_tokens)), s)) # position, best seq id, other seq ids
        # else:
        #     non_branches.append(len(curr_seq), best_token[1])
    else: # Not branching
        for token, value in subtree.items():
            branches.add((curr_seq_len, value[0], (), value[2]))

    for token, value in subtree.items():
        curr_seq_len += 1
        _dfs(value[1], rank, curr_seq_len, branches)
        curr_seq_len -= 1


def get_prefix(sequences, scores_all, pad_token_id, strategy='dominate', add_reference=False):
    all_branches, all_win_indices, all_lose_indices = [], [], []
    # first_diff_tok_idx = []
    for batch, scores in zip(sequences, scores_all):
        rank = 1 + len(batch) - rankdata(scores, method='max', axis=-1)
        # Build trie
        # key: token for transition (on edge)
        # value: [best sequence ID, subtree, num of sequences on this node]
        trie = {}
        for seq_id, seq in enumerate(batch):
            curr_trie = trie
            for tok in seq:
                if tok != pad_token_id:
                    if tok not in curr_trie:
                        curr_trie[tok] = [seq_id, {}, 1]
                    # Keep track of beam ID with highest rank
                    else:
                        curr_trie[tok][0] = seq_id if rank[seq_id] < rank[curr_trie[tok][0]] else curr_trie[tok][0]
                        curr_trie[tok][2] += 1
                    curr_trie = curr_trie[tok][1]

        # Extract prefix pairs and the branching token
        branches = set()
        _dfs(trie, rank, 0, branches)

        # branch: position, best_seq_id, other_seq_ids, num_seqs
        all_branches.append(branches)
        
        win_indices, lose_indices = [], []
        for position, best_seq_id, other_seq_ids, num_seqs in branches:
            if other_seq_ids:
                if strategy == 'dominate':
                    win_indices.extend([(best_seq_id, position)] * len(other_seq_ids))
                    lose_indices.extend([(o, position) for o in other_seq_ids])
                elif strategy == 'adjacent':
                    linearized = [(x, position) for x in [best_seq_id] + other_seq_ids]
                    win_indices = linearized[:-1]
                    lose_indices = linearized[1:]
                elif strategy == 'all':
                    ls = [best_seq_id] + other_seq_ids
                    for i in range(len(ls)):
                        for j in range(i+1, len(ls)):
                            win_indices.append(i)
                            lose_indices.append(j)
                else:
                    raise NotImplementedError

        all_win_indices.append(win_indices)
        all_lose_indices.append(lose_indices)

        # add reference
        # ref_branch_positions = []
        # seq_id, seq = -1, ref.tolist()
        # curr_trie = trie
        # for idx, tok in enumerate(seq):
        #     if tok not in curr_trie:
        #         # curr_trie[tok] = [seq_id, {}]
        #         ref_branch_positions.append(idx)
        #         break
        #     if len(curr_trie) > 1:
        #         ref_branch_positions.append(idx)
        #     curr_trie = curr_trie[tok][1]
        #     if tok == pad_token_id:
        #         break
        # all_ref_branch_positions.append(ref_branch_positions)

    # return all_branches, all_ref_branch_positions
    return all_branches, all_win_indices, all_lose_indices

class OfflineSupervisedDataset(Dataset):
    def __init__(self, tokenizer, data, beam_size, cache_path=None):
        # self.tokenizer = tokenizer
        # if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data_ = pickle.load(f)
            self.data = data_
        # else:
        except:
            for i, d in tqdm(enumerate(data)):
                src = tokenizer(d['source'], return_tensors='pt')['input_ids'][0]
                tgt = tokenizer(d['target'], return_tensors='pt')['input_ids'][0]
                hypos = tokenizer(d['outputs'])['input_ids']
                
                d['source'] = src
                d['target'] = tgt
                d['hypos'] = [torch.tensor(hypo) for hypo in hypos]
                del d['outputs']

            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.data = data
        self.bs = len(self.data[0]['hypos']) + 1
        assert self.bs == beam_size + 1
    
    def __len__(self):
        return len(self.data) * self.bs
    
    def __getitem__(self, i):
        i, j = i // self.bs, i % self.bs
        datum = self.data[i]
        if j == 0:
            return datum['source'], datum['target']
        else:
            return datum['source'], datum['hypos'][j-1]


class OfflineSupervisedFilteredDataset(Dataset):
    def __init__(self, tokenizer, data, scores, beam_size, cache_path=None, threshold=0.8):
        try:
            with open(cache_path, 'rb') as f:
                data_ = pickle.load(f)
            self.data = data_
        # else:
        except:
            data_ = []
            for i, d in tqdm(enumerate(data)):
                src = tokenizer(d['source'], return_tensors='pt')['input_ids'][0]
                tgt = tokenizer(d['target'], return_tensors='pt')['input_ids'][0]
                hypos = tokenizer(d['outputs'])['input_ids']
                
                data_.append((src, tgt))
                for j in range(beam_size):
                    if scores[i, j] > threshold:
                        data_.append((src, torch.tensor(hypos[j])))

            with open(cache_path, 'wb') as f:
                pickle.dump(data_, f)
            self.data = data_
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


class OfflineRLDataset(Dataset):
    def __init__(self, tokenizer, data, scores, cache_path=None, add_score=True):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            
            if add_score:
                for i, (d, score) in tqdm(enumerate(zip(self.data, scores))):
                    if 'score' in d:
                        break
                    d['score'] = score
                else:
                    with open(cache_path, 'wb') as f:
                            pickle.dump(self.data, f)

        else:
            for i, (d, score) in tqdm(enumerate(zip(data, scores))):
                src = tokenizer(d['source'], return_tensors='pt')['input_ids'][0]
                if 'target' in d:
                    tgt = tokenizer(d['target'], return_tensors='pt')['input_ids'][0]
                else:
                    tgt = tokenizer(d['targets'][0], return_tensors='pt')['input_ids'][0]
                    del d['targets']
                hypos = tokenizer(d['outputs'])['input_ids']
                all_branches, all_win_indices, all_lose_indices = get_prefix([hypos], [score], tokenizer.pad_token_id)
                d['source'] = src
                d['target'] = tgt
                d['hypos'] = [torch.tensor(hypo) for hypo in hypos]
                d['branches'] = all_branches[0]
                d['win_indices'] = all_win_indices[0]
                d['lose_indices'] = all_lose_indices[0]
                del d['outputs']
                if add_score:
                    d['score'] = score

            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.data = data

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        datum = self.data[i]
        if 'score' in datum:
            return datum['source'], datum['target'], datum['hypos'], datum['branches'], datum['win_indices'], datum['lose_indices'], datum['score']
        return datum['source'], datum['target'], datum['hypos'], datum['branches'], datum['win_indices'], datum['lose_indices']


def tg_collate_fn(batch):
    froms, tos = [], []
    for f, t in batch:
        froms.append(f)
        tos.append(t)
    return froms, tos

class BranchingEvalDataset():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        datum = self.data[i]
        return datum["input"], datum["output_prefix"], datum["better"], datum["worse"]
