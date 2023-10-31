"""
synonyms.json: https://www.kaggle.com/datasets/duketemon/wordnet-synonyms
"""

import argparse
import json
from collections import Counter
from ahocorapy.keywordtree import KeywordTree
from tqdm import tqdm
from multiprocessing import Pool

dataset = None

def get_word_semantic_map():
    with open("data/synonyms.json", "r", encoding="UTF-8") as file:
        # Load synonyms
        raw_synonyms = json.load(file)

        # Group synonyms by semantics
        dedup = set()
        synonyms = []
        for word, syngroups in raw_synonyms.items():
            word = word.split(":")[0] # discard PoS
            
            # split words by semantic groups
            syngroups = syngroups.split("|")
            syngroups = [sorted(group.split(";") + [word]) for group in syngroups] # sort words by order
            
            # If group not added previously,
            for group in syngroups:
                if "".join(group) not in dedup:
                    dedup.add("".join(group))
                    synonyms.append(group)
        
        # Generate synonym groups as a dicitonary
        word_semantic_map = {} # "word": [group ids]
        for i, group in enumerate(synonyms):
            for word in group:
                if word.lower() not in word_semantic_map:
                    word_semantic_map[word.lower()] = []
                word_semantic_map[word.lower()].append(i)
        
        # with open("word_semantic_map.json", "w", encoding="UTF-8") as file:
        #     json.dump(word_semantic_map, file, ensure_ascii=False)
        return word_semantic_map


searcher = None

def _find_para_pairs(pair, mode):
    global searcher
    paras = pair["paraphrases"]

    # Extract words that are in word_semantic_map from paraphrases 
    words_from_para = []
    for sent in paras:
        result = []
        for x in searcher.search_all(' ' + sent.strip() + ' '):
            word = x[0].strip()
            group_ids = word_semantic_map[word.lower()]
            for id in group_ids:
                result.append((id, word))
        result = sorted(result)
        words_from_para.append(result)
    
    result = []
    for i in range(len(words_from_para) - 1):
        for j in range(i + 1, len(words_from_para)):
            sent1 = words_from_para[i]
            sent2 = words_from_para[j]

            k, l = 0, 0
            while True:
                if k >= len(sent1) or l >= len(sent2):
                    break
                word1 = sent1[k]
                word2 = sent2[l]
                if word1[0] == word2[0]:
                    if word1[1] != word2[1] \
                         and (word1[1][:-1] not in word2[1][:-1]) and (word2[1][:-1] not in word1[1][:-1]) \
                         and (word1[1].lower() not in paras[1].lower()) and (word2[1].lower() not in paras[0].lower()):
                        # Add only if lexical forms are different
                        if mode == "train":
                            result.append(";".join(sorted([word1[1], word2[1]])))
                        elif mode == "test":
                            i_word1_index = paras[i].lower().find(word1[1].lower())
                            j_word2_index = paras[j].lower().find(word2[1].lower())
                            result.append({
                                "input": paras[i],
                                "output_prefix": paras[i][:(paras[i].lower().find(word1[1].lower()))],
                                "worse": paras[i][i_word1_index:i_word1_index + len(word1[1])],
                                "better": paras[j][j_word2_index:j_word2_index + len(word2[1])]
                            })
                            result.append({
                                "input": paras[j],
                                "output_prefix": paras[j][:(paras[j].lower().find(word2[1].lower()))],
                                "worse": paras[j][j_word2_index:j_word2_index + len(word2[1])],
                                "better": paras[i][i_word1_index:i_word1_index + len(word1[1])]
                            })

                    # Select all pairs of synonyms 
                    if k < len(sent1) - 1 and sent1[k+1][0] == word1[0]:
                        k += 1
                    elif l < len(sent2) - 1 and sent2[l+1][0] == word2[0]:
                        l += 1
                        while k > 0 and sent1[k-1][0] == word1[0]:
                            k -= 1
                    else:
                        k += 1
                        l += 1
                elif word1[0] > word2[0]:
                    l += 1
                elif word1[0] < word2[0]:
                    k += 1
    
    if mode == "train":
        return list(set(result))
    else:
        dedup = set()
        dedup_result = []
        for r in result:
            if r["worse"] + ";" + r["better"] not in dedup:
                dedup.add(r["worse"] + ";" + r["better"])
                dedup_result.append(r)
        return dedup_result


def get_valid_synonym_pairs(word_semantic_map, split="train"):
    global searcher
    with open(f"data/{dataset}_paragen_{split}.json", "r", encoding='UTF-8') as file:
        train_data = json.load(file)
    
    searcher = KeywordTree(case_insensitive=True)
    for word in word_semantic_map.keys():
        searcher.add(' ' + word + ' ')
    searcher.finalize()
        
    # Multi-processing
    # with Pool(8) as p:
    #     temp_results = p.map(_find_para_pairs, train_data)
    temp_results = [_find_para_pairs(x, 'train') for x in tqdm(train_data)]

    results = []
    for t in temp_results:
        results.extend(t)
        
    # synonyms_from_para = sorted(list(Counter(results).items()), key=lambda x: x[1])
    # with open(f"valid_synonym_pairs_{split}.tsv", "w", encoding="UTF-8") as file:
    #     file.writelines([
    #         "\t".join([word.split(";")[0], word.split(";")[1], str(count)])+"\n" for word, count in synonyms_from_para
    #     ])
    return set(results)

def generate_branch_dataset(synonyms_from_train, split="test"):
    global searcher
    with open(f"data/{dataset}_paragen_{split}.json", "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    
    temp_results = [_find_para_pairs(x, 'test') for x in tqdm(test_data)]
    results = []
    for t in temp_results:
        results.extend(t)
    
    # Leave only synonym pairs from train set
    results = [r for r in results if (r["worse"] + ";" + r["better"] in synonyms_from_train) or (r["better"] + ";" + r["worse"] in synonyms_from_train)]
    results = [r for r in results if len(r["worse"]) > 1 and len(r["better"]) > 1]
    
    with open(f"data/{dataset}_synonym_branching_test.json", "w", encoding="UTF-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["qqp", "mscoco"], help="Dataset to generate synonym branching test set.")
    args = parser.parse_args()
    dataset = args.dataset

    print(dataset)
    print("==========================================================")
    print("Preprocess synonyms...")
    word_semantic_map = get_word_semantic_map()
    print("Extract synonyms that appear from the training set...")
    synonyms_from_train = get_valid_synonym_pairs(word_semantic_map)
    print("Generate synonym-selection test set")
    synonyms_from_train = [s.lower() for s in synonyms_from_train]
    results = generate_branch_dataset(synonyms_from_train) 
    print(len(results))
    