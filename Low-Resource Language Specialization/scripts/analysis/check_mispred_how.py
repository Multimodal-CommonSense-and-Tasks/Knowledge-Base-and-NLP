import pickle
import numpy as np
from transformers import BertTokenizer
tok = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

all_dict = pickle.load(open("ug_corpus_dict.pkl", 'rb'))

ADDED_LAST_ID = 122821

for j in range(ADDED_LAST_ID):
    all_dict[j] += 1
    all_dict[j] -= 1

all_dict = dict(sorted(all_dict.items(), key=lambda x: x[1], reverse=True))
id_to_rank = {}
for i, key in enumerate(all_dict.keys()):
    id_to_rank[key] = i


freq_thresh = 0

# freq_thresh = 0.00001
special_ids = [0, 100, 101, 102, 103]
keephead_lines = open("keepheadopt_tfrecord.csv").readlines()[1:]
wohead_lines = open("woheadopt_tfrecord.csv").readlines()[1:]

all_freq = []
wo_all_freq = []
num = 0
freq_array = np.zeros(ADDED_LAST_ID)
wo_freq_array = np.zeros(ADDED_LAST_ID)
for keephead_l, wohead_l in zip(keephead_lines, wohead_lines):
    k_pred, l = keephead_l.strip().split(',')
    w_pred, l = wohead_l.strip().split(',')

    if k_pred != w_pred:
        k_pred = int(k_pred)
        w_pred = int(w_pred)
        all_freq.append(all_dict[k_pred])
        wo_all_freq.append(all_dict[w_pred])
        num += 1

print("Average Freq")
print(sum(all_freq) / num)
print(sum(wo_all_freq) / num)


wo_zero_freq = zero_freq = 0

for keephead_l, wohead_l in zip(keephead_lines, wohead_lines):
    k_pred, l = keephead_l.strip().split(',')
    w_pred, l = wohead_l.strip().split(',')

    if True:
        k_pred = int(k_pred)
        w_pred = int(w_pred)
        if all_dict[k_pred] <= freq_thresh:
            zero_freq += 1
            print(tok.ids_to_tokens[k_pred])
        if all_dict[w_pred] <= freq_thresh:
            wo_zero_freq += 1

print("Below Freq Thresh")
print(zero_freq)
print(wo_zero_freq)


