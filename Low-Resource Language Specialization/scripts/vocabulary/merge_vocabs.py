"""add vocabs, originally arabic from mBERT. We used init_vocabs_nolower, which is originated from the vocab-augmented mBERT vocab."""
import shutil
import sys

orig_vocab = sys.argv[1]
lines = open("init_vocabs_nolower.txt", "r", encoding='utf-8').readlines()
suffix = "_augment"
augment_vocab = orig_vocab + "_augment"

shutil.copy(orig_vocab, augment_vocab)
with open(augment_vocab, 'a', encoding='utf-8') as f:
    for line in lines:
        tid, tok = line.strip().split('\t')
        if int(tid) > 100: # ignore the augmented vocabs. We only get arabic tokens from mBERT.
            f.write(f"{tok}\n")
