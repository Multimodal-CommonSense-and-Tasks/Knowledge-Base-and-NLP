import glob
import json
import os
import sys
import unicodedata
from scripts.data.transliterate_nfc_ckb_panx import transliterate as _ckb_translit
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("lang", type=str)
parser.add_argument("tl_lang", type=str)
parser.add_argument("--lang_save_path", type=str, default=None)
parser.add_argument("--tl_lang_save_path", type=str, default=None)
parser.add_argument("--suffix", type=str, default='')
# parser.add_argument(
#     "--mode", choices=["NFC", "NFD", "NFKC", "NFKD"], default="NFKC"
# )
args = parser.parse_args()

if not args.lang_save_path:
    args.lang_save_path = f"/data2/{args.lang}_words"
    args.tl_lang_save_path = f"/data2/{args.tl_lang}_words"

lang_save_path = sorted(glob.glob(f'{args.lang_save_path}/*'))
tl_lang_save_path = sorted(glob.glob(f'{args.tl_lang_save_path}/*'))
assert len(lang_save_path) == len(tl_lang_save_path), f"{len(lang_save_path)} {len(tl_lang_save_path)}"

word_dict = {}
for s_p, t_p in zip(lang_save_path, tl_lang_save_path):
    s = open(s_p, encoding='utf-8').read().replace('\n', '')
    t = open(t_p, encoding='utf-8').read().replace('\n', '')
    if not t:
        print(f"something wrong with {t} ")
        # t = _ckb_translit(s.strip()).strip().replace('\n', '')
        # print(f"translited to {t}")
    else:
        word_dict[s] = t

save_folder = "translit_dict"
os.makedirs(save_folder, exist_ok=True)
with open(f"{save_folder}/{args.lang}_to_{args.tl_lang}{args.suffix}.txt", "w", encoding='utf-8') as f:
    for s, t in word_dict.items():
        f.write(f"{s}\t{t}\n")
