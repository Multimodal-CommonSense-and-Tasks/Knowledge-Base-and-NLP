import glob
import os
import sys
import subprocess
from multiprocessing import Pool
from functools import partial
from scripts.data.transliterate_nfc_ckb_panx import transliterate as _ckb_translit
from scripts.data.transliterate_nfc_ug_panx import transliterate as _ug_translit

def ug_nfc_translit(f):
    try:
        name = os.path.relpath(f, args.orig_dir)
        transliterated = _ug_translit(open(f, encoding='utf-8').read().strip())
        with open(os.path.join(args.save_dir, name), 'w', encoding='utf-8') as f:
            f.write(transliterated.strip())
    except Exception as e:
        print(e)

def ckb_translit(f):
    try:
        name = os.path.relpath(f, args.orig_dir)
        transliterated = _ckb_translit(open(f, encoding='utf-8').read().strip())
        with open(os.path.join(args.save_dir, name), 'w', encoding='utf-8') as f:
            f.write(transliterated.strip())
    except Exception as e:
        print(e)

def myvru_translit(f):
    import transliterate
    cyrl_translit = transliterate.get_translit_function("ru")
    try:
        name = os.path.relpath(f, args.orig_dir)
        transliterated = cyrl_translit(open(f, encoding='utf-8').read().strip(), reversed=True)
        with open(os.path.join(args.save_dir, name), 'w', encoding='utf-8') as f:
            f.write(transliterated.strip())
    except Exception as e:
        print(e)


def mt_translit(f):
    try:
        name = os.path.relpath(f, args.orig_dir)
        translit_output = subprocess.run(
            args=[
                'camel_transliterate',
                '-s', 'bw2ar',
                f
            ],
            capture_output=True,
            text=True,
        )
        transliterated = translit_output.stdout.replace('\n', ' ')
        with open(os.path.join(args.save_dir, name), 'w', encoding='utf-8') as f:
            f.write(transliterated.strip())
    except Exception as e:
        print(e)


def wiktratranslit(f):
    from scripts.data.transliterate_wiktra_panx import wiktra_translit
    try:
        name = os.path.relpath(f, args.orig_dir)
        word = open(f, encoding='utf-8').read()
        translit_output = wiktra_translit(word, args.lang)
        transliterated = translit_output.replace('\n', ' ')
        with open(os.path.join(args.save_dir, name), 'w', encoding='utf-8') as f:
            f.write(transliterated.strip())
    except Exception as e:
        print(e)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir", type=str, default='/data2/mt_words')
    parser.add_argument("--save_dir", type=str, default='/data2/mtarab_words')
    parser.add_argument("--lang", type=str, default='mt')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.orig_dir, "*"))
    pool = Pool(16)
    if args.lang == 'mt':
        translit = mt_translit
    elif args.lang == 'ckb':
        translit = ckb_translit
    elif args.lang == 'uglatinnfc':
        translit = ug_nfc_translit
    elif args.lang == 'myvru':
        translit = myvru_translit
    else:
        translit = wiktratranslit

    pool.map(translit, files)
