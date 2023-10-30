from scripts.data.transliterate_nfc_ug_panx import transliterate as transliterate_nfc_ug
from scripts.data.transliterate_nfc_ckb_panx import transliterate as transliterate_nfc_ckb
import argparse
import bert.tokenization as tokenization
import os
from multiprocessing import Pool

def translit(token):
  if args.lang == 'ug':
    return tokenization.convert_to_unicode(transliterate_nfc_ug(token))
  elif args.lang == 'ckb':
    return tokenization.convert_to_unicode(transliterate_nfc_ckb(token))
  else:
    raise NotImplementedError


def check(line):
  try:
    orig_w, translit_w = line.strip().split('\t')
    new_trans = translit(orig_w)
    if translit_w != new_trans:
      print(orig_w, translit_w, new_trans, sep='\n')
  except:
    pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dict_file", type=str, default="ckb_to_ckblatinnfc.txt")
  parser.add_argument("--lang", type=str, default="ckb")
  args = parser.parse_args()

  lines = open(args.dict_file, encoding='utf-8').readlines()
  pool = Pool(32)
  pool.map(check, lines)
