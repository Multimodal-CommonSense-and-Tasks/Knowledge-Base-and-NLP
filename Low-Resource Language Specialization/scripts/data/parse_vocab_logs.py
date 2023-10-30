import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("log_file", type=str, default='specializing-multilingual-data/data/si/unlabeled/bert_cleaned/5000-5-1000-99.log')
parser.add_argument("save_csv", type=str, default="specializing-multilingual-data/data/si/unlabeled/bert_cleaned/5000.csv")
parser.add_argument("--count", type=str, required=True, default="0")
args = parser.parse_args()

lines = open(args.log_file).readlines()
tok, fert, cont, unk, tfert, tcont, tunk, twunk, twconti = lines[4].strip().split('\t\t\t')
if not os.path.exists(args.save_csv):
    with open(args.save_csv, 'w') as f:
        f.write("added_vocs,fert,cont,unk,tfert,tcont,tunk,twunk,twconti\n")
with open(args.save_csv, 'a') as f:
    f.write(f"{args.count},{fert},{cont},{unk},{tfert},{tcont},{tunk},{twunk},{twconti}\n")
