import json
import argparse
import sys
sys.path.append("src")
from utils.data import prepare_dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--output_dir", type=str)

args = parser.parse_args()


dialog_dataset = prepare_dataset(args.dataset_name)

splits = ["train", "validation", "test"]


output_dataset = {spl: [] for spl in splits}
for split in splits:
    for dialog in dialog_dataset[split]:
        if args.dataset_name != "mutual" and args.dataset_name != "ed":
            for u_i in range(1, len(dialog)):
                target = dialog[u_i]
                context = dialog[:u_i]
                model_input = "\n".join(context)
                output_dataset[split].append({
                    "input": model_input,
                    "context": context,
                    "target": target
                })
        elif args.dataset_name == "ed":
            for u_i in range(1, len(dialog)):
                if u_i % 2 == 0: continue # for ED we only test listener's turns
                target = dialog[u_i]
                context = dialog[:u_i]
                model_input = "\n".join(context)
                output_dataset[split].append({
                    "input": model_input,
                    "context": context,
                    "target": target
                })
                
        else: # we only test the last utterance for MuTual
            target = dialog[-1]
            context = dialog[:-1]
            model_input = "\n".join(context)
            output_dataset[split].append({
                "input": model_input,
                "context": context,
                "target": target
                
            })

    with open(os.path.join(args.output_dir, split+".json"), "w") as f:
        json.dump({"data": output_dataset[split]}, f, indent=4)



