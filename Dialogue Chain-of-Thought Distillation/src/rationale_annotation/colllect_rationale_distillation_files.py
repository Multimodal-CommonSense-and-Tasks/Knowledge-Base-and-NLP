import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", type=str) 
parser.add_argument("--output_dir")
args = parser.parse_args()

dirs = args.dirs.split(",")

files = []
for d in dirs:
    json_files = [os.path.join(d,fn) for fn in os.listdir(d) if ".json" in fn]
    files.extend(json_files)


aggregated = []
for fn in files:
    data = json.load(open(fn, "r"))
    aggregated.extend(data)

for i in range(len(aggregated)):
    aggregated[i]['input'] = "\n".join(aggregated[i]['context'])
    aggregated[i]['label'] =  aggregated[i]['prediction'] if "prediction" in aggregated[i] else aggregated[i]['label']

total_len = len(aggregated)
print(f"Total instances: {total_len}") 
train_set = aggregated[:int(total_len*0.95)]
valid_set = aggregated[int(total_len*0.95):]


with open(os.path.join(args.output_dir, "train.json"), 'w') as f:
    json.dump({"data":train_set}, f, indent=4)


with open(os.path.join(args.output_dir, "validation.json"), 'w') as f:
    json.dump({"data":valid_set}, f, indent=4)

with open(os.path.join(args.output_dir, "total.json"), 'w') as f:
    json.dump({"data":aggregated}, f, indent=4)


