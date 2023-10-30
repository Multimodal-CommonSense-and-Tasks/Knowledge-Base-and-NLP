import argparse
import json
import os

fields = {
    "gpu": "training_gpu_0_memory_MB",
    "time": "training_duration",
}

buckets = {
    "gpu": [
        (10000, "2080ti"),
        (15000, "p100"),
        (25000, "titan"),
        (90000, "SPECIAL"),
    ],
    "time": [
        (10, "1h"),
        (15, "1h30"),
        (30, "2h30"),
        (45, "3h30"),
        (60, "4h30"),
        (135, "10h"),
        (180, "12h30"),
        (210, "15h"),
        (240, "16h30"),
        (270, "18h30"),
        (300, "20h30"),
        (900, "SPECIAL"),
    ],
}


def parse_time(time_str):
    h, m, s = time_str.split(":")
    return 60 * int(h) + int(m)


functions = {
    "gpu": lambda x: x,
    "time": parse_time,
}

parser = argparse.ArgumentParser()
parser.add_argument("--metrics", type=str)
parser.add_argument("--field", type=str, choices=fields.keys())
args = parser.parse_args()

if not os.path.exists(args.metrics):
    print("NOT DONE YET!")
    exit()

with open(args.metrics) as f:
    metrics = json.load(f)

raw_value = metrics[fields[args.field]]
transformed_value = functions[args.field](raw_value)

for upperbound, label in buckets[args.field]:
    if transformed_value < upperbound:
        print(label)
        break
