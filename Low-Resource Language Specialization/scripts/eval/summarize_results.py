import argparse
import collections
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--root-dir", type=str)
args = parser.parse_args()

TASK_METRICS = {
    "ner": "f1-measure-overall",
    "ud": "LAS",
}

LANGUAGES = ["be", "bg", "ga", "mhr", "mt", "ug", "ur", "vi", "wo"]
REPS = ["fasttext", "mbert", "roberta_best", "tva_best"]
SELFSUPS = ["", "mbert", "roberta", "tva_base"]

# task/selfsup/rep/language
results = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: {})
    )
)

for task, metric in TASK_METRICS.items():
    metric_key = f"best_validation_{metric}"
    for selfsup in SELFSUPS:
        if not selfsup:
            prefix = ""
        else:
            prefix = f"selfsup_{selfsup}_"

        for rep in REPS:
            for lang in LANGUAGES:
                config = f"{prefix}{task}_{lang}_{rep}"
                metric_file = os.path.join(
                    args.root_dir, config, "metrics.json"
                )

                if not os.path.exists(metric_file):
                    score = 0.0
                else:
                    with open(metric_file) as jsf:
                        metrics_obj = json.load(jsf)
                    score = metrics_obj[metric_key]
                results[task][selfsup][rep][lang] = score

max_key_len = max(len(rep) for rep in REPS)
for task, metric in TASK_METRICS.items():
    print(f"Results for {task}, based on {metric}")
    for selfsup in SELFSUPS:
        if not selfsup:
            print("Fully supervised")
        else:
            print(f"Augmented with {selfsup}")

        header_row = ["rep/".ljust(max_key_len)] + LANGUAGES
        print("\t".join(header_row))
        for rep in REPS:
            row = [rep.ljust(max_key_len)] + [
                "{:.2f}".format(100 * results[task][selfsup][rep][lang])
                for lang in LANGUAGES
            ]
            print("\t".join(row))

        print()
    print()
