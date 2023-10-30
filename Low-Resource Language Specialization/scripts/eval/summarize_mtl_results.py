import collections
import json
import os

results_dir = "/gscratch/cse/echau18/lr-ssmba/allennlp_outputs"

tasks = {
    "pos": "accuracy",
    "ner": "f1-measure-overall",
    "ud": "LAS",
}
langs = ["be", "bg", "ga", "mhr", "mt", "ug", "ur", "vi", "wo"]
reps = ["fasttext", "roberta_best", "mbert", "lapt_best", "tva_best"]

folds = 5

# task/lang/rep/score
results = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: {})
    )
)

for task, metric in tasks.items():
    for lang in langs:
        for rep in reps:
            score = 0.0
            for fold in range(folds):
                standalone_file = os.path.join(
                    results_dir,
                    f"mtl{task}_{lang}_{rep}.{fold}",
                    "metrics.json",
                )
                if not os.path.exists(standalone_file):
                    score += 0.0
                else:
                    with open(standalone_file) as f:
                        standalone_metrics = json.load(f)
                    score += standalone_metrics[
                        f"best_validation_{task}_{metric}"
                    ]
            results[task][lang][rep] = score / folds

            # mtl_file = os.path.join(
            #     results_dir, f"mtl_{lang}_{rep}", "metrics.json"
            # )
            # if not os.path.exists(mtl_file):
            #     score = 0.0
            # else:
            #     with open(mtl_file) as f:
            #         mtl_metrics = json.load(f)
            #     score = mtl_metrics[f"best_validation_{task}_{metric}"]
            # results[task][lang][f"{rep}_mtl"] = score

for task in tasks:
    print(f"===Results for {task}===")
    max_width = max(len(key) for key in results[task][langs[0]])
    separator = "\t"

    print(separator.join(["Rep".ljust(max_width)] + langs))
    for key in results[task][langs[0]]:
        print(
            separator.join(
                [key.ljust(max_width)]
                + [
                    "{:.2f}".format(100 * results[task][lang][key])
                    for lang in langs
                ]
            )
        )

    print()
