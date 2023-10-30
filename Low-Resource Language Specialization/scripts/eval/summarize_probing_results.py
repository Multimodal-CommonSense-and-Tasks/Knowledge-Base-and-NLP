import collections
import itertools
import json
import os

results_dir = "/m-pinotHD/echau18/lr-ssmba/allennlp_outputs"

metric = "accuracy_words_only"
langs = ["be", "bg", "ga", "mhr", "mt", "ug", "ur", "vi", "wo"]
sources = ["vanilla", "mtlner", "mtlud", "mtl"]
reps = ["mbert", "tva_best", "roberta_best"]

# lang/rep-source/score
results = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))
controls = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))
sensitivities = collections.defaultdict(
    lambda: collections.defaultdict(lambda: {})
)

for lang in langs:
    for rep in reps:
        for source in sources:
            probe_file = os.path.join(
                results_dir, f"probe_{lang}_{rep}_{source}", "metrics.json"
            )
            if not os.path.exists(probe_file):
                probe_score = 0.0
            else:
                with open(probe_file) as f:
                    probe_metrics = json.load(f)
                probe_score = probe_metrics[f"best_validation_{metric}"]
            results[lang][f"{rep}/{source}"] = probe_score

            control_file = os.path.join(
                results_dir,
                f"probecontrol_{lang}_{rep}_{source}",
                "metrics.json",
            )
            if not os.path.exists(control_file):
                control_score = 0.0
            else:
                with open(control_file) as f:
                    control_metrics = json.load(f)
                control_score = control_metrics[f"best_validation_{metric}"]
            controls[lang][f"{rep}/{source}"] = control_score
            sensitivities[lang][f"{rep}/{source}"] = (
                probe_score - control_score
            )

for metric, dict in zip(
    ("POS", "Control", "Sensitivity"), (results, controls, sensitivities)
):
    print(f"===Results for {metric}===")
    max_width = max(len(key) for key in dict[langs[0]])
    separator = "\t"

    print(separator.join(["Rep".ljust(max_width)] + langs))
    for rep, source in itertools.product(reps, sources):
        key = f"{rep}/{source}"
        print(
            separator.join(
                [key.ljust(max_width)]
                + ["{:.2f}".format(100 * dict[lang][key]) for lang in langs]
            )
        )

    print()
