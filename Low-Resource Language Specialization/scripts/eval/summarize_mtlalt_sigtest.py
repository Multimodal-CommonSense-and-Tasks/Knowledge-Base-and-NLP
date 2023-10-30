import collections
import json
import os
import sys

import numpy as np
import scipy.stats as scp

options = sys.argv[1].split("|")

# TODO: this probably doesn't handle averages right yet


tasks = {
    "pos": "accuracy",
    "ner": "f1-measure-overall",
    "ud": "LAS",
}
langs = ["be", "bg", "ga", "mt", "ug", "ur", "vi", "wo"]
reps = ["fasttext", "roberta_best", "mbert", "lapt_best", "tva_best"]

AVERAGE_KEY = "AVERAGE"
BASELINE_KEY = "BASELINE"

# bonferonni correction (incorrect for wo)
P_VAL = 0.05 / len(tasks)


def _json_lookup_thunk(metric_key):
    def _parser(filename):
        with open(filename) as f:
            metric_dict = json.load(f)
        return metric_dict[metric_key]

    return _parser


def _ud_parser(filename):
    with open(filename) as f:
        lines = f.readlines()
        assert len(lines) == 4
        raw_las_line = lines[-1].strip()
        las_str = raw_las_line[len("Raw LAS: ") :]
        return float(las_str)


def _get_prefix(lang):
    return "mtlalt" if lang != "wo" else "mtlalthom"


dirs = {
    "ms": "/gscratch/cse/echau18/lr-ssmba/result_files_ms",
    "mtl": "/gscratch/cse/echau18/lr-ssmba/result_files_mtl",
    "tva": "/gscratch/cse/echau18/lr-ssmba/result_files_tva",
}

parsers = {
    "ms": {
        "pos": _json_lookup_thunk("ud_tagger_accuracy"),
        "ner": _json_lookup_thunk("ner_f1-measure-overall"),
        "ud": _ud_parser,
        # "ud": _json_lookup_thunk("ud_LAS"),
    },
    "mtl": {
        "pos": _json_lookup_thunk("ud_tagger_accuracy"),
        "ner": _json_lookup_thunk("ner_f1-measure-overall"),
        "ud": _ud_parser,
        # "ud": _json_lookup_thunk("ud_LAS"),
    },
    "tva": {
        "pos": _json_lookup_thunk("pos_accuracy"),
        "ner": _json_lookup_thunk("ner_f1-measure-overall"),
        "ud": _ud_parser,
        # "ud": _json_lookup_thunk("ud_LAS"),
    },
}

filenames = {
    "ms": {
        "pos": "metrics_pos.json",
        "ner": "metrics_ner.json",
        "ud": "metrics_ud.txt",
        # "ud": "metrics_ud.json",
    },
    "mtl": {
        "pos": "metrics_mtl.json",
        "ner": "metrics_mtl.json",
        "ud": "metrics_ud.txt",
        # "ud": "metrics_ud.json",
    },
    "tva": {
        "pos": "metrics_pos.json",
        "ner": "metrics_ner.json",
        "ud": "metrics_ud.txt",
        # "ud": "metrics_ud.json",
    },
}

folds = 5

# task/lang/rep/score
results = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: {})
    )
)

all_results = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: [])
    )
)

# ! for best, * for sig, - for n/a
annotations = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: {})
    )
)

stddevs = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: {})
    )
)

if "nobase" not in options:
    for task, parser in parsers["tva"].items():
        for lang in langs:
            best_for_this_lang = ("fasttext", 0.0)
            for rep in reps:
                score = 0.0
                for fold in range(folds):
                    standalone_file = os.path.join(
                        dirs["tva"],
                        f"mtl{task}_{lang}_{rep}.{fold}.{filenames['tva'][task]}",
                    )
                    if not os.path.exists(standalone_file):
                        print(f"Missing {standalone_file}")
                        curr_fold_score = 0.0
                    else:
                        curr_fold_score = parser(standalone_file)
                    score += curr_fold_score
                    accumulated_scores = []
                    all_results[task][lang][rep].append(curr_fold_score)
                    all_results[task][AVERAGE_KEY][rep].append(curr_fold_score)

                results[task][lang][rep] = score / folds
                if (score / folds) > best_for_this_lang[1]:
                    best_for_this_lang = (rep, score / folds)
            results[task][lang][BASELINE_KEY] = best_for_this_lang[1]
            all_results[task][lang][BASELINE_KEY] = all_results[task][lang][
                best_for_this_lang[0]
            ]
            all_results[task][AVERAGE_KEY][BASELINE_KEY].extend(
                all_results[task][lang][best_for_this_lang[0]]
            )
        for rep in reps + [BASELINE_KEY]:
            results[task][AVERAGE_KEY][rep] = sum(
                all_results[task][AVERAGE_KEY][rep]
            ) / (
                len(all_results[task][AVERAGE_KEY][rep])
                - folds * int(task == "ner")
            )

# mtl
for task in tasks:
    for lang in langs:
        config = _get_prefix(lang)
        for rep in reps:
            rep_key = f"mtlalt(hom) {rep}"
            score = 0.0
            for fold in range(folds):
                standalone_file = os.path.join(
                    dirs["mtl"],
                    f"{config}_{lang}_{rep}.{fold}.{filenames['mtl'][task]}",
                )
                if not os.path.exists(standalone_file):
                    print(f"Missing {standalone_file}")
                    curr_fold_score = 0.0
                elif lang == "wo" and task == "ner":
                    print(f"Skipping wo/ner for {standalone_file}")
                    curr_fold_score = 0.0
                else:
                    curr_fold_score = parsers["mtl"][task](standalone_file)
                score += curr_fold_score
                all_results[task][lang][rep_key].append(curr_fold_score)
                all_results[task][AVERAGE_KEY][rep_key].append(curr_fold_score)
            results[task][lang][rep_key] = score / folds

    for rep in reps:
        rep_key = f"mtlalt(hom) {rep}"
        # NOTE: the -1 here is because there's going to be an extra 0.0 for the
        # empty language.  THis may not apply in future cases
        results[task][AVERAGE_KEY][rep_key] = sum(
            all_results[task][AVERAGE_KEY][rep_key]
        ) / (
            len(all_results[task][AVERAGE_KEY][rep_key])
            - folds * int(task == "ner")
        )

# ms
for task in tasks:
    for lang in langs:
        config = _get_prefix(lang)
        for rep in reps:
            rep_key = f"msmtlalt(hom) {rep}"
            score = 0.0
            for fold in range(folds):
                standalone_file = os.path.join(
                    dirs["ms"],
                    f"ms{task}_{config}_{lang}_{rep}.{fold}.{filenames['ms'][task]}",
                )
                if not os.path.exists(standalone_file):
                    print(f"Missing {standalone_file}")
                    curr_fold_score = 0.0
                elif lang == "wo" and task == "ner":
                    print(f"Skipping wo/ner for {standalone_file}")
                    curr_fold_score = 0.0
                else:
                    curr_fold_score = parsers["ms"][task](standalone_file)
                score += curr_fold_score
                all_results[task][lang][rep_key].append(curr_fold_score)
                all_results[task][AVERAGE_KEY][rep_key].append(curr_fold_score)
            results[task][lang][rep_key] = score / folds
    for rep in reps:
        rep_key = f"msmtlalt(hom) {rep}"
        # NOTE: the -1 here is because there's going to be an extra 0.0 for the
        # empty language.  THis may not apply in future cases
        results[task][AVERAGE_KEY][rep_key] = sum(
            all_results[task][AVERAGE_KEY][rep_key]
        ) / (
            len(all_results[task][AVERAGE_KEY][rep_key])
            - folds * int(task == "ner")
        )

for task in tasks:
    for lang in langs + [AVERAGE_KEY]:
        results_for_task_lang = list(results[task][lang].items())
        results_for_task_lang.sort(key=lambda x: x[1], reverse=True)

        best_key = results_for_task_lang[0][0]
        best_alls = np.array(all_results[task][lang][best_key])

        annotations[task][lang][best_key] = "!"
        # ddof=1 for sample stddev
        stddevs[task][lang][best_key] = np.std(best_alls, ddof=1)

        for key, _ in results_for_task_lang[1:]:
            alls = np.array(all_results[task][lang][key])
            stddevs[task][lang][key] = np.std(alls, ddof=1)

            # we use a 1-sided t-test because the difference is expected to be >0;
            # a difference <0 is bad
            _t, p_val = scp.ttest_1samp(
                best_alls - alls, 0.0, alternative="greater"
            )

            is_significant = p_val < P_VAL
            annotations[task][lang][key] = "-" if is_significant else "*"

for task in tasks:
    print(f"===Results for {task}===")
    max_width = max(len(key) for key in results[task][langs[0]])
    separator = " & " if "latex" in options else "\t"

    headings = langs + [AVERAGE_KEY]

    print(separator.join(["Rep".ljust(max_width)] + headings))
    for key in results[task][langs[0]]:
        print(
            separator.join(
                [key.ljust(max_width)]
                + [
                    "".join(
                        [
                            (
                                annotations[task][lang][key]
                                if "sig" in options
                                else ""
                            ),
                            "{:.2f}".format(100 * results[task][lang][key]),
                            (
                                " \\err{{{:.2f}}}".format(
                                    100 * stddevs[task][lang][key]
                                )
                                if "std" in options
                                else ""
                            ),
                        ]
                    )
                    for lang in headings
                ]
            )
        )

    print()
