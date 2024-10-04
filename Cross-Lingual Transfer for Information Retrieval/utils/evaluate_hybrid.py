import argparse
from collections import defaultdict
from trec_eval import trec_eval


def load_runs(fn):
    runs = defaultdict(dict)
    with open(fn) as f:
        for line in f:
            qid, _, docid, _, score, _ = line.rstrip().split()
            runs[qid][docid] = float(score)
    return runs


def main(args):
    runs_1 = load_runs(args.dense)
    runs_2 = load_runs(args.sparse)

    hybrid_result = {}
    output_f = open(args.output, "w")
    alpha = args.alpha

    for key in list(set(runs_1.keys()).union(set(runs_2.keys()))):
        dense_hits = (
            {docid: runs_1[key][docid] for docid in runs_1[key]}
            if key in runs_1
            else {}
        )
        sparse_hits = (
            {docid: runs_2[key][docid] for docid in runs_2[key]}
            if key in runs_2
            else {}
        )

        hybrid_result = []
        min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
        max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
        min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
        max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1
        for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
            if doc not in dense_hits:
                sparse_score = sparse_hits[doc]
                dense_score = min_dense_score
            elif doc not in sparse_hits:
                sparse_score = min_sparse_score
                dense_score = dense_hits[doc]
            else:
                sparse_score = sparse_hits[doc]
                dense_score = dense_hits[doc]

            if args.normalization:
                sparse_score = (
                    0
                    if (max_sparse_score - min_sparse_score) == 0
                    else (
                        (sparse_score - (min_sparse_score + max_sparse_score) / 2)
                        / (max_sparse_score - min_sparse_score)
                    )
                )
                dense_score = (
                    0
                    if (max_sparse_score - min_sparse_score) == 0
                    else (
                        (dense_score - (min_dense_score + max_dense_score) / 2)
                        / (max_dense_score - min_dense_score)
                    )
                )

            score = (
                alpha * sparse_score + dense_score
                if not args.weight_on_dense
                else sparse_score + alpha * dense_score
            )
            hybrid_result.append((doc, score))

        hybrid_result = sorted(hybrid_result, key=lambda x: x[1], reverse=True)[:100]
        for idx, item in enumerate(hybrid_result):
            output_f.write(f"{key} Q0 {item[0]} {idx+1} {item[1]} hybrid\n")
    output_f.close()
    recip_rank, recall_100 = trec_eval(
        f"-c -mrecip_rank -mrecall.100 {args.qrels} {args.output}"
    )
    return recip_rank, recall_100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate runs")
    parser.add_argument("--dense", required=True, help="retrieval run1")
    parser.add_argument("--sparse", required=True, help="retrieval run2")
    parser.add_argument("--output", required=False)
    parser.add_argument(
        "--lang_abbr",
        required=True,
        help="The language abbreviation. (e.g. en, ja, zh, etc.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="The language of the runfiles to compute hybrid on. Need to be set if alpha is not given.",
    )
    parser.add_argument("--normalization", action="store_true")
    parser.add_argument("--weight-on-dense", action="store_true")
    parser.add_argument("--set_name", type=str, default="dev")
    parser.add_argument("--qrels", type=str, default=None)

    args = parser.parse_args()
    alpha_candidates = [
        0.00,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]
    eval_best = {"eval/recip_rank": 0, "eval/recall_100": 0, "alpha": 0}
    if args.qrels is None:
        args.qrels = f"mrtydi-v1.1-{args.lang}-{args.set_name}" 
    recip_rank_dense, recall_100_dense = trec_eval(
        f"-c -mrecip_rank -mrecall.100 {args.qrels} {args.dense}"
    )
    log_prefix = f"eval/{args.set_name}"
    for alpha in alpha_candidates:
        args.alpha = alpha
        args.output = args.dense.replace(".txt", f"_hybrid_{alpha}.txt")
        recip_rank, recall_100 = main(args)
        if eval_best["eval/recip_rank"] < recip_rank:
            eval_best["eval/recip_rank"] = recip_rank
            eval_best["eval/recall_100"] = recall_100
            eval_best["alpha"] = alpha
    print(eval_best)
