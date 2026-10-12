"""Evaluate a model on ToolHaystack through any OpenAI-compatible endpoint.

Works with the OpenAI API, OpenRouter, and a local vLLM server. Predictions are
appended to a JSONL file as they arrive, so an interrupted run resumes where it
stopped.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from src.data import load_samples
from src.evaluate import error_breakdown, judge, parse_call, summarize
from src.prompt import build_messages


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model name served by the endpoint")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                   help="e.g. http://localhost:8000/v1 or https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--data-dir", default=None, help="local copy of the HF dataset (default: download)")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    p.add_argument("--extra-body", default=None,
                   help='JSON passed to the request, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    return p.parse_args()


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    extra_body = json.loads(args.extra_body) if args.extra_body else None

    samples = load_samples(args.data_dir)
    if args.limit:
        samples = samples[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.model.replace("/", "_")
    pred_path = os.path.join(args.output_dir, f"{tag}.jsonl")

    done = {}
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["idx"]] = r
    todo = [s for s in samples if s.idx not in done]
    print(f"{len(samples)} samples, {len(done)} cached, {len(todo)} to run")

    def run(sample):
        resp = client.chat.completions.create(
            model=args.model,
            messages=build_messages(sample),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        output = resp.choices[0].message.content or ""
        pred = parse_call(output)
        correct, error = judge(pred, sample.label)
        return {"idx": sample.idx, "scenario": sample.scenario, "label": sample.label,
                "output": output, "prediction": pred, "correct": correct, "error": error}

    with open(pred_path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, s) for s in todo]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            r = fut.result()
            done[r["idx"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

    records = [done[s.idx] for s in samples]
    scores = summarize(records)
    result = {"model": args.model, "n": len(records), "call_accuracy": scores,
              "errors": error_breakdown(records)}
    with open(os.path.join(args.output_dir, f"{tag}.score.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCall accuracy (%) - {args.model}")
    for k, v in scores.items():
        print(f"  {k:<12} {v:6.2f}")


if __name__ == "__main__":
    main()
