"""Parse model outputs and compute call accuracy (Eq. 1 of the paper)."""

import json
import re
from collections import defaultdict

from .data import MISSING, SCENARIOS


def parse_call(text):
    """Extract the first JSON object containing `api_name` from a model response."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "api_name" in obj:
            if not isinstance(obj.get("parameters"), dict):
                obj["parameters"] = {}
            return obj
    return None


def _norm(v):
    if isinstance(v, (dict, list)):
        v = json.dumps(v, sort_keys=True)
    return re.sub(r"\s+", " ", str(v)).strip().strip("\"'").lower()


def judge(pred, label):
    """Return (correct, error_type).

    A call is correct when the API name matches and every ground-truth
    parameter has the same value (case/whitespace-insensitive). A ground-truth
    value of "MISSING" (Missing Context scenarios) must be predicted as
    "MISSING" or left out; any concrete value counts as a hallucination.
    Extra optional parameters in the prediction are ignored.
    """
    if pred is None:
        return False, "parse"
    # Candidate lists contain distinct APIs whose names differ only in case
    # (e.g. "Get Product Reviews" vs "GET Product Reviews"), so match exactly.
    if str(pred["api_name"]).strip() != label["api_name"].strip():
        return False, "api"
    pred_params = {k.lower(): v for k, v in pred["parameters"].items()}
    for k, v in label["parameters"].items():
        p = pred_params.get(k.lower())
        if v == MISSING:
            if p is not None and _norm(p) not in ("missing", ""):
                return False, "hallucination"
        elif p is None or _norm(p) != _norm(v):
            return False, "argument"
    return True, None


def summarize(records):
    """Call accuracy per scenario; each sample has exactly one target call."""
    by = defaultdict(list)
    for r in records:
        by[r["scenario"]].append(r["correct"])
    out = {s: 100 * sum(by[s]) / len(by[s]) for s in SCENARIOS.values() if by[s]}
    out["Avg."] = sum(out.values()) / len(out)
    out["Overall"] = 100 * sum(r["correct"] for r in records) / len(records)
    return out


def error_breakdown(records):
    by = defaultdict(lambda: defaultdict(int))
    for r in records:
        if not r["correct"]:
            by[r["scenario"]][r["error"]] += 1
    return {s: dict(v) for s, v in by.items()}
