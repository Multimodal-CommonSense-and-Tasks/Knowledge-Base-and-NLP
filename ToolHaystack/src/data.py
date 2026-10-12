"""Load the ToolHaystack test set and build per-sample candidate API lists."""

import json
import os
import random
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO = "LangAGI-Lab/ToolHaystack"
PARQUET = "data/train-00000-of-00001.parquet"
API_FILE = "data/toolhaystack_apis.json"

# Scenario codes used in the `type` column -> names used in the paper.
SCENARIOS = {
    "cr_s": "CR-Single",
    "cr_m": "CR-Multi",
    "is_e": "IS-Explicit",
    "is_i": "IS-Implicit",
    "mc_e": "MC-Easy",
    "mc_h": "MC-Hard",
}

MISSING = "MISSING"


@dataclass
class Sample:
    idx: int
    scenario: str
    dialogue: str
    label: dict
    candidates: list  # API docs shown to the model (gt + distractors, shuffled)


def _fetch(filename, data_dir):
    if data_dir:
        return os.path.join(data_dir, filename)
    return hf_hub_download(HF_REPO, filename, repo_type="dataset")


def load_apis(data_dir=None):
    with open(_fetch(API_FILE, data_dir)) as f:
        return json.load(f)


def load_samples(data_dir=None, num_candidates=5, seed=42):
    """Return ToolHaystack samples in the paper's oracle retrieval setup.

    The paper gives the model 5 API documents: the ground-truth API plus 4 APIs
    semantically close to the target session. `top_api_indices` holds the
    retrieved APIs; the ground truth is added if missing and the list is
    shuffled so its position carries no signal.
    """
    df = pd.read_parquet(_fetch(PARQUET, data_dir))
    apis = load_apis(data_dir)
    rng = random.Random(seed)

    samples = []
    for i, row in df.iterrows():
        gt = row["gt_api_index"]
        others = [a for a in row["top_api_indices"] if a != gt]
        ids = [gt] + others[: num_candidates - 1]
        rng.shuffle(ids)
        samples.append(
            Sample(
                idx=int(i),
                scenario=SCENARIOS[row["type"]],
                dialogue=row["session_text"].strip(),
                label=json.loads(row["label"]),
                candidates=[apis[a] for a in ids],
            )
        )
    return samples
