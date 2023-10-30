import argparse
import torch
from allennlp.models.model import Model
from allennlp.common.util import import_module_and_submodules

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=str)
args = parser.parse_args()

import_module_and_submodules("modules-v2")

model = Model.from_archive(args.archive)

assert hasattr(model, "_backbone")
assert hasattr(model._backbone, "_text_field_embedder")
assert hasattr(model._backbone._text_field_embedder, "_token_embedders")

transformer = model._backbone._text_field_embedder._token_embedders[
    "transformer"
]
assert hasattr(transformer, "_matched_embedder")
assert hasattr(transformer._matched_embedder, "_scalar_mix")

scalar_mix = getattr(transformer._matched_embedder, "_scalar_mix")

weights = torch.cat([param for param in scalar_mix.scalar_parameters])

print(weights / weights.norm())
