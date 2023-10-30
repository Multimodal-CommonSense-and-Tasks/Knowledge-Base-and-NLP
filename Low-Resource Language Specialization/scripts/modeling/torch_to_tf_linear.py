import glob
import shutil

import numpy
import torch
import os
from util.io_utils import get_filename, get_folderpath
import tensorflow as tf
from transformers import HfArgumentParser
from scripts.modeling.manipulate_ckpt import CkptChanger
from dataclasses import dataclass, field

@dataclass
class Args:
    torch_layer_path: str
    tf_ckpt_path: str
    tf_ckpt_save_path: str
    linear_layer_path: str = field(default="bert/final_align_transform")

parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)

torch_layer = torch.load(args.torch_layer_path, map_location='cpu')
assert isinstance(torch_layer, torch.nn.Linear)
torch_weight = torch_layer.weight.detach().numpy()
if torch_layer.bias is None:
    torch_bias = numpy.zeros(torch_weight.shape[0], dtype=torch_weight.dtype)
else:
    torch_bias = torch_layer.bias.detach().numpy()

ckpt_changer = CkptChanger(args.tf_ckpt_path)
ckpt_changer.set_val(f'{args.linear_layer_path}/kernel', torch_weight.T)
ckpt_changer.set_val(f'{args.linear_layer_path}/bias', torch_bias)
ckpt_changer.save(args.tf_ckpt_save_path)

os.makedirs(args.tf_ckpt_save_path, exist_ok=True)
for f_suffix in [".data-00000-of-00001", '.index']:
    shutil.move(f"{args.tf_ckpt_save_path}{f_suffix}", args.tf_ckpt_save_path)

file_name = get_filename(args.tf_ckpt_save_path, remove_ext=False)
with open(os.path.join(args.tf_ckpt_save_path, "checkpoint"), "w") as f:
    string = f"""model_checkpoint_path: "{file_name}"
all_model_checkpoint_paths: "{file_name}"
"""
    f.write(string)
