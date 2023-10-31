# usage:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


import gc
import io
import json
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist

import deepspeed
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode


# the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--model_name", required=True, type=str, help="model_name")
parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--max_length", type=int)
args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


### Model loading and instantiating on GPUs


model_name = args.model_name
infer_dtype = args.dtype

# print(get_checkpoint_files(model_name))

print_rank0(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=1024,truncate_size="left", truncation=True, padding_side="left")
config = AutoConfig.from_pretrained(model_name)

# XXX: can't automatically derive dtype via config's `from_pretrained`
# dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16


# use one of these args to `init_inference`
# 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
# 2. replace_with_kernel_inject is the faster one (fast fused kernels)
# kernel_inject = True
kernel_inject = False

if kernel_inject:
    # XXX: for now ds-inference only works with fp16
    dtype = torch.float16
else:
    dtype = torch.bfloat16


# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=dtype, device="meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)



model = model.eval()

### Deepspeed-Inference Loading

checkpoints_json = "checkpoints.json"


checkpoints_json=None
model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    dtype=getattr(torch, infer_dtype),
    checkpoint=checkpoints_json,
    # **kwargs,
)

model = model.module


### Generate


print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

batch_size = args.batch_size

################################
# LOAD DATA 
################################

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def load_data():
    with open(args.input_file, "r") as f:
        data = json.load(f)
        if "data" in data:
            data = data["data"]
    
    return data

data = load_data()
my_dataset = ListDataset([d['input'] for d in data])
dataloader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=False)



generate_kwargs = dict(max_new_tokens=args.max_length, do_sample=False)


print_rank0(f"Generate args {generate_kwargs}")



def generate(batch):
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding="longest", truncation=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    output_seq= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    decoded_input = tokenizer.batch_decode(input_tokens["input_ids"], skip_special_tokens=True)
    for i in range(len(output_seq)):
        output_seq[i] = output_seq[i][len(decoded_input[i]):]
    return output_seq


batch_outputs = []
for batch in tqdm(dataloader):
    output_seq = generate(batch)
    torch.cuda.synchronize()
    batch_outputs.extend(output_seq)
    
### Benchmark
save_list = []
for i in range(len(batch_outputs)):
    dict_to_save = {"prediction": batch_outputs[i]}
    for k,v in data[i].items():
        dict_to_save[k] = v
    save_list.append(dict_to_save)
with open(args.output_file,"w") as f:
    json.dump(save_list, f, indent=4)


# # benchmark it!
# if args.benchmark:
#     print_rank0("*** Running benchmark")

#     # warm up
#     for i in range(1):
#         _ = generate()
#     torch.cuda.synchronize()

#     # benchmark
#     t0 = time.time()
#     cycles = 5
#     total_new_tokens_generated = 0
#     for i in range(cycles):
#         generated = generate()
#         total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)
#     torch.cuda.synchronize()
#     throughput = (time.time() - t0) / (total_new_tokens_generated)
#     print_rank0(
#         f"""
# *** Performance stats:
# Throughput per token including tokenize: {throughput*1000:.2f} msecs
# Start to ready to generate: {t_ready - t_start:.3f} secs
# Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
# Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
# """
#     )