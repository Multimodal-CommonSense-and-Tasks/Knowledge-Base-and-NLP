import argparse
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = None
    output_dir: str = "./output"
    model_name: str = "Qwen/Qwen3-8B-Instruct"
    max_seq_length: int = 4096
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    num_train_epochs: float = 3.0
    per_device_batch_size: int = 1
    gradient_accum_steps: int = 32
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "qwen3-sft"
    wandb_run_name: Optional[str] = None
    push_to_hub: bool = False
    use_fne: bool = False
    max_samples: Optional[int] = None
    eval_math: bool = False
    math_max_samples: Optional[int] = None
    eval_only: bool = False
    checkpoint_path: Optional[str] = None
    int_digit_len: int = 10
    frac_digit_len: int = 10
    period_base_list: Optional[List[float]] = None
    add_linear: bool = False
    len_gen_size: int = 10
    clip: float = 1.0
    gating_temperature: float = 1.0
    gating_num_layers: int = 2
    gating_dropout: float = 0.1
    gating_layer_index: int = -1
    freeze_prev_layers: bool = False
    eval_mmlu: bool = False
    mmlu_max_samples: Optional[int] = None
    gating_weight: float = 0.0
    balance_weight: float = 0.0
    freeze_backbone: bool = False
    general_dataset_path: Optional[str] = None
    num_general_samples: Optional[int] = None
    use_lora: bool = False
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    eval_gsm8k: bool = False
    gsm8k_max_samples: Optional[int] = None
    baseline: str = "none"
    xval_max_num: float = 1e9
    gating_encoder: str = "fne"
    baseline_layer_index: int = -1
    num_shots: int = 0
    gating_type: str = "sigmoid"
    anneal_temperature: bool = False
    temp_start: float = 2.0     # initial temperature (higher = softer)
    temp_min: float = 0.5       # final temperature (lower = harder)
    temp_decay_type: str = "cosine" # decay type: 'linear' or 'cosine'
    
def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Qwen3 / Llama SFT Training Script")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accum_steps", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_fne", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="qwen3-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_math", action="store_true")
    parser.add_argument("--math_max_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--int_digit_len", type=int, default=10)
    parser.add_argument("--frac_digit_len", type=int, default=10)
    parser.add_argument("--add_linear", action="store_true")
    parser.add_argument("--len_gen_size", type=int, default=10)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument('--period_base_list', type=str, nargs='+', default=[10])
    parser.add_argument("--gating_layer_index", type=int, default=-1)
    parser.add_argument("--freeze_prev_layers", action="store_true")
    parser.add_argument("--gating_weight", type=float, default=0.0)
    parser.add_argument("--balance_weight", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--gating_temperature", type=float, default=1.0)
    parser.add_argument("--gating_num_layers", type=int, default=2)
    parser.add_argument("--gating_dropout", type=float, default=0.1)
    parser.add_argument("--eval_mmlu", action="store_true")
    parser.add_argument("--mmlu_max_samples", type=int, default=None)
    parser.add_argument("--general_dataset_path", type=str, default=None)
    parser.add_argument("--num_general_samples", type=int, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--eval_gsm8k", action="store_true")
    parser.add_argument("--gsm8k_max_samples", type=int, default=None)
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument("--xval_max_num", type=float, default=1e9)
    parser.add_argument("--gating_encoder", type=str, default="fne")
    parser.add_argument("--baseline_layer_index", type=int, default=-1)
    parser.add_argument("--num_shots", type=int, default=0, help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--gating_type", type=str, default="sigmoid", choices=["sigmoid", "gumbel"], help="Gating mechanism type")
    parser.add_argument("--anneal_temperature", action="store_true", help="Enable Gating Temperature Annealing")
    parser.add_argument("--temp_start", type=float, default=2.0)
    parser.add_argument("--temp_min", type=float, default=0.5)
    parser.add_argument("--temp_decay_type", type=str, default="cosine", choices=["linear", "cosine"])
    args = parser.parse_args()
    args_dict = vars(args)
    if "local_rank" in args_dict:
        args_dict.pop("local_rank")
    
    # Without this return, None would be returned and cause runtime errors.
    return ScriptArguments(**args_dict)