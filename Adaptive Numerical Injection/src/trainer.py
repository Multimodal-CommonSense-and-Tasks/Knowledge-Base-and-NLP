import time
import torch
import os
import math  # required for temperature annealing
from transformers import Trainer, TrainerCallback
from src.eval import evaluate_math_dataset

class CompactProgressCallback(TrainerCallback):
    def __init__(self, script_args=None, model=None):
        self.script_args = script_args
        self.model = model  # keep a reference to the model instance
        self.epoch_start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"\n🚀 Epoch {state.epoch + 1} Start")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(f"Step {state.global_step} | Loss: {logs['loss']:.4f}")
            
            # Optionally print alpha statistics (with safety checks)
            if self.model:
                # Handle potential wrapping (e.g., DDP)
                real_model = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(real_model, 'current_alpha') and real_model.current_alpha is not None:
                    alpha = real_model.current_alpha
                    
                    # Guard against empty tensors before calling min/max
                    if alpha.numel() > 0:
                        non_zero = (alpha != 0).sum().item()
                        total = alpha.numel()
                        print(f"   Alpha stats: min={alpha.min():.6f}, max={alpha.max():.6f}, mean={alpha.mean():.6f}")
                        print(f"   Non-zero alpha: {non_zero}/{total}")
                    else:
                        print(f"   Alpha stats: (No numbers in this batch)")

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.epoch_start_time
        print(f"✅ Epoch {state.epoch} Done | Time: {elapsed:.0f}s")

class MathEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, script_args, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.script_args = script_args
        self.output_dir = output_dir
        
    def on_epoch_end(self, args, state, control, **kwargs):
        evaluate_math_dataset(self.model, self.tokenizer, self.script_args.math_eval_samples_during_train, self.script_args.use_fne, self.output_dir)

class GatingTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            from torch.optim import AdamW
            model = self.model.module if hasattr(self.model, "module") else self.model
            
            gating_params = []
            backbone_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if "gating_network" in name or "encoder" in name:
                    gating_params.append(param)
                else:
                    backbone_params.append(param)
            
            groups = [
                {"params": backbone_params, "lr": self.args.learning_rate},
                {"params": gating_params, "lr": self.args.learning_rate}
            ]
            self.optimizer = AdamW(groups, lr=self.args.learning_rate)
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

import time
import torch
import os
import math  # required for temperature annealing
from transformers import Trainer, TrainerCallback
from src.eval import evaluate_math_dataset

# Temperature annealing callback for gating networks
class GatingTemperatureCallback(TrainerCallback):
    def __init__(self, script_args, model):
        self.script_args = script_args
        self.model = model
    
    def on_step_begin(self, args, state, control, **kwargs):
        # Check whether annealing is enabled and state is valid
        if not self.script_args.anneal_temperature: return
        if state.max_steps is None or state.max_steps <= 0: return

        # 1. Compute progress (0.0 ~ 1.0)
        progress = state.global_step / state.max_steps
        if progress > 1.0: progress = 1.0
        
        t_start = self.script_args.temp_start
        t_end = self.script_args.temp_min
        
        # 2. Compute current temperature (linear or cosine schedule)
        if self.script_args.temp_decay_type == "linear":
            current_temp = t_start - (t_start - t_end) * progress
        elif self.script_args.temp_decay_type == "cosine":
            # Cosine decay: smooth transition
            cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
            current_temp = t_end + (t_start - t_end) * cosine_val
        else:
            current_temp = t_start
            
        # 3. Update temperature in the model
        # Model structure: Trainer -> Model (DDP) -> GatingIntegratedModel -> gating_network(Hybrid) -> gating_network(Core)
        
        real_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        target_network = None
        # GatingIntegratedModel case
        if hasattr(real_model, "gating_network") and hasattr(real_model.gating_network, "gating_network"):
            target_network = real_model.gating_network.gating_network
            
        if target_network and hasattr(target_network, "temperature"):
            target_network.temperature = current_temp