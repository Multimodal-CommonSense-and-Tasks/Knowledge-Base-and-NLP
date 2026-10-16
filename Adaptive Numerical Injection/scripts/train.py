import os
import sys
import signal
import torch
import shutil
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset

# -----------------------------------------------------------------------------
# Local module imports
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.models import GatingIntegratedModel, BaselineIntegratedModel, FoNEBaselineModel, setup_gating_structure
from src.modules import FNE, XVAL, VanillaEmbedding, GatingNetwork, HybridNumberEmbedding
from src.trainer import GatingTrainer, CompactProgressCallback, MathEvalCallback
from src.eval import run_evaluation_pipeline
from src.data import load_and_preprocess_dataset_new, create_gating_data_collator
from src.config import parse_args

# -----------------------------------------------------------------------------
# Environment settings and compatibility patches
# -----------------------------------------------------------------------------
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
signal.signal(signal.SIGHUP, signal.SIG_IGN)

if hasattr(torch, "is_autocast_enabled"):
    _original_is_autocast_enabled = torch.is_autocast_enabled
    def _is_autocast_enabled_patched(device_type=None):
        return _original_is_autocast_enabled()
    torch.is_autocast_enabled = _is_autocast_enabled_patched

def _set_submodule(self, target: str, module: torch.nn.Module):
    atoms = target.split(".")
    name = atoms.pop(-1)
    mod = self
    for item in atoms:
        mod = getattr(mod, item)
    setattr(mod, name, module)
if not hasattr(torch.nn.Module, "set_submodule"):
    torch.nn.Module.set_submodule = _set_submodule

# -----------------------------------------------------------------------------
# Model and tokenizer loading helper
# -----------------------------------------------------------------------------
def load_model_and_tokenizer(script_args, eval_only=False):
    model_name = script_args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    need_resize = False
    if script_args.use_fne and '[NUM]' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['[NUM]'])
        need_resize = True

    if script_args.baseline == 'numerologic':
        special_tokens = ["<sn>", "<mn>", "<en>"]
        if any(tok not in tokenizer.get_vocab() for tok in special_tokens):
            tokenizer.add_tokens(special_tokens)
            need_resize = True

    if tokenizer.chat_template is None:
        print("ℹ️ Chat template not found, applying a default template.")
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    bnb_config = None
    if getattr(script_args, 'use_qlora', False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if (eval_only or script_args.use_qlora) else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    if not (eval_only or script_args.use_qlora):
        model = model.cuda()
    
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    script_args = parse_args()
    
    import random
    import numpy as np
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(script_args.seed)
        torch.cuda.manual_seed_all(script_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if script_args.use_wandb:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        if script_args.wandb_run_name:
            os.environ["WANDB_NAME"] = script_args.wandb_run_name
    else:
        os.environ["WANDB_DISABLED"] = "true"

    if script_args.eval_only:
        print(f"🚀 Evaluation Only Mode Triggered")
        model, tokenizer = load_model_and_tokenizer(script_args, eval_only=True)
        device = next(model.parameters()).device
        
        if script_args.use_lora and script_args.checkpoint_path:
            print(f"🔄 Loading LoRA Adapter from: {script_args.checkpoint_path}")
            try:
                model = PeftModel.from_pretrained(model, script_args.checkpoint_path)
            except Exception as e:
                print(f"⚠️ Warning while loading LoRA adapter: {e}")
                
        load_dir = script_args.checkpoint_path if script_args.checkpoint_path else script_args.output_dir
        
        if script_args.baseline != "none" and script_args.baseline != "numerologic":
            print(f"🛠️ Restoring Baseline Model ({script_args.baseline})...")
            embedding_dim = model.config.hidden_size
            encoder = None
            if script_args.baseline == "fne_only":
                encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, add_linear=True, device=str(device)).to(device)
            elif script_args.baseline == "xval":
                encoder = XVAL(embedding_dim, script_args.xval_max_num, device=str(device)).to(device)
            elif script_args.baseline == "vanilla":
                encoder = VanillaEmbedding(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, device=str(device)).to(device)
            
            enc_path = os.path.join(load_dir, "baseline_encoder.pt")
            if os.path.exists(enc_path):
                encoder.load_state_dict(torch.load(enc_path, map_location=device))
                print("✅ Baseline Encoder weights loaded.")
            
            model = BaselineIntegratedModel(model, encoder, mode=script_args.baseline, layer_index=script_args.baseline_layer_index).to(device)

        elif script_args.use_fne:
            print(f"🛠️ Restoring Gating Network from {load_dir}...")
            model = setup_gating_structure(model, tokenizer, script_args, output_dir=load_dir)

        run_evaluation_pipeline(script_args, checkpoint_path=script_args.checkpoint_path, model=model, tokenizer=tokenizer)
        return

    # -------------------------------------------------------------------------
    # Training Mode
    # -------------------------------------------------------------------------
    print(f"🚀 Training Mode Start: {script_args.model_name}")
    use_fne_flag = (script_args.use_fne or (script_args.baseline != "none" and script_args.baseline != "numerologic"))
    
    model, tokenizer = load_model_and_tokenizer(script_args, eval_only=False)
    model.config.use_cache = False

    if script_args.use_lora:
        if getattr(script_args, 'use_qlora', False):
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(model, peft_config)

    device = next(model.parameters()).device
    embedding_dim = model.config.hidden_size

    if script_args.baseline != "none" and script_args.baseline != "numerologic":
        print(f"\n🚀 Baseline Mode: {script_args.baseline.upper()}")
        embedding_dim = model.config.hidden_size
        load_dir = script_args.checkpoint_path if script_args.checkpoint_path else script_args.output_dir
        # Configure encoder (unified as FoNE)
        encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, add_linear=True, device=str(device)).to(device)
        
        # Load pre-trained encoder weights if available
        enc_path = os.path.join(load_dir, "fone_encoder.pt")
        if os.path.exists(enc_path):
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
            print("✅ FoNE Encoder weights loaded.")

        # Baseline 선택 로직
        if script_args.baseline == "fone_input":
            # Method 1: Input layer replacement (layer index 0)
            print("Running Method 1: Original FoNE (Input Layer Replacement)")
            model = FoNEBaselineModel(model, encoder, layer_index=0).to(device)
            
        elif script_args.baseline == "fone_mid":
            # Method 2: Mid-Layer Replacement (Use gating_layer_index as target)
            target_layer = script_args.gating_layer_index if script_args.gating_layer_index != -1 else 12 # Default fallback
            print(f"Running Method 2: Mid-Layer Replacement at Layer {target_layer}")
            model = FoNEBaselineModel(model, encoder, layer_index=target_layer).to(device)
        else:
            if script_args.baseline == "fne_only":
                encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, add_linear=True, device=str(device)).to(device)
            elif script_args.baseline == "xval":
                encoder = XVAL(embedding_dim, script_args.xval_max_num, device=str(device)).to(device)
            elif script_args.baseline == "vanilla":
                encoder = VanillaEmbedding(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, device=str(device)).to(device)
            
            model = BaselineIntegratedModel(model, encoder, mode=script_args.baseline, layer_index=script_args.baseline_layer_index).to(device)
        
        if script_args.freeze_backbone:
            for param in model.base_model.parameters(): param.requires_grad = False
            for param in model.encoder.parameters(): param.requires_grad = True

    elif script_args.use_fne:
        print(f"\n🚀 Gating Mode: Hybrid (Residual) | Encoder: {script_args.gating_encoder.upper()}")
        fne_encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, script_args.add_linear, device=str(device)).to(device)
        gating_core = GatingNetwork(embedding_dim, embedding_dim, script_args.gating_num_layers, script_args.gating_dropout, script_args.gating_temperature).to(device)
        num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
        
        gating_network = HybridNumberEmbedding(fne_encoder, model.get_input_embeddings(), gating_core, num_token_id).to(device)
        model = GatingIntegratedModel(model, gating_network, script_args.gating_layer_index, script_args.gating_weight, script_args.balance_weight).to(device)

        if script_args.freeze_backbone:
            for param in model.base_model.parameters(): param.requires_grad = False
            for param in model.gating_network.parameters(): param.requires_grad = True

    # [NEW] Updated parameter counting logic (avoid negative counts)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all_params = sum(p.numel() for p in model.parameters())
    
    gating_trainable_params = 0
    gating_all_params = 0
    
    # Gating network parameters (excluding shared embedding layers; count only additional modules)
    if hasattr(model, "gating_network"):
        # 1. Gating core (MLP)
        if hasattr(model.gating_network, "gating_network"):
             gating_trainable_params += sum(p.numel() for p in model.gating_network.gating_network.parameters() if p.requires_grad)
             gating_all_params += sum(p.numel() for p in model.gating_network.gating_network.parameters())
        
        # 2. FNE Encoder
        if hasattr(model.gating_network, "fne_encoder"):
             gating_trainable_params += sum(p.numel() for p in model.gating_network.fne_encoder.parameters() if p.requires_grad)
             gating_all_params += sum(p.numel() for p in model.gating_network.fne_encoder.parameters())
             
    elif hasattr(model, "encoder"):
         gating_trainable_params += sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
         gating_all_params += sum(p.numel() for p in model.encoder.parameters())

    # LoRA params = (all trainable params) - (pure gating trainable params)
    lora_params = total_trainable_params - gating_trainable_params
    
    # Backbone params = (all params) - (LoRA) - (pure gating)
    backbone_params = total_all_params - lora_params - gating_all_params
    
    print(f"\n==================================================")
    print(f"📊 Detailed Parameter Statistics (Approx.)")
    print(f"==================================================")
    print(f"   • Backbone Model Params : {backbone_params:,}")
    print(f"     (Note: Lower than 7B/8B due to 4-bit packing)" if getattr(script_args, 'use_qlora', False) else "")
    print(f"   • LoRA Params           : {lora_params:,}")
    print(f"   • Gating/Encoder Params : {gating_all_params:,}")
    print(f"--------------------------------------------------")
    print(f"   • Total Trainable Params: {total_trainable_params:,}")
    print(f"   • All Params            : {total_all_params:,}")
    print(f"   • Trainable % (vs Backbone): {total_trainable_params / backbone_params * 100:.4f}%")
    print(f"==================================================\n")

    preprocess_method = "regular"
    if script_args.baseline != "none": preprocess_method = script_args.baseline
    elif script_args.use_fne: preprocess_method = "gating_full_finetuning"

    train_data_list = load_and_preprocess_dataset_new(
        dataset_name=script_args.dataset_path, 
        tokenizer=tokenizer, 
        num_train_samples=script_args.max_samples,
        method=preprocess_method, 
        split_type='train',
        max_seq_length=script_args.max_seq_length,
        general_dataset_name=script_args.general_dataset_path,
        num_general_samples=script_args.num_general_samples,
        seed=script_args.seed
    )
    tokenized_dataset = Dataset.from_list(train_data_list)
    
    if not use_fne_flag:
        remove_cols = [c for c in ["numbers", "answer_text", "solution", "source"] if c in tokenized_dataset.column_names]
        if remove_cols: tokenized_dataset = tokenized_dataset.remove_columns(remove_cols)

    data_collator = create_gating_data_collator(tokenizer) if use_fne_flag else DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_batch_size,
        gradient_accumulation_steps=script_args.gradient_accum_steps,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        logging_steps=script_args.logging_steps,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=["wandb"] if script_args.use_wandb else [],
        ddp_find_unused_parameters=True,
        remove_unused_columns=(not use_fne_flag),
        disable_tqdm=True    )

    callbacks = [CompactProgressCallback(script_args=script_args, model=model)]
    # if (script_args.eval_math or script_args.eval_gsm8k or script_args.eval_mmlu):
    #     callbacks.append(MathEvalCallback(model=model, tokenizer=tokenizer, script_args=script_args, output_dir=script_args.output_dir))
    if script_args.anneal_temperature:
        from src.trainer import GatingTemperatureCallback  # import here to avoid circular imports
        print(f"🔥 Temperature annealing enabled: {script_args.temp_start} -> {script_args.temp_min} ({script_args.temp_decay_type})")
        callbacks.append(GatingTemperatureCallback(script_args=script_args, model=model))
        
    trainer_cls = GatingTrainer if (script_args.baseline != "none" or script_args.use_fne) else Trainer
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    trainer.tokenizer = tokenizer

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)

    print("\n✅ Finished training and saved model.")

    if script_args.eval_math or script_args.eval_gsm8k or script_args.eval_mmlu:
        print("\n🧹 Cleaning up memory... (deleting Trainer and optimizer state)")
        
        # 1. Delete training-related objects (especially optimizer state)
        del trainer
        del model
        
        # 2. Run garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        
        print("🚀 Reloading model for evaluation... (inference mode)")
        
        # 3. Reload model in "evaluation mode"
        # load_model_and_tokenizer uses a more memory‑efficient configuration when eval_only=True.
        # Temporarily point script_args.checkpoint_path to the just‑saved output_dir.
        original_ckpt_path = script_args.checkpoint_path
        script_args.checkpoint_path = script_args.output_dir
        
        # Reload (eval_only=True)
        model_eval, tokenizer_eval = load_model_and_tokenizer(script_args, eval_only=True)
        
        # Restore Baseline / Gating structure on the reloaded model
        device = next(model_eval.parameters()).device
        load_dir = script_args.output_dir
        
        if script_args.baseline != "none" and script_args.baseline != "numerologic":
            print(f"🛠️ Restoring baseline structure ({script_args.baseline})...")
            # ... (Reuse the same encoder restoration logic as in the training path) ...
            embedding_dim = model_eval.config.hidden_size
            encoder = None
            # (Ideally, encoder initialization is factored out and reused;
            #  here we inline the minimal required logic.)
            if script_args.baseline == "fne_only":
                encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, add_linear=True, device=str(device)).to(device)
            elif script_args.baseline == "fone_input" or script_args.baseline == "fone_mid": 
                 encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, add_linear=True, device=str(device)).to(device)
            

            enc_path = os.path.join(load_dir, "fone_encoder.pt") if "fone" in script_args.baseline else os.path.join(load_dir, "baseline_encoder.pt")
            if os.path.exists(enc_path) and encoder is not None:
                encoder.load_state_dict(torch.load(enc_path, map_location=device))
            
            
            if script_args.baseline == "fone_input":
                model_eval = FoNEBaselineModel(model_eval, encoder, layer_index=0).to(device)
            elif script_args.baseline == "fone_mid":
                target_layer = script_args.gating_layer_index if script_args.gating_layer_index != -1 else 12
                model_eval = FoNEBaselineModel(model_eval, encoder, layer_index=target_layer).to(device)
            

        elif script_args.use_fne:
            print(f"🛠️ Restoring gating network structure...")
            # Use setup_gating_structure helper
            model_eval = setup_gating_structure(model_eval, tokenizer_eval, script_args, output_dir=load_dir)

        # 4. Run evaluation pipeline
        run_evaluation_pipeline(
            script_args, 
            checkpoint_path=script_args.output_dir, 
            model=model_eval,          
            tokenizer=tokenizer_eval   
        )
        
        # Restore original checkpoint path to avoid side effects
        script_args.checkpoint_path = original_ckpt_path
        print("\n✅ Finished full pipeline (training + reload + evaluation).")

if __name__ == "__main__":
    main()