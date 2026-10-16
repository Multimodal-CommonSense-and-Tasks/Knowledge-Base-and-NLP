import torch
import os
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.modules import (
    LoadBalancingLoss,
    GatingLoss,
    GatingNetwork,
    HybridNumberEmbedding,
    FNE,
    BinarizationLoss,
    VanillaEmbedding,
)

class GatingIntegratedModel(torch.nn.Module):
    def __init__(self, base_model, gating_network, gating_layer_index=-1, gating_weight=0.0, balance_weight=0.0):
        super().__init__()
        self.base_model = base_model        
        self.gating_network = gating_network 
        self.config = base_model.config
        self.alpha_history = []
        
        self.gating_weight = gating_weight
        self.balance_weight = balance_weight
        
        self.binarization_loss_fn = BinarizationLoss(weight=0.1)
        self.gating_type = "sigmoid" 

        self.current_alpha = None
        self.layers = self._find_layers(base_model)
        if gating_layer_index == -1: self.gating_layer_index = len(self.layers) - 1
        else: self.gating_layer_index = gating_layer_index
            
        target_layer = self.layers[self.gating_layer_index]
        self.hook_handle = target_layer.register_forward_hook(self._gating_hook)
        self._current_numbers = None
        self._current_number_indexs = None
        self.gating_analysis_data = [] 
        self._current_input_ids = None

    def _find_layers(self, model):
        if hasattr(model, "base_model"): model = model.base_model
        if hasattr(model, "model"): model = model.model
        if hasattr(model, "layers"): return model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
        raise AttributeError(f"Could not find 'layers' in model type: {type(model)}")

    def _gating_hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        if self._current_numbers and self._current_number_indexs:
            gating_output, alpha = self.gating_network(hidden_states.to(torch.float32), self._current_numbers, self._current_number_indexs)
            
            if not self.training and alpha is not None:
                alpha_list = alpha.detach().cpu().view(-1).tolist()
                idx_ptr = 0
                tokenizer = getattr(self, '_tokenizer', None)
                
                batch_full_texts = []
                if tokenizer and hasattr(self, '_current_input_ids') and self._current_input_ids is not None:
                    for i in range(len(self._current_numbers)):
                        if i < self._current_input_ids.shape[0]:
                            text = tokenizer.decode(self._current_input_ids[i], skip_special_tokens=True)
                            batch_full_texts.append(text)
                        else:
                            batch_full_texts.append("")
                
                for batch_idx, (nums, positions) in enumerate(zip(self._current_numbers, self._current_number_indexs)):
                    current_full_text = batch_full_texts[batch_idx] if batch_idx < len(batch_full_texts) else ""
                    for num, pos in zip(nums, positions):
                        if idx_ptr < len(alpha_list):
                            context = ""
                            if tokenizer and hasattr(self, '_current_input_ids') and self._current_input_ids is not None:
                                if batch_idx < self._current_input_ids.shape[0]:
                                    start = max(0, pos - 10)
                                    end = min(self._current_input_ids.shape[1], pos + 11)
                                    context_tokens = self._current_input_ids[batch_idx, start:end]
                                    context = tokenizer.decode(context_tokens, skip_special_tokens=False)

                            self.gating_analysis_data.append({
                                "number": num,
                                "alpha": round(alpha_list[idx_ptr], 4),
                                "context": context.strip(),
                                "full_text": current_full_text,
                                "pos": pos
                            })
                            idx_ptr += 1
            
            self.current_alpha = alpha
            if alpha is not None and not self.training:
                self.alpha_history.extend(alpha.detach().cpu().view(-1).tolist())
            modified = gating_output.to(hidden_states.dtype)
        else:
            modified = hidden_states
        return (modified,) + output[1:] if is_tuple else modified

    def forward(self, input_ids, attention_mask=None, labels=None, number_token_positions=None, numbers=None, **kwargs):
        if numbers is not None: self._current_numbers = numbers
        if number_token_positions is not None: self._current_number_indexs = number_token_positions
        self._current_input_ids = input_ids 
        self.current_alpha = None
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            aux_loss = 0.0
            if self.current_alpha is not None and hasattr(self, 'gating_type') and self.gating_type != 'gumbel':
                aux_loss = self.binarization_loss_fn(self.current_alpha)
            
            loss = main_loss + aux_loss
            
        self._current_numbers = None
        self._current_number_indexs = None
        self._current_input_ids = None
        
        return CausalLMOutputWithPast(loss=loss, logits=outputs.logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)
        torch.save(self.gating_network.state_dict(), os.path.join(output_dir, "gating_network.pt"))
    
    def generate(self, *args, **kwargs): return self.base_model.generate(*args, **kwargs)
    def gradient_checkpointing_enable(self, **kwargs): self.base_model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self): self.base_model.gradient_checkpointing_disable()

class BaselineIntegratedModel(torch.nn.Module):
    def __init__(self, base_model, encoder, mode="fne_only", layer_index=-1):
        super().__init__()
        self.base_model = base_model
        self.encoder = encoder
        self.mode = mode
        self.layer_index = layer_index
        self.config = base_model.config
        layers = self._find_layers(base_model)
        self.target_module = base_model.get_input_embeddings() if layer_index == -1 else layers[layer_index]
        self.hook_handle = self.target_module.register_forward_hook(self._forward_hook)
        self._current_numbers = None
        self._current_number_indexs = None

    def _find_layers(self, model):
        if hasattr(model, "base_model"): model = model.base_model
        if hasattr(model, "model"): model = model.model
        if hasattr(model, "layers"): return model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
        raise AttributeError(f"Could not find 'layers' in model type: {type(model)}")

    def _forward_hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        if not self._current_numbers: return output
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        scatter_numbers = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        has_valid = False
        for batch_idx, (num_strs, pos_list) in enumerate(zip(self._current_numbers, self._current_number_indexs)):
            for num, pos in zip(num_strs, pos_list):
                if pos < seq_len and num is not None:
                    try:
                        val = float(num)
                        if not (torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val))):
                            scatter_numbers[batch_idx, pos] = val
                            has_valid = True
                    except: continue
        if not has_valid: return output
        if self.mode == "xval":
            modified = self.encoder(scatter_numbers, hidden_states)
        elif self.mode in ["fne_only", "vanilla"]:
            numeric = self.encoder(scatter_numbers).to(hidden_states.dtype)
            mask = (scatter_numbers != 0).unsqueeze(-1).expand_as(hidden_states)
            modified = torch.where(mask, numeric, hidden_states)
        else: modified = hidden_states
        
        return (modified,) + output[1:] if is_tuple else modified

    def forward(self, input_ids, attention_mask=None, labels=None, number_token_positions=None, numbers=None, **kwargs):
        self._current_numbers = numbers
        self._current_number_indexs = number_token_positions
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
        self._current_numbers = None
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=outputs.logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def generate(self, *args, **kwargs): return self.base_model.generate(*args, **kwargs)
    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)
        if hasattr(self.encoder, 'state_dict'): torch.save(self.encoder.state_dict(), os.path.join(output_dir, "baseline_encoder.pt"))
    def gradient_checkpointing_enable(self, **kwargs): self.base_model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self): self.base_model.gradient_checkpointing_disable()

class FoNEBaselineModel(torch.nn.Module):
    """
    FoNE Baseline Model
    Mode 1 (Layer 0): Input Addition (+=)
    Mode 2 (Layer >0): Mid-Layer Addition (+=) [User Requested Update]
    """
    def __init__(self, base_model, encoder, layer_index=0):
        super().__init__()
        self.base_model = base_model
        self.encoder = encoder
        self.layer_index = layer_index
        self.config = base_model.config
        
        self._current_numbers = None
        self._current_number_indexs = None
        self._debug_logged = False

        layers = self._find_layers(base_model)
        
        if layer_index == 0:
            print(f"🔧 [FoNE Baseline] Hooking into INPUT Embedding Layer (Mode: ADDITION +=)")
            self.target_module = base_model.get_input_embeddings()
            self.hook_handle = self.target_module.register_forward_hook(self._input_hook)
        else:
            print(f"🔧 [FoNE Baseline] Hooking into Layer {layer_index} (Mode: ADDITION +=)")
            if layer_index >= len(layers):
                raise ValueError(f"Layer index {layer_index} is out of bounds (Total layers: {len(layers)})")
            self.target_module = layers[layer_index]
            self.hook_handle = self.target_module.register_forward_hook(self._mid_layer_hook)

    def _find_layers(self, model):
        if hasattr(model, "base_model"): model = model.base_model
        if hasattr(model, "model"): model = model.model
        if hasattr(model, "layers"): return model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
        raise AttributeError(f"Could not find 'layers' in model type: {type(model)}")

    def _input_hook(self, module, args, output):
        # Input level: base embedding + FoNE
        if not self._current_numbers: return output
        
        fne_embeddings, valid_mask = self._compute_fne_embeddings(output)
        if fne_embeddings is None: return output
        
        modified_output = output.clone()
        modified_output[valid_mask] += fne_embeddings[valid_mask].to(output.dtype)
        
        if not self._debug_logged:
            self._print_debug_info("Input (Addition)", output, modified_output, valid_mask)
            self._debug_logged = True
            
        return modified_output

    def _mid_layer_hook(self, module, args, output):
        # Mid-layer: base hidden state + FoNE (addition)
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        if not self._current_numbers: return output
        
        fne_embeddings, valid_mask = self._compute_fne_embeddings(hidden_states)
        if fne_embeddings is None: return output
        
        modified_hidden = hidden_states.clone()
        
        # Key change: use += (addition) instead of assignment
        modified_hidden[valid_mask] += fne_embeddings[valid_mask].to(hidden_states.dtype)
        
        if not self._debug_logged:
            self._print_debug_info(f"Layer {self.layer_index} (Addition)", hidden_states, modified_hidden, valid_mask)
            self._debug_logged = True
            
        return (modified_hidden,) + output[1:] if is_tuple else modified_hidden

    def _compute_fne_embeddings(self, reference_tensor):
        batch_size, seq_len, _ = reference_tensor.shape
        device = reference_tensor.device
        scatter_numbers = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        has_valid = False
        
        for batch_idx, (num_strs, pos_list) in enumerate(zip(self._current_numbers, self._current_number_indexs)):
            for num, pos in zip(num_strs, pos_list):
                if pos < seq_len and num is not None:
                    try:
                        val = float(num)
                        if not (torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val))):
                            scatter_numbers[batch_idx, pos] = val
                            has_valid = True
                    except: continue
                    
        if not has_valid: return None, None
        
        fne_out = self.encoder(scatter_numbers)
        valid_mask = (scatter_numbers != 0)
        return fne_out, valid_mask

    def _print_debug_info(self, location, original, modified, mask):
        print(f"\n🐞 [FoNE Baseline Debug] Location: {location}")
        num_replaced = mask.sum().item()
        print(f"   - Affected Tokens Count: {num_replaced}")
        if num_replaced > 0:
            b_idx, s_idx = torch.where(mask)
            b, s = b_idx[0], s_idx[0]
            orig_vec = original[b, s, :10].detach().cpu().numpy()
            mod_vec = modified[b, s, :10].detach().cpu().numpy()
            print(f"   - Sample Vector @ ({b},{s}) [First 10 dims]:")
            print(f"     Original: {orig_vec}")
            print(f"     Modified: {mod_vec}")
            
            diff = (original[mask] - modified[mask]).abs().mean().item()
            print(f"   - Mean Absolute Difference: {diff:.6f}")
        print("="*60 + "\n")

    def forward(self, input_ids, attention_mask=None, labels=None, number_token_positions=None, numbers=None, **kwargs):
        self._current_numbers = numbers
        self._current_number_indexs = number_token_positions
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
        self._current_numbers = None
        self._current_number_indexs = None
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(loss=loss, logits=outputs.logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def generate(self, *args, **kwargs): return self.base_model.generate(*args, **kwargs)
    def gradient_checkpointing_enable(self, **kwargs): self.base_model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self): self.base_model.gradient_checkpointing_disable()
    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)
        torch.save(self.encoder.state_dict(), os.path.join(output_dir, "fone_encoder.pt"))

def setup_gating_structure(model, tokenizer, script_args, output_dir=None):
    embedding_dim = model.config.hidden_size
    device = next(model.parameters()).device
    if getattr(script_args, 'gating_encoder', 'fne') == 'vanilla':
        print("🔧 Gating Encoder: Vanilla (Digit-wise)")
        fne_encoder = VanillaEmbedding(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, device=str(device)).to(device)
    else:
        print("🔧 Gating Encoder: FoNE (Fourier)")
        fne_encoder = FNE(embedding_dim, script_args.int_digit_len, script_args.frac_digit_len, script_args.period_base_list, script_args.add_linear, str(device)).to(device)
    
    gating_type = getattr(script_args, "gating_type", "sigmoid") 
    gating_core = GatingNetwork(
        hidden_dim=embedding_dim, 
        embedding_dim=embedding_dim, 
        num_layers=script_args.gating_num_layers, 
        dropout=script_args.gating_dropout, 
        temperature=script_args.gating_temperature,
        gating_type=gating_type
    ).to(device)
    
    num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
    gating_network = HybridNumberEmbedding(fne_encoder, model.get_input_embeddings(), gating_core, num_token_id).to(device)
    
    if output_dir:
        path = os.path.join(output_dir, "gating_network.pt")
        if os.path.exists(path):
            saved_state = torch.load(path, map_location=device)
            print(f"📦 Loading gating_network.pt from {path}")
            try:
                gating_network.load_state_dict(saved_state, strict=False)
            except Exception as e:
                print(f"❌ Error loading state_dict: {e}")
            
    integrated_model = GatingIntegratedModel(model, gating_network, script_args.gating_layer_index, script_args.gating_weight, script_args.balance_weight).to(device)
    integrated_model.gating_type = gating_type 
    
    return integrated_model