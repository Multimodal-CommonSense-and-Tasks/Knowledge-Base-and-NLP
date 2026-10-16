"""
Linear Probe Experiment for Diagnostic Dataset
"""

import json
import os
import numpy as np
import torch
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import re

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    DEPENDENCIES_AVAILABLE = True
    
    # Compatibility patch between PyTorch 2.3.0 and transformers 5.0.0.dev0
    # Fix for torch.is_autocast_enabled() not accepting device_type argument
    try:
        import transformers.utils.generic as generic_module
        
        # Backup original function
        _original_maybe_autocast = generic_module.maybe_autocast
        
        # Patched function
        def patched_maybe_autocast(device_type=None, enabled=True, dtype=None):
            # On PyTorch 2.3.0, torch.is_autocast_enabled() does not accept arguments
            try:
                # Try version that accepts device_type
                is_enabled = torch.is_autocast_enabled(device_type) if device_type else torch.is_autocast_enabled()
            except TypeError:
                # Fallback for versions without device_type argument (PyTorch 2.3.0)
                is_enabled = torch.is_autocast_enabled()
            
            if is_enabled or enabled:
                if device_type == "cuda" or (device_type is None and torch.cuda.is_available()):
                    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
                elif device_type == "cpu":
                    return torch.cpu.amp.autocast(enabled=enabled, dtype=dtype)
                else:
                    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
            else:
                # Context manager that does nothing
                from contextlib import nullcontext
                return nullcontext()
        
        # Apply patch
        generic_module.maybe_autocast = patched_maybe_autocast
    except Exception as e:
        print(f"Warning: Could not patch maybe_autocast: {e}")
    
    # Compatibility patch for bitsandbytes 4bit when torch.nn.Module.set_submodule is missing (PyTorch 2.3.0)
    if not hasattr(torch.nn.Module, 'set_submodule'):
        def _set_submodule(self, target: str, module) -> None:
            tokens = target.split('.')
            submodule = self
            for t in tokens[:-1]:
                submodule = getattr(submodule, t)
            setattr(submodule, tokens[-1], module)
        torch.nn.Module.set_submodule = _set_submodule
        
except ImportError as e:
    print(f"Warning: Some dependencies are not available: {e}")
    DEPENDENCIES_AVAILABLE = False


def load_datasets(data_dir: str = "/workspace/diagnostic_datasets", use_mmlu: bool = False, mmlu_subset: str = None) -> Tuple[List[str], List[int], List[List[str]]]:
    """
    Load dataset(s) and return texts, labels, and numeric information.
    
    Args:
        data_dir: dataset directory
        use_mmlu: if True, use MMLU data; otherwise use the synthetic diagnostic dataset
        mmlu_subset: MMLU subset name (when use_mmlu=True)
    
    Returns:
        texts: list of texts
        labels: list of labels (0: contextual, 1: numerical)
        numbers_list: list of number lists for each text
        contextual: contextual data entries
        numerical: numerical data entries
    """
    if use_mmlu and mmlu_subset:
        # Load MMLU data
        mmlu_file = f'{data_dir}/mmlu_{mmlu_subset}_labeled.json'
        try:
            with open(mmlu_file, 'r', encoding='utf-8') as f:
                mmlu_data = json.load(f)
            
            contextual_data = mmlu_data['contextual']
            numerical_data = mmlu_data['numerical']
            
            # Build text, label, and number lists
            texts = [item['text'] for item in contextual_data] + [item['text'] for item in numerical_data]
            labels = [0] * len(contextual_data) + [1] * len(numerical_data)
            numbers_list = [[item['number']] for item in contextual_data] + [[item['number']] for item in numerical_data]
            
            print(f"MMLU dataset loaded ({mmlu_subset}):")
            print(f"  - Contextual samples: {len(contextual_data)} (label=0)")
            print(f"  - Numerical samples: {len(numerical_data)} (label=1)")
            print(f"  - Total samples: {len(texts)}")
            
            # Numeric information statistics
            num_with_numbers = sum(1 for nums in numbers_list if nums)
            print(f"  - Samples with numeric info: {num_with_numbers}/{len(numbers_list)}")
            
            return texts, labels, numbers_list, contextual_data, numerical_data
            
        except FileNotFoundError:
            print(f"❌ MMLU file not found: {mmlu_file}")
            print("Please run mmlu_analysis.py first to generate the labeled MMLU data.")
            raise
    else:
        # Load existing synthetic diagnostic dataset
        with open(f'{data_dir}/contextual_dataset.json', 'r', encoding='utf-8') as f:
            contextual = json.load(f)
        
        with open(f'{data_dir}/numerical_dataset.json', 'r', encoding='utf-8') as f:
            numerical = json.load(f)
        
        # Build text, label, and number lists
        texts = [item['text'] for item in contextual] + [item['text'] for item in numerical]
        labels = [0] * len(contextual) + [1] * len(numerical)
        numbers_list = [item.get('numbers', []) for item in contextual] + [item.get('numbers', []) for item in numerical]
        
        print("Synthetic diagnostic dataset loaded:")
        print(f"  - Contextual samples: {len(contextual)} (label=0)")
        print(f"  - Numerical samples: {len(numerical)} (label=1)")
        print(f"  - Total samples: {len(texts)}")
        
        # Numeric information statistics
        num_with_numbers = sum(1 for nums in numbers_list if nums)
        print(f"  - Samples with numeric info: {num_with_numbers}/{len(numbers_list)}")
        return texts, labels, numbers_list, contextual, numerical


def train_pytorch_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    lr: float = 0.01,
    random_state: int = 42,
    save_path: str = None,
    early_stopping_patience: int = 20,
    max_epochs: int = None,
    min_epochs: int = 0,
    verbose: bool = True
):
    """
    Train a simple linear layer (nn.Linear) using PyTorch.
    
    This is a genuine "single linear layer":
    y = Wx + b (+ sigmoid for binary classification)
    
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        epochs: maximum number of epochs
        lr: learning rate
        random_state: random seed
        save_path: path to save the model checkpoint (None = do not save)
        early_stopping_patience: early stopping patience
        max_epochs: actual maximum training epochs (None = use epochs, used for generalization tests)
        min_epochs: minimum training epochs before early stopping is allowed
        verbose: whether to log every epoch
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import os
    
    torch.manual_seed(random_state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Define Linear Probe model (single linear layer)
    class LinearProbe(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)  # binary classification: input_dim -> 1
        
        def forward(self, x):
            return self.linear(x)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Determine actual maximum number of epochs
    actual_max_epochs = max_epochs if max_epochs is not None else epochs
    actual_max_epochs = min(actual_max_epochs, epochs)
    
    # Training loop
    print(f"\n  Starting PyTorch Linear Probe training...")
    print(f"  - Max epochs (requested): {epochs}, actual max epochs: {actual_max_epochs}, min epochs: {min_epochs}")
    print(f"  - Early stopping patience: {early_stopping_patience}")
    model.train()
    for epoch in range(actual_max_epochs):
        # Forward
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            # Validation loss and accuracy
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_pred = (torch.sigmoid(val_outputs) > 0.5).float()
            val_acc = (val_pred == y_val_tensor).float().mean().item()
            
            # Test accuracy (for reference)
            test_outputs = model(X_test_tensor)
            test_pred = (torch.sigmoid(test_outputs) > 0.5).float()
            test_acc = (test_pred == y_test_tensor).float().mean().item()
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_acc': test_acc,
            }
            patience_counter = 0
            
            # Save checkpoint
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
        
        model.train()
        
        # Print every epoch (verbose=True) or every 20 epochs
        if verbose or (epoch + 1) % 20 == 0:
            print(
                f"  Epoch [{epoch+1}/{actual_max_epochs}] Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
                f"Best Val Loss: {best_val_loss:.4f}"
            )
        
        # Early stopping (only after min_epochs)
        if epoch + 1 >= min_epochs and patience_counter >= early_stopping_patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch+1} (patience={early_stopping_patience}, min_epochs={min_epochs})")
            break
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(
            f"\n  ✓ Best model loaded "
            f"(Epoch {best_model_state['epoch']}, Val Loss: {best_val_loss:.4f}, "
            f"Val Acc: {best_model_state['val_acc']:.4f})"
        )
        if save_path:
            print(f"  ✓ Model saved to: {save_path}")
    
    return model


def extract_embeddings(
    texts: List[str],
    numbers_list: List[List[str]],
    model_name: str = "gpt2",
    layer_idx: int = -1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    use_num_token: bool = False,
    balance_samples: bool = True,
    device_map: str = None,
    load_in_4bit: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract embeddings **only at numeric token positions** from a language model.
    
    If a sentence contains multiple numbers, a separate sample is created for each number.
    
    Args:
        texts: list of input texts
        numbers_list: list of number lists contained in each text
        model_name: HuggingFace model name
        layer_idx: index of the layer to extract (-1: last layer)
        device: device (cuda/cpu)
        batch_size: batch size
        use_num_token: if True, replace numbers with a special [NUM] token and extract only its positions
        balance_samples: if True, at most one number is used per text (for balancing)
    Returns:
        embeddings: numpy array of shape (total_numbers, hidden_dim),
                    one vector per number
        sample_indices: indices indicating from which original text each embedding came
    """
    print(f"\nExtracting embeddings (numeric token positions only)...")
    print(f"  - Model: {model_name}")
    print(f"  - Layer: {layer_idx}")
    print(f"  - Device: {device}")
    print(f"  - Extraction mode: {'[NUM] token' if use_num_token else 'numeric span (last token only)'}")
    print(f"  - Sample balancing: {'enabled' if balance_samples else 'disabled'}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {}
    if device_map:
        load_kwargs["device_map"] = device_map
        print(f"  - Using multiple GPUs with device_map={device_map}")
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = load_kwargs.get("device_map") or "auto"
            load_kwargs["low_cpu_mem_usage"] = True
            print(f"  - Using 4-bit quantization (memory efficient)")
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization: pip install bitsandbytes"
            ) from e
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    
    # Register [PAD] and [NUM] special tokens and resize embeddings (kept in sync with model_utils.py)
    special_tokens_added = False
    if '[PAD]' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        special_tokens_added = True
    num_token_id = None
    if use_num_token and '[NUM]' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['[NUM]'])
        special_tokens_added = True
    
   
    final_vocab_size = len(tokenizer)
    try:
        if getattr(model.config, 'vocab_size', None) != final_vocab_size:
            model.config.vocab_size = final_vocab_size
        model.resize_token_embeddings(final_vocab_size)
    except Exception:

        pass
    
    if use_num_token:
        num_token_id = tokenizer.convert_tokens_to_ids('[NUM]')
    
    
    uses_device_map = bool(device_map or load_in_4bit)
    if not uses_device_map:
        model.to(device)
    model.eval()
    
    
    if uses_device_map:
        input_device = next(model.parameters()).device
    else:
        input_device = device
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else '[PAD]'
    
    embeddings = []
    sample_indices = []  # index of the original text for each embedding
    num_numbers_extracted = 0
    num_fallback_used = 0
    
    # Process each sample independently (to locate numeric positions)
    for idx, (text, numbers) in enumerate(zip(texts, numbers_list)):
        # In [NUM] mode, replace numbers with [NUM]
        processed_text = text
        if use_num_token:
            processed_text = re.sub(r"\d+(?:\.\d+)?", "[NUM]", text)
        
        # Tokenize
        inputs = tokenizer(
            processed_text, 
            return_tensors='pt', 
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        # Extract embeddings
        # Disable autocast to avoid compatibility issues between PyTorch 2.3.0 and transformers 5.0.0.dev0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(**inputs, output_hidden_states=True)
            
            # Take hidden states from the requested layer
            if layer_idx == -1:
                hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            else:
                hidden_states = outputs.hidden_states[layer_idx]
        
        # [NUM] mode: collect [NUM] token positions and create a sample per position
        if use_num_token and num_token_id is not None:
            token_ids = inputs['input_ids'][0].tolist()
            num_positions = [i for i, tid in enumerate(token_ids) if tid == num_token_id]
            if num_positions:
                if balance_samples:
                    # Sample balancing: at most one number per text
                    pos = num_positions[0]  # use the first [NUM] token
                    vec = hidden_states[0, pos, :].cpu().float().numpy()
                    embeddings.append(vec)
                    sample_indices.append(idx)
                    num_numbers_extracted += 1
                else:
                    # Original behavior: create a sample for every [NUM] token
                    for pos in num_positions:
                        vec = hidden_states[0, pos, :].cpu().float().numpy()
                        embeddings.append(vec)
                        sample_indices.append(idx)
                        num_numbers_extracted += 1
            else:
                # Fallback when there is no [NUM]: use the mean-pooled representation
                attention_mask = inputs['attention_mask'][0].unsqueeze(-1)
                masked_hidden = hidden_states[0] * attention_mask
                vec = (masked_hidden.sum(dim=0) / attention_mask.sum(dim=0)).cpu().float().numpy()
                embeddings.append(vec)
                sample_indices.append(idx)
                num_fallback_used += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx+1}/{len(texts)} samples processed")
            continue
        
        # Number-span based: use character offsets to average tokens per number span, then expand into samples
        if numbers and len(numbers) > 0:
            # Get token IDs
            token_ids = inputs['input_ids'][0].cpu().tolist()
            
            # Get character offsets per token
            offset_mapping = None
            try:
                inputs_with_offsets = tokenizer(
                    text,
                    return_tensors='pt',
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=512
                )
                offset_mapping = inputs_with_offsets['offset_mapping'][0].tolist()
            except:
                pass
            
            number_positions = []
            
            if offset_mapping:
                for num_str in numbers:
                    start_idx = text.find(num_str)
                    if start_idx == -1:
                        continue
                    end_idx = start_idx + len(num_str)
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        if token_start < end_idx and token_end > start_idx:
                            number_positions.append(token_idx)
            else:
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
                for num_str in numbers:
                    for i, token in enumerate(tokens):
                        if num_str in token.strip():
                            number_positions.append(i)
            
            if number_positions:
                number_to_positions = {}
                if offset_mapping:
                    for num_str in numbers:
                        start_idx = text.find(num_str)
                        if start_idx == -1:
                            continue
                        end_idx = start_idx + len(num_str)
                        positions = []
                        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                            if token_start < end_idx and token_end > start_idx:
                                positions.append(token_idx)
                        if positions:
                            number_to_positions[num_str] = positions
                else:
                    tokens = [tokenizer.decode([tid]) for tid in token_ids]
                    for num_str in numbers:
                        positions = []
                        for i, token in enumerate(tokens):
                            if num_str in token.strip():
                                positions.append(i)
                        if positions:
                            number_to_positions[num_str] = positions
                
                if balance_samples:
                    first_number = list(number_to_positions.items())[0]
                    positions = first_number[1]
                    if positions:
                        last_token_pos = positions[-1]  
                        vec = hidden_states[0, last_token_pos, :].cpu().float().numpy()
                        embeddings.append(vec)
                        sample_indices.append(idx)
                        num_numbers_extracted += 1
                else:
                    for _, positions in number_to_positions.items():
                        if positions:
                            last_token_pos = positions[-1]  
                            vec = hidden_states[0, last_token_pos, :].cpu().float().numpy()
                            embeddings.append(vec)
                            sample_indices.append(idx)
                            num_numbers_extracted += 1
                
                if not number_to_positions:
                    attention_mask = inputs['attention_mask'][0].unsqueeze(-1)
                    masked_hidden = hidden_states[0] * attention_mask
                    vec = (masked_hidden.sum(dim=0) / attention_mask.sum(dim=0)).cpu().float().numpy()
                    embeddings.append(vec)
                    sample_indices.append(idx)
                    num_fallback_used += 1
            else:
                attention_mask = inputs['attention_mask'][0].unsqueeze(-1)
                masked_hidden = hidden_states[0] * attention_mask
                vec = (masked_hidden.sum(dim=0) / attention_mask.sum(dim=0)).cpu().float().numpy()
                embeddings.append(vec)
                sample_indices.append(idx)
                num_fallback_used += 1
        else:

            attention_mask = inputs['attention_mask'][0].unsqueeze(-1)
            masked_hidden = hidden_states[0] * attention_mask
            vec = (masked_hidden.sum(dim=0) / attention_mask.sum(dim=0)).cpu().numpy()
            embeddings.append(vec)
            sample_indices.append(idx)
            num_fallback_used += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(texts)} samples processed")
    
    embeddings = np.vstack(embeddings)
    return embeddings, sample_indices


def train_linear_probe(
    embeddings: np.ndarray,
    labels: List[int],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    epochs: int = 100,
    lr: float = 0.01,
    save_path: str = None,
    use_num_token: bool = False,
    early_stopping_patience: int = 20,
    force_retrain: bool = False,
    max_epochs: int = None,
    min_epochs: int = 0,
    verbose: bool = True
) -> Tuple[object, float, Dict]:
    
    import os
    from datetime import datetime
    import torch
    import torch.nn as nn
    
    # Train/Val/Test split (3-way split)
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    input_dim = embeddings.shape[1]
    
    if save_path and os.path.exists(save_path) and not force_retrain:
       
        class LinearProbe(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        checkpoint = torch.load(save_path)
        probe = LinearProbe(input_dim)
        probe.load_state_dict(checkpoint['model_state_dict'])
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        probe.to(device)
        probe.eval()
        
        print(f"  ✓ Model loaded!")
        print(f"    - Trained epoch: {checkpoint['epoch']}")
        print(f"    - Val acc at save: {checkpoint['val_acc']:.4f}")
        print(f"    - Test acc at save: {checkpoint['test_acc']:.4f}")
        
    else:
        # Train model
        if force_retrain and save_path and os.path.exists(save_path):
            print(f"\n⚠ force_retrain=True: ignoring existing model and retraining")
        elif save_path:
            print(f"\n⚠ No saved model found: training from scratch")
        else:
            print(f"\nTraining Linear Probe...")
        
        print(f"  - Method: PyTorch nn.Linear")
        print(f"  - Train samples: {len(X_train)} ({len(X_train)/len(embeddings)*100:.1f}%)")
        print(f"  - Val samples: {len(X_val)} ({len(X_val)/len(embeddings)*100:.1f}%)")
        print(f"  - Test samples: {len(X_test)} ({len(X_test)/len(embeddings)*100:.1f}%)")
        
        # Train PyTorch Linear layer
        probe = train_pytorch_linear_probe(
            X_train, y_train, X_val, y_val, X_test, y_test, 
            epochs=epochs, lr=lr, random_state=random_state,
            save_path=save_path, early_stopping_patience=early_stopping_patience,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            verbose=verbose
        )
    
    # Predict
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        train_logits = probe(X_train_tensor)
        val_logits = probe(X_val_tensor)
        test_logits = probe(X_test_tensor)
        
        y_train_pred = (torch.sigmoid(train_logits) > 0.5).cpu().numpy().astype(int).flatten()
        y_val_pred = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(int).flatten()
        y_pred = (torch.sigmoid(test_logits) > 0.5).cpu().numpy().astype(int).flatten()
    
    train_accuracy = (y_train_pred == y_train).mean()
    val_accuracy = (y_val_pred == y_val).mean()
    test_accuracy = (y_pred == y_test).mean()
    
    print(f"\n✓ Training complete!")
    print(f"  - Train accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  - Val accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  - Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['Contextual', 'Numerical']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Context  Numerical")
    print(f"Actual Context    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"       Numerical  {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Build results dict
    results = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'save_path': save_path,
        'early_stopping_patience': early_stopping_patience
    }
    
    return probe, test_accuracy, results


def analyze_by_category(
    embeddings: np.ndarray,
    sample_indices: List[int],
    contextual_data: List[Dict],
    numerical_data: List[Dict],
    probe: object
) -> None:
    """
    Analyze classification performance by category.
    
    Args:
        embeddings: expanded embeddings (one sample per number)
        sample_indices: original text index for each embedding
        contextual_data: contextual dataset examples
        numerical_data: numerical dataset examples
        probe: trained linear probe
    """
    print("\n" + "="*80)
    print("Category analysis")
    print("="*80)
    
    from collections import defaultdict
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe.eval()
    
    # Start index of numerical data
    num_start_idx = len(contextual_data)
    
    # For each embedding
    for emb_idx, orig_idx in enumerate(sample_indices):
        # Check if this belongs to numerical data
        if orig_idx >= num_start_idx:
            numerical_idx = orig_idx - num_start_idx
            if numerical_idx < len(numerical_data):
                item = numerical_data[numerical_idx]
                embedding = embeddings[emb_idx:emb_idx+1]
                
                # PyTorch prediction
                with torch.no_grad():
                    embedding_tensor = torch.FloatTensor(embedding).to(device)
                    logits = probe(embedding_tensor)
                    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int).flatten()[0]
                
                # MMLU items don't have category labels; treat as 'numerical'
                category = 'numerical'
                
                category_results[category]['total'] += 1
                if pred == 1:  # correctly classified as 'numerical'
                    category_results[category]['correct'] += 1
    
    print("\nNumerical data accuracy by category:")
    for category in sorted(category_results.keys()):
        stats = category_results[category]
        accuracy = stats['correct'] / stats['total']
        print(f"  {category:15s}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")


def layer_wise_analysis(
    texts: List[str],
    labels: List[int],
    numbers_list: List[List[str]],
    model_name: str = "gpt2",
    num_layers: int = None,
    save_dir: str = "/workspace/diagnostic_datasets/saved_models",
    force_retrain: bool = False,
    device_map: str = None,
    load_in_4bit: bool = False,
    **train_kwargs
) -> Dict[int, float]:
    """
    Train/load and evaluate a separate linear probe for each layer.
    
    Args:
        texts: list of texts
        labels: list of labels
        numbers_list: list of numbers found in each text
        model_name: model name
        num_layers: number of layers to analyze (None for all layers)
        save_dir: directory to save probe models
        force_retrain: if True, ignore existing models and retrain
        **train_kwargs: extra args forwarded to train_linear_probe
    
    Returns:
        layer_accuracies: dict of accuracies by layer
    """
    import os
    
    print("\n" + "="*80)
    print("Layer-wise analysis (separate probe per layer)")
    print("="*80)
    
    # Determine number of layers (load config only to avoid OOM)
    if num_layers is None:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = config.num_hidden_layers
    
    # Sanitize model name (remove path separators)
    safe_model_name = model_name.replace('/', '_')
    
    layer_accuracies = {}
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_idx in range(num_layers + 1):  # +1 for embedding layer (0)
        print(f"\n{'='*80}")
        print(f"Analyzing layer {layer_idx}/{num_layers}...")
        print(f"{'='*80}")
        
        # Extract embeddings (number token positions only; one sample per number)
        embeddings, sample_indices = extract_embeddings(
            texts, numbers_list, model_name, layer_idx=layer_idx,
            use_num_token=train_kwargs.get('use_num_token', False),
            balance_samples=train_kwargs.get('balance_samples', True),
            device_map=device_map,
            load_in_4bit=load_in_4bit
        )
        
        # Expand labels (assign each number-sample the label of its source text)
        expanded_labels = [labels[idx] for idx in sample_indices]
        
        # Save path for this layer
        layer_save_path = os.path.join(save_dir, f"linear_probe_{safe_model_name}_layer{layer_idx}.pt")
        
        # Train/load probe for this layer
        # Remove balance_samples from train_kwargs (not used by train_linear_probe)
        filtered_kwargs = {k: v for k, v in train_kwargs.items() if k != 'balance_samples'}
        probe, accuracy, results = train_linear_probe(
            embeddings, 
            expanded_labels,
            save_path=layer_save_path,
            force_retrain=force_retrain,
            **filtered_kwargs
        )
        
        layer_accuracies[layer_idx] = accuracy
        
        print(f"\nFinal accuracy for layer {layer_idx}: {accuracy:.4f}")
    
    # Summarize results
    print("\n" + "="*80)
    print("Accuracy summary by layer")
    print("="*80)
    
    for layer_idx, accuracy in layer_accuracies.items():
        bar_length = int(accuracy * 50)
        bar = "█" * bar_length
        saved_mark = "💾" if os.path.exists(os.path.join(save_dir, f"linear_probe_{safe_model_name}_layer{layer_idx}.pt")) else "  "
        print(f"  {saved_mark} Layer {layer_idx:2d}: {accuracy:.4f} {bar}")
    
    print(f"\n✓ Saved probes for all layers to {save_dir}.")
    
    return layer_accuracies


def evaluate_all_layers_with_pretrained_probe(
    texts: List[str],
    labels: List[int],
    numbers_list: List[List[str]],
    probe_path: str,
    model_name: str = "Qwen/Qwen3-8B",
    num_layers: int = None,
    device_map: str = None,
    load_in_4bit: bool = False,
    **extract_kwargs
) -> Dict[int, float]:
    """
    Load a saved probe model and evaluate it across all layers' embeddings.
    
    Args:
        texts: list of texts
        labels: list of labels
        numbers_list: list of numbers found in each text
        probe_path: path to the saved probe model
        model_name: model name
        num_layers: number of layers to analyze (None for all layers)
        **extract_kwargs: extra args forwarded to extract_embeddings
    
    Returns:
        layer_accuracies: dict of accuracies by layer
    """
    import os
    import torch
    import torch.nn as nn
    
    print("\n" + "="*80)
    print("Evaluate all layers (using a saved probe)")
    print("="*80)
    
    # Verify saved model exists
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Saved model not found: {probe_path}")
    
    # Determine number of layers (load config only to avoid OOM)
    if num_layers is None:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = config.num_hidden_layers
    
    # Merge device_map/load_in_4bit into extract_kwargs
    if device_map is not None:
        extract_kwargs = {**extract_kwargs, "device_map": device_map}
    extract_kwargs["load_in_4bit"] = load_in_4bit
    
    # Load saved probe
    print(f"\nLoading saved probe: {probe_path}")
    checkpoint = torch.load(probe_path, map_location='cpu')
    
    # Define model architecture
    class LinearProbe(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # Infer input dimension from checkpoint
    checkpoint_input_dim = checkpoint['model_state_dict']['linear.weight'].shape[1]
    probe = LinearProbe(checkpoint_input_dim)
    probe.load_state_dict(checkpoint['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe.to(device)
    probe.eval()
    
    print(f"  ✓ Probe loaded (input dim: {checkpoint_input_dim})")
    print(f"    - Trained epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    - Val acc at save: {checkpoint.get('val_acc', 'N/A'):.4f}" if 'val_acc' in checkpoint else "")
    
    layer_accuracies = {}
    
    # Evaluate on each layer
    for layer_idx in range(num_layers + 1):  # +1 for embedding layer (0)
        print(f"\n{'='*80}")
        print(f"Evaluating layer {layer_idx}/{num_layers}...")
        print(f"{'='*80}")
        
        # Extract embeddings
        embeddings, sample_indices = extract_embeddings(
            texts, 
            numbers_list, 
            model_name=model_name, 
            layer_idx=layer_idx,
            **extract_kwargs
        )
        
        # Validate input dimension
        if embeddings.shape[1] != checkpoint_input_dim:
            print(f"  ⚠ Warning: layer {layer_idx} dim ({embeddings.shape[1]}) != probe input dim ({checkpoint_input_dim}). Skipping.")
            layer_accuracies[layer_idx] = None
            continue
        
        # Expand labels
        expanded_labels = [labels[idx] for idx in sample_indices]
        
        # Train/Val/Test split (use the same split)
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            embeddings, expanded_labels, test_size=0.2, random_state=42, stratify=expanded_labels
        )
        
        val_ratio = 0.1 / (1 - 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        # Evaluate
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            test_logits = probe(X_test_tensor)
            y_pred = (torch.sigmoid(test_logits) > 0.5).cpu().numpy().astype(int).flatten()
        
        test_accuracy = (y_pred == y_test).mean()
        layer_accuracies[layer_idx] = test_accuracy
        
        print(f"\nLayer {layer_idx} test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Context  Numerical")
        print(f"Actual Context    {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"       Numerical  {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Summarize results
    print("\n" + "="*80)
    print("Accuracy summary by layer")
    print("="*80)
    
    for layer_idx, accuracy in layer_accuracies.items():
        if accuracy is not None:
            bar_length = int(accuracy * 50)
            bar = "█" * bar_length
            print(f"  Layer {layer_idx:2d}: {accuracy:.4f} {bar}")
        else:
            print(f"  Layer {layer_idx:2d}: N/A (dimension mismatch)")
    
    return layer_accuracies


def main():
    parser = argparse.ArgumentParser(description="Linear Probe Experiment for Diagnostic Dataset (PyTorch only)")
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B', help='Model name from HuggingFace')
    parser.add_argument('--layer', type=int, default=-1, help='Layer index to extract (-1 for last layer)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--layer_wise', action='store_true', 
                       help='Run layer-wise analysis (train separate probe per layer)')
    parser.add_argument('--evaluate_all_layers', action='store_true',
                       help='Evaluate all layers using a pretrained probe model')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Force retrain even if saved models exist')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=20, 
                       help='Early stopping patience')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum training epochs (for generalization test, None=use --epochs)')
    parser.add_argument('--min_epochs', type=int, default=0,
                       help='Minimum training epochs before early stopping can trigger')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print every epoch (default: print every 20 epochs)')
    parser.add_argument('--save_path', type=str, 
                       default='/workspace/diagnostic_datasets/saved_models/linear_probe.pt',
                       help='Path to save/load linear probe model (for single layer mode)')
    parser.add_argument('--save_dir', type=str,
                       default='/workspace/diagnostic_datasets/saved_models',
                       help='Directory to save layer-wise probes')
    parser.add_argument('--data_dir', type=str, default='/workspace/diagnostic_datasets', 
                       help='Directory containing the datasets')
    parser.add_argument('--use_num_token', action='store_true',
                       help='Replace numbers with [NUM] special token and extract only its positions')
    parser.add_argument('--balance_samples', action='store_true', default=True,
                       help='Balance samples by extracting max 1 number per text (default: True)')
    parser.add_argument('--no-balance_samples', action='store_true',
                       help='Disable sample balancing (extract all numbers from each text)')
    parser.add_argument('--use_mmlu', action='store_true',
                       help='Use MMLU dataset instead of synthetic data')
    parser.add_argument('--mmlu_subset', type=str, default='high_school_mathematics',
                       help='MMLU subset name (e.g., high_school_mathematics, college_mathematics)')
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1" for multi-GPU). Sets CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--load_in_4bit', action='store_true',
                       help='Load model in 4-bit quantization (saves VRAM, requires bitsandbytes)')
    
    args = parser.parse_args()
    
    # GPU selection: if --gpus is set, only use those GPUs (use with device_map="auto" for sharding)
    device_map = None
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device_map = "auto"  # shard model across multiple GPUs
        print(f"Using GPUs: {args.gpus} (CUDA_VISIBLE_DEVICES={args.gpus}, device_map=auto)")
    
    # Handle --no-balance_samples
    if args.no_balance_samples:
        args.balance_samples = False
    
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies are not installed.")
        print("Please install: transformers, scikit-learn, matplotlib, seaborn")
        return
    
    print("="*80)
    print(" " * 15 + "Linear Probe Experiment (PyTorch)")
    print("="*80)
    
    # Load datasets
    texts, labels, numbers_list, contextual_data, numerical_data = load_datasets(
        args.data_dir, 
        use_mmlu=args.use_mmlu, 
        mmlu_subset=args.mmlu_subset
    )
    if args.evaluate_all_layers:
        # Evaluate-all-layers mode: evaluate all layers using the saved probe
        print("\n[Evaluate-all-layers mode] Evaluate all layers using the saved probe")
        print("  ✓ We extract and analyze only vectors at number token positions.")
        layer_accuracies = evaluate_all_layers_with_pretrained_probe(
            texts,
            labels,
            numbers_list,
            probe_path=args.save_path,
            model_name=args.model,
            batch_size=args.batch_size,
            use_num_token=args.use_num_token,
            balance_samples=args.balance_samples,
            device_map=device_map,
            load_in_4bit=args.load_in_4bit
        )
    elif args.layer_wise:
        # Layer-wise mode: train/load a separate probe per layer
        print("\n[Layer-wise mode] Train/load a separate probe per layer")
        print("  ✓ We extract and analyze only vectors at number token positions.")
        layer_accuracies = layer_wise_analysis(
            texts, 
            labels,
            numbers_list,
            model_name=args.model,
            save_dir=args.save_dir,
            force_retrain=args.force_retrain,
            device_map=device_map,
            load_in_4bit=args.load_in_4bit,
            test_size=args.test_size,
            val_size=args.val_size,
            epochs=args.epochs,
            lr=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            verbose=args.verbose,
            use_num_token=args.use_num_token,
            balance_samples=args.balance_samples
        )
    else:
        # Default mode: train and save a probe for a specific layer
        print("\n[Default mode] Train and save a probe for a specific layer")
        print("  ✓ We extract and analyze only vectors at number token positions.")
        
        # Extract embeddings (number token positions only; one sample per number)
        embeddings, sample_indices = extract_embeddings(
            texts,
            numbers_list,
            model_name=args.model,
            layer_idx=args.layer,
            batch_size=args.batch_size,
            use_num_token=args.use_num_token,
            balance_samples=args.balance_samples,
            device_map=device_map,
            load_in_4bit=args.load_in_4bit
        )
        
        # Expand labels (assign each number-sample the label of its source text)
        expanded_labels = [labels[idx] for idx in sample_indices]
        
        print(f"\nSample expansion:")
        print(f"  - Original texts: {len(texts)}")
        print(f"  - Expanded samples: {len(embeddings)} (one per number)")
        
        # Train and evaluate linear probe
        probe, accuracy, results = train_linear_probe(
            embeddings, 
            expanded_labels, 
            test_size=args.test_size,
            val_size=args.val_size,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.save_path,
            early_stopping_patience=args.early_stopping_patience,
            force_retrain=args.force_retrain,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            verbose=args.verbose
        )
        
        # Category analysis
        analyze_by_category(embeddings, sample_indices, contextual_data, numerical_data, probe)
    
    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)


if __name__ == "__main__":
    main()


