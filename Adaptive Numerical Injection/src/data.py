import torch
import random
import re
import json
import tqdm
import os
import logging
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence

# -----------------------------------------------------------------------------
# 1. Helper Functions & Constants
# -----------------------------------------------------------------------------

EXCLUDE_KEYWORDS = [
    "math", "calculation", "calculate", "equation", "algebra", "geometry", 
    "theorem", "arithmetic", "solve", "formula", "integral", "derivative",
    "compute", "numbers", "probability", "statistics",
    "python", "code", "java", "script", "function", "algorithm", 
    "programming", "debug", "compiler", "html", "css", "sql"
]

def filter_magpie_data(example):
    conversations = example.get('conversations', [])
    user_input = ""
    for turn in conversations:
        role = turn.get('role') or turn.get('from')
        if role in ['user', 'human']:
            content = turn.get('content') or turn.get('value')
            if content:
                user_input = str(content).lower()
                break
    if not user_input: return False
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in user_input: return False
    return True

def apply_numerologic_format(text):
    def replace_func(match):
        number_str = match.group()
        if '.' in number_str:
            integer_part, decimal_part = number_str.split('.')
            length_str = f"{len(integer_part)}.{len(decimal_part)}"
        else:
            length_str = str(len(number_str))
        return f"<sn>{length_str}<mn>{number_str}<en>"
    return re.sub(r'\d+(\.\d+)?', replace_func, text)

def clean_llama3_text(text):
    if not text: return text
    pattern = r"(?s)^.*?<\|start_header_id\|>user<\|end_header_id\|>\s*"
    if "<|start_header_id|>user<|end_header_id|>" in text:
        cleaned = re.sub(pattern, "", text)
        return cleaned
    return text

def format_math_with_think(text):
    if not text: return text
    split_pattern = r'(?i)(\n\s*the\s+(?:final\s+)?(?:answer|solution)\s+is[:\s]*)'
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) >= 3:
        reasoning = parts[0].strip()
        answer_intro = parts[1]
        answer_body = parts[2].strip()
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{answer_intro}{answer_body}"
    if r'\boxed{' in text:
        last_boxed_idx = text.rfind(r'\boxed{')
        last_newline_idx = text.rfind('\n', 0, last_boxed_idx)
        if last_newline_idx != -1:
            reasoning = text[:last_newline_idx].strip()
            answer_part = text[last_newline_idx:].strip()
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{answer_part}"
    return text

def format_general_with_think(text):
    if not text: return text
    if "<think>" in text: return text
    return f"<think>\n\n</think>\n\n{text}"

def _compute_number_info(input_ids, tokenizer):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    indices = []
    strings = []
    for idx, token in enumerate(words):
        if not isinstance(token, str): continue
        cleaned = token.replace('Ġ', '').replace('Ċ', '').strip()
        if not cleaned: continue
        try:
            float(cleaned)
            if cleaned in ['.', ',', '-', 'e']: continue
            indices.append(idx)
            strings.append(cleaned)
        except: pass
    return indices, strings

def _compute_number_token_positions_module(input_ids, tokenizer, debug=False, original_text=None, ignore_before_index=0):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    number_indices = []
    number = []
    # If debug is False, no additional logging is produced
    
    if debug and original_text is None:
        try:
            original_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        except:
            original_text = None
    
    numeric_token_indices = []
    
    for index, token in enumerate(words):
        if index < ignore_before_index:
            continue
            
        cleaned_token = token.lstrip('ĠĊ ')
        if not cleaned_token:
            continue
            
        is_numeric = True
        has_digit = False
        for char in cleaned_token:
            if char.isdigit():
                has_digit = True
            elif char not in ['.', '-', ',']:
                is_numeric = False
                break
        
        if is_numeric and has_digit:
            numeric_token_indices.append({
                'index': index,
                'token': token,
                'cleaned': cleaned_token
            })
    
    if numeric_token_indices:
        i = 0
        while i < len(numeric_token_indices):
            group = [numeric_token_indices[i]]
            j = i + 1
            while j < len(numeric_token_indices):
                if numeric_token_indices[j]['index'] <= group[-1]['index'] + 2:
                    group.append(numeric_token_indices[j])
                    j += 1
                else:
                    break
            
            merged_value = ''.join([item['cleaned'] for item in group])
            try:
                final_val = float(merged_value)
                target_idx = group[-1]['index']
                target_token = group[-1]['cleaned']
                
                number_indices.append(target_idx)
                number.append(str(final_val))
            except:
                pass
            
            i = j
            
    # Debug printing was intentionally removed
    return number_indices, number

# -----------------------------------------------------------------------------
# 2. Main Data Loading Function
# -----------------------------------------------------------------------------

def load_and_preprocess_dataset_new(
    dataset_name, 
    tokenizer, 
    num_train_samples=None, 
    method='regular', 
    split_type='train', 
    max_seq_length=4096, 
    general_dataset_name=None, 
    num_general_samples=None,
    seed=42
):
    train_data_list = []
    
    try:
        if isinstance(dataset_name, str): dataset_name = [dataset_name]
        d_name = dataset_name[0]
        if os.path.exists(d_name):
            logging.info(f"Loading Math Dataset from file: {d_name}")
            dataset = load_dataset('json', data_files=d_name, split='train')
        elif os.path.exists(os.path.basename(d_name)):
            logging.info(f"Loading Math Dataset from file: {os.path.basename(d_name)}")
            dataset = load_dataset('json', data_files=os.path.basename(d_name), split='train')
        else:
            logging.info(f"Loading Math Dataset from HuggingFace: {d_name}")
            try:
                dataset = load_dataset(d_name, split='train')
            except:
                dataset = load_dataset(d_name, split='train_1M', streaming=True)
                dataset = list(dataset.take(50000))
    except Exception as e:
        print(f"Data Load Error: {e}")
        return []

    if num_train_samples: 
        if isinstance(dataset, list):
            dataset = dataset[:num_train_samples]
        else:
            dataset = dataset.select(range(min(len(dataset), num_train_samples)))

    general_dataset = []
    if general_dataset_name:
        logging.info(f"Loading General Dataset: {general_dataset_name}")
        try:
            gen_ds = load_dataset(general_dataset_name, split='train')
            gen_ds = gen_ds.filter(filter_magpie_data)
            if num_general_samples is not None:
                if len(gen_ds) > num_general_samples:
                    gen_ds = gen_ds.shuffle(seed=42).select(range(num_general_samples))
            general_dataset = gen_ds
        except Exception as e:
            logging.error(f"Failed to load general dataset: {e}")

    def preprocess_entry(entry, is_general=False, debug=False):
        if is_general:
            raw_msgs = entry.get('conversations', [])
            msgs = []
            for msg in raw_msgs:
                role = msg.get('role') or msg.get('from')
                content = msg.get('content') or msg.get('value')
                if role == 'human': role = 'user'
                elif role == 'gpt': role = 'assistant'
                msgs.append({'role': role, 'content': content})
        else:
            if 'messages' in entry: 
                msgs = entry['messages']
            elif 'conversation' in entry: 
                msgs = entry['conversation']
            elif 'input' in entry and 'output' in entry:
                input_text = str(entry.get('input', ''))
                output_text = str(entry.get('output', ''))
                msgs = [{"role":"user", "content":input_text}, {"role":"assistant", "content":output_text}]
            else: 
                msgs = [{"role":"user", "content":str(entry.get('question',''))}, {"role":"assistant", "content":str(entry.get('answer',''))}]
        
        formatted_msgs = []
        for msg in msgs:
            content = msg['content']
            if msg['role'] == 'user':
                content = clean_llama3_text(content)
            if msg['role'] == 'assistant' and content:
                if is_general:
                    content = format_general_with_think(content)
                else:
                    content = format_math_with_think(content)
            formatted_msgs.append({'role': msg['role'], 'content': content})
        msgs = formatted_msgs

        if method == 'numerologic':
            new_msgs = []
            for msg in msgs:
                if msg['content']:
                    new_content = apply_numerologic_format(str(msg['content']))
                    new_msgs.append({'role': msg['role'], 'content': new_content})
                else:
                    new_msgs.append(msg)
            msgs = new_msgs
        
        training_input_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        if isinstance(training_input_ids, dict): training_input_ids = training_input_ids['input_ids']
        
        if len(training_input_ids) > max_seq_length: 
            training_input_ids = training_input_ids[:max_seq_length]
        
        q_msgs = msgs[:-1] if len(msgs) > 1 else msgs
        training_question_ids = tokenizer.apply_chat_template(q_msgs, tokenize=True, add_generation_prompt=True)
        if isinstance(training_question_ids, dict): training_question_ids = training_question_ids['input_ids']
        
        ignore_idx = 0
        if msgs and msgs[0]['role'] == 'system':
            sys_only_ids = tokenizer.apply_chat_template([msgs[0]], tokenize=True, add_generation_prompt=False)
            if isinstance(sys_only_ids, dict): sys_only_ids = sys_only_ids['input_ids']
            ignore_idx = len(sys_only_ids)
        
        full_text = " ".join([msg.get('content', '') for msg in msgs])
        
        # Force debug=False to avoid verbose logging
        pos, nums = _compute_number_token_positions_module(
            training_input_ids, 
            tokenizer, 
            debug=False,  # turn off debug logging 
            original_text=full_text,
            ignore_before_index=ignore_idx
        )
        
        labels = list(training_input_ids)
        q_len = len(training_question_ids)
        for i in range(min(q_len, len(labels))):
            labels[i] = -100

        if len(training_input_ids) < 20: return None
        actual_text = tokenizer.decode(training_input_ids, skip_special_tokens=True)
        if len(actual_text.strip()) < 10: return None
        if '<redacted' in actual_text.lower() and len(actual_text.strip()) < 50: return None

        return {
            'input_ids': training_input_ids,
            'labels': labels,
            'number_token_positions': pos,
            'numbers': nums,
            'original_text': q_msgs[0]['content'] if q_msgs else ""
        }

    processed_math = [preprocess_entry(item, False, debug=False) for item in dataset]
    processed_math = [item for item in processed_math if item is not None]
    train_data_list.extend(processed_math)
    
    if general_dataset:
        processed_gen = [preprocess_entry(item, True, debug=False) for item in general_dataset]
        processed_gen = [item for item in processed_gen if item is not None]
        train_data_list.extend(processed_gen)
        random.seed(seed)
        random.shuffle(train_data_list)
    
    if len(train_data_list) == 0:
        logging.warning("⚠️ All samples were filtered out!")
    else:
        print(f"\n✅ Final training dataset size: {len(train_data_list)} samples")
        
    return train_data_list

def create_gating_data_collator(tokenizer):
    def collate_fn(examples):
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        labels = [torch.tensor(ex["labels"]) for ex in examples]
        
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "number_token_positions": [ex.get("number_token_positions", []) for ex in examples], 
            "numbers": [ex.get("numbers", []) for ex in examples],
        }
    return collate_fn