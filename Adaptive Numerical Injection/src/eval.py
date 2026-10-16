import torch
import os
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
# Import the numeric‑token detection helper used during training
from src.data import _compute_number_token_positions_module

# -----------------------------------------------------------------------------
# 1. Prompt Constants (Full Version)
# -----------------------------------------------------------------------------

# Restored: MATH few‑shot prompt (4‑shot) for base models
# FEW_SHOT_PROMPT = r"""Question:
# Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# My solution:
# First, calculate the total cost of the bagels. Olivia bought 5 bagels at $3 each.
# Cost = 5 * 3 = 15.
# She started with $23 and spent $15.
# Money left = 23 - 15 = 8.
# She has $8 left.
# The final answer is: \boxed{8}

# ---

# Question:
# A pen and its ink refill together cost $1.10. The pen costs $1 more than the ink refill. What is the cost of the pen in dollars?

# My solution:
# Let p be the cost of the pen and i be the cost of the ink refill.
# We are given:
# 1. p + i = 1.10
# 2. p = i + 1
# Substitute equation 2 into equation 1:
# (i + 1) + i = 1.10
# 2i + 1 = 1.10
# 2i = 0.10
# i = 0.05
# Now find the cost of the pen, p:
# p = i + 1 = 0.05 + 1 = 1.05
# The cost of the pen is $1.05.
# The final answer is: \boxed{1.05}

# ---

# Question:
# What is the value of $f(f(1))$ if $f(x) = 2x + 3$?

# My solution:
# We need to find the value of f(f(1)).
# First, we calculate f(1):
# f(1) = 2(1) + 3 = 2 + 3 = 5.
# Now we substitute this result back into f(x):
# f(f(1)) = f(5).
# f(5) = 2(5) + 3 = 10 + 3 = 13.
# The final answer is: \boxed{13}

# ---

# Question:
# How many ways can 3 people (Alice, Bob, Charlie) line up in a single file line?

# My solution:
# This is a permutation problem.
# For the first spot in line, there are 3 choices.
# For the second spot, there are 2 remaining choices.
# For the last spot, there is only 1 choice left.
# The total number of ways is 3 * 2 * 1 = 6.
# The final answer is: \boxed{6}

# ---
# """
MATH_EXAMPLES_LIST = [
    {"q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "a": "First, calculate the total cost of the bagels. Olivia bought 5 bagels at $3 each. Cost = 5 * 3 = 15. She started with $23 and spent $15. Money left = 23 - 15 = 8. The final answer is: \\boxed{8}"},
    {"q": "A pen and its ink refill together cost $1.10. The pen costs $1 more than the ink refill. What is the cost of the pen in dollars?", "a": "Let p be the cost of the pen and i be the cost of the ink refill. We are given: 1. p + i = 1.10, 2. p = i + 1. Substitute equation 2 into equation 1: (i + 1) + i = 1.10 -> 2i + 1 = 1.10 -> 2i = 0.10 -> i = 0.05. Now find p: p = i + 1 = 0.05 + 1 = 1.05. The final answer is: \\boxed{1.05}"},
    {"q": "What is the value of f(f(1)) if f(x) = 2x + 3?", "a": "First, we calculate f(1): f(1) = 2(1) + 3 = 2 + 3 = 5. Now substitute this into f(x): f(f(1)) = f(5) = 2(5) + 3 = 10 + 3 = 13. The final answer is: \\boxed{13}"},
    {"q": "How many ways can 3 people (Alice, Bob, Charlie) line up in a single file line?", "a": "This is a permutation problem. For the first spot, there are 3 choices. For the second, 2 remaining. For the last, 1 choice left. Total ways = 3 * 2 * 1 = 6. The final answer is: \\boxed{6}"}
]
# Restored: GSM8K 8‑shot prompt (kept for base models)
# GSM8K_8_SHOT_PROMPT = r"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
# A: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6. The answer is 6.

# Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
# A: There are 3 cars originally. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

# Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
# A: Leah had 32 chocolates and her sister had 42. That means there were 32 + 42 = 74 chocolates. 35 were eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

# Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
# A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he gave to Denny must have been 20 - 12 = 8. The answer is 8.

# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
# A: He has 5 toys. As he got 2 from mom and 2 from dad, he got 2 + 2 = 4 more. 5 + 4 = 9. The answer is 9.

# Q: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
# A: There were 9 computers. Then 5 more were installed for 4 days, which means 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
# A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he had 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
# A: She bought 5 bagels for $3 each. This means she spent 5 * 3 = $15. She had $23 so she has 23 - 15 = 8. The answer is 8.

# Q: """

GSM8K_EXAMPLES_LIST = [
    {"q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "a": "There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6. The answer is 6."},
    {"q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "a": "There are 3 cars originally. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
    {"q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "a": "Leah had 32 chocolates and her sister had 42. That means there were 32 + 42 = 74 chocolates. 35 were eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."},
    {"q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "a": "Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he gave to Denny must have been 20 - 12 = 8. The answer is 8."},
    {"q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "a": "He has 5 toys. As he got 2 from mom and 2 from dad, he got 2 + 2 = 4 more. 5 + 4 = 9. The answer is 9."},
    {"q": "There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?", "a": "There were 9 computers. Then 5 more were installed for 4 days, which means 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."},
    {"q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "a": "Michael initially had 58 balls. He lost 23 on Tuesday, so after that he had 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33."},
    {"q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "a": "She bought 5 bagels for $3 each. This means she spent 5 * 3 = $15. She had $23 so she has 23 - 15 = 8. The answer is 8."}
]

# Restored: full list of MMLU subjects (57)
MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 
    'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 
    'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 
    'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 
    'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 
    'virology', 'world_religions'
]

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str:
    start_keyword = "\\boxed{"
    start_index = text.find(start_keyword)
    if start_index == -1: return ""
    content_start = start_index + len(start_keyword)
    brace_count = 1
    current_index = content_start
    while brace_count > 0 and current_index < len(text):
        char = text[current_index]
        if char == '{': brace_count += 1
        elif char == '}': brace_count -= 1
        current_index += 1
    if brace_count == 0: return text[content_start:current_index-1].strip()
    return ""

def get_math_few_shot_text(num_shots):
    if num_shots <= 0: return ""
    text = ""
    for i in range(min(num_shots, len(MATH_EXAMPLES_LIST))):
        ex = MATH_EXAMPLES_LIST[i]
        text += f"Question:\n{ex['q']}\n\nMy solution:\n{ex['a']}\n\n---\n\n"
    return text

def get_few_shot_text(num_shots):
    """Generate a few-shot text block with the given number of examples."""
    if num_shots <= 0: return ""
    text = ""
    for i in range(min(num_shots, len(GSM8K_EXAMPLES_LIST))):
        ex = GSM8K_EXAMPLES_LIST[i]
        text += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    return text

def normalize_answer(answer: str) -> str:
    return answer.strip().replace('\\frac', '').replace('{', '').replace('}', '').replace(' ', '')

def extract_gsm8k_answer(text: str) -> str:
    """Enhanced GSM8K answer extractor (keeps the previously validated logic)."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    text = text.replace(',', '')
    
    # Prioritize the "#### [number]" pattern
    if "####" in text:
        res = text.split("####")[-1].strip().split()[0]
        res = re.sub(r"[^\d\.]", "", res)
        if res: return res.rstrip('.')
    
    # Fallback: \\boxed{number} pattern
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed: return boxed[-1].strip().rstrip('.')
    
    # Final fallback: last numeric substring (regex avoids stray dots)
    numbers = re.findall(r"-?\d+\.?\d*", text) 
    if numbers:
        val = numbers[-1].strip().rstrip('.')
        if val and val != '.': return val
    return ""


def format_example_mmlu(question, choices, answer_idx=None):
    options = ['A', 'B', 'C', 'D']
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{options[i]}. {choice}\n"
    prompt += "Answer:"
    if answer_idx is not None:
        prompt += f" {options[answer_idx]}\n\n"
    else:
        prompt += " " 
    return prompt

def gen_few_shot_prompt(subject, dev_dataset, target_example, k=5):
    prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    k = min(k, len(dev_dataset))
    for i in range(k):
        example = dev_dataset[i]
        prompt += format_example_mmlu(example['question'], example['choices'], example['answer'])
    prompt += format_example_mmlu(target_example['question'], target_example['choices'], answer_idx=None)
    return prompt

# -----------------------------------------------------------------------------
# 3. Evaluation Functions
# -----------------------------------------------------------------------------

def evaluate_math_dataset(model, tokenizer, max_samples=None, use_fne=False, save_dir=".", batch_size=4, num_shots=0):
    import json
    print(f"\n📊 Starting MATH evaluation (num_shots={num_shots})...")
    try:
        dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    except:
        dataset = load_dataset("qwedsacf/competition_math", split="test")
        
    if max_samples: dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    model.eval()
    device = next(model.parameters()).device
    real_model = model.module if hasattr(model, "module") else model
    
    # Initialize analysis buffer and attach tokenizer
    real_model.gating_analysis_data = [] 
    real_model._tokenizer = tokenizer 
    
    correct, total = 0, 0
    use_chat_template = tokenizer.chat_template is not None
    pbar = tqdm(range(0, len(dataset), batch_size), desc="MATH Eval", ncols=100, mininterval=60)
    debug_printed = False

    for i in pbar:
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        prompts = []
        truths = []
        
        for idx in range(len(batch['problem'])):
            prob = batch['problem'][idx]
            truth = extract_boxed_answer(batch['solution'][idx])
            if not truth: continue
            
            few_shot_text = get_math_few_shot_text(num_shots)
            
            if use_chat_template:
                instruction_base = "Solve the following math problem step-by-step. Put your final answer in \\boxed{}."
                if num_shots > 0:
                    user_content = f"Below are some examples:\n\n{few_shot_text}Now, solve this problem:\nQuestion: {prob}"
                else:
                    user_content = f"Solve the following math problem step-by-step.\nQuestion: {prob}"

                messages = [
                    {"role": "system", "content": "You are a helpful math reasoning assistant. Please reason step by step."},
                    {"role": "user", "content": f"{instruction_base}\n\n{user_content}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = few_shot_text + f"Question:\n{prob}\n\nMy solution:"
                
            prompts.append(prompt)
            truths.append(truth)
            
        if not prompts: continue
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        
        if use_fne:
            # Provide current input IDs so the model can extract numeric context
            real_model._current_input_ids = inputs.input_ids
            indices_batch, strs_batch = [], []
            for seq in inputs.input_ids:
                idx, st = _compute_number_token_positions_module(seq.tolist(), tokenizer)
                indices_batch.append(idx)
                strs_batch.append(st)
            real_model._current_numbers = strs_batch
            real_model._current_number_indexs = indices_batch
            
            if not debug_printed:
                print(f"\n🐞 MATH Debug (Shots={num_shots})\nDetected: {strs_batch[0]}")
                debug_printed = True

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            

        if use_fne and hasattr(real_model, "gating_analysis_data") and real_model.gating_analysis_data:
            log_path = os.path.join(save_dir, "gating_analysis_math.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                for entry in real_model.gating_analysis_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            real_model.gating_analysis_data = []
        if use_fne: 
            real_model._current_numbers = None
            real_model._current_number_indexs = None
            real_model._current_input_ids = None

        inp_len = inputs.input_ids.shape[1]
        for out, truth in zip(outputs, truths):
            gen = tokenizer.decode(out[inp_len:], skip_special_tokens=True)
            pred = extract_boxed_answer(gen)
            if normalize_answer(pred) == normalize_answer(truth): correct += 1
            total += 1
        pbar.set_postfix({"Acc": f"{correct/total:.2%}"})

    acc = correct / total if total > 0 else 0
    print(f"✅ MATH Acc: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc}

def evaluate_gsm8k_dataset(model, tokenizer, max_samples=None, use_fne=False, save_dir=".", batch_size=8, num_shots=0):
    import json
    print(f"\n📊 Starting GSM8K evaluation (num_shots={num_shots})...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
    except:
        return {"accuracy": 0.0}

    if max_samples: dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    model.eval()
    device = next(model.parameters()).device
    real_model = model.module if hasattr(model, "module") else model
    
    # Initialize analysis buffer and attach tokenizer
    real_model.gating_analysis_data = [] 
    real_model._tokenizer = tokenizer 
    
    correct, total = 0, 0
    use_chat_template = tokenizer.chat_template is not None
    
    pbar = tqdm(range(0, len(dataset), batch_size), desc="GSM8K Eval", ncols=100, mininterval=60)
    debug_printed = False
    
    for i in pbar:
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        prompts = []
        answers = batch['answer']
        
        for q in batch['question']:
            few_shot_content = get_few_shot_text(num_shots)
            if use_chat_template:
                instruction_base = (
                    "Solve the following math problem step-by-step. "
                    "At the end of your response, provide the final answer in the format '#### [value]'."
                )
                user_content = f"Below are some examples of math problems:\n\n{few_shot_content}Now, solve this problem:\nQ: {q}\nA:" if num_shots > 0 else f"Solve the following math problem step-by-step.\nQ: {q}\nA:"
                messages = [
                    {"role": "system", "content": "You are a helpful math assistant. Please reason step by step."},
                    {"role": "user", "content": f"{instruction_base}\n\n{user_content}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = few_shot_content + f"Q: {q}\nA:"
            prompts.append(prompt)
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        
        if use_fne:
            # Provide current input IDs so the model can extract numeric context
            real_model._current_input_ids = inputs.input_ids
            indices_batch, strs_batch = [], []
            for seq in inputs.input_ids:
                idx, st = _compute_number_token_positions_module(seq.tolist(), tokenizer)
                indices_batch.append(idx)
                strs_batch.append(st)
            real_model._current_numbers = strs_batch
            real_model._current_number_indexs = indices_batch
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        

        if use_fne and hasattr(real_model, "gating_analysis_data") and real_model.gating_analysis_data:
            log_path = os.path.join(save_dir, "gating_analysis_gsm8k.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                for entry in real_model.gating_analysis_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            real_model.gating_analysis_data = [] 
        
        if use_fne: 
            real_model._current_numbers = None
            real_model._current_number_indexs = None
            real_model._current_input_ids = None

        inp_len = inputs.input_ids.shape[1]
        for idx, (out, ans) in enumerate(zip(outputs, answers)):
            gen = tokenizer.decode(out[inp_len:], skip_special_tokens=True)
            for stop in ["Q:", "\n\nQ:", "<|im_end|>", "<|eot_id|>"]:
                if stop in gen: gen = gen.split(stop)[0]
                
            pred = extract_gsm8k_answer(gen)
            gt = extract_gsm8k_answer(ans)
            
            if not debug_printed and idx == 0:
                print("\n" + "="*50)
                print(f"🐞 GSM8K Debug (Shots={num_shots}, Template={use_chat_template})")
                print("="*50)
                print(f"📌 Extracted: '{pred}' | GT: '{gt}'")
                print(f"📌 Detected Numbers: {strs_batch[0] if use_fne else 'N/A'}")
                print("="*50 + "\n")
                debug_printed = True
            
            is_correct = False
            if pred and gt:
                if pred == gt: is_correct = True
                else:
                    try: 
                        if float(pred) == float(gt): is_correct = True
                    except: pass
            
            if is_correct: correct += 1
            total += 1
            
        pbar.set_postfix({"Acc": f"{correct/total:.2%}"})
        
    acc = correct / total if total > 0 else 0
    print(f"✅ GSM8K Acc (Shots={num_shots}): {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc}

def evaluate_mmlu_dataset(model, tokenizer, max_samples=None, use_fne=False, save_dir=".", batch_size=2, num_shots=5):
    import json
    print(f"\n📊 Starting MMLU evaluation (standard raw {num_shots}-shot)...")
    
    model.eval()
    device = next(model.parameters()).device
    real_model = model.module if hasattr(model, "module") else model
    
    # Initialize analysis buffer and attach tokenizer
    real_model.gating_analysis_data = [] 
    real_model._tokenizer = tokenizer
    
    # MMLU uses logit-based evaluation, so we avoid chat templates (raw few‑shot is recommended)
    use_chat_template = False 
    
    choices = ["A", "B", "C", "D"]
    choice_ids = []
    for c in choices:
        ids = tokenizer.encode(c, add_special_tokens=False)
        choice_ids.append(ids[-1])
    choice_ids_tensor = torch.tensor(choice_ids).to(device)

    total_correct = 0
    total_count = 0
    subject_results = {}
    
    pbar_subjects = tqdm(MMLU_SUBJECTS, desc="MMLU Subjects", ncols=100, leave=False)
    debug_printed = False
    
    for subject in pbar_subjects:
        try:
            dataset_test = load_dataset("cais/mmlu", subject, split="test")
            dataset_dev = load_dataset("cais/mmlu", subject, split="dev")
        except Exception as e:
            continue

        if max_samples: dataset_test = dataset_test.select(range(min(max_samples, len(dataset_test))))

        subject_correct = 0
        subject_total = 0
        
        examples = []
        for i, item in enumerate(dataset_test):
            prompt = gen_few_shot_prompt(subject, dataset_dev, item, k=num_shots)
            examples.append({"prompt": prompt, "answer": item["answer"]})
            
        # Batch Processing
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i : i + batch_size]
            batch_prompts = [ex["prompt"] for ex in batch_examples]
            batch_answers = [ex["answer"] for ex in batch_examples]
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            if use_fne:
                # Provide current input IDs so the model can extract numeric context
                real_model._current_input_ids = inputs.input_ids
                indices_batch, strs_batch = [], []
                for seq in inputs.input_ids:
                    idx, st = _compute_number_token_positions_module(seq.tolist(), tokenizer)
                    indices_batch.append(idx)
                    strs_batch.append(st)
                real_model._current_numbers = strs_batch
                real_model._current_number_indexs = indices_batch
                
                if not debug_printed:
                    print("\n" + "="*50)
                    print(f"🐞 MMLU Evaluation Debug (num_shots={num_shots})")
                    print(f"📌 Subject: {subject}")
                    print(f"📌 Detected Numbers (FNE): {strs_batch[0]}")
                    print("="*50 + "\n")
                    debug_printed = True

            with torch.no_grad():
                outputs = model(**inputs)
            if use_fne and hasattr(real_model, "gating_analysis_data") and real_model.gating_analysis_data:
                log_path = os.path.join(save_dir, "gating_analysis_mmlu.jsonl")
                with open(log_path, "a", encoding="utf-8") as f:
                    for entry in real_model.gating_analysis_data:
                        entry["subject"] = subject
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                real_model.gating_analysis_data = [] 
            
            if use_fne: 
                real_model._current_numbers = None
                real_model._current_number_indexs = None
                real_model._current_input_ids = None

            logits = outputs.logits
            
            for batch_idx in range(logits.shape[0]):
                if tokenizer.padding_side == "left":
                    last_token_idx = -1
                else:
                    last_token_idx = inputs["attention_mask"][batch_idx].sum() - 1
                
                last_token_logits = logits[batch_idx, last_token_idx, :]
                choice_logits = last_token_logits[choice_ids_tensor]
                pred_idx = torch.argmax(choice_logits).item()
                
                if pred_idx == batch_answers[batch_idx]:
                    subject_correct += 1
                subject_total += 1

        acc = subject_correct / subject_total if subject_total > 0 else 0
        subject_results[subject] = acc
        total_correct += subject_correct
        total_count += subject_total
        
        pbar_subjects.set_postfix({"Acc": f"{acc:.2%}"})

    pbar_subjects.close()
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"✅ MMLU evaluation finished: {overall_acc:.4f} ({total_correct}/{total_count})")
    
    with open(os.path.join(save_dir, "mmlu_results.txt"), "w") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
        for subj, score in subject_results.items():
            f.write(f"{subj}: {score:.4f}\n")
            
    return {"accuracy": overall_acc, "results": subject_results}

def run_evaluation_pipeline(script_args, checkpoint_path=None, model=None, tokenizer=None):
    if model is None: return 
    save_dir = checkpoint_path if checkpoint_path else script_args.output_dir
    
    num_shots = getattr(script_args, 'num_shots', 0)
    
    if script_args.eval_gsm8k:
        evaluate_gsm8k_dataset(
            model, tokenizer, 
            max_samples=script_args.gsm8k_max_samples, 
            use_fne=script_args.use_fne, 
            save_dir=save_dir, 
            batch_size=32,
            num_shots=num_shots
        )
    if script_args.eval_mmlu: 
        evaluate_mmlu_dataset(
            model, tokenizer, 
            max_samples=script_args.mmlu_max_samples, 
            use_fne=script_args.use_fne, 
            save_dir=save_dir,
            batch_size=4, 
            num_shots=num_shots                   
        )
    if script_args.eval_math: 
        evaluate_math_dataset(
            model, tokenizer, 
            max_samples=script_args.math_max_samples, 
            use_fne=script_args.use_fne, 
            save_dir=save_dir, 
            batch_size=16,
            num_shots=num_shots  
        )