# ANI: Adaptive Numerical Injection

This repository contains the official implementation of the **ANI (Adaptive Numerical Injection)** framework, designed to diagnose and enhance the numerical reasoning capabilities of Large Language Models (LLMs). The project includes a diagnostic tool (Linear Probe) to analyze layer-wise numerical encoding, as well as training scripts for adaptive numerical embedding injection via a Gating mechanism.


## 📂 Repository Structure

* `diagnostic_datasets/`: Scripts and datasets for numerical analysis and linear probe experiments.
* `scripts/`: Main execution scripts for model training (`train.py`).
* `src/`: Core source code for the ANI framework.
  * `models.py`: Architectures for Gating and Baseline integrated models.
  * `modules.py`: Implementation of Fourier Number Embedding (FoNE) and the Gating Network.
  * `data.py`: Data preprocessing and logic for numerical token position detection.
  * `eval.py` / `trainer.py`: Custom evaluation pipelines and training loops.
* `*.sh`: Shell scripts for hyperparameter search and batch evaluation.

---
## 🧰 Environment Setup (Conda)

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

---
## 🗄️ Data Preparation

Before running the experiments and training, you need to prepare the required datasets.

### 1. Diagnostic Datasets for Linear Probe
To evaluate the model's ability to distinguish between different usages of numbers, you must generate the diagnostic datasets. 
The `generate_diagnostic_dataset.py` script automatically generates two types of synthetic data:
1. **Contextual/Semantic Data**: Numbers used as identifiers, dates, times, locations, etc.
2. **Numerical/Mathematical Data**: Numbers used directly in mathematical operations or calculations.

Run the following command to generate `contextual_dataset.json` and `numerical_dataset.json` in the `/workspace/diagnostic_datasets` directory:

```bash
python /workspace/diagnostic_datasets/generate_diagnostic_dataset.py
```

## 🚀 Layer-wise Linear Probe Experiments

To understand how well each layer of an LLM preserves numerical information, you can run layer-wise Linear Probe experiments. Use the following commands depending on your model size.

### 1. Single GPU Setup (Small Models)
Use this command to analyze models like `Qwen/Qwen3-8B` on a single GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python /workspace/diagnostic_datasets/linear_probe_experient.py \
    --layer_wise \
    --max_epochs 1 \
    --verbose \
    --gpus 0 \
    --model Qwen/Qwen3-8B \
    --start_layer 0
```

### 1-1. Multi-GPU & Quantization Setup (Large Models)

For memory-intensive models like `Qwen/Qwen3-32B`, apply 4-bit quantization (`--load_in_4bit`) and distribute the workload across multiple GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 python /workspace/diagnostic_datasets/linear_probe_experient.py \
    --layer_wise \
    --max_epochs 1 \
    --verbose \
    --gpus 0,1 \
    --load_in_4bit \
    --model Qwen/Qwen3-32B \
    --start_layer 0
```

### 2. Training Dataset (OpenMathInstruct-2)
For training, the model requires the `openmathinstruct2-sft.jsonl` dataset. You can download the original OpenMathInstruct-2 dataset from Hugging Face and convert it into the required JSONL format using the following Python snippet:



### 3. Model Training & Evaluation
You can train the integrated model (combining the FoNE encoder and Gating Network) to reinforce numerical reasoning using the ANI framework. The following command runs the training with QLoRA, Gumbel gating, cosine temperature annealing, and evaluates on MMLU, GSM8K, and MATH benchmarks.

```bash
python -u /workspace/scripts/train.py \
    --general_dataset_path "Magpie-Align/Magpie-Pro-MT-300K-v0.1" \
    --model_name "Qwen/Qwen3-8B" \
    --max_samples 10000 \
    --num_general_samples 10000 \
    --learning_rate 1e-4 \
    --use_lora \
    --use_qlora \
    --lora_r 64 \
    --lora_alpha 128 \
    --use_fne \
    --add_linear \
    --anneal_temperature \
    --temp_start 2.0 \
    --temp_min 0.5 \
    --temp_decay_type "cosine" \
    --gating_layer_index 23 \
    --dataset_path "/workspace/openmathinstruct2-sft-10k.jsonl" \
    --output_dir "/workspace/checkpoints/ani_model" \
    --per_device_batch_size 1 \
    --gradient_accum_steps 32 \
    --num_train_epochs 1 \
    --int_digit_len 10 \
    --frac_digit_len 6 \
    --len_gen_size 0 \
    --eval_mmlu \
    --eval_gsm8k \
    --eval_math \
    --gating_type "gumbel" \
    --seed 23 \
    --logging_steps 1
```