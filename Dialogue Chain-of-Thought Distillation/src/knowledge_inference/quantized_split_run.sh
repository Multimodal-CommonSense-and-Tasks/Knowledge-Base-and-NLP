#!/bin/bash
dataset_name=$1
for split_idx in {0..8}
do
    log_file="${dataset_name}_process_${split_idx}.log"
    CUDA_VISIBLE_DEVICES=$split_idx nohup  python src/knowledge_inference/qlora_inference.py --model_name=/home/hjchae/nas2/EMNLP2023/MindMap/checkpoints/opt_1_3b_cicero_helpfulness_filtered_8bit/checkpoint-1485 --batch_size=32 --input_file=/home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_input/$dataset_name/test.json --output_file=/home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_output/$dataset_name/test_ours_opt_cicero_helpfulness.json --max_length=400 --split=8 --split_idx=$split_idx --is_clm > "$log_file" 2>&1 &
done
