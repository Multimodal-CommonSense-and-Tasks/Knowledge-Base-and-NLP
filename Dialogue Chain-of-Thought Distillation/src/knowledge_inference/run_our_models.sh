#!/bin/bash
# filter_type=$1
declare -a datasets=("daily_dialog")
# declare -a datasets=("daily_dialog" "dream")
# declare -a datasets=("mutual")
# declare -a filter_types=("consist_filter" "helpful_filter" "both" "none")

# for dataset_name in ${datasets[@]};do
#     echo "Start running $dataset_name"
#     for split_idx in {0..7};do
#         log_file="logs/additional/${dataset_name}_process_${split_idx}.log"
#         CUDA_VISIBLE_DEVICES=$split_idx python src/knowledge_inference/qlora_inference.py \
#         --model_name /home/hjchae/nas2/EMNLP2023/MindMap/checkpoints/opt_1_3b_ours \
#         --input_file /home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_input/$dataset_name/test.json \
#         --output_file /home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_output/additional/$dataset_name/test_ours.json \
#         --batch_size=16 --max_length=400 --split=8 --split_idx=$split_idx --is_clm > "$log_file" 2>&1 &
#     done

#     wait

#     echo "Finished running $dataset_name and Collecting inference files"

#     python src/knowledge_inference/collect_inference_files.py \
#         --file_name test_ours \
#         --input_dir /home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_output/additional/$dataset_name \
#         --output_file /home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_output/additional/$dataset_name/test_ours.json
    
#     echo "Finished collecting inference files for $dataset_name"
# done

declare -a dataset_paths=("/home/hjchae/nas2/EMNLP2023/MindMap/data/knowledge_inference_output/additional/daily_dialog/test_ours.json")
declare -a devices=("1" "2")

# Run Cosmo
for index in ${!datasets[@]}; do
    input_path=${dataset_paths[$index]}
    dataset_name=${datasets[$index]}
    device=${devices[$index]}
    echo "Start dataset $dataset_name"
    CUDA_VISIBLE_DEVICES=$device python src/rg_ours/run_ours_cosmo.py \
        --input_file $input_path \
        --batch_size 32 --min_context_len 0 \
        --cs_type ours \
        --output_file /home/hjchae/nas2/EMNLP2023/MindMap/results/additional/$dataset_name/cosmo/test > "logs/additional/cosmo_${dataset_name}.log" 2>&1 &
done


# ## Run ChatGPT
# for dataset_name in ${datasets[@]};do
#     CUDA_VISIBLE_DEVICES=3 python src/rg_ours/run_ours_chatgpt.py \
#         --min_context_len 0 \
#         --input_file $input_path \
#         --prompt /home/hjchae/nas2/EMNLP2023/MindMap/src/rg_ours/prompt.txt \
#         --cs_type ours \
#         --save_dir /home/hjchae/nas2/EMNLP2023/MindMap/results/additional/$dataset_name/chatgpt/test &
# done