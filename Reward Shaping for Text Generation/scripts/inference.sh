base_model=$2 # e.g., t5, bart
dataset=$3 # e.g., qqp, mscoco
model=${base_model}_${dataset}
output_dir=results/${model}_online_from_scratch_fluency_10000

cd ..

if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

CUDA_VISIBLE_DEVICES=$1 python new_inference.py \
  --test_data data/${dataset}_paragen_test.json \
  --batch_size 40 \
  --num_beams 16 \
  --model_path checkpoints/2023-05-24_20-24-22_bart_mscoco_online_from_scratch_fluency_seed0/checkpoint-10000/pytorch_model.bin \
  --result_path $output_dir \
  --model_postfix $model \
  --base_model $base_model
