gpu=$1
mp=${2}_${3}_${4}

cd ..

run() {
  CUDA_VISIBLE_DEVICES=$gpu python eval_bert_ibleu.py \
    --model_store_path results \
    --model_postfix $mp \
    --eval_file result.json \
    --quick \
    --eval_postfix $1
}

run 'default' # raw bert-ibleu with unsmoothed bleu
run 'sacre --use_sacre' # raw bert-ibleu with sacrebleu (by default smoothed)
run 'fluency_soft --fluency --use_sacre' # bert-ibleu-ppl with soft threshold
run 'fluency_hard --fluency --use_sacre --fluency_hard_threshold' # bert-ibleu-ppl with hard threshold
