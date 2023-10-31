import json
import logging
import os
import pickle
import random
import sys
from functools import partial

from typing import Any, Dict, List, Literal, Optional, Union

import datasets
import numpy as np
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import (
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    MarianMTModel,
    EarlyStoppingCallback,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoConfig,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

from model.arguments import *
from model.dataset import *
from model.model_mrt import MRTParaphraser
from model.model_brio import BRIOParaphraser
from model.model_triecl import TrieCLParaphraser
from model.metrics import *
from model.perplexity import Perplexity

logger = logging.getLogger(__name__)

class TrieCLTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss = 0
        if model.generative:
            new_loss = model.get_generation_loss(*inputs[:2])
            loss += new_loss

        if model.contrastive:
            # try:
            new_loss = model.get_contrastive_loss(*inputs)
            loss += new_loss * model.mix_rate # Multiply mix_rate for weighted sum

            # except RuntimeError as e:
            #     print(e)
            #     logger.info(f"Recovering from OOM (contrastive loss = {args.loss_fn})")
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     gc.collect()

        return loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        # result = []
        # first_batch=True
        with torch.inference_mode():
            total_score = 0
            # total_nll_loss = total_loss = total_score = 0
            
            for data in tqdm(eval_dataloader):
                sources, targets = data
                outputs = self.model.generate(sources)
                hypos = [output[0] for output in outputs]
                sources_decode = self.tokenizer.batch_decode(sources, skip_special_tokens=True)
                bert_ibleu_score = self.model.metric(sources_decode, None, hypos, shape=(len(hypos), 1), extended=True)
                total_score += bert_ibleu_score.sum().item()

        # total_batch_size = self.args.eval_batch_size * self.args.world_size

        n = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        # final_nll_loss = total_nll_loss / n
        # final_loss = total_loss / n
        final_score = total_score / n
        metrics = {f'{metric_key_prefix}_bert_ibleu': final_score}
        self.log(metrics)
        return metrics

MODEL_ID = {
    'bart': 'facebook/bart-base',
    't5': 't5-small',
    'marian-ende': "Helsinki-NLP/opus-mt-en-de",
    'marian-enfr': "Helsinki-NLP/opus-mt-en-fr",
    'marian-enro': "Helsinki-NLP/opus-mt-en-ro",
}
MODEL_CLASS = {
    'bart': BartForConditionalGeneration,
    't5': T5ForConditionalGeneration,
    'marian-ende': MarianMTModel,
    'marian-enfr': MarianMTModel,
    'marian-enro': MarianMTModel,
}
KEY_TO_METRIC = {
    "bert-ibleu": bert_ibleu,
    "bert-ibleu-fluency": bert_ibleu_fluency,
    "bleu": get_bleu_score,
    "bleurt": get_bleurt_score,
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, TrieCLArguments, EvaluationArguments))
    model_args, data_args, training_args, args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # load model
    model_id = MODEL_ID[args.base_model]
    model_class = MODEL_CLASS[args.base_model]
    base_model = model_class.from_pretrained(model_id)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_config = AutoConfig.from_pretrained(model_id)

    if args.loss_fn == "triecl":
        paraphraser = TrieCLParaphraser
    elif args.loss_fn == "brio":
        paraphraser = BRIOParaphraser
    elif args.loss_fn == "mrt":
        paraphraser = MRTParaphraser
    else:
        raise ValueError("loss_fn should be in: 'triecl', 'brio', 'mrt'")
    
    metric = KEY_TO_METRIC[args.metric](eval_args)
    
    model = paraphraser(
        base_model,
        base_tokenizer,
        metric,
        args,
    )

    if args.from_checkpoint is not None:
        # Fine-tune from a local checkpoint
        # model_load_path = os.path.join('checkpoints_trainer', args.from_checkpoint)
        model.load_state_dict(torch.load(args.from_checkpoint))

    def collate_fn(batch):
        if len(batch[0]) == 2:
            src, tgt = zip(*batch)
            src = pad_sequence(src, batch_first=True, padding_value=base_tokenizer.pad_token_id)
            tgt = pad_sequence(tgt, batch_first=True, padding_value=base_tokenizer.pad_token_id)
            return src, tgt
        elif len(batch[0]) == 6 or len(batch[0]) == 7:
            src, tgt, hypos, *bwls = zip(*batch)
            src = pad_sequence(src, batch_first=True, padding_value=base_tokenizer.pad_token_id)
            tgt = pad_sequence(tgt, batch_first=True, padding_value=base_tokenizer.pad_token_id)
            hypos = [pad_sequence(hs, batch_first=True, padding_value=base_tokenizer.pad_token_id) for hs in hypos]
            return src, tgt, hypos, *bwls
        else:
            raise ValueError

    ls = [x for x in args.model_postfix.split('_') if 'seed' not in x]
    cpath = '.cache/' + '_'.join(ls)
    cpath2 = '.cache/' + '_'.join(ls[:2])

    if training_args.do_train:
        logger.info("Start building(loading) train set.")
        with open(args.train_data, "r", encoding='UTF-8') as file:
            train_data = json.load(file)
        
        if args.contrastive:
            if args.learning_mode == 'offline':
                with open(os.path.join('/'.join(args.train_data.split('/')[:-1]), 'ibleu_train.pkl'), 'rb') as f:
                    scores = pickle.load(f)
                train_dataset = OfflineRLDataset(base_tokenizer, train_data, scores, cpath + '_train.pkl')
            else:
                train_dataset = TextGenerationDataset(base_tokenizer, train_data, cpath + '_train.pkl', shuffle=False)
        else:
            if args.learning_mode == 'offline_filtered':
                with open(os.path.join('/'.join(args.train_data.split('/')[:-1]), 'ibleu_train.pkl'), 'rb') as f:
                    scores = pickle.load(f)
                train_dataset = OfflineSupervisedFilteredDataset(base_tokenizer, train_data, scores, args.num_beams, cpath + '_train.pkl')
            elif args.learning_mode == 'offline':
                train_dataset = OfflineSupervisedDataset(base_tokenizer, train_data, args.num_beams, cpath + '_train.pkl')
            else:
                train_dataset = TextGenerationDataset(base_tokenizer, train_data, cpath + '_train.pkl', shuffle=False)
        logger.info("Finished building(loading) train set.")

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = Subset(train_dataset, np.arange(max_train_samples))

    if training_args.do_eval:
        logger.info("Start building(loading) validation set.")
        with open(args.dev_data, "r", encoding='UTF-8') as file:
            dev_data = json.load(file)
        eval_dataset = TextGenerationDataset(base_tokenizer, dev_data, cpath2 + '_dev.pkl', shuffle=False)
        logger.info("Finished building(loading) validation set.")
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = Subset(eval_dataset, np.arange(max_eval_samples))
    
    # Log a few random samples from the training set:
    if training_args.do_train:
        logger.info(f"Training set size: {len(train_dataset)}.")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Initialize our Trainer
    trainer = TrieCLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=base_tokenizer,
        data_collator=collate_fn,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
