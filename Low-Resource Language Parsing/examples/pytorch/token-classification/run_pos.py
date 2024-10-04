#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

from seqeval.metrics import accuracy_score
import transformers
import shutil
import transformers.adapters.composition as ac
from typing import Dict, List, Optional, Tuple
from torch import nn
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    EvalPrediction,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from pos_tagging_dataset import POSDataset, Split, get_file
from utils_pos import UPOS_LABELS
from transformers.adapters.configuration import AdapterConfig, IA3Config, ConfigUnion, PrefixTuningConfig, PfeifferConfig, LoRAConfig, PfeifferInvConfig



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class AdapterArgs(MultiLingAdapterArguments):
    adapter_r: int = field(default=1, metadata={"help": "How much you want to scale the adapter."})
    lang_adapter_r: int = field(default=1, metadata={"help": "How much you want to scale the adapter."})
    prefixtune_shared_gating: bool = field(default=True)


def create_adapter_config(adapter_args: AdapterArgs, use_lang=False, leave_out=None):
    adapter_config = adapter_args.lang_adapter_config if use_lang else adapter_args.adapter_config
    adapter_r = adapter_args.lang_adapter_r if use_lang else adapter_args.adapter_r
    if adapter_config == 'ia3':
        adapter_config = IA3Config(r=adapter_r, alpha=adapter_r)
    elif adapter_config == 'unipelt':
        adapter_config = ConfigUnion(
            LoRAConfig(r=8 * adapter_r, alpha=8 * adapter_r, use_gating=True),
            PrefixTuningConfig(prefix_length=10 * adapter_r, use_gating=True, shared_gating=adapter_args.prefixtune_shared_gating),
            PfeifferConfig(reduction_factor=16 / adapter_r, use_gating=True),
        )
    elif adapter_config == 'unipelt+inv':
        adapter_config = ConfigUnion(
            LoRAConfig(r=8 * adapter_args.adapter_r, alpha=8 * adapter_args.adapter_r, use_gating=True),
            PrefixTuningConfig(prefix_length=10 * adapter_args.adapter_r, use_gating=True, shared_gating=adapter_args.prefixtune_shared_gating),
            PfeifferInvConfig(reduction_factor=16 / adapter_args.adapter_r, use_gating=True),
        )
    else:
        adapter_config = AdapterConfig.load(
            adapter_config,
            non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=adapter_args.adapter_reduction_factor,
            leave_out=leave_out,
        )

    return adapter_config

from collections import OrderedDict
def get_optim_grouped_params_only_requires_grad(model, lr, wd, consider_adapter_params, omit_names_from: List[dict] = []):
    from transformers.trainer_pt_utils import get_parameter_names
    import torch.nn as nn
    import re
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    omitted_names = []
    for groups in omit_names_from:
        omitted_names.extend(groups['named_params'].keys())

    if consider_adapter_params:
        # if hasattr(model, "config") and hasattr(model.config, "adapters"):
        match_str = r"adapter_fusion_layer\..*\.value"
        decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]
    optimizer_grouped_parameters = [
        {
            "lr": lr,
            "named_params": OrderedDict([(n, p) for n, p in model.named_parameters() if p.requires_grad and n in decay_parameters and n not in omitted_names]),
            "weight_decay": wd,
        },
        {
            "lr": lr,
            "named_params": OrderedDict([(n, p) for n, p in model.named_parameters() if p.requires_grad and n not in decay_parameters and n not in omitted_names]),
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

from typing import List
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    leave_out: List[int] = field(default_factory=lambda: [])
    load_best: bool = field(default=False)
    load_last: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    pos_data_dir: Optional[str] = field(
        default=None
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    head_learning_rate: float = field(default=None)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.pos_data_dir is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "conllu"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "conllu"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()




def setup_log(training_args, parser):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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

def check_output_dir_and_get_last_ckpt(training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if is_main_process(training_args.local_rank):
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
    return last_checkpoint



def get_config_tok_model(model_args, data_args, num_labels, label_to_id, label_map):
    ################ Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label=label_map,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    #
    #
    # # Tokenizer check: this script requires a fast tokenizer.
    # # Model has labels -> use them.
    # if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    #     if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
    #         # Reorganize `label_list` to match the ordering of the model.
    #         if labels_are_int:
    #             label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
    #             label_list = [model.config.id2label[i] for i in range(num_labels)]
    #         else:
    #             label_list = [model.config.id2label[i] for i in range(num_labels)]
    #             label_to_id = {l: i for i, l in enumerate(label_list)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #             "\nIgnoring the model labels as a result.",
    #         )
    #
    # # Set the correspondences label/ID inside the model config
    # model.config.label2id = {l: i for i, l in enumerate(label_list)}
    # model.config.id2label = {i: l for i, l in enumerate(label_list)}

    return config, tokenizer, model


def get_pos_model_and_dataset(model_args, data_args, training_args):
    labels = UPOS_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label_to_id = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    config, tokenizer, model = get_config_tok_model(model_args, data_args, num_labels, label_to_id, label_map)

    train_dataset = (
        POSDataset(
            data_dir=data_args.pos_data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            file_name=data_args.train_file,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        POSDataset(
            data_dir=data_args.pos_data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            file_name=data_args.validation_file,
        )
        if training_args.do_eval
        else None
    )

    predict_dataset = (
        POSDataset(
            data_dir=data_args.pos_data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            file_name=data_args.test_file,
        )
        if training_args.do_predict
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        pred_result = {"accuracy": accuracy_score(out_label_list, preds_list)}
        return pred_result


    data_collator = None


    return config, model, tokenizer, train_dataset, eval_dataset, predict_dataset, label_map, compute_metrics, data_collator

def setup_adapters(model, tokenizer, model_args, data_args, adapter_args):
    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.task_name
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = create_adapter_config(adapter_args, leave_out=model_args.leave_out)
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                    # leave_out=model_args.leave_out,
                    with_head=False

                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = create_adapter_config(adapter_args, use_lang=True, leave_out=model_args.leave_out)
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
                # leave_out=model_args.leave_out,
                with_head=False
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        else:
            model.set_active_adapters([task_name])
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    assert isinstance(model_args, ModelArguments)
    assert isinstance(data_args, DataTrainingArguments)
    assert isinstance(training_args, TrainingArguments)
    assert isinstance(adapter_args, AdapterArgs)

    setup_log(training_args, parser)
    # Set seed before initializing model.
    set_seed(training_args.seed)
    last_checkpoint = check_output_dir_and_get_last_ckpt(training_args)

    if not data_args.pos_data_dir and not data_args.test_file.endswith('conllu'):
        raise NotImplementedError
    else:
        config, model, tokenizer, train_dataset, eval_dataset, predict_dataset, label_map, compute_metrics, data_collator = get_pos_model_and_dataset(
            model_args, data_args, training_args)

    ############## setup adapters

    # Setup adapters
    setup_adapters(model, tokenizer, model_args, data_args, adapter_args)

    # Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer if data_collator else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    groups = []
    if data_args.head_learning_rate:
        head_params = [(n, p) for n, p in model.named_parameters() if "classifier" in n]
        print([n[0] for n in head_params])
        decay_parameters = [name_param[0] for name_param in head_params if "bias" not in name_param[0]]
        optimizer_grouped_parameters = [
            {
                "lr": data_args.head_learning_rate,
                "named_params": OrderedDict(
                    [(n, p) for n, p in head_params if n in decay_parameters]),
                "weight_decay": training_args.weight_decay,
            },
            {
                "lr": data_args.head_learning_rate,
                "named_params": OrderedDict(
                    [(n, p) for n, p in head_params if n not in decay_parameters]),
                "weight_decay": 0.0,
            },
        ]
        groups.extend(optimizer_grouped_parameters)

        task_name = data_args.task_name

        model.train_adapter([task_name])
        groups.extend(get_optim_grouped_params_only_requires_grad(model, training_args.learning_rate, training_args.weight_decay, consider_adapter_params=True,
                                                                  omit_names_from=groups))
        if adapter_args.load_lang_adapter:
            model.set_active_adapters([adapter_args.language, task_name])
        else:
            model.set_active_adapters([task_name])

    if groups:
        for param_group in groups:
            param_group['params'] = list(param_group.pop('named_params').values())

        from transformers.optimization import Adafactor, AdamW
        from transformers.trainer_utils import ShardedDDPOption

        if training_args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (training_args.adam_beta1, training_args.adam_beta2),
                "eps": training_args.adam_epsilon,
            }
        if trainer.sharded_ddp == ShardedDDPOption.SIMPLE:
            raise NotImplementedError
            # trainer.optimizer = OSS(
            #     params=groups,
            #     optim=optimizer_cls,
            #     **optimizer_kwargs,
            # )
        else:
            trainer.optimizer = optimizer_cls(groups, **optimizer_kwargs)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        if is_main_process(training_args.local_rank):
            shutil.copytree(trainer.state.best_model_checkpoint, os.path.join(training_args.output_dir, "best-ckpt"))

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if model_args.load_best:
        from transformers.trainer import TRAINER_STATE_NAME
        best_ckpt = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME)))['best_model_checkpoint']
        trainer._load_from_checkpoint(best_ckpt)
    if model_args.load_last:
        best_ckpt = get_last_checkpoint(training_args.output_dir)
        trainer._load_from_checkpoint(best_ckpt)
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        print(metrics)
        # predictions = np.argmax(predictions, axis=2)
        # # Remove ignored index (special tokens)
        # true_predictions = [
        #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # # Save predictions
        # output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        # if trainer.is_world_process_zero():
        #     with open(output_predictions_file, "w") as writer:
        #         for prediction in true_predictions:
        #             writer.write(" ".join(prediction) + "\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
