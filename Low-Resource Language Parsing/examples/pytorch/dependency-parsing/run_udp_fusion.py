"""
Code taken and modified from: https://github.com/Adapter-Hub/hgiyt.
Credits: "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models" (Rust et al., 2021)
https://arxiv.org/abs/2012.15613
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List



from datasets import load_dataset

import transformers.adapters.composition as ac
from preprocessing import preprocess_dataset
from transformers import (
    AutoAdapterModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    MultiLingAdapterArguments,
    set_seed,
)
from utils_udp import UD_HEAD_LABELS, DependencyParsingAdapterTrainer, DependencyParsingTrainer, UDTrainingArguments

from transformers.adapters.configuration import AdapterConfig, IA3Config, ConfigUnion, PrefixTuningConfig, PfeifferConfig, LoRAConfig, PfeifferInvConfig, StaticAdapterFusionConfig


logger = logging.getLogger(__name__)


def train_adapter_custom(model, adapter_setup, freeze_model):
    """Sets the model into mode for training the given adapters."""
    if model.base_model is model:
        model.train()
        model.freeze_model(freeze_model)
        adapter_setup = ac.parse_composition(adapter_setup)
        model.apply_to_adapter_layers(lambda i, layer: layer.enable_adapters(adapter_setup, True, False))
        for adapter_name in adapter_setup:
            if adapter_name in model.shared_parameters:
                for param in model.shared_parameters[adapter_name].values():
                    param.requires_grad = True
    else:
        train_adapter_custom(model.base_model, adapter_setup, freeze_model)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    replace_embeddings: bool = field(default=False, metadata={"help": "Whether or not to replace embeddings."})
    leave_out_twelvth: bool = field(
        default=False, metadata={"help": "Whether or not to leave out adapters in twelvth layer"}
    )
    do_lower_case: bool = field(default=False, metadata={"help": "Set this flag when using uncased model/tokenizer"})
    is_japanese: bool = field(default=False, metadata={"help": "Set this to true when using Japanese model/tokenizer"})
    mecab_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to mecab installation. Required when using Japanese model/tokenizer"}
    )
    mecab_dic_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to mecab dictionary. Required when using Japanese model/tokenizer"}
    )
    load_best: bool = field(default=False, metadata={"help": "Set this flag when using uncased model/tokenizer"})
    load_last: bool = field(default=False, metadata={"help": "Set this flag when using uncased model/tokenizer"})
    load_last_dir: str = field(default="")
    reduce_params: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The identifier of the Universal Dependencies dataset to train on."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
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

from collections import OrderedDict
def get_optim_grouped_params_only_requires_grad(model, lr, wd, consider_adapter_params, omit_names_from: List[dict] = []):
    import torch.nn as nn
    import re
    from transformers.trainer_pt_utils import get_parameter_names
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
            "named_params": OrderedDict(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad and n not in decay_parameters and n not in omitted_names]),
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


@dataclass
class AdapterArgs(MultiLingAdapterArguments):
    train_adapter: bool = field(default=False, metadata={"help": "Train an adapter instead of the full model."})
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[float] = field(
        default=None, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language adapter configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language adapter configuration."}
    )

    load_adapter: List[str] = field(
        default_factory=lambda: [], metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    language: List[str] = field(default_factory=lambda: [], metadata={"help": "The training language, e.g. 'en' for English."})

    load_lang_adapter: List[str] = field(
        default_factory=lambda: [], metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    fuse_mode: str = field(default='')
    with_head: bool = field(default=False)

    adapter_r: int = field(default=1, metadata={"help": "How much you want to scale the adapter."})
    lang_adapter_r: int = field(default=1, metadata={"help": "How much you want to scale the adapter."})
    prefixtune_shared_gating: bool = field(default=True)
    use_avg_fusion: bool = field(default=False)
    invert_fusion_score: bool = field(default=False)
    inv_fusion: bool = field(default=False)
    mlp_fusion_mode: str = field(default=None)
    mlp_fusion_depth: int = field(default=1)
    value_matrix_in_fusion: bool = field(default=True)


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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UDTrainingArguments, AdapterArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    assert isinstance(training_args, UDTrainingArguments)
    assert isinstance(adapter_args, AdapterArgs)
    assert isinstance(data_args, DataTrainingArguments)
    assert isinstance(model_args, ModelArguments)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare for UD dependency parsing task
    labels = UD_HEAD_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        pad_token_id=-1,
    )
    config.use_avg_fusion = adapter_args.use_avg_fusion
    config.invert_fusion_score = adapter_args.invert_fusion_score
    config.inv_fusion = adapter_args.inv_fusion
    config.mlp_fusion_mode = adapter_args.mlp_fusion_mode
    config.mlp_fusion_depth = adapter_args.mlp_fusion_depth

    if model_args.is_japanese:
        assert model_args.mecab_dir is not None
        assert model_args.mecab_dic_dir is not None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
        do_lower_case=model_args.do_lower_case,
        add_prefix_space=True,  # Used e.g. for RoBERTa
        mecab_kwargs={"mecab_option": f"-r {model_args.mecab_dir} -d {model_args.mecab_dic_dir}"}
        if model_args.is_japanese
        else None,
    )

    # The task name (with prefix)
    task_name = "ud_" + data_args.task_name

    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    groups = []
    head_name = task_name

    model.add_dependency_parsing_head(
        head_name,
        num_labels=num_labels,
        id2label=label_map,
    )

    if model_args.leave_out_twelvth:
        logger.info("Leaving out 12")
        leave_out = [11]
    else:
        leave_out = []

    # Setup adapters
    adapter_config = create_adapter_config(adapter_args, leave_out=leave_out)
    lang_adapter_config = create_adapter_config(adapter_args, use_lang=True, leave_out=leave_out)

    stacked_adapters = []
    task_adapters = []
    lang_adapters = []
    # load a pre-trained from Hub if specified

    def normalize_name(s, limit_len=120//(len(adapter_args.load_adapter) + len(adapter_args.load_lang_adapter))):
        s = s.replace("@", "")
        s = s.replace("/", "")
        s = s.replace(".", "")
        if len(s) > limit_len:
            from util.log_utils import get_hash
            s = get_hash(s, limit_len // 2 * 2)
        return s


    fusion_config = StaticAdapterFusionConfig() if not adapter_args.value_matrix_in_fusion else None
    model.model_name = 'bert-base-multilingual-cased'
    if not adapter_args.fuse_mode:
        assert len(adapter_args.load_lang_adapter) == 1
        for task_adapter in adapter_args.load_adapter:
            task_adapter_name = model.load_adapter(
                task_adapter,
                config=adapter_config,
                load_as=normalize_name(task_adapter),
                with_head=adapter_args.with_head
            )
            task_adapters.append(task_adapter_name)

        lang_adapter = adapter_args.load_lang_adapter[0]
        # if language is mhr, it seems critical to use relu-pfeiffer
        lang_adapter_name = model.load_adapter(
            lang_adapter,
            config=lang_adapter_config,
            load_as=normalize_name(lang_adapter),
            with_head=adapter_args.with_head
        )
        fuse_adapter_setup = ac.Fuse(*task_adapters) # note that the invertible adapter is used for only the first adapter
        model.add_adapter_fusion(fuse_adapter_setup, config=fusion_config)
        # if not adapter_args.with_head:
        #     model.add_classification_head('fuse', num_labels=num_labels)
        # Freeze all model weights except of those of this adapter
        # model.set_active_adapters(fuse_adapter_setup)
        model.train_adapter_fusion(fuse_adapter_setup) # fusion의 weight만 사용함
        model.set_active_adapters(ac.Stack(lang_adapter_name, fuse_adapter_setup))

    if 'lang_fuse' in adapter_args.fuse_mode:
        # if language is mhr, it seems critical to use relu-pfeiffer
        for lang_adapter, lang in zip(adapter_args.load_lang_adapter, adapter_args.language):
            lang_adapter_name = model.load_adapter(
                lang_adapter,
                config=lang_adapter_config,
                load_as=lang,
                with_head=adapter_args.with_head
            )
            lang_adapters.append(lang_adapter_name)

        if adapter_args.load_adapter:
            assert len(adapter_args.load_adapter) == 1
            task_adapter = adapter_args.load_adapter[0]
            task_adapter_name = model.load_adapter(
                task_adapter,
                config=adapter_config,
                load_as=normalize_name(task_adapter),
                with_head=adapter_args.with_head
            )
        else:
            task_adapter_name = task_name
            model.add_adapter(task_adapter_name, config=adapter_config)
        stacked_adapters.append(task_adapter_name)

        fuse_adapter_setup = ac.Fuse(*lang_adapters)  # note that the invertible adapter is used for only the first adapter
        model.add_adapter_fusion(fuse_adapter_setup, config=fusion_config)
        stacked_adapters.insert(0, fuse_adapter_setup)

        if adapter_args.fuse_mode == 'lang_fuse_train_fuse_and_task':
            if training_args.head_learning_rate:
                head_params = [(n, p) for n, p  in model.named_parameters() if head_name in n]
                decay_parameters = [name_param[0] for name_param in head_params if "bias" not in name_param[0]]
                optimizer_grouped_parameters = [
                    {
                        "lr": training_args.head_learning_rate,
                        "named_params": OrderedDict(
                            [(n, p) for n, p in head_params if n in decay_parameters]),
                        "weight_decay": training_args.weight_decay,
                    },
                    {
                        "lr": training_args.head_learning_rate,
                        "named_params": OrderedDict(
                            [(n, p) for n, p in head_params if n not in decay_parameters]),
                        "weight_decay": 0.0,
                    },
                ]
                groups.extend(optimizer_grouped_parameters)

            model.train_adapter_fusion(fuse_adapter_setup)
            groups.extend(
                get_optim_grouped_params_only_requires_grad(model, training_args.fusion_learning_rate, training_args.weight_decay, consider_adapter_params=True,
                                                            omit_names_from=groups)) # add fusion (the new layers) only first

            model.apply_to_adapter_layers(lambda i, layer: layer.enable_adapters(ac.parse_composition([task_adapter_name]), True, False))
            if task_adapter_name in model.shared_parameters:
                for param in model.shared_parameters[task_adapter_name].values():
                    param.requires_grad = True
            groups.extend(get_optim_grouped_params_only_requires_grad(model, training_args.learning_rate, training_args.weight_decay,
                                                                      consider_adapter_params=True, omit_names_from=groups))

        model.set_active_adapters(stacked_adapters)

    # Load and preprocess dataset
    if data_args.train_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files)
    else:
        dataset = load_dataset("universal_dependencies", data_args.task_name)
    dataset = preprocess_dataset(dataset, tokenizer, labels, data_args, pad_token_id=-1)

    # Initialize our Trainer
    # HACK: Set this attribute to False to prevent label columns from being deleted
    training_args.remove_unused_columns = False
    trainer_class = DependencyParsingAdapterTrainer if adapter_args.train_adapter else DependencyParsingTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

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

    if model_args.load_best:
        from transformers.trainer import TRAINER_STATE_NAME
        import json
        best_ckpt = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME)))['best_model_checkpoint']
        trainer._load_from_checkpoint(best_ckpt) # This resets model.active_head to some random adapter
        # trainer.model.active_head = head_name
        # trainer.model.to(training_args.device)
    if model_args.load_last: # for resuming with adapter fusion
        from transformers.trainer_utils import get_last_checkpoint
        best_ckpt = get_last_checkpoint(model_args.load_last_dir or training_args.output_dir)
        trainer._load_from_checkpoint(best_ckpt) # This resets model.active_head to some random adapter
    trainer.model.active_head = head_name
    trainer.model.to(training_args.device)


    if model_args.reduce_params:
        trainer.model.eject_prefix_tuning(task_name)
        # trainer.model.merge_adapter(task_name)
        trainer.save_model()


    # Training
    if training_args.do_train:
        train_result = trainer.train(
            # model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)


    # Predict
    if training_args.do_predict:
        logging.info("*** Test ***")

        predictions, _, metrics = trainer.predict(dataset["test"])

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")

        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))


    return results


if __name__ == "__main__":
    main()
