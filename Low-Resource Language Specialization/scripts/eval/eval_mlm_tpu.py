from accelerate import Accelerator
import argparse
import os
import sys
from dataclasses import dataclass, field

import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import random, math
import transformers
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
import torch
from scripts.modeling.convert_bert_to_hf import ROTATION_LAYER_NAME
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput

class BertForMaskedLMWithFinalTransform(BertForMaskedLM):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.final_align_transform(sequence_output) # added
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class DataCollatorForLanguageModelingMaxLen:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            raise NotImplementedError
            # batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


logger = logging.getLogger(__name__)


def set_(dir, args, accelerator):
    if args.is_roberta:
        tokenizer = RobertaTokenizerFast.from_pretrained(dir, max_len=512)

    else:
        tokenizer = BertTokenizerFast.from_pretrained(
            dir,
            clean_text=True,
            tokenize_chinese_chars=True,
            strip_accents=False,
            do_lower_case=False,
        )

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer, file_path=args.eval_file, block_size=128
    )
    if not args.tpu:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
        )
    else:
        data_collator = DataCollatorForLanguageModelingMaxLen(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability, padding='max_length', max_length=128,
        )

    eval_dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(
        eval_dataloader
    )
    return tokenizer, dataset, eval_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--best_epoch_save_name", default="best_epoch.txt")
    parser.add_argument("--min_loss_save_name", default="min_loss.txt")
    parser.add_argument("--start_eps", type=int, default=0)
    parser.add_argument("--end_eps", type=int, default=20)
    parser.add_argument("--is-roberta", action="store_true")
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
    parser.add_argument("--tpu", default=False, action='store_true')
    parser.add_argument("--has_rotation", default=False, action='store_true')
    parser.add_argument("--additional_linears", default="")

    args = parser.parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    best_i = 0
    min_loss = float("inf")
    if os.path.exists(os.path.join(args.model_dir, args.best_epoch_save_name)):
        best_i = int(open(os.path.join(args.model_dir, args.best_epoch_save_name)).read().strip())
        min_loss = float(open(os.path.join(args.model_dir, args.min_loss_save_name)).read().strip())
    tokenizer = None
    with torch.no_grad():
        for i in range(args.start_eps, args.end_eps + 1):
            dir = os.path.join(args.model_dir, f"epoch_{i}")
            if not os.path.exists(dir):
                continue

            if tokenizer is None:
                tokenizer, dataset, eval_dataloader = set_(dir, args, accelerator)


            if args.is_roberta:
                model = RobertaForMaskedLM.from_pretrained(dir)

            elif args.additional_linears:
                import importlib
                modules_v2 = importlib.import_module("modules-v2.custom_transformer")
                BertForMaskedLMWithLinear = modules_v2.BertForMaskedLMWithLinear
                model = BertForMaskedLMWithLinear.from_pretrained(dir)
                assert model.config.additional_linears == [int(i) for i in args.additional_linears.split(',')]
            else:
                model = BertForMaskedLM.from_pretrained(dir)

            # Prepare everything with our `accelerator`.
            model = accelerator.prepare(
                model
            )

            model.eval()
            losses = []
            progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
                progress_bar.update(1)

            losses = torch.cat(losses)
            losses = losses[:len(dataset)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {i}: perplexity: {perplexity}")
            if min_loss > perplexity:
                min_loss = perplexity
                best_i = i

            accelerator.free_memory() # this frees memory !!!
            del model

    if accelerator.is_local_main_process:
        with open(os.path.join(args.model_dir, args.best_epoch_save_name), 'w') as f:
            f.write(f"{best_i}")
        with open(os.path.join(args.model_dir, args.min_loss_save_name), 'w') as f:
            f.write(f"{min_loss}")

if __name__ == '__main__':
    main()