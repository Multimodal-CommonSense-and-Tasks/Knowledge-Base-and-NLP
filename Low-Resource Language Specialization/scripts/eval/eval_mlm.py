import argparse
import os

from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    TrainingArguments,
    Trainer,
)
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

parser = argparse.ArgumentParser()
parser.add_argument("--eval-file", type=str, required=True)
parser.add_argument("--model-dir", type=str, required=True)
parser.add_argument("--best_epoch_save_name", default="best_epoch.txt")
parser.add_argument("--is-roberta", action="store_true")
parser.add_argument("--mlm-probability", type=float, default=0.15)
parser.add_argument("--tpu", default=False, action='store_true')
parser.add_argument("--has_rotation", default=False, action='store_true')
args = parser.parse_args()


results = {}


best_i = 0
min_loss = float("inf")
for i in range(21):
    dir = os.path.join(args.model_dir, f"epoch_{i}")
    if not os.path.exists(dir):
        continue

    if args.is_roberta:
        tokenizer = RobertaTokenizerFast.from_pretrained(dir, max_len=512)
        model = RobertaForMaskedLM.from_pretrained(dir)

    else:
        tokenizer = BertTokenizerFast.from_pretrained(
            dir,
            clean_text=True,
            tokenize_chinese_chars=True,
            strip_accents=False,
            do_lower_case=False,
        )
        if args.has_rotation:
            model = BertForMaskedLMWithFinalTransform.from_pretrained(dir)
            rotation_layer_path = os.path.join(dir, ROTATION_LAYER_NAME)
            trans_layer = torch.load(rotation_layer_path, map_location='cpu')
            model.final_align_transform = trans_layer
        else:
            model = BertForMaskedLM.from_pretrained(dir)

    dataset = LineByLineTextDataset(
            tokenizer=tokenizer, file_path=args.eval_file, block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=8 if args.tpu else 32,
        do_train=False,
        do_eval=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        data_collator=data_collator,
    )
    metrics = trainer.evaluate()
    results[i] = metrics

with open(os.path.join(args.model_dir, args.best_epoch_save_name), 'w') as f:
    f.write(f"{best_i}")

print(results)
