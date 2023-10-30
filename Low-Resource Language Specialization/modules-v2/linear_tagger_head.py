from overrides import overrides
from typing import Any, Dict, List

import torch
import torch.nn as nn

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads import Head
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import (
    get_text_field_mask,
    sequence_cross_entropy_with_logits,
)
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy


@Head.register("linear_tagger")
class LinearTagger(Head):
    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            encoder.get_output_dim(), vocab.get_vocab_size("pos")
        )

        self.accuracy = CategoricalAccuracy()
        self.accuracy_words_only = CategoricalAccuracy()
        self.non_word_tags = set(
            vocab.get_token_index(tag, "pos") for tag in {"PUNCT", "SYM", "X"}
        )

        initializer(self)

    @overrides
    def forward(
        self,
        embedded_text_input: torch.Tensor,
        mask: torch.Tensor,
        pos_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch = self.dropout(encoded_text)
        logits = self.linear(batch)

        outputs = {"logits": logits, "mask": mask}

        if pos_tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, pos_tags, mask)
            self.accuracy(logits, pos_tags, mask)
            word_mask = mask.clone()
            for label in self.non_word_tags:
                label_mask = pos_tags.eq(label)
                word_mask = word_mask & ~label_mask
            self.accuracy_words_only(logits, pos_tags, word_mask)

            outputs["loss"] = loss
        if metadata is not None:
            outputs["metadata"] = metadata
        return outputs

    # @overrides
    # def make_output_human_readable(
    #     self, output_dict: Dict[str, torch.Tensor]
    # ) -> Dict[str, torch.Tensor]:
    #     raise NotImplementedError

    @overrides
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset),
            "accuracy_words_only": self.accuracy_words_only.get_metric(reset),
        }
