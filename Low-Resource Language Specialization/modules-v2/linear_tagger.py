from overrides import overrides
from typing import Any, Dict, List

import torch
import torch.nn as nn

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import (
    get_text_field_mask,
    sequence_cross_entropy_with_logits,
)
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy


@Model.register("linear-tagger")
class LinearTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        input_dim: int,
        max_rank: int,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        # The following is based on
        # https://github.com/john-hewitt/control-tasks/blob/master/control-tasks/probe.py#L366
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(input_dim, max_rank)
        self.linear_2 = nn.Linear(max_rank, vocab.get_vocab_size("pos"))

        self.accuracy = CategoricalAccuracy()
        self.accuracy_words_only = CategoricalAccuracy()
        self.non_word_tags = set(
            vocab.get_token_index(tag, "pos") for tag in {"PUNCT", "SYM", "X"}
        )

        initializer(self)

    @overrides
    def forward(
        self,
        words: TextFieldTensors,
        pos_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        embedded_input = self.text_field_embedder(words)
        mask = get_text_field_mask(words)

        batch = self.dropout(embedded_input)
        batch = self.linear_1(batch)
        logits = self.linear_2(batch)

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

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @overrides
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset),
            "accuracy_words_only": self.accuracy_words_only.get_metric(reset),
        }
