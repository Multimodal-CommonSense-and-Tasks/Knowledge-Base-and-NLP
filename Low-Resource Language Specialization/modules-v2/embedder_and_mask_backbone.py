import torch 
from typing import Dict

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.backbones import Backbone
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors


@Backbone.register("embedder_and_mask")
class EmbedderAndMaskBackbone(Backbone):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
    ) -> None:
        super().__init__()
        self._vocab = vocab
        self._text_field_embedder = text_field_embedder

    def forward(self, text: TextFieldTensors) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(text)
        embedded_text_input = self._text_field_embedder(text)
        token_ids = get_token_ids_from_text_field_tensors(text)
        return {
            "mask": mask,
            "embedded_text_input": embedded_text_input,
            "token_ids": token_ids,
        }

    