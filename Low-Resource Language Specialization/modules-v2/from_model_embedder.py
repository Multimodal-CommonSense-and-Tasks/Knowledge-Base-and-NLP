from typing import List
import torch

from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder


@TextFieldEmbedder.register("from_model")
class FromModelTokenEmbedder(TextFieldEmbedder):
    """
    Grabs the token embedder portion of a pretrained model
    """

    def __init__(
        self,
        original_model: Model,
        model_field: str,
        model_subfields: List[str] = [],
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        embedder = getattr(original_model, model_field)
        for subfield in model_subfields:
            embedder = getattr(embedder, subfield)
        self.copied_embedder = embedder
        self.copied_embedder.requires_grad_(requires_grad)
        self.requires_grad = requires_grad

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if not self.requires_grad:
            self.copied_embedder.eval()
        return self.copied_embedder(*args, **kwargs)

    def get_output_dim(self) -> int:
        return self.copied_embedder.get_output_dim()
