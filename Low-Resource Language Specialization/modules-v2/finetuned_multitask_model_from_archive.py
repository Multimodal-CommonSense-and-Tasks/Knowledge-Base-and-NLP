import logging
from typing import Any, Dict

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.multitask import MultiTaskModel
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


@Model.register(
    "finetuned_multitask_from_archive", constructor="multitask_from_archive"
)
class FinetunedModelDummyClass(MultiTaskModel):
    @classmethod
    def multitask_from_archive(
        cls,
        archive_file: str,
        head_name: str,
        head_overrides: Dict[str, Any] = {},
        vocab: Vocabulary = None,
    ) -> MultiTaskModel:
        model = load_archive(archive_file).model

        if not isinstance(model, MultiTaskModel):
            raise ConfigurationError(
                "Finetuned models used this way must have "
                "originally been trained as multitask."
            )

        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()

        # prevent mutation during iteration
        all_model_heads = list(model._heads.keys())
        for head in all_model_heads:
            if head != head_name:
                logger.info(f"Removing model head {head}")
                # free up memory we won't use
                del model._heads[head]

        for key, val in head_overrides.items():
            old_val = getattr(model._heads[head_name], key)
            logger.info(
                f"For head {head_name}, changing property {key} "
                f"from {old_val} to {val}"
            )
            setattr(model._heads[head_name], key, val)

        return model
