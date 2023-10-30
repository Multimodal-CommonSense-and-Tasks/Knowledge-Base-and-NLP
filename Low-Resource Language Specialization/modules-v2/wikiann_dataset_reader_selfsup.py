import logging
from overrides import overrides
from typing import Dict, Iterable, List

from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import (
    LabelField,
    MetadataField,
    SequenceLabelField,
    TextField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer

logger = logging.getLogger(__name__)

_SEPARATOR_LINE = "===END UNLABELED DATA==="


@DatasetReader.register("wikiann_selfsup")
class WikiannSelfSupervisedDatasetReader(DatasetReader):
    """
    Reads mixed instances from a file with the following two parts, separated by
    ===END UNLABELED DATA===

    The first part consists of unlabeled sentences, where each token is separated
    by a space.
    
    The second part are from the Rahimi et al. (2019) partitioning of the Wikiann
    NER dataset, which uses an IOB2 tagging scheme.  Sentences are on contiguous
    lines, separated by a blank line, and each token is of the format:

    `{lang}:{token}\t{label}`
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        label_namespace: str = "labels",
        is_gold_namespace: str = "is_gold",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace
        self.is_gold_namespace = is_gold_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path) as data_file:
            logger.info(
                f"Reading instances from lines in file at: {file_path}"
            )
            # first part: unlabeled data
            for line in data_file:
                line = line.strip()
                if line == _SEPARATOR_LINE:
                    # cut off the file here
                    break
                elif line:
                    tokens = [Token(tok) for tok in line.split(" ")]
                    tags = ["O" for _ in tokens]
                    yield self.text_to_instance(
                        tokens, tags, is_gold_labeled=False
                    )

            # second part: labeled data
            tokens = []
            tags = []

            for line in data_file:
                line = line.strip()
                if (not line) and tokens:
                    if len(tokens) != len(tags):
                        logger.error(
                            f"Mismatch in length between tokens "
                            f"{tokens} and tags {tags}"
                        )
                    tok_input = [Token(tok) for tok in tokens]
                    tag_input = tags
                    tokens = []
                    tags = []
                    yield self.text_to_instance(tok_input, tag_input)
                elif line:
                    _prefix, content = line.split(":", maxsplit=1)
                    token, label = content.split("\t")
                    tokens.append(token)
                    tags.append(label)
                # else the line is a (potentially preceding) blank line
            if tokens:
                if len(tokens) != len(tags):
                    logger.error(
                        f"Mismatch in length between tokens "
                        f"{tokens} and tags {tags}"
                    )
                tok_input = [Token(tok) for tok in tokens]
                tag_input = tags
                tokens = []
                tags = []
                yield self.text_to_instance(
                    tok_input, tag_input, is_gold_labeled=True
                )

    @overrides
    def text_to_instance(
        self,
        tokens: List[Token],
        tags: List[str] = None,
        is_gold_labeled: bool = False,
    ) -> Instance:
        # tokens, tags, metadata "words"
        token_field = TextField(tokens)
        tag_field = SequenceLabelField(tags, token_field, self.label_namespace)
        meta_field = MetadataField({"words": [tok.text for tok in tokens]})
        gold_label_field = LabelField(
            int(is_gold_labeled),
            label_namespace=self.is_gold_namespace,
            skip_indexing=True,
        )

        fields = {
            "tokens": token_field,
            "tags": tag_field,
            "metadata": meta_field,
            "gold_label_mask": gold_label_field,
        }
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self._token_indexers
