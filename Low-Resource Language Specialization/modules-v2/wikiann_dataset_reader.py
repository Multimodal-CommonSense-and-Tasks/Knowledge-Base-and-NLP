import logging
from overrides import overrides
from typing import Dict, Iterable, List

from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("wikiann")
class WikiannDatasetReader(DatasetReader):
    """
    Reads instances from the Rahimi et al. (2019) partitioning of the Wikiann
    NER dataset, which uses an IOB2 tagging scheme.  Sentences are on contiguous
    lines, separated by a blank line, and each token is of the format:

    `{lang}:{token}\t{label}`
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path) as data_file:
            logger.info(
                f"Reading instances from lines in file at: {file_path}"
            )

            tokens = []
            tags = []

            for line in data_file:
                line = line.strip()
                if not line:
                    tok_input = [Token(tok) for tok in tokens]
                    tag_input = tags
                    tokens = []
                    tags = []
                    yield self.text_to_instance(tok_input, tag_input)
                else:
                    _prefix, content = line.split(":", maxsplit=1)
                    token, label = content.split("\t")
                    tokens.append(token)
                    tags.append(label)
            if tokens:
                assert tags
                tok_input = [Token(tok) for tok in tokens]
                tag_input = tags
                tokens = []
                tags = []
                yield self.text_to_instance(tok_input, tag_input)

    @overrides
    def text_to_instance(
        self, tokens: List[Token], tags: List[str] = None
    ) -> Instance:
        # tokens, tags, metadata "words"
        token_field = TextField(tokens)
        tag_field = SequenceLabelField(tags, token_field, self.label_namespace)
        meta_field = MetadataField({"words": [tok.text for tok in tokens]})

        fields = {
            "tokens": token_field,
            "tags": tag_field,
            "metadata": meta_field,
        }
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self._token_indexers
