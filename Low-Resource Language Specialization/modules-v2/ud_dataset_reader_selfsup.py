from typing import Dict, Tuple, List
import logging
import unicodedata

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    LabelField,
    TextField,
    SequenceLabelField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)

_SEPARATOR_LINE = "===END UNLABELED DATA==="


def _is_punctuation(char):
    """
    Checks whether `char` is a punctuation character.
    Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py#L77
    """
    if len(char) > 1:
        return False
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


@DatasetReader.register("universal_dependencies_selfsup")
class UniversalDependenciesSelfSupervisedDatasetReader(DatasetReader):
    """
    Reads mixed instances from a file with the following two parts, separated by
    ===END UNLABELED DATA===

    The first part consists of unlabeled sentences, where each token is
    separated by a space.

    The second part is conllu UD format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : `bool`, optional (default = `False`)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `Tokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        tokenizer: Tokenizer = None,
        is_gold_namespace: str = "is_gold",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer
        self.is_gold_namespace = is_gold_namespace

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info(
                "Reading UD instances from conllu dataset at: %s", file_path
            )

            # first part: unlabeled data
            line = conllu_file.readline()
            while line:
                line = line.strip()
                if line == _SEPARATOR_LINE:
                    break
                elif line:
                    tokens = line.split(" ")
                    # This effectively serves to mask out everything when
                    # calculating LAS, since those are undefined for unknowns
                    # anyways
                    pos_tags = ["PUNCT" for _ in tokens]
                    # # we're not using POS tag embeddings, so this is really
                    # # to filter on the evaluation mask
                    # pos_tags = [
                    #     "PUNCT" if _is_punctuation(tok) else "NOUN"
                    #     for tok in tokens
                    # ]
                    heads = [0 for _ in tokens]
                    tags = ["root" for _ in tokens]
                    yield self.text_to_instance(
                        tokens,
                        pos_tags,
                        list(zip(tags, heads)),
                        is_gold_labeled=False,
                    )
                line = conllu_file.readline()

            # second part: conllu data
            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                annotation = [
                    x for x in annotation if isinstance(x["id"], int)
                ]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
                yield self.text_to_instance(
                    words,
                    pos_tags,
                    list(zip(tags, heads)),
                    is_gold_labeled=True,
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
        is_gold_labeled: bool = False,
    ) -> Instance:

        """
        # Parameters

        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies : `List[Tuple[str, int]]`, optional (default = `None`)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        # Returns

        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]

        text_field = TextField(tokens)
        fields["words"] = text_field
        fields["pos_tags"] = SequenceLabelField(
            upos_tags, text_field, label_namespace="pos"
        )
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies],
                text_field,
                label_namespace="head_tags",
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies],
                text_field,
                label_namespace="head_index_tags",
            )

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags})
        fields["gold_label_mask"] = LabelField(
            int(is_gold_labeled),
            label_namespace=self.is_gold_namespace,
            skip_indexing=True,
        )
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["words"].token_indexers = self._token_indexers
