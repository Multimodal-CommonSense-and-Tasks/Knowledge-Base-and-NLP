import unicodedata
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


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


@Predictor.register("dependency-parser-unlabeled")
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the
    [`BiaffineDependencyParser`](../models/biaffine_dependency_parser.md) model.
    Adapted from
    https://github.com/allenai/allennlp-models/blob/main/allennlp_models/structured_prediction/predictors/biaffine_dependency_parser.py
    (majority) and
    https://github.com/ethch18/parsing-mbert/blob/master/modules/dependency_parser_predictor.py
    (dump_line)
    """

    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        language: str = "en_core_web_sm",
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        # Parameters

        sentence The sentence to parse.

        # Returns

        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            json_dict["tokens"], json_dict["pos_tags"]
        )

    @overrides
    def load_line(self, line: str) -> JsonDict:
        # adapted from ud_dataset_reader_selfsup.py
        results: JsonDict = {}
        results["tokens"] = [tok for tok in line.strip().split(" ") if tok]
        results["pos_tags"] = [
            "PUNCT" if _is_punctuation(tok) else "NOUN"
            for tok in results["tokens"]
        ]
        return results

    # overrides for predict_instance, predict_batch_instances,
    # _build_hierplane_tree used to be here, but removed because they're not
    # useful for our use case

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        words = outputs["words"]
        pos = outputs["pos"]
        heads = outputs["predicted_heads"]
        tags = outputs["predicted_dependencies"]
        return (
            "".join(
                [
                    (
                        "{0}\t{1}\t{1}\t{2}\t{2}\t_\t{3}\t{4}\t_\t_\n".format(
                            i + 1, words[i].strip(), pos[i], heads[i], tags[i]
                        )
                    )
                    for i in range(len(words))
                ]
            )
            + "\n"
        )
