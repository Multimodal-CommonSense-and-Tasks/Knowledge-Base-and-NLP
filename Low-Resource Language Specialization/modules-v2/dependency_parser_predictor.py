from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("dependency-parser")
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
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        """
        spacy_tokens = self._tokenizer.tokenize(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        if self._dataset_reader.use_language_specific_pos:  # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    # overrides for predict_instance, predict_batch_instances,
    # _build_hierplane_tree used to be here, but removed because they're not
    # useful for our use case

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        words = outputs["ud_words"]
        pos = outputs["ud_pos"]
        heads = outputs["ud_predicted_heads"]
        tags = outputs["ud_predicted_dependencies"]
        return (
            "".join(
                [
                    (
                        "{0}\t{1}\t{1}\t{2}\t{2}\t_\t{3}\t{4}\t_\t_\n".format(
                            i + 1, words[i], pos[i], heads[i], tags[i]
                        )
                    )
                    for i in range(len(words))
                ]
            )
            + "\n"
        )
