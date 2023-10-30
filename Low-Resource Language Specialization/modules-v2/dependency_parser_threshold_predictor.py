from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("dependency-parser-threshold")
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
        # not calculating attachment scores here
        results["pos_tags"] = ["PUNCT" for _ in results["tokens"]]
        return results

    # overrides for predict_instance, predict_batch_instances,
    # _build_hierplane_tree used to be here, but removed because they're not
    # useful for our use case

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        joint_loss = outputs["loss"]
        arc_loss = outputs["arc_loss"]
        tag_loss = outputs["tag_loss"]
        return f"{joint_loss}\t{arc_loss}\t{tag_loss}\n"
