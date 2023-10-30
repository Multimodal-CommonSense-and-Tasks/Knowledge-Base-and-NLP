from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
import numpy

from .biaffine_parser import BiaffineDependencyParser

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import (
    Seq2SeqEncoder,
    TextFieldEmbedder,
    Embedding,
    InputVariationalDropout,
)
from allennlp.modules.matrix_attention.bilinear_matrix_attention import (
    BilinearMatrixAttention,
)
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import (
    get_device_of,
    masked_log_softmax,
    get_lengths_from_binary_sequence_mask,
)
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores

logger = logging.getLogger(__name__)

POS_TO_IGNORE = {"`", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}


@Model.register("biaffine_parser_selfsup")
class BiaffineDependencyParserSelfSupervised(BiaffineDependencyParser):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    teacher : `Model`
        The pretrained teacher model, which must be a BiaffineDependencyParser.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : `int`, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : `Embedding`, optional.
        Used to embed the `pos_tags` `SequenceLabelField` we get as input to the model.
    use_mst_decoding_for_validation : `bool`, optional (default = `True`).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = `0.0`)
        The dropout applied to the embedded text input.
    gold_interpolation : `float`, optional, (default = None)
        If present, interpolates the KL-divergence loss with the standard losses
        as gold_interp * KL + (1 - gold_interp) * loss.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        teacher: Model,
        teacher_prefix: str = "teacher_",
        # vocab: Vocabulary,
        # text_field_embedder: TextFieldEmbedder,
        # encoder: Seq2SeqEncoder,
        # tag_representation_dim: int,
        # arc_representation_dim: int,
        # tag_feedforward: FeedForward = None,
        # arc_feedforward: FeedForward = None,
        # pos_tag_embedding: Embedding = None,
        # use_mst_decoding_for_validation: bool = True,
        # dropout: float = 0.0,
        # input_dropout: float = 0.0,
        gold_interpolation: Optional[float] = None,
        # initializer: InitializerApplicator = InitializerApplicator(),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if type(teacher) is not BiaffineDependencyParser:
            logger.error(
                "Teacher model is not of type BiaffineDependencyParser"
            )
        self.teacher_model: BiaffineDependencyParser = teacher
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        self.teacher_prefix = teacher_prefix
        self.gold_interpolation = gold_interpolation

    @overrides
    def forward(
        self,  # type: ignore
        words: TextFieldTensors,
        pos_tags: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        gold_label_mask: torch.LongTensor = None,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        words : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, sequence_length)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : `torch.LongTensor`, required
            The output of a `SequenceLabelField` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        metadata : `List[Dict[str, Any]]`, optional (default=`None`)
            A dictionary of metadata for each batch element which has keys:
                words : `List[str]`, required.
                    The tokens in the original sentence.
                pos : `List[str]`, required.
                    The dependencies POS tags for each word.
        gold_label_mask : `torch.LongTensor`, optional (default = `None`)
            A tensor of shape `(batch_size,)` that has 1 for entries in the
            batch that have gold labels, and 0 for those that don't.
        head_tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape `(batch_size, sequence_length)`.
        head_indices : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length)`.

        # Returns

        An output dictionary consisting of:

        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        arc_loss : `torch.FloatTensor`
            The loss contribution from the unlabeled arcs.
        loss : `torch.FloatTensor`, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : `torch.FloatTensor`
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : `torch.FloatTensor`
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : `torch.BoolTensor`
            A mask denoting the padded elements in the batch.
        """
        self.teacher_model.eval()
        teacher_words = {
            key[len(self.teacher_prefix) :]: val
            for key, val in words.items()
            if self.teacher_prefix in key
        }
        words = {
            key: val
            for key, val in words.items()
            if self.teacher_prefix not in key
        }
        embedded_text_input = self.text_field_embedder(words)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat(
                [embedded_text_input, embedded_pos_tags], -1
            )
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError(
                "Model uses a POS embedding, but no POS tags were passed."
            )

        mask = get_text_field_mask(words)
        teacher_mask = get_text_field_mask(teacher_words)
        assert torch.all(mask == teacher_mask)

        # we won't need the gold loss unless we're doing interpolation,
        # so skip the computation
        tag_input = head_tags if self.gold_interpolation is not None else None
        idx_input = (
            head_indices if self.gold_interpolation is not None else None
        )
        (
            predicted_heads,
            predicted_head_tags,
            mask,
            normalized_arc_logits,
            normalized_head_tag_logits,
            _arc_raw_loss,
            _tag_raw_loss,
            arc_raw_loss_gold,
            tag_raw_loss_gold,
        ) = self._parse(
            embedded_text_input,
            mask,
            head_tags=tag_input,
            head_indices=idx_input,
        )

        with torch.no_grad():
            teacher_input = self.teacher_model.text_field_embedder(
                teacher_words
            )
            (
                _,  # predicted_heads
                _,  # predicted_head_tags
                _,  # mask
                _,  # arc_nll
                _,  # tag_nll
                teacher_arc,
                teacher_tag,
                _,  # arc_raw
                _,  # tag_raw
                _,  # valid_positions
            ) = self.teacher_model._parse(
                # force the use of predicted heads/indices
                teacher_input,
                # same as original mask prior to being munged by last call to
                # _parse()
                teacher_mask,
                head_tags=None,
                head_indices=None,
            )

        # remove the first element of the sequence length dimension, since
        # it represents the symbolic ROOT token head that was added by
        # self._parse
        # (batch_size, sequence_length, sequence_length + 1)
        normalized_arc_logits = normalized_arc_logits[:, 1:, :]
        teacher_arc = teacher_arc[:, 1:, :]
        # (batch_size, sequence_length, num_head_tags)
        normalized_head_tag_logits = normalized_head_tag_logits[:, 1:, :]
        teacher_tag = teacher_tag[:, 1:, :]

        if (
            normalized_arc_logits is None
            or teacher_arc is None
            or normalized_head_tag_logits is None
            or teacher_tag is None
        ):
            logger.warn(
                "One of {student, teacher} x {arc, tag} logits is None; "
                "skipping loss calculation..."
            )
            arc_kl = torch.tensor(0.0)
            tag_kl = torch.tensor(0.0)
            loss = torch.tensor(0.0)
        else:
            if normalized_arc_logits.shape != teacher_arc.shape:
                logger.error(
                    "Size mismatch between self/teacher arc shapes"
                    f"{normalized_arc_logits.shape} and {teacher_arc.shape}"
                )

            if normalized_head_tag_logits.shape != teacher_tag.shape:
                logger.error(
                    "Size mismatch between self/teacher tag shapes"
                    f"{normalized_head_tag_logits.shape} and {teacher_tag.shape}"
                )

            # arc_kl and tag_kl on distributions; multiplied by mask[:, 1:] because
            # the mask has been corrupted by self._parse
            # everything is already in log space
            arc_kl = (
                F.kl_div(
                    normalized_arc_logits,
                    teacher_arc,
                    reduction="none",
                    log_target=True,
                )
                * mask[:, 1:].unsqueeze(-1)
            )
            tag_kl = (
                F.kl_div(
                    normalized_head_tag_logits,
                    teacher_tag,
                    reduction="none",
                    log_target=True,
                )
                * mask[:, 1:].unsqueeze(-1)
            )
            # the base loss is mean
            arc_kl = arc_kl.mean()
            tag_kl = tag_kl.mean()
            loss = arc_kl + tag_kl

            if self.gold_interpolation is not None:
                if arc_raw_loss_gold is None or tag_raw_loss_gold is None:
                    logger.error(
                        "Attempted to compute interpolated loss, "
                        "but no gold indices/tags provided"
                    )
                arc_loss = -arc_raw_loss_gold
                tag_loss = -tag_raw_loss_gold
                if gold_label_mask is not None:
                    arc_loss *= gold_label_mask.unsqueeze(-1)
                    tag_loss *= gold_label_mask.unsqueeze(-1)
                    # valid positions are ones that are non-padding, not ROOT, and
                    # have gold labels
                    valid_positions = (
                        mask[:, 1:] * gold_label_mask.unsqueeze(-1)
                    ).sum()
                else:
                    valid_positions = mask[:, 1:].sum()

                arc_loss = arc_loss.sum() / valid_positions.float()
                tag_loss = tag_loss.sum() / valid_positions.float()

                loss = self.gold_interpolation * loss + (
                    1 - self.gold_interpolation
                ) * (arc_loss + tag_loss)

        if head_indices is not None and head_tags is not None:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(
                predicted_heads[:, 1:],
                predicted_head_tags[:, 1:],
                head_indices,
                head_tags,
                evaluation_mask,
            )

        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "arc_loss_kl": arc_kl,
            "tag_loss_kl": tag_kl,
            "loss": loss,
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
            "pos": [meta["pos"] for meta in metadata],
        }

        return output_dict

    def _parse(
        self,
        embedded_text_input: torch.Tensor,
        mask: torch.BoolTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], 1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], 1
            )
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(
            self.head_arc_feedforward(encoded_text)
        )
        child_arc_representation = self._dropout(
            self.child_arc_feedforward(encoded_text)
        )

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(
            self.head_tag_feedforward(encoded_text)
        )
        child_tag_representation = self._dropout(
            self.child_tag_feedforward(encoded_text)
        )
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(
            head_arc_representation, child_arc_representation
        )

        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = (
            attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        )

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation,
                child_tag_representation,
                attended_arcs,
                mask,
            )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation,
                child_tag_representation,
                attended_arcs,
                mask,
            )
        if head_indices is not None and head_tags is not None:
            # how many different tags are predicted by our model (0-indexed)
            max_head_tag_index = self.tag_bilinear.out_features - 1
            if torch.any(head_tags > max_head_tag_index):
                logger.info(
                    "Unseen gold tag!  Skipping head loss computation "
                    "to avoid tensor errors"
                )
                arc_raw_gold = None
                tag_raw_gold = None
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    arc_raw_gold,
                    tag_raw_gold,
                    _,
                ) = self._construct_loss(
                    head_tag_representation=head_tag_representation,
                    child_tag_representation=child_tag_representation,
                    attended_arcs=attended_arcs,
                    head_indices=head_indices,
                    head_tags=head_tags,
                    mask=mask,
                )
        else:
            arc_raw_gold = None
            tag_raw_gold = None

        (
            _arc_nll,
            _tag_nll,
            normalized_arc_logits,
            normalized_head_tag_logits,
            arc_raw,
            tag_raw,
            _valid_positions,
        ) = self._construct_loss(
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            attended_arcs=attended_arcs,
            head_indices=predicted_heads.long(),
            head_tags=predicted_head_tags.long(),
            mask=mask,
        )

        return (
            predicted_heads,
            predicted_head_tags,
            mask,
            normalized_arc_logits,
            normalized_head_tag_logits,
            arc_raw,
            tag_raw,
            arc_raw_gold,
            tag_raw_gold,
        )
