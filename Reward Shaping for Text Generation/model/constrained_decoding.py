import torch
from transformers import LogitsProcessor


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, sample_p: float, exclude_id: list):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        gen_tokens = [tok for tok in gen_tokens if tok not in exclude_id]
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            # if sample_p != 1, apply sampling
            if sample_p != 1 and torch.rand((1,)).item() >= sample_p:
                continue
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


class PenalizeNgramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that penalizes repetition of encoder input ids n-grams for the decoder ids.
        Papers:
            Niu (2020) Unsupervised Paraphrase Generation with Pretrained Language Models
                n_gram_penalty = [0, inf, 0, 0]
                sample_p = 0.5
            Thompson (2020) Paraphrase Generation as Zero-Shot Multilingual Translation
                n_gram_penalty = [0.003, 0.048, 0.243, 0.768]
                sample_p = 1
    Args:
        penalty_for_ngrams (List(`int`)):
            List of length 4, denoting penalty for n-gram of size 1~4.
        sample_p (`float`):
            If given between 0~1, only sampled n_grams are penalized rather than all.
        exclude_id (`List[float]`):
            Token ids that should not be penalized. typically: PAD, EOS
    """

    def __init__(self, penalty_for_ngrams, sample_p):
        if len(penalty_for_ngrams) == 4:
            penalty_for_ngrams.insert(0, 0)
        self.penalty = penalty_for_ngrams
        self.sample_p = sample_p
        self.exclude_id=None
    
    def update(self, encoder_input_ids: torch.LongTensor):
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]

        self.generated_ngrams = [None] + [
            _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size, self.sample_p, self.exclude_id)
            for encoder_ngram_size in range(1, 4+1)    
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]

        for ngram_size in range(1, 4+1):
            if self.penalty[ngram_size] == 0:
                continue

            banned_batch_tokens = [
                _get_generated_ngrams(
                    self.generated_ngrams[ngram_size][hypo_idx // num_beams], input_ids[hypo_idx], ngram_size, cur_len
                )
                for hypo_idx in range(num_hypos)
            ]

            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] -= self.penalty[ngram_size]

        return scores


class PositionalConstraintLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that penalizes if first n tokens generated are repeated from input.
        Papers:
            Hu (2019) ParaBank: Monolingual Bitext Generation and Sentential Paraphrasing via Lexically-constrained Neural Machine Translation
    Args:
        position_n:
            Initial tokens to penalize if being repeated from input
        penalty (List(`int`)):
            Size of the penalty
    """

    def __init__(self, position_n, penalty):
        self.position_n = position_n
        self.penalty = penalty
        self.exclude_id=None

    def update(self, encoder_input_ids: torch.LongTensor):
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]

        self.banned_tokens = encoder_input_ids[:, 1:1+self.position_n].tolist()
        assert len(self.banned_tokens) == self.batch_size
        for batch in self.banned_tokens:
            for exclude in self.exclude_id:
                if exclude in batch:
                    batch.remove(exclude)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]

        if cur_len <= self.position_n:
            for i, banned_token in enumerate(self.banned_tokens): # i: batch index
                for j in range(num_beams):
                    scores[i * num_beams + j, banned_token] -= self.penalty

        return scores