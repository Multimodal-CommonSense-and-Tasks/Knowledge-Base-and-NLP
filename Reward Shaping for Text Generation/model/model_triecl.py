import random

from typing import List, Set, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

# from scipy.stats import rankdata

from .arguments import TrieCLArguments
from .model import ParaphraserBase
from .dataset import get_prefix
from .metrics import SequenceEvaluationMetric

class TrieCLParaphraser(ParaphraserBase):
    """
    Implementation of TrieCL(proposed) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            metric: SequenceEvaluationMetric,
            args: TrieCLArguments,
            **kwargs):
        super(TrieCLParaphraser, self).__init__(base, tokenizer, args.num_beams)

        self.metric = metric
        self.pad_id = self.base.config.pad_token_id
        self.eos_id = self.base.config.eos_token_id
        self.bos_id = self.base.config.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.pad_id # T5 hotfix

        self.contrast_lambda = args.contrast_lambda

        self.generative = args.generative
        self.contrastive = args.contrastive
        self.mix_rate = args.mix_rate


    def get_contrastive_loss(self, src, tgt, hypos=None, all_branches=None, all_win_indices=None, all_lose_indices=None, _=None, return_scores=False):
        """
        Calculates the token_wise contrastive loss.
        @return loss
        """

        batch_size, _ = src.shape
        # beam_size = hypos[0].shape[0]
        beam_size = self.num_beams

        if hypos is None:
            with torch.no_grad():
                # Generate in beam sequences(beam size = batch size)
                output = self.base.generate(
                    src.to(self.base.device),
                    num_beams=self.num_beams,
                    # Output control
                    # max_new_tokens=int(input_ids.size(1)),
                    num_return_sequences=self.num_beams,
                    return_dict_in_generate=True,
                    output_scores=True,
                    early_stopping=True
                )
                sequences = output.sequences.reshape(batch_size, self.num_beams, -1)
                if self.tokenizer.bos_token_id is not None:
                    bos_index = sequences[0, 0].tolist().index(self.tokenizer.bos_token_id)
                    sequences = sequences[:, :, bos_index:].contiguous()
                else:
                    pad_index = sequences[0, 0].tolist().index(self.tokenizer.pad_token_id)
                    sequences = sequences[:, :, pad_index+1:].contiguous()

                samples_str = self.tokenizer.batch_decode(sequences.view(-1, sequences.size(-1)), skip_special_tokens=True) # aggregate batch & sample IDs
                hypos = sequences
                sequences = sequences.tolist()

            # Rank the outputs
            sources_decode = self.tokenizer.batch_decode(src, skip_special_tokens=True) # [B]
            extended_inputs = [x for x in sources_decode for _ in range(beam_size)]

            scores = self.metric(extended_inputs, None, samples_str, (batch_size, beam_size), extended=True).reshape(batch_size, beam_size).cpu() # batch_size * num_beams
            # Extract common prefixes out of the prefix tree
            all_branches, all_win_indices, all_lose_indices = get_prefix(sequences, scores, self.tokenizer.pad_token_id)

        contrast_loss = 0
        # cnt = 0
        for i in range(batch_size):
            ith_input_ids = src[i].repeat(beam_size, 1)
            # branches = all_branches[i]
            win_indices = all_win_indices[i]
            lose_indices = all_lose_indices[i]
            target = hypos[i]
            logits = self.base(
                input_ids=ith_input_ids.to(self.base.device),
                labels=target.to(self.base.device),
            ).logits # num_beams, seq_len, vocab_size
            probs = logits.softmax(dim=-1).reshape(-1, logits.shape[-1]) # [B*T,V]
            probs = torch.gather(probs, -1, index=target.reshape(-1).unsqueeze(-1)).reshape(target.shape) # [B, T]
            lose_x, lose_y = zip(*lose_indices)
            win_x, win_y = zip(*win_indices)
            c_loss = probs[lose_x, lose_y] - probs[win_x, win_y] + self.contrast_lambda
            c_loss = (c_loss > 0) * c_loss
            contrast_loss += c_loss.sum()
        
        mean_loss = contrast_loss / (batch_size * (beam_size - 1))
        if return_scores:
            return mean_loss, scores[:, 0]
        else:
            return mean_loss
