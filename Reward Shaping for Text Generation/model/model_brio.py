import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from scipy.stats import rankdata

from .arguments import TrieCLArguments
from .model import ParaphraserBase
from .metrics import SequenceEvaluationMetric

class BRIOParaphraser(ParaphraserBase):
    """
    Implementation of BRIO(Bringing Order to Abstractive Summarization) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            metric: SequenceEvaluationMetric,
            args: TrieCLArguments,
            **kwargs):
        super(BRIOParaphraser, self).__init__(base, tokenizer, num_beams=args.num_beams)

        self.metric = metric
        self.pad_id = self.base.config.pad_token_id
        self.len_penalty = args.len_penalty

        self.contrast_lambda = args.contrast_lambda
        
        self.generative = args.generative
        self.contrastive = args.contrastive
        self.mix_rate = args.mix_rate


    def get_contrastive_loss(self, inputs, outputs, hypos=None, _=None, __=None, ___=None):
        """
        Calculates the token_wise contrastive loss.
        """
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        batch_size = len(inputs)
        beam_size = self.num_beams

        
        if hypos is None:
            with torch.no_grad():
                # Generate in beam sequences(beam size = batch size)
                output = self.base.generate(
                    inputs,
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

            # Rank the outputs                
            sources_decode = self.tokenizer.batch_decode(inputs, skip_special_tokens=True) # [B]
            extended_inputs = [x for x in sources_decode for _ in range(beam_size)]
            
            metrics = self.metric(extended_inputs, _, samples_str, (batch_size, beam_size), extended=True).reshape(batch_size, beam_size).cpu() # batch_size * num_beams
            ranks = torch.tensor(1 + batch_size - rankdata(metrics, method='max', axis=-1)).to(torch.float32).to(self.base.device)

            # Generate sequence pair differences
            rank_diff_matrix = ranks.unsqueeze(2) - ranks.unsqueeze(1) # batch_size * num_beams * num_beams
            rank_diff_matrix *= self.contrast_lambda
            rank_diff_mask = (rank_diff_matrix > 0).to(torch.float32)

            decoder_mask = hypos != self.pad_id

        # Compare beams according to their rank
        # we compute single input and its output beams one by one(that's why we set beam_size to batch_size)
        contrast_loss = 0
        cnt = 0
        for i in range(batch_size):
            ith_input_ids = inputs[i].repeat(beam_size, 1)
            target = hypos[i]
            dmask = decoder_mask[i]
            rmask = rank_diff_mask[i]
            logits = self.base(
                input_ids=ith_input_ids.to(self.base.device),
                labels=target.to(self.base.device),
            ).logits # num_beams * seq_len * vocab_size

            # Calculate NLL losses and length penalty
            losses = - loss_fct(logits.reshape(-1, logits.shape[2]), target.reshape(-1))
            losses = losses.reshape(logits.shape[0], logits.shape[1]) * dmask # num_beams * seq_len
            losses = losses.sum(dim=-1) / torch.pow(dmask.sum(dim=1), self.len_penalty)
            
            # calculate pairwise loss
            loss_diff_matrix = losses[:, None] - losses # loss_diff[i, j] = lprobs[i] - lprobs[j]
            # if rmask[i,j] == true, it means that rank[i] > rank[j], i.e., j is ranked higher
            # we want to minimize max(0, lprobs[i] - lprobs[j] + lambda_ij), i.e.,
            # want lprobs[j] to be larger than lprobs[i] + lambda_ij
            loss_terms = torch.max(torch.zeros_like(loss_diff_matrix), rank_diff_matrix[i] + loss_diff_matrix) * rmask
            update_val = loss_terms.sum() / rmask.sum() # Normalize by (seq1, seq2) combination count
            if not torch.isnan(update_val): # NaN prevention
                contrast_loss += update_val
                cnt += 1
        
        if cnt > 0:
            return contrast_loss / cnt # Normalize by batch size
        else:
            return 0
