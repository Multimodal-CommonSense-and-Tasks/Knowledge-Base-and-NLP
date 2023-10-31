import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    LogitsProcessorList
)

class ParaphraserBase(nn.Module):
    """
    BART based module
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            # device: torch.device = torch.device("cpu"),
            **kwargs):
        super(ParaphraserBase, self).__init__()

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id

        self.num_beams = num_beams
        # self.device = device

    def get_generation_loss(self, inputs, outputs):
        """
        Calculates classic teacher-forced generation loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        # torch.cuda.empty_cache()
        # assert len(inputs) == len(outputs)
        # batch_size = len(inputs)

        # Tokenize
        # inputs = {k:v.to(self.device) for k,v in self.tokenizer(inputs, return_tensors='pt', padding=True).items()}
        # input_ids = [torch.tensor(idx) for idx in input_ids]
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)
        # attention_mask = input_ids != self.pad_id
        # decoder_input_ids = self.tokenizer(outputs, truncation=True)["input_ids"]
        # decoder_input_ids = [torch.tensor(idx) for idx in decoder_input_ids]
        # decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)
        # decoder_attention_mask = decoder_input_ids != self.pad_id
        # target = self.tokenizer(outputs, return_tensors='pt', padding=True)['input_ids'].to(self.device)

        # Run forward pass with teacher forcing
        loss = self.base(
            inputs.to(self.base.device),
            labels=outputs.to(self.base.device),
        ).loss
        return loss
    
    def generate(self, inputs, skip_special_tokens=True, sampling=False, **kwargs):
        batch_size = len(inputs)

        # Tokenize
        # input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        # input_ids = [torch.tensor(idx) for idx in input_ids]
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)

        # Run BART generation: DFA based generation
        if sampling:
            new_inputs = inputs.to(self.base.device).repeat(self.num_beams, 1)
            output = self.base.generate(
                new_inputs,
                do_sample=True,
                num_beams=1,
                # Output control
                max_new_tokens=int(inputs.shape[1] * 1.5),
                num_return_sequences=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
        else:
            output = self.base.generate(
                inputs.to(self.base.device),
                num_beams=self.num_beams,
                # Output control
                max_new_tokens=int(inputs.shape[1] * 1.5),
                num_return_sequences=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
        # Convert ids to tokens
        output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=skip_special_tokens)
        
        # Reshape
        results = []
        i = 0
        for _ in range(batch_size):
            results.append([])
            for __ in range(self.num_beams):
                results[-1].append(output[i])
                i += 1
        return results

    def generate_ngram_constrained(self, inputs, logits_processor, skip_special_tokens=True):
        """
        Implements various constrained decoding that differentiates the output from input.
        Applies penalty to certain repetitions from input.
        """
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.base.device)

        # Set LogitsProcessor
        logits_processor.exclude_id=[self.pad_id, self.tokenizer.eos_token_id]
        logits_processor.update(input_ids)
        logits_processors = LogitsProcessorList([logits_processor])

        # Run BART generation
        output = self.base.generate(
            input_ids,
            num_beams=self.num_beams,
            # Output control
            max_new_tokens=int(input_ids.size(1) * 1.5),
            num_return_sequences=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            # N-gram penalize
            logits_processor=logits_processors
        )
        # Convert ids to tokens
        output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=skip_special_tokens)
        
        # Reshape
        results = []
        i = 0
        for _ in range(batch_size):
            results.append([])
            for __ in range(self.num_beams):
                results[-1].append(output[i])
                i += 1
        return results

    def branching_test(self, inputs, output_prefixes, better, worse):
        """
        Script for testing branching.
        @param inputs List[str]
        @param output_prefixes List[List[int]]
        @param better List[int]
        @param worse List[int]

        @return loss
        """
        # Transform batch to list
        inputs, output_prefixes, better, worse = list(inputs), list(output_prefixes), list(better), list(worse)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx, device=self.base.device) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        input_attention_mask = input_ids != self.pad_id

        # output_ids = self.tokenizer(output_prefixes, truncation=True)["input_ids"]
        # better_ids = self.tokenizer(better, add_special_tokens=False)["input_ids"]
        # worse_ids = self.tokenizer(worse, add_special_tokens=False)["input_ids"]
        output_ids = output_prefixes
        better_ids = better
        worse_ids = worse
        first_diff_tok_idx = list(zip(better_ids, worse_ids))
        # If first few tokens overlap, add the overlaping region to output_ids
        # for out, o, s in zip(output_ids, better_ids, worse_ids):
        #     i = 0
        #     try:
        #         while o[i] == s[i]:
        #             out.append(o[i])
        #             i+=1
        #         assert o[i] != s[i]
        #         first_diff_tok_idx.append([o[i], s[i]])
        #     except IndexError:
        #         raise ValueError(f"better & worse must be different: better={self.tokenizer.decode(o)}, worse={self.tokenizer.decode(s)}")

        output_ids = [torch.tensor(idx, device=self.base.device) for idx in output_ids]
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=self.pad_id)
        output_ids[output_ids==self.tokenizer.eos_token_id] = self.pad_id # replace EOS to PAD
        output_attention_mask = output_ids != self.pad_id
        boundaries = torch.sum(output_attention_mask, dim=1)-1

        first_diff_tok_idx = torch.tensor(first_diff_tok_idx, dtype=torch.long, device=self.device)

        logits = self.base(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            decoder_input_ids=output_ids,
            decoder_attention_mask = output_attention_mask
        ).logits # batch_size, seq_len, vocab_size
        logits_gather_index = torch.tile(boundaries.unsqueeze(1).unsqueeze(2), (1, 1, logits.size(2)))
        logits = torch.gather(logits, 1, logits_gather_index).squeeze(1) # batch_size, vocab_size

        # Calculate logprob_diff
        # logprob_diff = logp(better) - logp(worse)
        logprob = F.log_softmax(logits, dim=1)
        compare_logprob = torch.gather(logprob, 1, first_diff_tok_idx) # batch_size, 2
        logprob_diff = compare_logprob[:, 0] - compare_logprob[:, 1]
        better_prob = torch.gather(logprob, 1, first_diff_tok_idx[:, 0].unsqueeze(1)).squeeze(1) # batch_size, 1
        worse_prob = torch.gather(logprob, 1, first_diff_tok_idx[:, 1].unsqueeze(1)).squeeze(1) # batch_size, 1

        # Calculate worse_rank
        ranks = torch.argsort(logprob, dim=1, descending=True)
        better_rank = torch.gather(ranks, 1, first_diff_tok_idx[:, 0].unsqueeze(1)).squeeze(1) # batch_size, 1
        worse_rank = torch.gather(ranks, 1, first_diff_tok_idx[:, 1].unsqueeze(1)).squeeze(1) # batch_size, 1
        rank_diff = better_rank - worse_rank
        
        return {
            "better_prob": better_prob.tolist(),
            "worse_prob": worse_prob.tolist(),
            "logprob_diff": logprob_diff.tolist(),
            "better_rank": better_rank.tolist(),
            "worse_rank": worse_rank.tolist(),
            "rank_diff": rank_diff.tolist()
        }