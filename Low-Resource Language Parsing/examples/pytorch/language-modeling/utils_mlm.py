import sys
from dataclasses import dataclass, field
from torch.utils.data.dataset import Dataset, IterableDataset
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
from typing import List, Dict, Tuple
from transformers.tokenization_utils_base import BatchEncoding
from torch.nn.functional import cosine_similarity
from filelock import FileLock
import os, itertools
import numpy as np
import logging
logger = logging.getLogger(__name__)
import time, pickle, random, collections
from torch.utils.tensorboard import SummaryWriter  # noqa: F401
from transformers.trainer import Trainer
def torch_default_data_collator_cat(features, pop_language=False):
    import torch

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if pop_language and k == "language":
            continue
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch
@dataclass
class DataCollatorForLanguageModelingOld:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class WeightedSampleDataset(IterableDataset):

    def __init__(self,
                 batch_size,
                 datasets: dict,
                 weighted_sampling: bool,
                 smoothing: float,
                 training: bool,
                 n_gpus: int,
                 n_steps: int,
                 do_shuffle: bool,
                 n_epochs: int = None,
                 sample_weights: list = None,
                 ):
        self.do_shuffle = do_shuffle
        self.sample_weights = sample_weights
        self.training = training
        self.weighted_sampling = weighted_sampling
        self.smoothing = smoothing
        self.datasets = datasets
        self.length = sum(
            len(dataset) // batch_size for dataset in self.datasets.values())
        self.n_steps = n_steps
        self.n_gpus = n_gpus
        if n_epochs is not None:
            assert n_steps == -1
            self.n_steps = int(n_epochs * self.length)

    def __len__(self):
        if self.training:
            return self.n_steps
        else:
            return self.length

    def __iter__(self):
        return _WeightedSampleIterator(self)


class _WeightedSampleIterator:

    def __init__(self, ws_dataset: WeightedSampleDataset):
        datasets = sorted(list(ws_dataset.datasets.items()))
        self.monolingual_generators = \
            itertools.cycle([
                dataset.gen(repeat=ws_dataset.training,
                                  shuffle=ws_dataset.do_shuffle
                                  )
                for _, dataset in datasets
            ])

        if ws_dataset.sample_weights:
            self.sample_prob = np.array(ws_dataset.sample_weights) / np.sum(ws_dataset.sample_weights)
        elif ws_dataset.weighted_sampling:
            lengths = np.array([len(dataset) for _, dataset in datasets], dtype=np.float32)
            smoothed_lengths = lengths ** ws_dataset.smoothing
            self.sample_prob = smoothed_lengths / np.sum(smoothed_lengths)
        else:
            self.sample_prob = np.ones([len(datasets)], dtype=np.float32) / len(datasets)
        logging.info('Languages will be sampled in the following proportions:')
        for i, (language, _) in enumerate(datasets):
            logging.info('%s: %.7f' % (language, self.sample_prob[i]))

        self.training = ws_dataset.training
        self.n_gpus = ws_dataset.n_gpus
        self.n_steps = ws_dataset.n_steps
        self.step_count = 0
        self.gpu_count = 0
        self.current = 0

    def __next__(self):
        if self.training:
            # if self.step_count >= self.n_steps:
            #     raise StopIteration()

            if self.gpu_count == 0:
                self.dataset = next(self.monolingual_generators)
            batch = next(self.dataset)
            self.gpu_count += 1
            # self.step_count += 1
            if self.gpu_count == self.n_gpus:
                self.gpu_count = 0
            return batch

        else:
            raise NotImplementedError
            # while True:
            #     if self.current >= len(self.monolingual_generators):
            #         raise StopIteration()
            #
            #     try:
            #         batch = next(self.monolingual_generators[self.current])
            #     except StopIteration:
            #         self.current += 1
            #         continue
            #
            #     return batch


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename, ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                lines = []
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        tokens = line.strip().split()
                        if len(tokens) < 10:
                            continue
                        lines.append(line.strip())
                text = '\n'.join(lines)

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class MonolingualDataset(TextDataset):

    def __init__(
            self, tokenizer, file_path, block_size, language, batch_size, overwrite_cache=False,
            mlm_probability=0.15, padding='max_length',
            seed=42
    ):
        super().__init__(tokenizer, file_path, block_size, overwrite_cache=overwrite_cache)
        self.language = language
        self.batch_size = batch_size
        logging.info('Initialised %s dataset with %d examples' % (language, len(self)))
        self.data_collator = DataCollatorForLanguageModelingOld(tokenizer=tokenizer,
                                                                mlm_probability=mlm_probability,
                                                                # padding=padding,
                                                                # max_length=block_size,
                                                                # pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
                                                                )
        self.seed = seed

    def gen(self, repeat=False, shuffle=False):
        n_examples = len(self)
        while True:
            indices = list(range(n_examples))
            if shuffle:
                random.seed(self.seed)
                # print(f"set seed with {self.seed}")
                random.shuffle(indices)
            for batch_begin in range(0, n_examples - self.batch_size, self.batch_size):
                batch_end = batch_begin + self.batch_size
                batch = [torch.tensor(self.examples[indices[i]], dtype=torch.long)
                         for i in range(batch_begin, batch_end)]
                batch = self.data_collator(batch)
                batch['language'] = self.language
                yield batch
            self.seed += 1
            if not repeat:
                break


class MultilingualDataset(WeightedSampleDataset):

    def __init__(
            self, tokenizer: PreTrainedTokenizer, files_by_language, block_size, batch_size,
            overwrite_cache=False, training=False, weighted_sampling=True, smoothing=0.7, n_gpus=1, n_steps=250000,
            mlm_probability=0.15, padding='max_length',
            do_shuffle=False, seed=42, seed_sync=True,

    ):
        # languages = ', '.join(sorted(list(files_by_language.keys())))
        # logging.info('Initialising multilingual dataset with languages ' + languages)
        datasets = {
            language: MonolingualDataset(tokenizer, file_path, block_size, language,
                                         batch_size, overwrite_cache=overwrite_cache,
                                         mlm_probability=mlm_probability, padding=padding,
                                         seed=seed + (0 if seed_sync else random.randint(0, sys.maxsize)))
            for language, file_path in files_by_language.items()
        }
        super().__init__(batch_size, datasets, weighted_sampling, smoothing, training, n_gpus, n_steps,
                         do_shuffle)


class DualScriptTrainer(Trainer):
    # args: CustomTrainingArgs
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            super()._set_signature_columns_if_needed()
            self._signature_columns += ['language']

    def __init__(self, *args, accumulate_grads_how_much, **kwargs):
        super().__init__(*args, **kwargs)

        self.accumulate_grads_how_much = accumulate_grads_how_much
        self.accumulated_grads = 0
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        self.save_csv = open(os.path.join(self.args.logging_dir, 'cos_sim.csv'), 'w')
        # for p in self.model.parameters():
        #     if p.requires_grad:
        #         self.trainable_params.append(p)
        # for layer in self.model.encoder.layer: #     parameters = []
        #     for p in layer.parameters():
        #         parameters.append(p)
        #     self.trainable_params.append(parameters)

        self.loss_per_languages = collections.defaultdict(list)
        # self.zero_grad_emb_indicies = zero_grad_emb_indicies
        # if self.zero_grad_emb_indicies is not None:
        #     self.zero_grad_emb_indicies = torch.LongTensor(zero_grad_emb_indicies).to(next(self.model.parameters()).device)

    def flatten_tensors(self, tensors):
        return torch.cat([t.view(-1) for t in tensors]).cpu()

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     raise NotImplementedError
        #     # scaler = self.scaler if self.use_amp else None
        #     # loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
        #     # return loss_mb.reduce_mean().detach().to(self.args.device)
        #
        # if self.use_amp:
        #     raise NotImplementedError
        #     # with autocast():
        #     #     loss = self.compute_loss(model, inputs)
        # else:
        language = torch.sum(inputs.pop("language")).item()
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        # grads = []
        grads = torch.autograd.grad(loss, self.trainable_params, retain_graph=True)
        self.loss_per_languages[language].append(self.flatten_tensors(grads))
        self.accumulated_grads += 1
        if self.accumulated_grads == self.accumulate_grads_how_much:
            vals = list(self.loss_per_languages.values())
            first = self.flatten_tensors(vals[0])
            second = self.flatten_tensors(vals[1])
            cos_sim = self.cos(first, second).item()
            self.tb_writer.add_scalar("cosine_similarity", cos_sim, self.state.global_step)
            self.save_csv.write("{}\n".format(cos_sim))
            print(cos_sim)
            want_remove_keys = list(self.loss_per_languages.keys())
            for k in want_remove_keys:
                del self.loss_per_languages[k]
            del self.loss_per_languages
            self.loss_per_languages = collections.defaultdict(list)
            self.accumulated_grads = 0

        # if self.use_amp:
        #     raise NotImplementedError
        #     # self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     raise NotImplementedError
        #     # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     #     scaled_loss.backward()
        # elif self.deepspeed:
        #     raise NotImplementedError
        #     # loss gets scaled under gradient_accumulation_steps in deepspeed
        #     # loss = self.deepspeed.backward(loss)
        # else:
        loss.backward()


        return loss.detach()
