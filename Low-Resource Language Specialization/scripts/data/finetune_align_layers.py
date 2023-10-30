import os
from transformers import BertTokenizer, BertModel, BertForPreTraining, HfArgumentParser, SchedulerType
from transformers import set_seed
from transformers.optimization import get_scheduler, AdamW
from dataclasses import dataclass, field
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from util.other_utils import wait_for_everyone
from scripts.data.common import OrigTLDataset, Collator, CLDataset

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    orig_corpus: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_cleaned/train.txt")
    tl_dict: str = field(default="translit_dict/ug_to_uglatinnfc_tok.txt")
    model_tok_cfg: str = field(default="bert-base-multilingual-cased")
    tok_cfg: str = field(default=None)
    reg_model_tok_cfg: str = field(default=None)
    align_tgt_model_tok_cfg: str = field(default=None)
    align_tok_cfg: str = field(default=None)
    update_src_emb_also: bool = field(default=False)
    output_dir: str = field(default="")
    start_i: int = field(default=0)
    end_i: int = field(default=-1)

    lang: str = field(default='ug', metadata={"help": "Used for finding the cache dir"})
    regularlize_on_orig_embeddings: str = field(default=False)
    reg_w: float = field(default=1.0)
    ignore_unk_words: bool = field(default=True)

    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0)
    no_weight_decay_on_bias: bool = field(default=True)
    cache_dir: str = field(default=None)
    num_workers: int = field(default=4)
    per_device_train_batch_size: int = field(default=8)
    dataloader_pin_memory: bool = field(default=True)
    start_layer: int = field(default=9)
    end_layer: int = field(default=12)

    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use.",
            "choices": ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
        }
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of steps for the warmup in the lr scheduler.",
        }
    )
    num_warmup_epoch: float = field(
        default=0.1,
        metadata={
            "help": "Number of steps for the warmup in the lr scheduler.",
        }
    )
    epoch: int = field(default=1)
    align_src_to_tgt: bool = field(default=True, metadata={"help": "Align orig corpus to transliterated/parallel corpus"})
    use_basic_tok:bool = field(default=True)
    reset_head: bool = field(default=True, metadata={"help": "for backward compatibility..."})
    seed: int = field(default=42)

    cross_ling: bool = field(default=False)
    tgt_corpus: str = field(default="")
    src_lang: str = field(default="")
    src_corpus: str = field(default="")
    word_alignment: str = field(default="")

    reduce_word_strategy: str = field(default='avg', metadata={"choices": ['avg', 'last']})



from collections import OrderedDict
def prepare_optimizer(model, train_args: Args):
    # Optimizer
    optimizer_grouped_parameters = [
        {"named_params": OrderedDict(model.named_parameters()),
         "lr": train_args.learning_rate,
         "weight_decay": train_args.weight_decay,
         }
    ]

    # Split weights in two groups, one with weight decay and the other not.
    if train_args.no_weight_decay_on_bias:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "named_params": OrderedDict([(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]),
                "lr": train_args.learning_rate,
                "weight_decay": train_args.weight_decay,
            },
            {
                "named_params": OrderedDict([(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]),
                "lr": train_args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

    for param_group in optimizer_grouped_parameters:
        param_group['params'] = list(param_group.pop('named_params').values())

    optimizer = AdamW(optimizer_grouped_parameters)

    return optimizer

import math
def get_num_update_steps_per_epoch(train_dataloader, gradient_accum_steps=1):
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accum_steps)
    return num_update_steps_per_epoch

def save_model(model, accelerator: Accelerator, save_dir):
    wait_for_everyone("save_model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir)
    # unwrapped_model.save_pretrained(save_dir, save_config=accelerator.is_local_main_process, save_function=accelerator.save)

if __name__ == '__main__':
    parser = HfArgumentParser(Args,)
    args, = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    if args.tok_cfg is None:
        args.tok_cfg = args.model_tok_cfg
    tok = BertTokenizer.from_pretrained(args.tok_cfg)
    bert_cls = BertModel if args.reset_head else BertForPreTraining
    print(f"USING {bert_cls.__name__}")
    train_model = bert_cls.from_pretrained(args.model_tok_cfg)

    set_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


    if args.update_src_emb_also:
        assert args.align_tgt_model_tok_cfg == args.model_tok_cfg or args.align_tgt_model_tok_cfg is None
        align_tgt_model = train_model
        align_tgt_tok = tok
    else:
        if args.align_tgt_model_tok_cfg is None:
            args.align_tgt_model_tok_cfg = args.model_tok_cfg
        if args.align_tok_cfg is None:
            args.align_tok_cfg = args.align_tgt_model_tok_cfg
        align_tgt_model = bert_cls.from_pretrained(args.align_tok_cfg)
        align_tgt_tok = BertTokenizer.from_pretrained(args.align_tok_cfg)

    if args.reg_model_tok_cfg is None:
        args.reg_model_tok_cfg = args.model_tok_cfg
    regularze_from_model = bert_cls.from_pretrained(args.reg_model_tok_cfg)

    optimizer = prepare_optimizer(train_model, args)

    accelerator = Accelerator()
    if not args.cross_ling:
        train_dataset = OrigTLDataset(args.orig_corpus, args.tl_dict, tok, align_tgt_tok, args, args.cache_dir,
                                      regularlize_on_orig_embeddings=args.regularlize_on_orig_embeddings)
    else:
        train_dataset = CLDataset(args.src_lang, args.lang, args.src_corpus, args.orig_corpus, args.word_alignment,
                                  tok, align_tgt_tok, args, args.cache_dir,
                                  regularlize_on_orig_embeddings=args.regularlize_on_orig_embeddings)


    train_data_collator = Collator(tok, align_tgt_tok, max_length=tok.model_max_length, pad_to_multiple_of=8 if accelerator.use_fp16 else None)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=args.per_device_train_batch_size,
                                  num_workers=args.num_workers, pin_memory=args.dataloader_pin_memory)

    train_model, align_tgt_model, regularze_from_model, optimizer, train_dataloader =\
        accelerator.prepare(train_model, align_tgt_model, regularze_from_model, optimizer, train_dataloader)

    num_update_steps_per_epoch = get_num_update_steps_per_epoch(train_dataloader)
    if args.num_warmup_epoch:
        assert not args.num_warmup_steps
        args.num_warmup_steps = int(args.num_warmup_epoch * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.epoch * num_update_steps_per_epoch,
    )


    def get_contextualized_emb(model, batch, layers=None):
        if (layers is None) or (layers == [12]):
            out = model(**batch).last_hidden_state
            return [out[:, 1:-1]] # remove special embs
        else:
            batch['output_hidden_states'] = True
            out = model(**batch).hidden_states
            outs = []
            for l in layers:
                outs.append(out[l][:, 1:-1])
            return outs

    def train_loss_fn(emb1, emb2):
        return torch.norm(emb1 - emb2)

    def reg_loss_fn(emb1, emb2):
        return torch.norm(emb1 - emb2)

    def reduce_word(emb1, method):
        if method == 'avg':
            return torch.mean(emb1, dim=0)
        elif method == 'last':
            return emb1[-1]

    regularze_from_model.eval()
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    global_step = 0
    layers = list(range(args.start_layer, args.end_layer + 1))
    for ep in range(args.epoch):
        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)

        for i, (orig_batch, tl_batch, orig_indicies_per_batch, tl_indicies_per_batch,
                ) in enumerate(train_dataloader):
            train_orig_out = get_contextualized_emb(train_model, orig_batch, layers)
            train_tl_out = get_contextualized_emb(align_tgt_model, tl_batch, layers)

            if args.regularlize_on_orig_embeddings:
                with torch.no_grad():
                    if args.update_src_emb_also:
                        fixed_orig_out = get_contextualized_emb(regularze_from_model, orig_batch, layers)
                        fixed_tl_out = get_contextualized_emb(regularze_from_model, tl_batch, layers)
                    else:
                        if args.align_src_to_tgt:
                            fixed_out = get_contextualized_emb(regularze_from_model, orig_batch, layers)
                        else:
                            fixed_out = get_contextualized_emb(regularze_from_model, tl_batch, layers)

            losses = []
            reg_losses = []
            for sent_i_in_batch, (orig_idxs_per_sent, tl_idxs_per_sent) in \
                    enumerate(zip(orig_indicies_per_batch, tl_indicies_per_batch)):
                for l in range(len(layers)):
                    if args.regularlize_on_orig_embeddings:
                        reg_losses.append(reg_loss_fn(train_orig_out[l], fixed_orig_out[l].detach()))
                        reg_losses.append(reg_loss_fn(train_tl_out[l], fixed_tl_out[l].detach()))

                    for word_i, (orig_idx_per_word, tl_idx_per_word) in enumerate(zip(orig_idxs_per_sent, tl_idxs_per_sent)):
                        train_orig_embs = reduce_word(train_orig_out[l][sent_i_in_batch][orig_idx_per_word], args.reduce_word_strategy)
                        train_tl_embs = reduce_word(train_tl_out[l][sent_i_in_batch][tl_idx_per_word], args.reduce_word_strategy)
                        if not args.update_src_emb_also:
                            if args.align_src_to_tgt: # we need to fix all other than train targets, i.e., align targets
                                train_tl_embs = train_tl_embs.detach()
                            else:
                                train_orig_embs = train_orig_embs.detach()

                        losses.append(train_loss_fn(train_orig_embs, train_tl_embs))


            loss = 0
            if losses:
                loss = torch.mean(torch.stack(losses))
            reg_loss = 0
            if reg_losses:
                reg_loss = torch.mean(torch.stack(reg_losses))
            tb_writer.add_scalar("loss", float(loss), global_step=global_step)
            tb_writer.add_scalar("reg_loss", float(reg_loss), global_step=global_step)
            global_step += 1

            if losses:
                accelerator.backward(loss + args.reg_w * reg_loss)
                optimizer.step()
                lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    os.makedirs(args.output_dir, exist_ok=True)
    save_model(train_model, accelerator, args.output_dir)
