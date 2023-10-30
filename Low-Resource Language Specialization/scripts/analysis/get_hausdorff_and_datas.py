import os
from transformers import BertTokenizer, BertModel, BertForPreTraining, HfArgumentParser, SchedulerType
from transformers import set_seed
from transformers.optimization import AdamW
from dataclasses import dataclass, field
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from scripts.data.common import OrigTLDataset, Collator, CLDataset

from torch.utils.data.dataloader import DataLoader



@dataclass
class Args:
    orig_corpus: str = field(default="specializing-multilingual-data/data/ug/unlabeled/bert_cleaned/train.txt")
    tl_dict: str = field(default="translit_dict/ug_to_uglatinnfc_tok.txt")
    model_tok_cfg: str = field(default="")
    reg_model_tok_cfg: str = field(default=None)
    align_tgt_model_tok_cfg: str = field(default=None)
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
    num_workers: int = field(default=0)
    per_device_train_batch_size: int = field(default=64)
    dataloader_pin_memory: bool = field(default=True)
    start_layer: int = field(default=12)
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
    align_src_to_tgt: bool = field(default=True)
    use_basic_tok: bool = field(default=True)
    reset_head: bool = field(default=True, metadata={"help": "for backward compatibility..."})
    seed: int = field(default=42)

    cross_ling: bool = field(default=False)
    tgt_corpus: str = field(default="")
    src_lang: str = field(default="")
    src_corpus: str = field(default="")
    word_alignment: str = field(default="")


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


## check hausdorff
def get_hausdorff(src_embeds, tgt_embeds, max_dots=-1):
    from hausdorff import hausdorff_distance
    src_embeds = src_embeds[:max_dots]
    tgt_embeds = tgt_embeds[:max_dots]
    src_embeds = torch.cat(src_embeds).detach().cpu().numpy()
    tgt_embeds = torch.cat(tgt_embeds).detach().cpu().numpy()
    hd = hausdorff_distance(src_embeds, tgt_embeds, distance='cosine')
    return hd



if __name__ == '__main__':
    parser = HfArgumentParser(Args, )
    args, = parser.parse_args_into_dataclasses()

    import pickle
    orig_pkl_file = f"{args.output_dir}/orig_embs"
    tl_pkl_file = f"{args.output_dir}/tl_embs"
    layers = list(range(args.start_layer, args.end_layer + 1))
    if os.path.exists(orig_pkl_file):
        print("LOADING!!!")
        orig_embs = pickle.load(open(orig_pkl_file, 'rb'))
        tl_embs = pickle.load(open(tl_pkl_file, 'rb'))
    else:
        assert isinstance(args, Args)
        tok = BertTokenizer.from_pretrained(base_config_path)
        bert_cls = BertModel if args.reset_head else BertForPreTraining
        print(f"USING {bert_cls.__name__}")
        train_model = bert_cls.from_pretrained(args.model_tok_cfg)

        set_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

        align_tgt_model = train_model
        align_tgt_tok = tok

        optimizer = prepare_optimizer(train_model, args)

        accelerator = Accelerator()
        if not args.cross_ling:
            train_dataset = OrigTLDataset(args.orig_corpus, args.tl_dict, tok, align_tgt_tok, args, args.cache_dir,
                                          regularlize_on_orig_embeddings=args.regularlize_on_orig_embeddings)
        else:
            train_dataset = CLDataset(args.src_lang, args.lang, args.src_corpus, args.orig_corpus, args.word_alignment,
                                      tok, align_tgt_tok, args, args.cache_dir,
                                      regularlize_on_orig_embeddings=args.regularlize_on_orig_embeddings
                                      )

        train_data_collator = Collator(tok, align_tgt_tok, max_length=tok.model_max_length, pad_to_multiple_of=8 if accelerator.use_fp16 else None)
        train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=train_data_collator, batch_size=args.per_device_train_batch_size,
                                      num_workers=args.num_workers, pin_memory=args.dataloader_pin_memory)

        train_model, train_dataloader = \
            accelerator.prepare(train_model, train_dataloader)


        def get_contextualized_emb(model, batch, layers=None):
            if (layers is None) or (layers == [12]):
                out = model(**batch).last_hidden_state
                return [out]  # remove special embs
            else:
                batch['output_hidden_states'] = True
                out = model(**batch).hidden_states
                outs = []
                for l in layers:
                    outs.append(out[l])
                return outs


        train_model.eval()
        global_step = 0

        from collections import defaultdict

        orig_embs = defaultdict(list)
        tl_embs = defaultdict(list)


        def get_num_update_steps_per_epoch(train_dataloader, gradient_accum_steps=1):
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accum_steps)
            return num_update_steps_per_epoch

        num_update_steps_per_epoch = get_num_update_steps_per_epoch(train_dataloader)
        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)

        with torch.no_grad():
            for i, (orig_batch, tl_batch, orig_indicies_per_batch, tl_indicies_per_batch, orig_has_init_tids_per_batch, tl_has_init_tids_per_batch
                    ) in enumerate(train_dataloader):
                train_orig_out = get_contextualized_emb(train_model, orig_batch, layers)
                train_tl_out = get_contextualized_emb(train_model, tl_batch, layers)
                # check whether the shape is

                for l_i, l in enumerate(layers):
                    orig_embs[f'sent_avg_{l}'].append(torch.mean(train_orig_out[l_i], dim=1).detach().cpu())
                    tl_embs[f'sent_avg_{l}'].append(torch.mean(train_tl_out[l_i], dim=1).detach().cpu())

                    orig_embs[f'sent_cls_{l}'].append(train_orig_out[l_i][:, 0].detach().cpu())
                    tl_embs[f'sent_cls_{l}'].append(train_tl_out[l_i][:, 0].detach().cpu())

                for sent_i_in_batch, (orig_idxs_per_sent, tl_idxs_per_sent) in \
                        enumerate(zip(orig_indicies_per_batch, tl_indicies_per_batch)):
                    for l_i, l in enumerate(layers):
                        for word_i, (orig_idx_per_word, tl_idx_per_word) in enumerate(zip(orig_idxs_per_sent, tl_idxs_per_sent)):
                            train_orig_embs = torch.mean(train_orig_out[l_i][sent_i_in_batch][orig_idx_per_word], dim=0, keepdim=True).detach().cpu()
                            train_tl_embs = torch.mean(train_tl_out[l_i][sent_i_in_batch][tl_idx_per_word], dim=0, keepdim=True).detach().cpu()

                            orig_embs[f'word_{l}'].append(train_orig_embs)
                            tl_embs[f'word_{l}'].append(train_tl_embs)
                progress_bar.update(1)

        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(orig_embs, open(orig_pkl_file, "wb"))
        pickle.dump(tl_embs, open(tl_pkl_file, "wb"))

    for l in reversed(layers):
        key = f'{l}'
        orig_emb = orig_embs[key]
        tl_emb = tl_embs[key]
        try:
            for max_dots in [5000]:
                if not os.path.exists(os.path.join(args.output_dir, f'{key}_hausdorff_{max_dots}')):
                    print(f"doing {key}_hausdorff_{max_dots}")
                    hd = get_hausdorff(orig_emb, tl_emb, max_dots)
                    with open(os.path.join(args.output_dir, f'{key}_hausdorff_{max_dots}'), 'w') as f:
                        f.write(f"{hd}")
        except:
            pass
        break
