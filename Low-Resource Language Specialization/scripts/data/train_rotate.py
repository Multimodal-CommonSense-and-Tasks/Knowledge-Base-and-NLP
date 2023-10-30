import glob
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, HfArgumentParser, AdamW, DataCollatorWithPadding
from pytorch_transformers import WarmupCosineSchedule
from util.other_utils import AverageMeter
from dataclasses import dataclass, field
import torch
import scipy

cos_sim = torch.nn.CosineSimilarity(dim=-1)
def calc_loss(src_emb_mean, tgt_emb_mean, method):
    if method == 'cossim':
        assert src_emb_mean.shape[1] == 1
        sim = cos_sim(src_emb_mean, tgt_emb_mean)
        sim = torch.sum(sim)
        loss = -sim
    elif method == 'norm':
        loss = torch.norm(src_emb_mean - tgt_emb_mean)
    elif 'cka' in method:
        src_emb_mean = torch.squeeze(src_emb_mean)
        tgt_emb_mean = torch.squeeze(tgt_emb_mean)
        debiased = 'debiased' in method
        loss = - cka_torch(src_emb_mean, tgt_emb_mean, debiased=debiased)
    else:

        raise NotImplementedError
    return loss

def get_procrustes(orig: np.ndarray, tgt: np.ndarray):
    """
    get mapping from orig to tgt, i.e. get min|R*orig - tgt|
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    A = orig
    B = tgt
    M = np.matmul(B, A.T)
    U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
    R = U.dot(V_t)
    return R


class DatsetMeanEmb(Dataset):
    def __init__(self, src_mean_embs, tgt_mean_embs):
        assert len(src_mean_embs) == len(tgt_mean_embs)
        self.tgt_mean_embs = tgt_mean_embs
        self.src_mean_embs = src_mean_embs

    def __getitem__(self, index):
        return self.src_mean_embs[index], self.tgt_mean_embs[index]

    def __len__(self):
        return len(self.src_mean_embs)


def get_dat(data_src_emb, data_tgt_emb, batch_size, num_workers=8, load_reverse=False):
    if load_reverse:
        data_tgt_emb, data_src_emb = data_src_emb, data_tgt_emb
    my_dataset = DatsetMeanEmb(data_src_emb, data_tgt_emb)
    dataloader = DataLoader(my_dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=True)
    return dataloader


@dataclass
class Args:
    pkl_dir: str = field(default="cross_script_align/rotate/ug")
    train_method: str = field(default='gd', metadata={"choices": ['svd', 'gd']})
    centroid_normalize: bool = field(default=False)
    save_path: str = field(default='cross_script_align/rotate_mat/ug_gd.pkl')
    batch_size: int = field(default=256)
    epochs: int = field(default=100,
                        metadata={"help": "", }
                        )
    warmup: int = field(default=0,
                        metadata={"help": "", }
                        )
    num_workers: int = field(default=8,
                             metadata={"help": "", }
                             )
    lr: float = field(default=0.001,
                      metadata={"help": "", }
                      )
    wd: float = field(default=1e-02,
                      metadata={"help": "", }
                      )
    lr_sched: str = field(default="cosine",
                          metadata={"help": "",
                                    "choices": ['exp', 'cosine']
                                    }
                          )
    loss_ftn: str = field(default="norm",
                          metadata={"help": "",
                                    "choices": ['cossim', 'norm', 'cka_biased', 'cka_debiased']
                                    }
                          )
    use_bias: bool = field(
        default=False,
    )
    init_pattern: str = field(default="one",
                              metadata={"help": "CLBT used identity matrix as default.. check build_model.py#103",
                                        "choices": ['zero_random', 'one_random', 'one']
                                        }
                              )

if __name__ == '__main__':
    parser = HfArgumentParser(Args,)
    args, = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    prefixes = ['orig', 'tl']
    all_datas = {}
    centroids = {}
    for prefix in prefixes:
        all_files = glob.glob(os.path.join(args.pkl_dir, f"{prefix}*.pkl"))
        all_nps = []
        for f in all_files:
            all_nps.extend(pickle.load(open(f, 'rb')))
        all_datas[prefix] = np.stack(all_nps)
        centroids[prefix] = np.mean(all_datas[prefix], axis=0)
        if args.centroid_normalize:
            all_datas[prefix] -= centroids[prefix]

    if args.train_method == 'svd':
        map_mat = get_procrustes(all_datas['orig'], all_datas['tl'])
        pickle.dump(map_mat, open(args.save_path, 'wb'))

    elif args.train_method == 'gd':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        emb_size = all_datas['orig'].shape[1]
        dataloader = get_dat(all_datas['orig'], all_datas['tl'], batch_size=args.batch_size)
        trans_layer = torch.nn.Linear(emb_size, emb_size, bias=args.use_bias)

        if args.init_pattern == 'one_random':
            with torch.no_grad():
                w = trans_layer.weight.detach()  # must be some random
                w += torch.eye(emb_size)  # Identity matrix init
            trans_layer.weight.data.copy_(w)
        elif args.init_pattern == 'one':
            with torch.no_grad():
                w = torch.eye(emb_size)  # Identity matrix init
            trans_layer.weight.data.copy_(w)

        trans_layer = trans_layer.to(device)
        optimizer = AdamW(trans_layer.parameters(), lr=args.lr, weight_decay=args.wd)

        if args.lr_sched == 'exp':
            lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            if args.warmup:
                raise NotImplementedError
            sched_every_step = False

        elif args.lr_sched == 'cosine':
            iter_per_one_ep = len(dataloader)
            total_iter = iter_per_one_ep * args.epochs
            print(total_iter)
            warmup_steps = iter_per_one_ep * args.warmup
            lr_sched = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_iter)
            # lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter)
            sched_every_step = True

        loss_meter = AverageMeter('loss_meter')
        min_loss = float("inf")
        for e in range(args.epochs):
            loss_meter.reset()
            for i, batch in enumerate(dataloader):
                src_emb_mean, tgt_emb_mean = [x.to(device) for x in batch]
                src_emb = trans_layer(src_emb_mean)
                loss = calc_loss(src_emb, tgt_emb_mean, args.loss_ftn)

                loss_meter.update(loss, src_emb.shape[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if sched_every_step:
                    lr_sched.step()

                # if i % 50 == 0:
                #     print(f"current_lr : {optimizer.param_groups[0]['lr']}")

            print(f"cosine sim of epoch {e} : {loss_meter.avg}")
            if not sched_every_step:
                lr_sched.step()
            if loss_meter.avg < min_loss:
                min_loss = loss_meter.avg
                torch.save(trans_layer, args.save_path)
                with open(args.save_path + '_min_loss', 'w') as f:
                    f.write(f"{min_loss}")
                # trans_layer.cpu()
                # pickle.dump(trans_layer.weight.data.numpy(), open(args.save_path+"_np", 'wb'))

    else:
        raise NotImplementedError