import importlib.util
import inspect
import itertools
from collections import OrderedDict
from collections import OrderedDict
from itertools import tee
from typing import List
import torch
from torch.nn import Linear


def load_dict(dict_path, load_reverse=False):
    lines = open(dict_path, 'r', encoding='utf-8').readlines()
    src2tgt = {}
    for i, line in enumerate(lines):
        line = line.strip()
        try:
            src, tgt = line.split("\t")
        except:
            try:
                src, tgt = line.split(" ")
            except:
                src = line.strip()
                tgt = ''

        if load_reverse:
            src, tgt = tgt, src

        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)
    return src2tgt


def is_torch_tpu_available():
    # copied from https://github.com/huggingface/transformers/blob/024cd19bb7c188a0e4aa681d248ad9f47587ddab/src/transformers/file_utils.py#L280
    # if not _torch_available:
    #     return False
    # This test is probably enough, but just in case, we unpack a bit.
    if importlib.util.find_spec("torch_xla") is None:
        return False
    if importlib.util.find_spec("torch_xla.core") is None:
        return False
    return importlib.util.find_spec("torch_xla.core.xla_model") is not None


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


def wait_for_everyone(msg="accelerate.utils.wait_for_everyone"):
    from accelerate.state import AcceleratorState, DistributedType
    # copied from https://github.com/huggingface/accelerate/blob/f1333b54ad0141d162de6e6f04893ff7fd1f7e36/src/accelerate/utils.py#L252
    # Modified because series of xm.rendezvous with same msg seemed to be synced in different lines...
    if AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        torch.distributed.barrier()
    elif AcceleratorState().distributed_type == DistributedType.TPU:
        xm.rendezvous(msg)


def cat_lists(lists):
    return list(itertools.chain.from_iterable(lists))


def str_to_list(space_splitted_str, type=float):
    space_splitted_str = str(space_splitted_str)
    return [type(o) for o in space_splitted_str.split()]


def l_of_dic_to_dic_of_l(l_of_dic: List[dict]):
    result = OrderedDict([(k, []) for k in l_of_dic[0].keys()])
    for dic in l_of_dic:
        for k, v in dic.items():
            assert k in result
            result[k].append(v)
    return result


def test_l_of_dic_to_dic_of_l():
    l_of_dict = [{1: "min"},
                 {1: "max"}]
    dict_of_l = l_of_dic_to_dic_of_l(l_of_dict)
    assert dict_of_l == {1: ["min", "max"]}


class OrderedSet:
    def __init__(self, iterable):
        self._ordered_dict = OrderedDict()
        for i in iterable:
            self._ordered_dict[i] = ''

    def remove(self, iterable):
        for i in iterable:
            assert i in self._ordered_dict
            self._ordered_dict.pop(i)
        return self

    def to_list(self):
        return list(self._ordered_dict.keys())


def get_inverted_dense_layer(l: Linear):
    assert isinstance(l, Linear)
    with torch.no_grad():
        w = l.weight.detach()
        b = l.bias.detach().unsqueeze(-1)

        w_inv = torch.linalg.inv(w)  # w * w_inv = I
        b_inv = - torch.mm(w_inv, b)  # w_inv * (w * x + b) + b_inv = x
        b_inv = b_inv.squeeze()

        inv_l = Linear(l.out_features, l.in_features)
        inv_l.weight.data.copy_(w_inv)
        inv_l.bias.data.copy_(b_inv)

    return inv_l


def test_get_inverted_dense_layer():
    in_out = 5
    l = Linear(in_out, in_out)
    inv_l = get_inverted_dense_layer(l)

    for _ in range(1000):
        features = torch.randn((in_out, in_out))
        inverted_features = inv_l(l(features))
        print(features)
        print(inverted_features)
        assert torch.allclose(features, inverted_features, rtol=5e-2)


if __name__ == '__main__':
    test_get_inverted_dense_layer()
    # test_l_of_dic_to_dic_of_l()


def prev_curr(iterable):
    # from https://docs.python.org/3/library/itertools.html#itertools-recipes
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_kwargs():
    """
    Gets kwargs of given called function.
    You can use this instead "local()"
    got idea from https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
    """
    frame = inspect.stack()[1][0]
    varnames, _, _, values = inspect.getargvalues(frame)

    called_from_class_method = (varnames[0] == 'self')
    if called_from_class_method:
        varnames = varnames[1:]

    kwargs = {i: values[i] for i in varnames}
    return kwargs


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, total_sum, n=1):
        self.sum += total_sum
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MaxMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.max = float("-inf")
        self.count = 0

    def update(self, total_sum, n=1):
        self.max = max(self.max, total_sum)
        self.count += n

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
