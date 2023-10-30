"""
convert the ckpts of trained bert to huggingface
"""
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import os
from pathlib import Path
from util.io_utils import get_ckpt_old_to_new, write_ckpts, get_folderpath
from scripts.modeling.convert_bert_to_hf import convert_tf_checkpoint_to_pytorch
from multiprocessing import Pool
import shutil


@dataclass
class Args:
    output_dir: str
    bert_config_path: str
    start_epoch: int = 0
    last_only: bool = False
    has_rotation: bool = field(default=False)


suffix = ""


def convert(epoch, ckpt, output_dir, bert_tf_config, bert_hf_config, bert_vocab, is_last=False):
    save_folder = os.path.join(output_dir, f"epoch_{epoch}")
    save_path = os.path.join(save_folder, "pytorch_model.bin")
    os.makedirs(save_folder, exist_ok=True)
    print(save_folder)
    convert_tf_checkpoint_to_pytorch(ckpt, bert_tf_config, save_path)
    # shutil.copy(bert_hf_config, os.path.join(save_folder, "config.json"))
    shutil.copy("bert/scripts/convert_to_hf/tokenizer_config.json", os.path.join(save_folder, "tokenizer_config.json"))
    shutil.copy(bert_vocab, os.path.join(save_folder, "vocab.txt"))
    if is_last:
        Path(f"{output_dir}/epoch_last").unlink(missing_ok=True)
        os.symlink(f"epoch_{epoch}", os.path.join(output_dir, 'epoch_last'))


if __name__ == '__main__':
    parser = HfArgumentParser(Args)
    args, = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)

    args_list = []

    pool = Pool(64)

    output_dir = args.output_dir
    bert_config_path = args.bert_config_path
    start_epoch = args.start_epoch
    last_only = args.last_only

    ckpts = get_ckpt_old_to_new(output_dir)
    bert_tf_config = f"{bert_config_path}/bert_config.json"
    bert_hf_config = f"{bert_config_path}/config.json"
    bert_vocab = f"{bert_config_path}/vocab.txt"
    if last_only:
        args_list.append((len(ckpts) - 1 - start_epoch, ckpts[-1], output_dir, bert_tf_config, bert_hf_config, bert_vocab, last_only))
    else:
        for epoch, ckpt in enumerate(ckpts, start=start_epoch):
            args_list.append((epoch, ckpt, output_dir, bert_tf_config, bert_hf_config, bert_vocab))
    pool.starmap(convert, args_list)
