import os, glob
import re


def open_file_in_path(path, filename_in_path, mode='w'):
    """
    Automatically creates path if the path doesn't exist,
    then open filename_in_path with mode,
    and write content
    """
    filepath = os.path.join(path, filename_in_path)
    if not os.path.exists(filepath):
        if not os.path.exists(path):
            os.makedirs(path)
    return open(filepath, mode)


def tf_open_file_in_path(path, filename_in_path, mode='w'):
    """
    Automatically creates path if the path doesn't exist,
    then open filename_in_path with mode,
    and write content
    """
    import tensorflow as tf
    filepath = os.path.join(path, filename_in_path)
    if not tf.io.gfile.exists(filepath):
        if not tf.io.gfile.exists(path):
            tf.io.gfile.makedirs(path)
    return tf.io.gfile.GFile(filepath, mode)


def get_filename(file: str, remove_ext=True):
    if file.endswith('/'):
        file = file[:-1]
    name = os.path.split(file)[1]
    if remove_ext:
        return rm_ext(name)
    return name


def get_folderpath(file):
    name = os.path.split(file)[0]
    return name


def rm_ext(filename):
    filename = os.path.splitext(filename)[0]
    return filename


# copied and modified from https://github.com/huggingface/transformers/blob/cd56f3fe7eae4a53a9880e3f5e8f91877a78271c/src/transformers/trainer_utils.py#L96
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)[/]?$")


def get_checkpoint_subdir_name(steps):
    return f'{PREFIX_CHECKPOINT_DIR}-{steps}'


def get_checkpoint_dirs_from(folder):
    return glob.glob(os.path.join(folder, f"{PREFIX_CHECKPOINT_DIR}-*"))


def get_last_checkpoint(folder):
    """Supports google cloud bucket also"""
    import tensorflow as tf
    content = tf.io.gfile.listdir(folder)
    # content = ['check-789/']
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and tf.io.gfile.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def get_step_from_checkpoint(ckpt_dir):
    return int(_re_checkpoint.search(ckpt_dir).groups()[0])
