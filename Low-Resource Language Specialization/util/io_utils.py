import os, glob
import re
import json
import os
from util.string_utils import natural_keys
import tensorflow.compat.v1 as tf



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
PREFIX_CHECKPOINT_DIR = "check"
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


# def tf_open_file_in_path(path, filename_in_path, mode='w'):
#     """
#     Automatically creates path if the path doesn't exist,
#     then open filename_in_path with mode,
#     and write content
#     """
#     filepath = os.path.join(path, filename_in_path)
#     if not tf.gfile.Exists(filepath):
#         if not tf.gfile.Exists(path):
#             tf.gfile.MakeDirs(path)
#     return tf.gfile.GFile(filepath, mode)


### Code below originated from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py
def archive_ckpt(ckpt_eval_result_dict, ckpt_objective, ckpt_dir, keep_archives=2):
    """Archive a checkpoint and ckpt before, if the metric is better."""
    archive_dir = 'archive'
    archive_oldest_available_dir = 'archive_oldest_available'

    saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')

    if not check_is_improved(ckpt_objective, saved_objective_path):
        return False

    all_ckpts_available = get_ckpt_old_to_new(ckpt_dir)
    latest_ckpt = all_ckpts_available[-1]
    if not update_one_ckpt_and_remove_old_ones(latest_ckpt, os.path.join(ckpt_dir, archive_dir),
                                               keep_archives, ckpt_eval_result_dict):
        return False

    oldest_ckpt_available = all_ckpts_available[0]
    if not update_one_ckpt_and_remove_old_ones(oldest_ckpt_available,
                                               os.path.join(ckpt_dir, archive_oldest_available_dir),
                                               keep_archives):
        return False

    # Update the best objective.
    with tf.gfile.GFile(saved_objective_path, 'w') as f:
        f.write('%f' % ckpt_objective)

    return True


def check_is_improved(ckpt_objective, saved_objective_path):
    saved_objective = float('-inf')
    if tf.gfile.Exists(saved_objective_path):
        with tf.gfile.GFile(saved_objective_path, 'r') as f:
            saved_objective = float(f.read())
    if saved_objective > ckpt_objective:
        tf.logging.info('Ckpt %s is worse than %s', ckpt_objective, saved_objective)
        return False
    else:
        return True


def get_ckpt_old_to_new(target_dir):
    """Returns ckpt names from newest to oldest. Returns [] if nothing exists"""
    prev_ckpt_state = tf.train.get_checkpoint_state(target_dir)
    all_ckpts = []
    if prev_ckpt_state:
        all_ckpts = sorted(prev_ckpt_state.all_model_checkpoint_paths, key=natural_keys, reverse=False)
        tf.logging.info('got all_model_ckpt_paths %s' % str(all_ckpts))
    return all_ckpts


def update_one_ckpt_and_remove_old_ones(ckpt_name_path, dst_dir, num_want_to_keep_ckpts, ckpt_eval_result_dict=""):
    """
    :param ckpt_eval_result_dict: provide a evaluation informations if you want to write there.
    """
    filenames = tf.gfile.Glob(ckpt_name_path + '.*')
    if filenames is None:
        tf.logging.info('No files to copy for checkpoint %s', ckpt_name_path)
        return False

    tf.gfile.MakeDirs(dst_dir)

    num_want_to_keep_prev_ckpts = num_want_to_keep_ckpts - 1
    remaining_ckpts = remove_old_ckpts_and_get_remaining_names(
        dst_dir, num_want_to_keep_ckpts=num_want_to_keep_prev_ckpts)

    write_ckpts(ckpt_name_path, dst_dir, remaining_ckpts)

    if ckpt_eval_result_dict:
        with tf.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
            f.write('%s' % ckpt_eval_result_dict)

    return True


def remove_old_ckpts_and_get_remaining_names(dst_dir, num_want_to_keep_ckpts):
    # Remove old ckpt files. get_checkpoint_state returns absolute path. refer to
    # https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/commit/2843a7867d51c2cf065b85899ea0b9564e4d9db9
    all_ckpts = get_ckpt_old_to_new(dst_dir)
    if all_ckpts:
        want_to_rm_ckpts = all_ckpts[:-num_want_to_keep_ckpts]
        for want_to_rm_ckpt in want_to_rm_ckpts:
            want_to_rm = tf.gfile.Glob(want_to_rm_ckpt + "*")
            for f in want_to_rm:
                tf.logging.info('Removing checkpoint %s', f)
                tf.gfile.Remove(f)
        remaining_ckpts = all_ckpts[-num_want_to_keep_ckpts:]
    else:
        remaining_ckpts = []

    return remaining_ckpts


def write_ckpts(ckpt_path, dst_dir, remaining_ckpts):
    filenames = tf.gfile.Glob(ckpt_path + '.*')
    tf.logging.info('Copying checkpoint %s to %s', ckpt_path, dst_dir)
    for f in filenames:
        dest = os.path.join(dst_dir, os.path.basename(f))
        tf.gfile.Copy(f, dest, overwrite=True)

    ckpt_path = get_filename(ckpt_path, remove_ext=False)
    ckpt_state = tf.train.generate_checkpoint_state_proto(
        dst_dir,
        model_checkpoint_path=ckpt_path,
        all_model_checkpoint_paths=remaining_ckpts)
    with tf.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
        str_ckpt_state = str(ckpt_state)
        str_ckpt_state = str_ckpt_state.replace('../', '')
        tf.logging.info('str_ckpt_state %s' % str_ckpt_state)
        f.write(str_ckpt_state)
