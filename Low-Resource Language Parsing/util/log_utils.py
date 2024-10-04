import datetime
import os
import subprocess
import sys

from util.io_utils import open_file_in_path
from util.string_utils import add_pre_suf_to_keys_of
from hashlib import blake2s
import time


class TimeChecker:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"{self.name} consumed {end_time - self.start_time}")


def log_verbose(folder_path, name_prefix=""):
    verbose_filename = name_prefix + 'verbose'
    gitdiff_filename = name_prefix + 'gitdiff'
    f = open_file_in_path(folder_path, verbose_filename, "w")
    f.write(" ".join(sys.argv) + '\n')

    write_git_revision(f)
    f.close()

    f = open_file_in_path(folder_path, gitdiff_filename, "wb")
    write_gitdiff(f)
    f.close()


def write_git_revision(opened_file):
    git_tag = _try_getting_output_or_blank(["git", "describe", "--tags", "--exact-match"])
    git_branch = _try_getting_output_or_blank(["git", "symbolic-ref", "-q", "--short", "HEAD"])
    git_revision = _try_getting_output_or_blank(["git", "rev-parse", "HEAD"])
    opened_file.write(git_tag + git_branch + git_revision)


def write_gitdiff(opened_binary_file):
    git_diff = str.encode(_try_getting_output_or_blank(["git", "diff", "--ignore-space-at-eol"]))
    opened_binary_file.write(git_diff)


def _try_getting_output_or_blank(command):
    try:
        output = subprocess.check_output(command, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = ""
    return output


def get_starttime():
    if not hasattr(get_starttime, "starttime"):
        now = datetime.datetime.now()
        get_starttime.starttime = now.strftime('%y-%m-%d.%H.%M.%S')
    return get_starttime.starttime


def init_log(save_path):
    """
    Helper function to log detailed settings in a experiment.
    - it creates a path named current time in the save_path
    - it creates a symbolic link in it. It's called 'latest'. This makes it easier to cd in latest experiment.
    - (if you use gpus I assume you work on local.) it makes a local log file and let tensorflow to log in that file.
    """
    base_path = os.path.join(save_path, get_starttime())

    log_verbose(folder_path=base_path)
    link_path_to_savepath = os.path.join(save_path, "latest")
    if os.path.islink(link_path_to_savepath):
        os.remove(link_path_to_savepath)
    os.symlink(get_starttime(), link_path_to_savepath)




def get_hash(tag: str, wanted_len_even_num=30):
    assert wanted_len_even_num % 2 == 0
    digest_size = wanted_len_even_num // 2
    h = blake2s(digest_size=digest_size, salt=b'salt')
    h.update(tag.encode())
    return h.hexdigest()
