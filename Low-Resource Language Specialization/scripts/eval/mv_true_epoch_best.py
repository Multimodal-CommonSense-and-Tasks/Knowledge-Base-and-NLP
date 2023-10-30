import glob
import os
import shutil
from pathlib import Path
from util.io_utils import get_filename
import argparse
from util.string_utils import natural_keys
from util.io_utils import get_ckpt_old_to_new, write_ckpts, get_folderpath
from distutils.dir_util import copy_tree


def archive(orig_file, new_link):
    if os.path.islink(new_link):
        Path(new_link).unlink(missing_ok=True)
    if os.path.isdir(orig_file):
        copy_tree(orig_file, new_link)
    else:
        shutil.copy2(orig_file, new_link)
    if args.gs_output_dir:
        if os.path.isdir(orig_file):
            os.system(f"gsutil cp -r {orig_file}/\* {args.gs_output_dir}/{get_filename(new_link, remove_ext=False)}/")
        else:
            os.system(f"gsutil cp {orig_file} {args.gs_output_dir}/{get_filename(new_link, remove_ext=False)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_prefix", default="")
    parser.add_argument("--best_epoch_save_name", default="best_epoch.txt")
    parser.add_argument("--gs_output_dir", type=str, default=None)
    parser.add_argument("--remove_gs", default=False, action='store_true')
    parser.add_argument("--last_only", default=False, action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    save_prefix = args.save_prefix

    all_ckpts = sorted(glob.glob(f"{output_dir}/epoch_*"), key=natural_keys, reverse=False)
    while not all_ckpts[-1].split('_')[-1].isnumeric():
        all_ckpts.pop()
    archive(f"{output_dir}/{get_filename(all_ckpts[-1], remove_ext=False)}", f"{output_dir}/{save_prefix}epoch_last")

    if not args.last_only:
        best_file = os.path.join(args.output_dir, args.best_epoch_save_name)
        best_epoch = int(open(best_file).read().strip())
        if args.remove_gs:
            os.system(f"gsutil -m rm -r {args.gs_output_dir}")
        if args.gs_output_dir:
            os.system(f"gsutil cp {best_file} {args.gs_output_dir}/")
        archive(f"{output_dir}/epoch_{best_epoch}", f"{output_dir}/{save_prefix}epoch_best")

        tf_ckpts = get_ckpt_old_to_new(output_dir)
        best_ckpt = tf_ckpts[best_epoch]
        prefix = best_ckpt
        print(glob.glob(f"{prefix}*"))
        for f in glob.glob(f"{prefix}*"):
            print(f)
            suffix = f[len(prefix):]
            tgt_path = f"{output_dir}/{save_prefix}model_best.ckpt{suffix}"
            archive(f"{output_dir}/{get_filename(best_ckpt, remove_ext=False)}{suffix}", tgt_path)
