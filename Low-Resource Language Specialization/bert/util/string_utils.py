import re
import configparser, pickle


# def compare_jsons(json1, json2):
#     from deepdiff import DeepDiff
#     js.append(json.load(open(json_file, "r")))
#
#     ddiff = DeepDiff(*js, ignore_order=True)
#


def unparse_args_to_cmd(args, prefix='python'):
    import argunparse, sys
    kwargs = {}
    suffix = ' '
    for key, val in vars(args).items():
        if isinstance(val, list):
            suffix += f'--{key} {" ".join([str(v) for v in val])} '
        elif isinstance(val, bool):
            suffix += f'--{key}={val} '
        else:
            kwargs[key] = val
    unparser = argunparse.ArgumentUnparser()
    arg_string = unparser.unparse(**kwargs)
    prefix = f'{prefix} '
    return prefix + arg_string + suffix


def deep_copy_cfg(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """deep copy config"""
    rep = pickle.dumps(config)
    new_config = pickle.loads(rep)
    return new_config


def natural_keys(text):
    # from https://stackoverflow.com/a/5967539
    return [atoi(c) for c in re.split('(\d+)', text)]


def atoi(text):
    return int(text) if text.isdigit() else text


def add_pre_suf_to_keys_of(dict, prefix="", suffix=""):
    result = {}
    for name, value in dict.items():
        result[str(prefix + name + suffix)] = value
    return result


def grab_str_between_pre_suf(string, prefix, suffix):
    assert isinstance(string, str)
    left = string.find(prefix)
    if left < 0:
        return ''
    left += len(prefix)
    right = string.find(suffix, left)
    if right < 0:
        return ''
    return string[left:right]
