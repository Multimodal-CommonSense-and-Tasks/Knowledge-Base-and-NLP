import fileinput
import subprocess
from wiktra.common_remove_dummy import remove_texts, restore_texts
def transliterate(token):
    token, dummy_dict = remove_texts(token, ['"', '\\'])
    res = subprocess.check_output(
        f'echo "{token}" | camel_transliterate -s bw2ar',
        # capture_output=True,
        text=True,
        shell=True,
    )
    res = restore_texts(res, dummy_dict, "NFC")
    return res.strip()


if __name__ == '__main__':
    for line in fileinput.input():
        line = line.strip()
        if line:
            prefix, content = line.split(":", maxsplit=1)
            token, label = content.split("\t")
            token = transliterate(token)
            print(f"{prefix}:{token}\t{label}")
        else:
            print()
