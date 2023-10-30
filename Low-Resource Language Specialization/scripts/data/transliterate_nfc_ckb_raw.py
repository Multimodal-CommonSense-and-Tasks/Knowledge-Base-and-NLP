import argparse
import fileinput
import unicodedata
import subprocess


# NOTE: this doesn't actually transliterate; it just does NFKC.  It's named
# this way so it groups well with other files in the directory.
def transliterate(token):
    token = unicodedata.normalize("NFC", token)
    translit_output = subprocess.run(
        [
            "perl",
            "alTranscribe.pl",
            "-f",
            "ckb",
            "-t",
            "ku",
        ],
        capture_output=True,
        input=token,
        text=True,
        cwd="../mbert-unseen-languages/transfer/transliteration/",
    )
    transliterated = translit_output.stdout
    return transliterated.strip()


if __name__ == '__main__':
    for line in fileinput.input():
        if line:
            print(transliterate(line))
