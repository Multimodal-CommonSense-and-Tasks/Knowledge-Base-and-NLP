import fileinput
import subprocess
import unicodedata


def transliterate(token):
    token = unicodedata.normalize("NFC", token)
    translit_output = subprocess.run(
        [
            "perl",
            "alTranscribe.pl",
            "-f",
            "ug",
            "-t",
            "tr",
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
        line = line.strip()
        if line:
            prefix, content = line.split(":", maxsplit=1)
            token, label = content.split("\t")
            token = transliterate(token)
            print(f"{prefix}:{token}\t{label}")
        else:
            print()