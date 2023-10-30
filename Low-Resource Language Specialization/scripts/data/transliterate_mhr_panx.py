import fileinput
import transliterate

cyr_translit = transliterate.get_translit_function("ru")

for line in fileinput.input():
    line = line.strip()
    if line:
        prefix, content = line.split(":", maxsplit=1)
        token, label = content.split("\t")
        token = cyr_translit(token, reversed=True)
        print(f"{prefix}:{token}\t{label}")
    else:
        print()