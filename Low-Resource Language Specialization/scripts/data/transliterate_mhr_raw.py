import fileinput
import transliterate

cyr_translit = transliterate.get_translit_function("ru")

for line in fileinput.input():
    line = line.strip()
    print(cyr_translit(line, reversed=True))
