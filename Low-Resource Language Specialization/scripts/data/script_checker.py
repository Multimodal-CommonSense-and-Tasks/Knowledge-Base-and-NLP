# https://github.com/EliFinkelshteyn/alphabet-detector/blob/master/alphabet_detector/alphabet_detector.py
import glob
import unicodedata as ud
import argparse
from collections import defaultdict, OrderedDict


class AlphabetDetector:
    all_langs = ['ARABIC', 'LATIN', 'CYRILLIC', 'CJK', 'KATAKANA', 'GREEK', 'HIRAGANA', 'HEBREW', 'HANGUL', 'FULLWIDTH', 'MODIFIER', 'GEORGIAN', 'MONGOLIAN',
                 'THAI', 'DEVANAGARI', 'ARMENIAN', 'MYANMAR', 'GUJARATI', 'SYRIAC', 'KATAKANA-HIRAGANA', 'ETHIOPIC', 'BENGALI', 'OLD', 'MICRO']

    def __init__(self, no_memory=False):
        self.alphabet_letters = defaultdict(dict)
        self.no_memory = no_memory

    def is_in_alphabet(self, uchr, alphabet):
        if self.no_memory:
            return alphabet in ud.name(uchr)
        try:
            return self.alphabet_letters[alphabet][uchr]
        except KeyError:
            return self.alphabet_letters[alphabet].setdefault(
                uchr, alphabet in ud.name(uchr))

    def only_alphabet_chars(self, unistr, alphabet):
        return all(self.is_in_alphabet(uchr, alphabet)
                   for uchr in unistr if uchr.isalpha())

    def has_chars(self, unistr, alphabet):
        return any(self.is_in_alphabet(uchr, alphabet)
                   for uchr in unistr)

    def only_chars(self, unistr, alphabet):
        return all(self.is_in_alphabet(uchr, alphabet)
                   for uchr in unistr)

    def has_alphabet_chars(self, unistr, alphabet):
        return any([uchr.isalpha() for uchr in unistr]) and \
               any(self.is_in_alphabet(uchr, alphabet)
                   for uchr in unistr if uchr.isalpha())

    def real_alphabet_chars(self, unistr, alphabet):
        return any([uchr.isalpha() for uchr in unistr]) and \
               self.only_alphabet_chars(unistr, alphabet)

    def detect_alphabet(self, unistr):
        return set(ud.name(char).split(' ')[0]
                   for char in unistr if char.isalpha())

    def detect_alphabet_ratio(self, unistr):
        return self.detect_ratio(unistr, check_isalpha=True)

    def detect_ratio(self, unistr, check_isalpha=False):
        lang_script_counts = defaultdict(int)
        for char in unistr:
            if (not check_isalpha) or char.isalpha():
                lang_script = ud.name(char).split(' ')[0]
                lang_script_counts[lang_script] += 1
        total_scripts = sum(list(lang_script_counts.values()))
        return OrderedDict(sorted([(k, v / total_scripts) for k, v in lang_script_counts.items()], key=lambda t: t[1], reverse=True))

    def is_greek(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'GREEK') else False

    def is_cyrillic(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'CYRILLIC') else False

    def is_latin(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'LATIN') else False

    def is_arabic(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'ARABIC') else False

    def is_hebrew(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'HEBREW') else False

    # NOTE: this only detects Chinese script characters (Hanzi/Kanji/Hanja).
    # it does not detect other CJK script characters like Hangul or Katakana
    def is_cjk(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'CJK') else False

    def is_hangul(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'HANGUL') else False

    def is_hiragana(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'HIRAGANA') else False

    def is_katakana(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'KATAKANA') else False

    def is_thai(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'THAI') else False


detector = AlphabetDetector()


def detect_ratio(input_text):
    return detector.detect_ratio(input_text)

def detect_alphabet_ratio(input_text):
    return detector.detect_alphabet_ratio(input_text)

def most_significant(text):
    result = detect_ratio(text)
    return list(result.keys())[0]

def most_significant_except(text, remove=("LATIN")):
    result = detect_ratio(text)
    for r in remove:
        del result[r]
    return list(result.keys())[0]

def most_significant_alphabet(text):
    result = detect_alphabet_ratio(text)
    return list(result.keys())[0]

def most_significant_alphabet_except(text, remove=("LATIN")):
    result = detect_alphabet_ratio(text)
    for r in remove:
        del result[r]
    return list(result.keys())[0]

def agg_files(glob_pattern):
    text = ""
    for f in sorted(glob.glob(glob_pattern)):
        text += open(f, encoding='utf-8').read()
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", default="specializing-multilingual-data/data/ug/unlabeled/bert_cleaned/*")
    parser.add_argument("--vocab", default="bert/scripts/convert_to_hf/mbert_vocab.txt")
    args = parser.parse_args()

    text = "،"
    print(detect_ratio(text))

    for text in [".", '1', '!', '"', '(']:
        print(ud.name(text))
        # print(detect_ratio(text))
    #
    text = "##한"
    print(detector.only_alphabet_chars(text, "HANGUL"))
    text = "##한"
    print(detector.only_chars(text, "HANGUL"))
    text = "##한AA"
    print(detector.has_chars(text, "HANGUL"))
    # text = "###LA한"
    # print(detector.has_alphabet_chars(text, "LATIN"))
    # print(detector.has_alphabet_chars(text, "HANGUL"))
    # text = "###"
    # print(detector.has_alphabet_chars(text, "LATIN"))
    # print(detector.has_alphabet_chars(text, "HANGUL"))

#     text = """
# ＤｓｔｉｍｕｌｕｓｔｒｉｇｇｅｒｅｄａｃｑｕｉｓｉｔｉｏｎｏｆｐｌｕｒｉｐｏｔｅｎｃｙＩＩＩＩＡ𐱃𐰆𐰪𐰸𐰸    """
#
#     for c in text:
#         if c.isalpha():
#             for form in ["NFC", "NFKC", "NFD", "NFKD"]:
#                 print(ud.name(c))
#                 print(ud.name(ud.normalize(form, c)))
#                 break
