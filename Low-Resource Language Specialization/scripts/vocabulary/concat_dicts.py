import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--dict_file_prefix", type=str, default="ug_to_uglatinnfc/ug_to_uglatinnfc_tok.txt")
args = parser.parse_args()

result_dict = {}


def add_to_dict(orig_word, translit_word):
    if orig_word in result_dict:
        assert translit_word == result_dict[orig_word]
    else:
        result_dict[orig_word] = translit_word


if __name__ == '__main__':

    for file in glob.glob(args.dict_file_prefix + ".*"):
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip():
                    orig_words = line.strip().split('\t')
                    if len(orig_words) == 1:
                        add_to_dict(orig_words[0], "")
                    else:
                        assert len(orig_words) == 2
                        add_to_dict(*orig_words)


    with open(args.dict_file_prefix, 'w', encoding='utf-8') as f:
        for orig_w, translit_w in result_dict.items():
            f.write(f"{orig_w}\t{translit_w}\n")
