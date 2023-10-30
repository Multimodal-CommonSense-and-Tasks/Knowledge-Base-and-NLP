from urllib.request import urlopen

import pandas as pd

IS_UNLABELED = False

COLUMNS = [
    "lang",
    "split",
    "tokenizer",
    "fertility",
    "continuation",
    "unkproportion",
    "tfertility",
    "tcontinuation",
    "tunkprop",
    "tokwithunkprop",
]


def get_vocab_data():
    # url =
    # "https://gist.githubusercontent.com/ethch18/1c5ffc5b3344642dd027b4138c0f57bb/raw/7144546e7532fd2bac70dede8c0fe7b4d6053ae9/gistfile0.txt"
    url = "https://gist.githubusercontent.com/ethch18/d6dda87e29ae29f1a43d67f927347729/raw/8dec2be2bf9aa9c4801d59526a15778d038e3e62/gistfile0.txt"
    fileobj = urlopen(url)

    unlabeled_rows = []
    ud_rows = []
    panx_rows = []

    lang = None
    split = None
    current_rowset = None

    def get_final_token(line: str):
        toks = line.split()
        return toks[-1]

    for line_raw in fileobj:
        line: str = line_raw.decode("utf-8")
        line = line.strip()
        if line.startswith("BEGIN VOCABULARY"):
            lang = get_final_token(line)
        elif line.startswith("Evaluating dataset"):
            dataset = get_final_token(line)
            if dataset == "unlabeled":
                current_rowset = unlabeled_rows
            elif dataset == "ud":
                current_rowset = ud_rows
            elif dataset == "ner":
                current_rowset = panx_rows
            else:
                raise Exception("Bad dataset", line)
        elif line.startswith("For data split"):
            lbrack = line.index("{")
            rbrack = line.index("}")
            split = line[lbrack + 1 : rbrack]
        elif line.startswith("Data split"):
            # does not exist case
            continue
        elif line.startswith("Tokenizer"):
            # header row
            continue
        elif line.startswith("END VOCABULARY"):
            lang = None
            split = None
            current_rowset = None
        else:
            assert lang != None and split != None and current_rowset != None
            fields = line.split()
            tokenizer = fields[0]
            remainder = list(map(lambda num_str: float(num_str), fields[1:]))
            final_row = [lang, split, tokenizer] + remainder
            current_rowset.append(tuple(final_row))

    unlabeled_df = pd.DataFrame(unlabeled_rows, columns=COLUMNS)
    unlabeled_df = unlabeled_df.set_index(["lang"])
    ud_df = pd.DataFrame(ud_rows, columns=COLUMNS)
    ud_df = ud_df.set_index(["lang"])
    panx_df = pd.DataFrame(panx_rows, columns=COLUMNS)
    panx_df = panx_df.set_index(["lang"])

    return {
        "unlabeled_df": unlabeled_df,
        "ud_df": unlabeled_df if IS_UNLABELED else ud_df,
        "panx_df": unlabeled_df if IS_UNLABELED else panx_df,
        "columns": COLUMNS,
    }
