import random
from utils.code_mix import get_codemixed_ids


class TrainDataset:
    def __init__(self, train_data, data_args, tokenizer):
        self.group_size = data_args.train_n_passages
        self.data = train_data
        self.data_args = data_args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def get_example(self, i, epoch):
        example = self.data[i]
        q = example["query_input_ids"]

        pp = example["pos_psgs_input_ids"]
        p = pp[0]

        nn = example["neg_psgs_input_ids"]
        off = epoch * (self.group_size - 1) % len(nn)
        nn = nn * 2
        nn = nn[off : off + self.group_size - 1]

        return q, [p] + nn

    def get_batch(self, indices, epoch):
        qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
        dd = sum(dd, [])
        return dict(
            self.tokenizer.pad(
                qq,
                max_length=self.data_args.q_max_len,
                padding="max_length",
                return_tensors="np",
            )
        ), dict(
            self.tokenizer.pad(
                dd,
                max_length=self.data_args.p_max_len,
                padding="max_length",
                return_tensors="np",
            )
        )


class TrainDatasetNaive:
    def __init__(self, train_data, data_args, tokenizer, basic_tokenizer, src2tgt):
        self.group_size = data_args.train_n_passages
        self.data = train_data
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.basic_tokenizer = basic_tokenizer
        self.src2tgt = src2tgt

    def __len__(self):
        return len(self.data)

    def get_example(self, i, epoch):
        example = self.data[i]
        q = example["query_input_ids"]

        pp = example["pos_psgs_input_ids"]
        p = pp[0]

        nn = example["neg_psgs_input_ids"]
        off = epoch * (self.group_size - 1) % len(nn)
        nn = nn * 2
        nn = nn[off : off + self.group_size - 1]

        if not self.data_args.codemix_in_runtime:
            return q, [p] + nn

        if (
            self.data_args.codemix_ratio_query > 0
            and random.random() < self.data_args.codemix_sentence_ratio_query
        ):
            q = dict(
                {
                    "input_ids": get_codemixed_ids(
                        self.tokenizer,
                        self.basic_tokenizer,
                        self.src2tgt,
                        q["input_ids"],
                        self.data_args.q_max_len,
                        self.data_args.codemix_ratio_query,
                    )
                }
            )
        if (
            self.data_args.codemix_ratio_document > 0
            and random.random() < self.data_args.codemix_sentence_ratio_document
        ):
            p = dict(
                {
                    "input_ids": get_codemixed_ids(
                        self.tokenizer,
                        self.basic_tokenizer,
                        self.src2tgt,
                        p["input_ids"],
                        self.data_args.p_max_len,
                        self.data_args.codemix_ratio_document,
                    )
                }
            )
            nn = [
                dict(
                    {
                        "input_ids": get_codemixed_ids(
                            self.tokenizer,
                            self.basic_tokenizer,
                            self.src2tgt,
                            n["input_ids"],
                            self.data_args.p_max_len,
                            self.data_args.codemix_ratio_document,
                        )
                    }
                )
                for n in nn
            ]

        return q, [p] + nn

    def get_batch(self, indices, epoch):
        qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
        dd = sum(dd, [])
        return dict(
            self.tokenizer.pad(
                qq,
                max_length=self.data_args.q_max_len,
                padding="max_length",
                return_tensors="np",
            )
        ), dict(
            self.tokenizer.pad(
                dd,
                max_length=self.data_args.p_max_len,
                padding="max_length",
                return_tensors="np",
            )
        )


class TrainDatasetContrastive:
    def __init__(self, train_data, data_args, tokenizer, basic_tokenizer, src2tgt):
        self.group_size = data_args.train_n_passages
        self.data = train_data
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.basic_tokenizer = basic_tokenizer
        self.src2tgt = src2tgt

    def __len__(self):
        return len(self.data)

    def get_example(self, i, epoch):
        example = self.data[i]
        q = example["query_input_ids"]

        pp = example["pos_psgs_input_ids"]
        p = pp[0]

        nn = example["neg_psgs_input_ids"]
        off = epoch * (self.group_size - 1) % len(nn)
        nn = nn * 2
        nn = nn[off : off + self.group_size - 1]

        cm_q = q
        cm_p = p
        cm_nn = nn

        if not self.data_args.codemix_in_runtime:
            return q, [p] + nn, cm_q, [cm_p] + cm_nn

        if (
            self.data_args.codemix_ratio_query > 0
            and random.random() < self.data_args.codemix_sentence_ratio_query
        ):
            cm_q = dict(
                {
                    "input_ids": get_codemixed_ids(
                        self.tokenizer,
                        self.basic_tokenizer,
                        self.src2tgt,
                        q["input_ids"],
                        self.data_args.q_max_len,
                        self.data_args.codemix_ratio_query,
                    )
                }
            )
        if (
            self.data_args.codemix_ratio_document > 0
            and random.random() < self.data_args.codemix_sentence_ratio_document
        ):
            cm_p = dict(
                {
                    "input_ids": get_codemixed_ids(
                        self.tokenizer,
                        self.basic_tokenizer,
                        self.src2tgt,
                        p["input_ids"],
                        self.data_args.p_max_len,
                        self.data_args.codemix_ratio_document,
                    )
                }
            )
            cm_nn = [
                dict(
                    {
                        "input_ids": get_codemixed_ids(
                            self.tokenizer,
                            self.basic_tokenizer,
                            self.src2tgt,
                            n["input_ids"],
                            self.data_args.p_max_len,
                            self.data_args.codemix_ratio_document,
                        )
                    }
                )
                for n in nn
            ]

        return q, [p] + nn, cm_q, [cm_p] + cm_nn

    def get_batch(self, indices, epoch):
        qq, dd, cm_qq, cm_dd = zip(
            *[self.get_example(i, epoch) for i in map(int, indices)]
        )
        dd = sum(dd, [])
        cm_dd = sum(cm_dd, [])
        return (
            dict(
                self.tokenizer.pad(
                    qq,
                    max_length=self.data_args.q_max_len,
                    padding="max_length",
                    return_tensors="np",
                )
            ),
            dict(
                self.tokenizer.pad(
                    dd,
                    max_length=self.data_args.p_max_len,
                    padding="max_length",
                    return_tensors="np",
                )
            ),
            dict(
                self.tokenizer.pad(
                    cm_qq,
                    max_length=self.data_args.q_max_len,
                    padding="max_length",
                    return_tensors="np",
                )
            ),
            dict(
                self.tokenizer.pad(
                    cm_dd,
                    max_length=self.data_args.p_max_len,
                    padding="max_length",
                    return_tensors="np",
                )
            ),
        )
