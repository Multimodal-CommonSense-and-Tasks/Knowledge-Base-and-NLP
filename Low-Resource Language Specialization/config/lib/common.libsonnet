local seed_set = std.extVar("SEED_SET");
local seeds = {
    "allentune": {
        "numpy_seed": std.extVar("NUMPY_SEED"),
        "pytorch_seed": std.extVar("PYTORCH_SEED"),
        "random_seed": std.extVar("RANDOM_SEED"),
    },
    "0": {
        "numpy_seed": 1337,
        "pytorch_seed": 133,
        "random_seed": 13370,
    },
    "1": {
        "numpy_seed": 337,
        "pytorch_seed": 33,
        "random_seed": 3370,
    },
    "2": {
        "numpy_seed": 1537,
        "pytorch_seed": 153,
        "random_seed": 15370,
    },
    "3": {
        "numpy_seed": 2460,
        "pytorch_seed": 246,
        "random_seed": 24601,
    },
    "4": {
        "numpy_seed": 1279,
        "pytorch_seed": 127,
        "random_seed": 12790,
    },
    "5": {
        "numpy_seed": 5480,
        "pytorch_seed": 548,
        "random_seed": 54808
    },
    "6": {
        "numpy_seed": 4615,
        "pytorch_seed": 461,
        "random_seed": 46154
    },
    "7": {
        "numpy_seed": 6358,
        "pytorch_seed": 635,
        "random_seed": 63580
    },
    "8": {
        "numpy_seed": 6187,
        "pytorch_seed": 618,
        "random_seed": 61878
    },
    "9": {
        "numpy_seed": 4828,
        "pytorch_seed": 482,
        "random_seed": 48280
    },
    "10": {
        "numpy_seed": 793,
        "pytorch_seed": 79,
        "random_seed": 7937
    },
    "11": {
        "numpy_seed": 6685,
        "pytorch_seed": 668,
        "random_seed": 66857
    },
    "12": {
        "numpy_seed": 2153,
        "pytorch_seed": 215,
        "random_seed": 21535
    },
    "13": {
        "numpy_seed": 5914,
        "pytorch_seed": 591,
        "random_seed": 59145
    },
    "14": {
        "numpy_seed": 5947,
        "pytorch_seed": 594,
        "random_seed": 59470
    },
    "15": {
        "numpy_seed": 1440,
        "pytorch_seed": 144,
        "random_seed": 14408
    },
    "16": {
        "numpy_seed": 9226,
        "pytorch_seed": 922,
        "random_seed": 92263
    },
    "17": {
        "numpy_seed": 6689,
        "pytorch_seed": 668,
        "random_seed": 66894
    },
    "18": {
        "numpy_seed": 5198,
        "pytorch_seed": 519,
        "random_seed": 51986
    },
    "19": {
        "numpy_seed": 2881,
        "pytorch_seed": 288,
        "random_seed": 28817
    },
    "20": {
        "numpy_seed": 717,
        "pytorch_seed": 71,
        "random_seed": 7170
    },
    "21": {
        "numpy_seed": 171,
        "pytorch_seed": 17,
        "random_seed": 1719
    },
    "22": {
        "numpy_seed": 396,
        "pytorch_seed": 39,
        "random_seed": 3965
    },
    "23": {
        "numpy_seed": 7207,
        "pytorch_seed": 720,
        "random_seed": 72071
    },
    "24": {
        "numpy_seed": 1467,
        "pytorch_seed": 146,
        "random_seed": 14670
    },
};

seeds[seed_set]
