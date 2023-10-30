import pandas as pd

ATTRIBUTES = {
    "type0": {"be", "bg", "ur", "vi"},
    "type1": {"ga"},
    "type2": {"mhr", "mt", "ug", "wo"},
    "latin": {"ga", "mt", "vi", "wo"},
    "cyrillic": {"be", "bg", "mhr"},
    "arabic": {"ug", "ur"},
    "all": {"be", "bg", "ga", "mhr", "mt", "ug", "ur", "vi", "wo"},
}

COMMON_LANGS = ["be", "bg", "ga", "mt", "ug", "ur", "vi"]
UD_COLUMNS = COMMON_LANGS + ["wo", "avg"]
PANX_COLUMNS = COMMON_LANGS + ["mhr", "avg"]
ROWS = ["fasttext", "roberta", "mbert", "lapt", "tva"]

POS_TUPLES = [
    (68.84, 88.86, 86.87, 89.68, 89.45, 90.81, 81.84, 87.48, 85.48),
    (91.00, 94.48, 90.36, 92.61, 90.87, 89.88, 84.73, 87.71, 90.20),
    (94.57, 96.98, 91.91, 94.01, 78.07, 91.77, 88.97, 93.04, 91.16),
    (95.74, 97.15, 93.28, 95.76, 79.88, 92.18, 89.64, 94.58, 92.28),
    (95.28, 97.20, 93.33, 96.33, 91.49, 92.24, 89.49, 94.48, 93.73),
]
UD_TUPLES = [
    (35.81, 84.03, 65.58, 68.45, 54.52, 79.33, 54.91, 70.39, 64.13),
    (45.77, 84.61, 64.02, 65.92, 60.34, 78.07, 54.70, 60.12, 64.19),
    (71.83, 91.62, 71.68, 76.63, 47.70, 81.45, 64.58, 76.24, 72.72),
    (72.77, 92.08, 74.79, 81.53, 50.67, 81.78, 66.15, 80.34, 75.01),
    (73.22, 91.90, 74.35, 82.00, 67.55, 81.88, 65.64, 80.22, 77.09),
]
NER_TUPLES = [
    (84.26, 87.98, 67.21, 33.53, 00.00, 92.85, 85.57, 35.28, 60.84),
    (88.08, 90.31, 76.58, 54.64, 61.54, 94.04, 88.08, 54.17, 75.93),
    (91.13, 92.56, 82.82, 61.86, 50.76, 94.60, 92.13, 61.85, 78.46),
    (91.61, 92.96, 84.13, 81.53, 56.76, 95.17, 92.41, 59.17, 81.72),
    (91.38, 92.70, 84.82, 80.00, 68.93, 95.43, 92.43, 64.23, 83.74),
]

TRANSLIT_COLUMNS = [
    "mhr_ner",
    "mhrlatin_ner",
    "ug_ner",
    "uglatin_ner",
    "ug_pos",
    "uglatin_pos",
    "ug_ud",
    "uglatin_ud",
]

TRANSLIT_TUPLES = [
    (35.28, 41.32, 00.00, 00.00, 89.45, 89.03, 54.52, 54.45),
    (54.17, 48.45, 61.54, 63.05, 90.87, 90.76, 60.34, 60.08),
    (61.85, 63.84, 50.76, 56.80, 78.07, 91.34, 47.70, 65.85),
    (59.17, 63.68, 56.76, 67.57, 79.88, 92.59, 50.67, 69.39),
    (64.23, 63.19, 68.93, 67.10, 91.49, 92.64, 67.55, 68.58),
]


def get_task_data():
    pos_df = pd.DataFrame(POS_TUPLES, columns=UD_COLUMNS, index=ROWS)
    ud_df = pd.DataFrame(UD_TUPLES, columns=UD_COLUMNS, index=ROWS)
    ner_df = pd.DataFrame(NER_TUPLES, columns=PANX_COLUMNS, index=ROWS)
    translit_df = pd.DataFrame(
        TRANSLIT_TUPLES, columns=TRANSLIT_COLUMNS, index=ROWS
    )

    return {
        "attributes": ATTRIBUTES,
        "pos": pos_df,
        "ud": ud_df,
        "ner": ner_df,
        "translit": translit_df,
    }
