from datasets import *
import json

for set, subset in [("wmt14", "de-en"), ("wmt15", "fr-en"), ("wmt16", "ro-en")]:
    print(set, subset)
    ds_builder = load_dataset_builder(set, subset)
    second_lang = subset[0:2]
    cnt = 0
    for split in ["train", "validation", "test"]:
        dataset = load_dataset(set, subset, split=split)
        new_dataset = []
        for datum in dataset:
            new_dataset.append({
                "id": cnt,
                "source": datum["translation"]["en"],
                "targets": [datum["translation"][second_lang]],
            })
            cnt += 1

        split = "dev" if split == "validation" else split
        with open(f"data/{set}_en{second_lang}_translation_{split}.json", "w", encoding="UTF-8") as file:
            json.dump(new_dataset, file, ensure_ascii=False, indent=4)

for set, subset in [("wmt14", "de-en"), ("wmt15", "fr-en"), ("wmt16", "ro-en")]:
    print(set, subset)
    ds_builder = load_dataset_builder(set, subset)
    second_lang = subset[0:2]
    cnt = 0
    for split in ["train", "validation", "test"]:
        dataset = load_dataset(set, subset, split=split)
        new_dataset = []
        for datum in dataset:
            new_dataset.append({
                "id": cnt,
                "source": datum["translation"][second_lang],
                "targets": [datum["translation"]["en"]],
            })
            cnt += 1

        split = "dev" if split == "validation" else split
        with open(f"data/{set}_{second_lang}en_translation_{split}.json", "w", encoding="UTF-8") as file:
            json.dump(new_dataset, file, ensure_ascii=False, indent=4)