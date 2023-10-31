from datasets import *
import json

for split in ["train", "validation", "test"]:
    new_dataset = []
    cnt = 0

    for set, subset in [("paws", "labeled_final"), ("paws", "labeled_swap")]:
        try:
            ds_builder = load_dataset_builder(set, subset)
            dataset = load_dataset(set, subset, split=split)
            for datum in dataset:
                if datum["label"]:
                    new_dataset.append({
                        "id": cnt,  
                        "source": datum["sentence1"],
                        "targets": [datum["sentence2"]],
                    })
                    cnt += 1
        except Exception as e:
            # print(e)
            continue

    split = "dev" if split == "validation" else split
    print(split, len(new_dataset))
    with open(f"data/paws_paragen_{split}.json", "w", encoding="UTF-8") as file:
        json.dump(new_dataset, file, ensure_ascii=False, indent=4)
