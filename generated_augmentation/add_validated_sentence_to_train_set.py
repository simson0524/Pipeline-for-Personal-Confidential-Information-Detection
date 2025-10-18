# generated_augmentation/add_validated_sentence_to_train_set.py

import json
import os


def add_validated_sentence_to_train_set(config, sentence, dataset_id, span_token, char_start, char_end, label):
    save_dir = config['data']['train_data_dir']
    sentence_dataset = {
        "data": {
            "sentence": sentence,
            "id": dataset_id,
            "filename": "AUGMENTED_DATA",
            "caseField": 99,
            "detailField": 99,
            "sequence": 1
        },
        "annotations": [
            {
                "id": dataset_id,
                "annotations": [
                    {
                        "start": char_start,
                        "end": char_end,
                        "label": label,
                        "score": 1.0,
                        "span_token": span_token
                    }
                ]
            }
        ]
    }

    file_name = f"{dataset_id}.json"

    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(sentence_dataset, f, ensure_ascii=False, indent=2)