### 프로젝트 한줄소개
문서 내 개인정보 및 기밀정보 추출을 위한 파이프라인을 구성한 프로젝트입니다.

### [Project Structure]
root/
│
├── classifier/  ### 리팩토링완료 ###
│   ├── model.py 
│   └── {customized}_model.py
│
├── dataset/
│   └── pipeline_dataset.py
│
├── model_train/
│   └── model_train_process_1.py
│
├── model_validation/
│   ├── model_train_validation_process_2.py
│   └── model_validation_process_5.py
│
├── dictionary_matching/
│   ├── dictionary_matching_process_3.py
│   └── update_dictionary.py # 이거 추가해야함(250930)
│
├── ner_regex_matching/
│   ├── ner_logics/
│   │   ├── ner_main.py
│   │   └── PACKAGE_SOURCE_FILES
│   │   
│   ├── regex_logics/
│   │   ├── regex_main.py
│   │   └── PACKAGE_SOURCE_FILES
│   │
│   └── ner_regex_matching_process_4.py
│
├── generated_augmentation/
│   ├── generated_augmentation_process_6.py ## 수동검증 미완 - 라벨링툴스
│   ├── generate_sentences.py
│   ├── add_validated_sentence_to_train_set.py
│   ├── auto_validation.py
│   └── manual_validation.py
│
├── labeling_tools/
│   ├── dictionary_labeler.py
│   ├── metric_viewer.py
│   └── manual_labeler.py
│
├── database/
│   ├── create_dbs.py
│   ├── export_dbs.py
│   └── db_logics.py
│
├── checkpoints/
│   │   ...
│   └── experiment_{experiment_name}/
│       │   ...
│       └── fold{fold}_epoch{epoch}_model.pt
│
├── data/
│   ├── raw_data/
│   │   │   ...
│   │   └── {부서명}_{문서명}.json
│   │
│   ├── answer_sheet/
│   │   │   ...
│   │   └── {부서명}_answer_sheet.csv
│   │
│   ├── train_set_{D}/
│   │   │   ...
│   │   └── {sentence_id}.json
│   │
│   ├── experiment_log/
│   │   │   ...
│   │   └── {experiment_name}_run_log.txt
│   │
│   └── process_log/
│       │   ...
│       └── {experiment_name}_{table_name}_log.csv  or  .txt
│
├── run_pipeline.py
├── run_config.yaml
│
├── README.md
└── requirements.txt