# init_dictionary.py

from ner_regex_matching.regex_logics.regex_main import run_regex_detection
from ner_regex_matching.ner_logics.ner_main import run_ner_detection
from database.db_logics import insert_many_rows
from database.create_dbs import get_connection, create_nessesary_tables
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import yaml
import json

# 실험 Config
config_file_path = "run_config.yaml"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

conn = get_connection(config)

# Create DB tables
create_nessesary_tables(conn)

answer_sheet = config['dictionary_init']['answer_sheet_dir']
raw_data = config['dictionary_init']['raw_data_based_dictionary_init_data_dir']

if answer_sheet is not None:
    for domain_id, answer_sheet_path in answer_sheet.items():
        info_dict_list = []
        info_dict_set = set()
        
        answer_sheet_df = pd.read_csv( answer_sheet_path )
        
        if config['exp']['is_pii']:
            filtered_info_df = answer_sheet_df[answer_sheet_df['개인정보/준식별자/기밀정보'] == '개인정보'] # 개인정보/준식별자 or 개인정보/준식별자/기밀정보
            table_name = 'personal_info_dictionary'
        else:
            filtered_info_df = answer_sheet_df[answer_sheet_df['개인정보/준식별자/기밀정보'] == '기밀정보'] # 기밀정보 or 개인정보/준식별자/기밀정보
            table_name = 'confidential_info_dictionary'

        for index, row in tqdm(filtered_info_df.iterrows(), desc=f"{domain_id}도메인 사전 추가중"):
            span_token = row['단어']
            first_inserted_experiment_name = "INIT"
            insertion_counts = 1
            deletion_counts = 0
            if span_token in info_dict_set:
                continue
            else:
                info_dict_scheme = (
                    domain_id,
                    span_token,
                    0,
                    first_inserted_experiment_name,
                    insertion_counts,
                    deletion_counts
                )
                info_dict_list.append( info_dict_scheme )
                info_dict_set.add( span_token )

        insert_many_rows(conn, table_name, info_dict_list)

if raw_data is not None:
    for domain_id, raw_data_dir in raw_data.items():
        info_dict_list = []
        info_dict_set = set()

        dir_path = Path(raw_data_dir)

        all_sentences = []

        for file_path in tqdm(dir_path.glob('*.json'), desc=f"{domain_id} 데이터 json 읽는중"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sentence = data['data']['sentence']
                all_sentences.append( sentence )

        for sentence in tqdm(all_sentences, desc=f"{domain_id} ner/regex기반 추가중"):
            regex_texts = run_regex_detection(sentence)
            ner_texts = run_ner_detection(sentence)
            texts = regex_texts + ner_texts
            for text in texts:
                span_token = text['단어']
                first_inserted_experiment_name = "INIT"
                insertion_counts = 1
                deletion_counts = 0
                if span_token in info_dict_set:
                    continue
                else:
                    info_dict_scheme = (
                        domain_id,
                        span_token,
                        0,
                        first_inserted_experiment_name,
                        insertion_counts,
                        deletion_counts
                    )
                    info_dict_list.append( info_dict_scheme )
                    info_dict_set.add( span_token )
        
        insert_many_rows(conn, table_name, info_dict_list)

conn.close()