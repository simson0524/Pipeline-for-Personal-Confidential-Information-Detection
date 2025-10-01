# generated_augmentation/generated_augmentation_process_6.py

from generated_augmentation.generate_sentences import generate_n_sentences
from generated_augmentation.auto_validation import auto_validation
from generated_augmentation.manual_validation import manual_validation
from generated_augmentation.add_validated_sentence_to_train_set import add_validated_sentence_to_train_set
from database.db_logics import *
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import yaml
import os


def generated_augmentation_process_6(conn, config):
    start_time = datetime.now()

    # Config
    train_data_dir = config['data']['train_data_dir']
    experiment_name = config['exp']['name']
    is_pii = config['exp']['is_pii']
    if is_pii:
        id_2_label = config['label_mapping']['pii_id_2_label']
    else:
        id_2_label = config['label_mapping']['confid_id_2_label']

    # 오탐/미탐 항목을 모두 가져옴
    generation_candidates = fetch_generation_candidates(
        conn=conn,
        experiment_name=experiment_name
    ) # [(experiment_name, sentence_id, sentence, span_token, index_in_datset_class, ground_truth, prediction, source_file_name, sentence_sequence), ]

    manual_chk_list = []
    generation_augmented_sent_dataset_log = []

    # 문장생성 및 자동검증
    for _, _, _, span_token, _, gt, pred, _, _ in tqdm(generation_candidates, desc="문맥 문장 데이터 생성중..."):
        wrong_test = 0
        wrong_priority = 0

        # 문맥 문장 데이터 생성
        gt_samples, pred_samples = generate_n_sentences(
            n=config['exp']['generation_num'],
            span_token=span_token,
            gt_label=gt,
            pred_label=pred,
            is_pii=is_pii
        )

        # gt_samples 검증
        valid_results_1, samples_1 = auto_validation(
            span_token=span_token,
            samples=gt_samples,
            target_label=gt,
            is_pii=is_pii
        )

        # pred_samples 검증
        valid_results_2, samples_2 = auto_validation(
            span_token=span_token,
            samples=pred_samples,
            target_label=pred,
            is_pii=is_pii
        )

        # 모든 검증 결과 병합
        valid_results = valid_results_1 + valid_results_2 # [(사용가능여부(bool), 검증된 라벨), ...]
        samples = samples_1 + samples_2 # [문장1, ...]

        # 자동검증결과에 따라 검증된 친구는 자동으로 합치기, 안 된 친구는 수동 검증으로
        for i, (valid_result, sample) in enumerate(zip(valid_results, samples)):
            dataset_id = f"sample_99_{(experiment_name.split('_')[-1])}_{str(i).zfill(6)}"
            if valid_result[0]:
                char_start = sample.find(span_token, 0)
                char_end = char_start + len(span_token)
                add_validated_sentence_to_train_set(
                    config=config, 
                    sentence=sample, 
                    dataset_id=dataset_id, 
                    span_token=span_token, 
                    char_start=char_start, 
                    char_end=char_end, 
                    label=valid_result[1]
                )
            elif (valid_result[0] == False) and (valid_result[1] != None):
                manual_chk_info = (sample, span_token, valid_result[1], dataset_id)
                manual_chk_list.append( manual_chk_info )

            generation_augmented_sent_dataset_log_scheme = (
                dataset_id,
                experiment_name,
                sample,
                span_token,
                "None",
                valid_result[1],
                dataset_id + '.json'
            )

            generation_augmented_sent_dataset_log.append( generation_augmented_sent_dataset_log_scheme )
    
    insert_many_rows(conn=conn, table_name='generation_augmented_sent_dataset_log', data_list=generation_augmented_sent_dataset_log)
                
    # # 수동검증
    # manual_validation(manual_chk_list=manual_chk_list)