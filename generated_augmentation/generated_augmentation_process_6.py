# generated_augmentation/generated_augmentation_process_6.py

from generated_augmentation.generate_sentences import generate_n_sentences
from generated_augmentation.auto_validation import auto_validation
from generated_augmentation.manual_validation import manual_validation
from generated_augmentation.add_validated_sentence_to_train_set import add_validated_sentence_to_train_set
from labeling_tools.manual_validation_labeler import manual_validation_labeler
from database.db_logics import *
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import yaml
import os


def generated_augmentation_process_6(conn, experiment_name, config):
    start_time = datetime.now()

    # Config
    train_data_dir = config['data']['train_data_dir']
    is_pii = config['exp']['is_pii']
    if is_pii:
        id_2_label = config['label_mapping']['pii_id_2_label']
    else:
        id_2_label = config['label_mapping']['confid_id_2_label']

    # 오탐/미탐 항목을 모두 가져옴
    generation_candidates = fetch_generation_candidates(
        conn=conn,
        experiment_name=experiment_name
    )

    manual_chk_list = []
    generation_augmented_sent_dataset_log = []

    generation_start_time = datetime.now()

    # 문장생성 및 자동검증
    for idx, (_, _, _, span_token, _, gt, pred, _, _) in enumerate(tqdm(generation_candidates, desc="문맥 문장 데이터 생성중...")):
        # 빠른 실험을 위한...
        if idx % 5 != 0:
            continue
            
        # 일반정보는 너무 많으니까 굳이 추가할 이유가 없음
        if pred == "일반정보":
            continue

        # 문맥 문장 데이터 생성
        pred_samples = generate_n_sentences(
            n=config['exp']['generation_num'],
            span_token=span_token,
            gt_label=gt,
            pred_label=pred,
            is_pii=is_pii
        )

        # pred_samples 검증
        valid_results, samples, start_time, end_time, duration = auto_validation(
            span_token=span_token,
            samples=pred_samples,
            target_label=pred,
            is_pii=is_pii
        )

        # 자동검증결과에 따라 검증된 친구는 자동으로 합치기, 안 된 친구는 수동 검증으로
        for i, (valid_result, sample) in enumerate(zip(valid_results, samples)):
            dataset_id = f"sample_00_{(experiment_name)}_{str(i).zfill(6)}"
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
                manual_chk_info = {
                    "generated_sentence": sample, 
                    "span_token": span_token, 
                    "validated_label": valid_result[1], 
                    "dataset_id": dataset_id
                    }
                manual_chk_list.append( manual_chk_info )

            generation_augmented_sent_dataset_log_scheme = (
                dataset_id,
                experiment_name,
                sample,
                "00", # 증강된 문장은 도메인이 별도로 없으므로 "00"으로 통일
                span_token,
                pred,
                valid_result[1]
            )
            generation_augmented_sent_dataset_log.append( generation_augmented_sent_dataset_log_scheme )

    generation_duration = datetime.now() - 
    
    manual_validation_start_time = datetime.now()

    # 수동검증
    manual_chk_completed_list = manual_validation_labeler(
        config=config, 
        manual_chk_list=manual_chk_list
        )

    target_label = "개인정보" if config['exp']['is_pii'] else "기밀정보"

    for manual_chk_dict in manual_chk_completed_list:
        generation_augmented_sent_dataset_log_scheme = (
            manual_chk_dict['dataset_id'],
            experiment_name,
            manual_chk_dict['generated_sentence'],
            "00", # 증강된 문장은 도메인이 별도로 없으므로 "00"으로 통일
            manual_chk_dict['span_token'],
            target_label,
            manual_chk_dict['validated_label']
        )
        generation_augmented_sent_dataset_log.append( generation_augmented_sent_dataset_log_scheme )

    manual_validation_duration = datetime.now() - manual_validation_start_time

    insert_many_rows(conn=conn, table_name='generation_augmented_sent_dataset_log', data_list=generation_augmented_sent_dataset_log)

    return generation_duration, manual_validation_duration