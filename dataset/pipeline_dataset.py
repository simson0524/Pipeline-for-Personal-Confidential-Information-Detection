# dataset/pipeline_dataset.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.create_dbs import *
from database.db_logics import *
from torch.utils.data import Dataset
from konlpy.tag import Okt
from tqdm.auto import tqdm
import pandas as pd
import random
import torch
import json
import os

class PipelineDataset(Dataset):
    def __init__(self, experiment_name, json_data, tokenizer, label_2_id, id_2_label, sampling_ratio, conn, is_pii=True, max_length=256):
        self.samples         = {data['id']:data for data in json_data['data']}
        self.annotations     = {data[0]['id']:data[0]['annotations'] for data in json_data['annotations']}
        self.tokenizer       = tokenizer
        self.label_2_id      = label_2_id
        self.id_2_label      = id_2_label
        self.max_length      = max_length
        self.experiment_name = experiment_name
        self.instances       = []
        self.okt             = Okt()
        self.is_pii          = is_pii
        self.conn            = conn
        self._create_instances()
        self._instance_post_processing(ratio=sampling_ratio)

    def _create_instances(self):
        def find_char_idx_in_sentence(str_find_start_idx, sentence, span_token):
            char_start = sentence.find(span_token, str_find_start_idx)

            if char_start == -1: # 만약 문장 내 span_token 없다면
                return None, None, str_find_start_idx
            else: # 만약 문장 내 span_token 있다면
                char_end = char_start + len(span_token)
                str_find_start_idx = char_end
                return char_start, char_end, str_find_start_idx
            

        def return_char_idx_to_token_idx(char_start, char_end, offset_mapping):
            token_start, token_end = None, None
            
            # offset mapping을 순회하며 각 토큰별 idx와 char_start, char_end를 추출
            for i, (offset_char_start, offset_char_end) in enumerate(offset_mapping):
                if offset_char_start <= char_start < offset_char_end:
                    token_start = i
                if offset_char_start < char_end <= offset_char_end:
                    token_end = i+1
            
            return token_start, token_end
        
        # 로그용
        no_span_skipped = 0
        span_truncated_sent_skipped = 0
        samples_num = [0 for _ in range( len(self.label_2_id) )]

        # self.samples에 접근하여 각 문장데이터별로 span후보 모두 추출
        for id, sample_data in tqdm(self.samples.items(), desc="Create Trainset instances"):
            sentence = sample_data['sentence']
            sentence_id = sample_data['id']
            sentence_seq = sample_data['sequence']
            filename = sample_data['filename']

            # 토큰화(tokenizing)
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
            input_ids = encoding['input_ids']
            decoded_input_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]

            # 정답지에 있는 친구들 먼저 추출하고 남은 친구 중에 Okt로 추출해야하므로 추출 여부 확인용 리스트
            special_ids = set(self.tokenizer.all_special_ids)
            is_extracted = [ token_id in special_ids for token_id in input_ids ]

            # 1. annotations에 있는 친구들 기준으로 우선 추출
            span_tokens = []
            for annotations_dict in self.annotations[id]:
                span_tokens.append( (annotations_dict['span_text'], annotations_dict['label']) ) # 아직 원본 테이터는 span_token으로 변경 안됨
            

            # 1-1. annotations에서 추출한 친구들을 인스턴스 추출
            str_find_start_idx = 0
            for span_token, label in span_tokens:
                if self.is_pii:
                    if label == "기밀정보":
                        print("기밀정보 탐지. 사용 불가능 라벨.")
                        continue
                if not self.is_pii:
                    if label == "준식별자" or label == "개인정보":
                        print("개인정보/준식별자 탐지. 사용 불가능 라벨.")
                        continue
                if label not in self.label_2_id:
                    print(f"사용불가 ({span_token}, {label})")
                    continue

                # sentence내 span_token의 char idx(start, end)를 추출
                char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                    str_find_start_idx=str_find_start_idx,
                    sentence=sentence,
                    span_token=span_token
                )

                # 문장 내 span_token가 없으므로 진행불가(LOG as "no_span_skipped")
                if char_start == None:
                    no_span_skipped += 1
                    continue

                # char idx(start, end)를 token idx(start, end)로 변환
                token_start, token_end = return_char_idx_to_token_idx(
                    char_start=char_start,
                    char_end=char_end,
                    offset_mapping=offset_mapping
                    )
                
                # "max_length truncation"으로 인해 해당 span_token가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                if (token_start is None) or (token_end is None):
                    span_truncated_sent_skipped += 1
                    continue

                # 찾은 token idx(start, end)로 해당 span_token 토큰 정보 확인(token_id 및 decoded_token_id)
                span_ids = input_ids[token_start:token_end]
                decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                # 해당 span_token label
                label_id = self.label_2_id[label]
                samples_num[label_id] += 1

                # 2번 프로세스(Okt 기반 span_token 추출)에서 중복 추출 방지를 위한 token flag 설정
                for token_idx in range(token_start, token_end):
                    is_extracted[token_idx] = True

                # 인스턴스 추가
                self.instances.append({
                        "sentence": sentence,
                        "sentence_id": sentence_id,
                        "sentence_seq": sentence_seq,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_token": span_token,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids,
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "is_validated": False,
                        "file_name": filename
                    })



            # 2. okt를 이용한 Span후보가 될 수 있는 POS tag 목록(only for "일반정보")
            target_pos = {"Noun", "Number", "Email", "URL", "Foreign", "Alpha"}
            POS_in_sentence = self.okt.pos( sentence )

            str_find_start_idx = 0

            # 다음 loop에서 사용할 이전 loop 정보
            PRE_LOOP_SPAN_TOKEN = ''
            PRE_LOOP_SPAN_POS = '' 
            PRE_LOOP_TOKEN_START = None
            PRE_LOOP_TOKEN_END = None
            PRE_LOOP_SPAN_IDs = []
            PRE_LOOP_DECODED_SPAN_IDs = []

            for span_token, pos in POS_in_sentence:
                # 현재 loop에서 사용할 현재 loop정보
                CURR_LOOP_SPAN_TOKEN = span_token
                CURR_LOOP_SPAN_POS = pos
                
                if CURR_LOOP_SPAN_POS in target_pos:
                    # sentence내 span_token의 char idx(start, end)를 추출
                    char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                        str_find_start_idx=str_find_start_idx,
                        sentence=sentence,
                        span_token=CURR_LOOP_SPAN_TOKEN
                    )

                    # 문장 내 span_token가 없으므로 진행불가(LOG as "no_span_skipped")
                    if char_start == None:
                        no_span_skipped += 1
                        continue

                    # char idx(start, end)를 token idx(start, end)로 변환
                    CURR_LOOP_TOKEN_START, CURR_LOOP_TOKEN_END = return_char_idx_to_token_idx(
                        char_start=char_start,
                        char_end=char_end,
                        offset_mapping=offset_mapping
                        )

                    # "max_length truncation"으로 인해 해당 span_token가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                    if (CURR_LOOP_TOKEN_START is None) or (CURR_LOOP_TOKEN_END is None):
                        span_truncated_sent_skipped += 1
                        continue

                    # 찾은 token idx(start, end)로 해당 span_token 토큰 정보 확인(token_id 및 decoded_token_id)
                    CURR_LOOP_SPAN_IDs = input_ids[CURR_LOOP_TOKEN_START:CURR_LOOP_TOKEN_END]
                    CURR_LOOP_DECODED_SPAN_IDs = self.tokenizer.convert_ids_to_tokens(CURR_LOOP_SPAN_IDs)

                    # 만약 1번 프로세스(정답지 기반 span_token 추출)에서 해당 token이 이미 추출된 경우 건너띔
                    if is_extracted[CURR_LOOP_TOKEN_START] == True and is_extracted[CURR_LOOP_TOKEN_END-1] == True:
                        continue

                    # 만약 CURR_LOOP_DECODED_SPAN_IDs[0]이 ## prefix로 시작하는 경우, pre_loop_token과 합쳐야 하므로
                    if CURR_LOOP_DECODED_SPAN_IDs[0].startswith("##") and PRE_LOOP_TOKEN_END == CURR_LOOP_TOKEN_START:
                        CURR_LOOP_SPAN_TOKEN = PRE_LOOP_SPAN_TOKEN + CURR_LOOP_SPAN_TOKEN
                        CURR_LOOP_SPAN_IDs = PRE_LOOP_SPAN_IDs + CURR_LOOP_SPAN_IDs
                        CURR_LOOP_DECODED_SPAN_IDs = PRE_LOOP_DECODED_SPAN_IDs + CURR_LOOP_DECODED_SPAN_IDs
                        CURR_LOOP_TOKEN_START = PRE_LOOP_TOKEN_START

                    samples_num[0] += 1

                    self.instances.append({
                        "sentence": sentence,
                        "sentence_id": sentence_id,
                        "sentence_seq": sentence_seq,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_token": CURR_LOOP_SPAN_TOKEN,
                        "span_ids": CURR_LOOP_SPAN_IDs,
                        "decoded_span_ids": CURR_LOOP_DECODED_SPAN_IDs,
                        "token_start": CURR_LOOP_TOKEN_START,
                        "token_end": CURR_LOOP_TOKEN_END,
                        "label": self.label_2_id["일반정보"],
                        "is_validated": False,
                        "file_name": filename
                    })
                    
                    # 현재 loop의 정보를 저장
                    PRE_LOOP_SPAN_TOKEN = CURR_LOOP_SPAN_TOKEN
                    PRE_LOOP_SPAN_POS = CURR_LOOP_SPAN_POS
                    PRE_LOOP_TOKEN_START = CURR_LOOP_TOKEN_START
                    PRE_LOOP_TOKEN_END = CURR_LOOP_TOKEN_END
                    PRE_LOOP_SPAN_IDs = CURR_LOOP_SPAN_IDs
                    PRE_LOOP_DECODED_SPAN_IDs = CURR_LOOP_DECODED_SPAN_IDs
                    
            
        print(f"[Total instances]\nLabel & ids : {self.label_2_id}\nnums : {samples_num}\ntruncated sents : {span_truncated_sent_skipped}\n")
        
    
    def _instance_post_processing(self, ratio=1.0):
        # 기존 self.instance에 최종 결과를 저장하기 위해 memory 재할당
        instances = self.instances

        zero_instances = []
        non_zero_instance = []

        # 로그용
        skipped = 0
        total_instances = 0
        samples_num = [0 for _ in range( len(self.label_2_id) )]


        # instance별 순회
        for i in range( len(instances)-1 ):
            # 만약 현재 instance가 다음 instance의 subset일 경우 제외(최종만 포함하면 되므로)
            if (instances[i]["sentence"] == instances[i+1]["sentence"]) and (instances[i]["token_start"] == instances[i+1]["token_start"]) and (instances[i]["token_end"] <= instances[i+1]["token_end"]):
                skipped += 1
                continue
            
            # 만약 현재 instance의 "일반정보" 인스턴스의 토큰길이가 1인 경우, 굳이 데이터셋 포함하지 않고 제외하여 label imbalace 최소화
            if (instances[i]['label'] == 0) and (instances[i]['token_end'] - instances[i]['token_start'] <= 1):
                skipped += 1
                continue
            
            curr_instance = instances[i]
            total_instances += 1
            samples_num[curr_instance['label']] += 1

            if curr_instance['label'] == 0:
                zero_instances.append( curr_instance )
            else:
                non_zero_instance.append( curr_instance )
        
        # label:0에 대한 Downsampling target 개수 설정
        sampling_target_num = min(int(max(samples_num[1:])*ratio), samples_num[0]) if len(samples_num) > 1 else 0
        
        # label:0에 대한 Downsampling
        zero_instances = random.sample( zero_instances, sampling_target_num )
        
        # 제외된 인스턴스 수
        gap = samples_num[0] - len(zero_instances)

        # Downsampling된 숫자로 갱신
        samples_num[0] = len(zero_instances)

        # 데이터 재할당 및 shuffle
        self.instances = zero_instances + non_zero_instance
        random.shuffle( self.instances )


        # 후처리 결과
        print(f"[instance 후처리 완료 \n건너뛴 instance 수 : {skipped}\n사용되는 instance 수 : {total_instances}\n제외된 instance 수 : {gap}\nLabel & ids : {self.label_2_id}\nnums : {samples_num}\n\n\n\n")

    def edit_is_validated(self, idx, edit_to):
        self.instances[idx]["is_validated"] = edit_to

    def __getitem__(self, idx):
        item = self.instances[idx]
        return {
            "idx": torch.tensor(idx),
            "sentence": item['sentence'],
            "sentence_id": item['sentence_id'],
            "sentence_seq": item['sentence_seq'],
            "input_ids": torch.tensor(item["input_ids"]),
            "decoded_input_ids": item["decoded_input_ids"],
            "attention_mask": torch.tensor(item["attention_mask"]),
            "span_token": item["span_token"],
            "token_start": torch.tensor(item["token_start"]),
            "token_end": torch.tensor(item["token_end"]),
            "label": torch.tensor(item["label"]),
            "is_validated": item["is_validated"],
            "file_name": item["file_name"]
        }
    
    def __len__(self):
        return len(self.instances)


def load_all_json(json_dir="data/train_set_1"):
    all_data = {"data": [], 'annotations': []}
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(json_dir, file_name), "r", encoding='utf-8') as f:
                json_file = json.load(f)
                all_data["data"].append( json_file["data"] )
                all_data["annotations"].append( json_file['annotations'] )
                    
    return all_data