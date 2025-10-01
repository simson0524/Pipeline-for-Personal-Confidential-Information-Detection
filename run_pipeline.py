# run_pipeline.py

from database.create_dbs import get_connection, create_nessesary_tables
from dataset.pipeline_dataset import PipelineDataset, load_all_json
from classifier.model import Classifier
from model_train.model_train_process_1 import model_train_process_1
from model_validation.model_train_validation_process_2 import model_train_validation_process_2
from dictionary_matching.dictionary_matching_process_3 import dictionary_matching_process_3
from ner_regex_matching.ner_regex_matching_process_4 import ner_regex_matching_process_4
from model_validation.model_validation_process_5 import model_validation_process_5
from generated_augmentation.generated_augmentation_process_6 import generated_augmentation_process_6
from labeling_tools.metric_viewer import metric_viewer
from database.db_logics import *
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import numpy as np
import torch
import yaml


# 실험 Config
config_file_path = "run_config.yaml"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Train_set_{D}
data_dir = config['data']['train_data_dir']

# Train set_{D}에 대한 파이프라인 루프_{L}
PIPELINE_LOOP_L = 0

# DB connection
conn = get_connection(config)

# Create DB tables
create_nessesary_tables(conn)

# 이전 실험명(변화율 관련 지표 계산 시 사용)
previous_experiment_name = "None"


### ------------------------------------------------------------
### 파이프라인
### ------------------------------------------------------------
while True:
    # 루프_{L} 회차 업데이트
    PIPELINE_LOOP_L += 1

    # 현재실험명 : YYMMDD_{당일실험순서}_{L}
    experiment_name = f"{config['exp']['name']}_{(PIPELINE_LOOP_L):02d}"

    # 현재실험 Config
    is_pii        = config['exp']['is_pii']
    batch_size    = config['exp']['batch_size']
    num_epochs    = config['exp']['num_epochs']
    learning_rate = config['exp']['learning_rate']
    device        = config['exp']['device']
    data_dir      = config['data']['train_data_dir']
    if is_pii:
        label_2_id = config['label_mapping']['pii_label_2_id']
        id_2_label = config['label_mapping']['pii_id_2_label']
    else:
        label_2_id = config['label_mapping']['confid_label_2_id']
        id_2_label = config['label_mapping']['confid_id_2_label']

    # 실험시작시각(KST)
    experiment_start_time = datetime.now()



    ### ------------------------------------------------------------
    ### 0. 모델 학습 INPUT DATA
    ### ------------------------------------------------------------
    model_name = config['model']['model_name']
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load All Data
    all_json_data = load_all_json(data_dir)
    all_dataset = PipelineDataset(
        experiment_name=experiment_name,
        json_data=all_json_data,
        tokenizer=tokenizer,
        label_2_id=label_2_id,
        id_2_label=id_2_label,
        sampling_ratio=config['exp']['sampling_ratio'],
        conn=conn,
        is_pii=is_pii
    )

    # K-Fold 준비
    labels = [instance['label'] for instance in all_dataset.instances]
    k_fold = config['exp']['k_fold']
    skf = StratifiedKFold(n_split=k_fold, shuffle=True, random_state=42)
    kfold_splits = list(skf.split(np.zeros(len(all_dataset)), labels)) # Fold 조합별로 어떤 Idx가 학습/검증인지 저장



    ### ------------------------------------------------------------
    ### 1. 모델 학습 & 2. 모델 학습 검증
    ### ------------------------------------------------------------
    model_train_start_time = datetime.now()

    # model_train_performance 테이블에 들어갈 데이터들 
    model_train_performance_log = []

    # 각 fold별 최고성능 epoch
    best_performed_epoch_per_fold = {}

    for fold in range(k_fold):
        print(f"\n===== Fold {fold+1}/{k_fold} 학습 시작 =====")

        train_indices, valid_indices = kfold_splits[fold]

        train_subset = Subset(all_dataset, train_indices)
        valid_subset = Subset(all_dataset, valid_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        print(f"=====[DATA CONFIG INFO]=====\nTrain 데이터: {len(train_subset)}개, Valid 데이터: {len(valid_subset)}개\n")

        # Setting and Print Config Log
        model = AutoModel.from_pretrained(model_name)
        print(f"=====[MODEL CONFIG INFO]=====\n{AutoConfig.from_pretrained(model_name)}\n")
        max_length = 256
        print(f"=====[Train CONFIG INFO]====\nBatch_size : {batch_size}({type(batch_size)})\nNum_epochs : {num_epochs}({type(num_epochs)})\nLearning_rate : {learning_rate}({type(learning_rate)})\nMax_length : {max_length}({type(max_length)})\nDevice : {device}({type(device)})\n")

        if is_pii:
            classifier = Classifier(model=model, num_labels=3).to(device)
        else:
            classifier = Classifier(model=model, num_labels=2).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # 최고성능 저장
        best_f1 = 0
        best_prec = 0
        pest_rec = 0
        best_metric = ''
        best_epoch = 0

        for epoch in range(1, num_epochs+1):
            # 학습 및 검증
            model_train_result = model_train_process_1(
                model=classifier,
                epoch=epoch,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device
            )
            model_train_validation_result = model_train_validation_process_2(
                conn=conn,
                experiment_name=experiment_name,
                model=classifier,
                epoch=epoch,
                dataloader=valid_dataloader,
                device=device,
                label_2_id=label_2_id,
                id_2_label=id_2_label
            )

            print(f"[Train] Loss: {model_train_result['avg_loss']:.4f}")
            print(f"[Valid] Loss: {model_train_validation_result['avg_loss']:.4f} | Precision: {model_train_validation_result['precision']:.4f} | Recall: {model_train_validation_result['recall']:.4f} | F1: {model_train_validation_result['f1']:.4f} | Metric: {model_train_validation_result['metric']}")

            model_path = "None"

            # 최고성능 달성시 저장
            if best_f1 < model_train_validation_result['f1']:
                best_f1 = model_train_validation_result['f1']
                best_prec = model_train_validation_result['precision']
                best_rec = model_train_validation_result['recall']
                best_metric = model_train_validation_result['metric']
                best_epoch = epoch
                print(f"✅ 성능갱신 {model_train_validation_result}")
            
                # 모델 가중치 저장
                save_dir = f'checkpoints/experiment_{experiment_name}'
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"fold{fold}_epoch{epoch}_model.pt")
                torch.save(classifier.state_dict(), model_path)

            # 현재 fold+epoch의 model_train_performance 저장
            model_train_performance_log_scheme = (
                experiment_name,
                epoch,
                fold,
                model_train_result['start_time'],
                model_train_validation_result['end_time'],
                model_path,
                model_train_result['avg_loss'],
                model_train_validation_result['avg_loss'],
                model_train_validation_result['precision'],
                model_train_validation_result['recall'],
                model_train_validation_result['f1'],
                model_train_validation_result['metric']
            )
            model_train_performance_log.append( model_train_performance_log_scheme )

        # 이번 fold의 최고성능 epoch 저장
        best_performed_epoch_per_fold[f'fold{fold}'] = f'epoch{best_epoch}'
    
    # 학습데이터 model_train_performance에 전부 삽입
    insert_many_rows(conn=conn, table_name='model_train_performance', data_list=model_train_performance_log)

    model_train_end_time = datetime.now()
    model_train_duration = model_train_end_time - model_train_start_time



    ### ------------------------------------------------------------
    ### 3. 사전 매칭 검증
    ### ------------------------------------------------------------
    dictionary_matching_start_time = datetime.now()

    all_dataloader_3 = DataLoader(all_dataset, batch_size=1, shuffle=False)

    dictionary_matching_result = dictionary_matching_process_3(
        conn=conn,
        experiment_name=experiment_name,
        dataloader=all_dataloader_3,
        label_2_id=label_2_id,
        id_2_label=id_2_label,
        is_pii=is_pii
    )

    # previous_experiment_name(이전실험)에서 dictionary_matching_performance로그 조회
    _, previous_dictionary_matching_performance = select_specific_row(
        conn=conn,
        table_name='dictionary_matching_performance',
        select_column_name='experiment_name',
        select_value=previous_experiment_name,
        experiment_name=experiment_name,
        return_csv=False
    )

    # 정탐/미탐/오탐 변화율 기록
    epsilon = 0.00001
    if previous_dictionary_matching_performance:
        prev_hit_counts      = previous_dictionary_matching_performance[0][3]
        prev_wrong_counts    = previous_dictionary_matching_performance[0][5]
        prev_mismatch_counts = previous_dictionary_matching_performance[0][7]
        hit_delta_rate_3             = len(dictionary_matching_result['hit']) / (prev_hit_counts+epsilon)
        wrong_delta_rate_3           = len(dictionary_matching_result['wrong']) / (prev_wrong_counts+epsilon)
        mismatch_delta_rate_3        = len(dictionary_matching_result['mismatch']) / (prev_mismatch_counts+epsilon)
    else:
        hit_delta_rate_3             = None
        wrong_delta_rate_3           = None
        mismatch_delta_rate_3        = None

    # 정탐인 것 이후 프로세스에서 보지 않도록 is_valided True로 만들기
    for _, _, _, _, idx, _, _, _, _ in dictionary_matching_result['hit']:
        all_dataset.edit_is_validated(
            idx=idx,
            edit_to=True
        )

    # 4. NER/REGEX 매칭검증 5. 모델검증에서도 정탐 후보인것을 올릴때 table_dict에 먼저 기록하고 한번에 DB에 올릴것
    if is_pii:
        dict_table_name = 'personal_info_dictionary'
    else:
        dict_table_name = 'confidential_info_dictionary'
    table_dict = fetch_table_as_dict(
            conn=conn,
            table_name=dict_table_name,
            key_column='span_token'
        )
    
    # 오탐사항 사전제외하기 (단순히 *_info_dictionary 테이블의 deletion_counts=+1 로 갈음함)
    for wrong_log in dictionary_matching_result['wrong']:
        if wrong_log[3] in table_dict:
            if table_dict[wrong_log[3]]['insertion_counts'] > table_dict[wrong_log[3]]['deletion_counts']:
                table_dict[wrong_log[3]]['deletion_counts'] = table_dict[wrong_log[3]]['insertion_counts']
        
    dictionary_matching_end_time = datetime.now()
    dictionary_matching_duration = dictionary_matching_end_time - dictionary_matching_start_time

    print(f"\n[Metric] 3. 사전매칭검증\n{dictionary_matching_result['metric']}\nRow : label | Column : 정탐/오탐/미탐 순서\n")



    ### ------------------------------------------------------------
    ### 4. NER/REGEX 매칭 검증
    ### ------------------------------------------------------------
    ner_regex_matching_start_time = datetime.now()
    
    all_dataloader_4 = DataLoader(all_dataset, batch_size=1, shuffle=False)

    ner_regex_matching_result = ner_regex_matching_process_4(
        conn=conn,
        experiment_name=experiment_name,
        dataloader=all_dataloader_4,
        label_2_id=label_2_id,
        id_2_label=id_2_label,
        is_pii=is_pii
    )

    # previous_experiment_name(이전실험)에서 ner_regex_matching_performance 조회
    _, previous_ner_regex_matching_performance = select_specific_row(
        conn=conn,
        table_name='ner_regex_matching_performance',
        select_column_name='experiment_name',
        select_value=previous_experiment_name,
        experiment_name=experiment_name,
        return_csv=False
    )

    # 정탐/미탐/오탐 변화율 기록
    epsilon = 0.00001
    if previous_ner_regex_matching_performance:
        prev_hit_counts      = previous_ner_regex_matching_performance[0][3]
        prev_wrong_counts    = previous_ner_regex_matching_performance[0][5]
        prev_mismatch_counts = previous_ner_regex_matching_performance[0][7]
        hit_delta_rate_4             = len(ner_regex_matching_result['hit']) / (prev_hit_counts+epsilon)
        wrong_delta_rate_4           = len(ner_regex_matching_result['wrong']) / (prev_wrong_counts+epsilon)
        mismatch_delta_rate_4        = len(ner_regex_matching_result['mismatch']) / (prev_mismatch_counts+epsilon)
    else:
        hit_delta_rate_4             = None
        wrong_delta_rate_4           = None
        mismatch_delta_rate_4        = None

    # 정탐인 것 이후 프로세스에서 보지 않도록 is_valided True로 만들기
    for _, _, _, _, idx, _, _, _, _ in ner_regex_matching_result['hit']:
        all_dataset.edit_is_validated(
            idx=idx,
            edit_to=True
        )

    ner_regex_matching_end_time = datetime.now()
    ner_regex_matching_duration = ner_regex_matching_end_time - ner_regex_matching_start_time

    print(f"\n[Metric] 4. 사전매칭검증\n{ner_regex_matching_result['metric']}\nRow : label | Column : 정탐/오탐/미탐 순서\n")
    


    ### ------------------------------------------------------------
    ### 5. 모델 검증
    ### ------------------------------------------------------------
    model_validation_start_time = datetime.now()

    # 아직 3.사전매칭검증, 4.NER/REGEX매칭검증 프로세스에서 검증되지 않은 친구들 추출
    is_not_validated_indices = [
        i for i, instance in enumerate(all_dataset.instances)
        if not instance['is_validated']
    ]
    is_not_validated_set = set( is_not_validated_indices )
    print(f"[LOG] 5. 모델검증\n전체 데이터 {len(all_dataset)}개 중, is_validated=False인 데이터는 {len(is_not_validated_set)}개입니다.\n")

    # 각 fold별 model_validation_result를 저장하기 위한 리스트
    model_validation_result_list = []

    # K-Fold 루프 실행
    for fold in range(k_fold):
        print(f"\n===== Fold {fold+1} 처리 시작 =====")

        # 현재 fold의 원본 훈련/검증 인덱스 가져오기
        _, original_val_indices = kfold_splits[fold]

        # original_val_indices와 is_not_validated_indices의 교집합
        filtered_val_indices = [idx for idx in original_val_indices if idx in is_not_validated_set]
        print(f"검증 데이터: {len(filtered_val_indices)}개")

        # Subset + Dataloader 구성
        valid_subset = Subset(all_dataset, filtered_val_indices)
        valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        if is_pii:
            classifier = Classifier(model=model, num_labels=3).to(device)
        else:
            classifier = Classifier(model=model, num_labels=2).to(device)

        model_path = f"checkpoints/experiment_{experiment_name}/fold{fold}_{best_performed_epoch_per_fold[f'fold{fold}']}_model.pt"

        state_dict = torch.load(model_path)

        classifier.load_state_dict(state_dict)

        model_validation_result = model_validation_process_5(
            conn=conn,
            experiment_name=experiment_name,
            model=classifier,
            dataloader=valid_dataloader,
            device=device,
            label_2_id=label_2_id,
            id_2_label=id_2_label
        )

        print(f"[Valid] Loss: {model_validation_result['avg_loss']:.4f} | Precision: {model_validation_result['precision']:.4f} | Recall: {model_validation_result['recall']:.4f} | F1: {model_validation_result['f1']:.4f} | Metric: {model_validation_result['metric']}")

        model_validation_result_list.append( model_validation_result )

    model_validation_end_time = datetime.now()
    model_validation_duration = model_validation_end_time - model_validation_start_time

    ### ------------------------------------------------------------
    ### 사전등재리스트 운용
    ### ------------------------------------------------------------
    
    # 4. NER/REGEX 매칭검증, 5. 모델검증 에서 개인/기밀정보로 정탐인 친구들만 사전등재리스트에 후보로 추림 
    if is_pii:
        dict_label = '개인정보'
        dict_table_name = 'personal_info_dictionary'
    else:
        dict_label = '기밀정보'
        dict_table_name = 'confidential_info_dictionary'

    ner_regex_matching_candidates = fetch_matching_predictions(conn=conn, table_name='ner_regex_matching_sent_dataset_log', experiment_name=experiment_name, dict_label=dict_label)
    model_validation_candidates = fetch_matching_predictions(conn=conn, table_name='model_validation_sent_dataset_log', experiment_name=experiment_name, dict_label=dict_label)
    all_candidates = ner_regex_matching_candidates + model_validation_candidates

    ### 조건만족하는 span_token 모음
    condition_satisfied_span_token_set = set()

    # 조건에 만족하는 후보는 사전에 등재
    for candidate in all_candidates:
        span_token = candidate[3]
        if span_token in condition_satisfied_span_token_set:
            if span_token in table_dict:
                table_dict[span_token]['insertion_counts'] += 1
            else:
                table_dict[span_token] = {
                    'span_token': span_token,
                    'first_insertion_loop': PIPELINE_LOOP_L,
                    'insertion_counts': 1,
                    'deletion_counts': 0
                }

    # 프로세스 이후 사전 크기
    after_process_dictionary_size = 0
    for value in table_dict.values():
        if value['insertion_counts'] > value['deletion_counts']:
            after_process_dictionary_size += 1
        else:
            print(value,'\n')

    

    ### ------------------------------------------------------------
    ### *_performance 테이블 업데이트
    ### ------------------------------------------------------------
    # dictionary_matching_performance 로그
    dictionary_matching_performance_log = [(
        experiment_name,
        dictionary_matching_start_time,
        dictionary_matching_end_time,
        len(dictionary_matching_result['hit']),
        hit_delta_rate_3,
        len(dictionary_matching_result['wrong']),
        wrong_delta_rate_3,
        len(dictionary_matching_result['mismatch']),
        mismatch_delta_rate_3,
        dictionary_matching_result['fetched_dictionary_size'],
        after_process_dictionary_size / dictionary_matching_result['fetched_dictionary_size'],
        dictionary_matching_result['metric']
    )]
    insert_many_rows(conn=conn, table_name='dictionary_matching_performance', data_list=dictionary_matching_performance_log)
    print("dictionary_matching_performance 테이블 업데이트 완료")

    # ner_regex_matching_performance 로그
    ner_regex_matching_performance_log = [(
        experiment_name,
        ner_regex_matching_start_time,
        ner_regex_matching_end_time,
        len(ner_regex_matching_result['hit']),
        hit_delta_rate_4,
        len(ner_regex_matching_result['wrong']),
        wrong_delta_rate_4,
        len(ner_regex_matching_result['mismatch']),
        mismatch_delta_rate_4,
        ner_regex_matching_result['metric']
    )]
    insert_many_rows(conn=conn, table_name='ner_regex_matching_performance', data_list=ner_regex_matching_performance_log)
    print("ner_regex_matching_performance 테이블 업데이트 완료")

    # model_validation_performance 로그
    model_validation_performance_log = [(
        experiment_name,
        model_validation_start_time,
        model_validation_end_time,
        None,
        best_performed_epoch_per_fold,
        model_validation_result['precision'],
        model_validation_result['recall'],
        model_validation_result['f1'],
        model_validation_result['metric']
    )]
    insert_many_rows(conn=conn, table_name='model_validation_performance', data_list=model_validation_performance_log)
    print("model_validation_performance 테이블 업데이트 완료")


    
    ### ------------------------------------------------------------
    ### [HUMAN 개입] 지표 확인 후 종료 설정
    ### ------------------------------------------------------------
    want_to_stop = metric_viewer(config=config, experiment_name=experiment_name, is_pii=is_pii)
    if want_to_stop:
        print("[파이프라인 종료] train_set_{D+1}로 다음 파이프라인을 진행하세요.")
        break



    ### ------------------------------------------------------------
    ### 6. 문서 증강 + [HUMAN 개입] 수동 검증
    ### ------------------------------------------------------------
    generated_augmentation_process_6(conn, config)



    ### ------------------------------------------------------------
    ### 루프_L 마무리
    ### ------------------------------------------------------------
    experiment_end_time = datetime.now()

    