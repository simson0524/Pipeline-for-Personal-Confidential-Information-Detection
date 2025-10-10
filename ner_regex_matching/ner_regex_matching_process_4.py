# ner_regex_matching/ner_regex_matching_process_4.py

from database.db_logics import *
from ner_regex_matching.regex_logics.regex_main import run_regex_detection
from ner_regex_matching.ner_logics.ner_main import run_ner_detection
from datetime import datetime
from tqdm.auto import tqdm


def ner_regex_matching_process_4(conn, experiment_name, dataloader, label_2_id, id_2_label, is_pii=True):
    # 4. NER/REGEX매칭검증 시작시간
    start_time = datetime.now()

    # 정탐, 오탐, 미탐 로그를 담을 리스트
    hit, wrong, mismatch = [], [], []

    # column_name = [정탐, 오탐, 미탐]
    # row_name = label_id of values
    metric = [ [0, 0, 0] for _ in range(len(label_2_id)) ]

    for label, _ in label_2_id.items():
        # 모델별 가능한 라벨 솎아내기
        if label == "개인정보" and is_pii==True:
            print("4. NER/REGEX매칭검증(개인정보) 진행")
        elif label == "기밀정보" and is_pii==False:
            print("4. NER/REGEX매칭검증(기밀정보) 진행")
        else:
            continue

        for batch in tqdm(dataloader, desc=f"4. NER/REGEX매칭검증({label})"):
            batch_size = len( batch['sentence'] )
            for i in range( batch_size ):
                curr_is_validated = batch['is_validated'][i]
                curr_sentence_id = batch['sentence_id'][i]
                curr_sentence = batch['sentence'][i]
                curr_span_token = batch['span_token'][i]
                curr_dataset_idx = batch['idx'][i].item()
                curr_gt_label_id = batch['label'][i].item()
                curr_pred_label_id = label_2_id[label]
                curr_file_name = batch['file_name'][i]
                curr_sent_seq = batch['sentence_seq'][i]

                # ner_regex_matching_sent_dataset_log 테이블에 들어가는 scheme
                curr_data_log = (
                    experiment_name,
                    curr_sentence_id,
                    curr_sentence,
                    curr_span_token,
                    curr_dataset_idx,
                    id_2_label[curr_gt_label_id],
                    id_2_label[curr_pred_label_id],
                    curr_file_name,
                    curr_sent_seq
                )

                # 앞 단계에서 이미 검증된 경우 건너뜀
                # print(type(curr_is_validated), curr_is_validated) # 로그용 : 지금 is_validated가 반영이 되는지 안되는지를 모르겠음
                if curr_is_validated:
                    continue
                
                # 정탐인 경우 - REGEX 추출
                regex_texts = run_regex_detection(curr_sentence)
                regex_spans = {regex_dict['단어'] for regex_dict in regex_texts}

                # REGEX 매칭되는 것이 있다면
                if (curr_span_token in regex_spans) and (label_2_id[label] == curr_gt_label_id):
                    hit.append( curr_data_log )
                    metric[ curr_pred_label_id ][ 0 ] += 1
                    continue

                # 정탐인 경우 - NER 추출
                ner_texts = run_ner_detection(curr_sentence)
                ner_spans = {ner_dict['단어'] for ner_dict in ner_texts}

                # NER 매칭되는 것이 있다면
                if (curr_span_token in ner_spans) and (curr_pred_label_id == curr_gt_label_id):
                    hit.append( curr_data_log )  
                    metric[ curr_pred_label_id ][ 0 ] += 1
                    continue

                # 오탐인 경우
                if ((curr_span_token in regex_spans) or (curr_span_token in ner_spans)) and (curr_pred_label_id != curr_gt_label_id):
                    wrong.append( curr_data_log )
                    metric[ curr_pred_label_id ][ 1 ] += 1
                    continue 

                # 미탐인 경우
                if (curr_pred_label_id == curr_gt_label_id) and (curr_span_token not in regex_spans) and (curr_span_token not in ner_spans):
                    mismatch.append( curr_data_log )
                    metric[ curr_pred_label_id ][ 2 ] += 1
                    continue             
    
    # DB(ner_regex_matching_sent_dataset_log)에 정보 추가하기
    insert_many_rows(conn, "ner_regex_matching_sent_dataset_log", hit)
    insert_many_rows(conn, "ner_regex_matching_sent_dataset_log", wrong)
    insert_many_rows(conn, "ner_regex_matching_sent_dataset_log", mismatch)
    
    # 4. NER/REGEX매칭검증 종료시간
    end_time = datetime.now()

    # 4. NER/REGEX매칭검증 소요시간
    duration = end_time - start_time

    # return metric, hit, wrong, mismatch, start_time, end_time, duration
    return {
        'metric': metric,
        'hit': hit,
        'wrong': wrong,
        'mismatch': mismatch,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration
    }