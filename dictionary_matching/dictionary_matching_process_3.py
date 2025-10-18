# dictionary_matching/dictionary_matching_process_3.py

from database.db_logics import *
from datetime import datetime
from tqdm.auto import tqdm


def dictionary_matching_process_3(conn, experiment_name, dataloader, label_2_id, id_2_label, is_pii=True):
    # 3. 사전매칭검증 시작시간
    start_time = datetime.now()

    # 정탐, 오탐, 미탐 로그를 담을 리스트
    hit, wrong, mismatch = [], [], []

    # column_name = [정탐, 오탐, 미탐]
    # row_name = label_id
    metric = [ [0, 0, 0] for _ in range(len(label_2_id)) ]
    
    for label, _ in label_2_id.items():
        # 모델별 가능한 라벨 솎아내기
        if label == "개인정보" and is_pii==True:
            print("3. 사전매칭검증(개인정보) 진행")
            dict_table_name = 'personal_info_dictionary'
        elif label == "기밀정보" and is_pii==False:
            print("3. 사전매칭검증(기밀정보) 진행")
            dict_table_name = 'confidential_info_dictionary'
        else:
            continue

        # 데이터베이스 내용 미리 dict으로 추출하여 Hash로 비교
        table_dict = fetch_table_as_dict(
            conn=conn,
            table_name=dict_table_name,
            key_column='span_token'
        )

        # 3. 사전매칭검증 시작 전 사전의 크기(로그로 사용), insertion == deletion이 같으면 최종적으로 deletion되었다는 의미
        fetched_dictionary_size = 0
        for value in table_dict.values():
            if value['insertion_counts'] > value['deletion_counts']:
                fetched_dictionary_size += 1
            else:
                print(value,'\n')

        for batch in tqdm(dataloader, desc=f"3. 사전매칭검증 프로세스({label})"):
            batch_size = len( batch['sentence'] )
            for i in range( batch_size ):
                curr_sentence_id = batch['sentence_id'][i]
                curr_sentence = batch['sentence'][i]
                curr_span_token = batch['span_token'][i]
                curr_dataset_idx = batch['idx'][i].item()
                curr_gt_label_id = batch['label'][i].item()
                curr_pred_label_id = label_2_id[label]
                curr_file_name = batch['file_name'][i]
                curr_sent_seq = batch['sentence_seq'][i]

                # dictionary_matching_sent_dataset_log 테이블에 들어가는 scheme
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

                # 정탐인 경우
                if (curr_pred_label_id == curr_gt_label_id) and (curr_span_token in table_dict):
                    hit.append( curr_data_log )
                    metric[curr_pred_label_id][ 0 ] += 1
                    continue
                
                # 오탐인 경우
                if (curr_span_token in table_dict) and (curr_pred_label_id != curr_gt_label_id):
                    wrong.append( curr_data_log )
                    metric[curr_pred_label_id][ 1 ] += 1
                    continue 

                # 미탐인 경우
                if (curr_span_token not in table_dict) and (curr_pred_label_id == curr_gt_label_id):
                    mismatch.append( curr_data_log )
                    metric[curr_pred_label_id][ 2 ] += 1
                    continue

    # DB(dictionary_matching_sent_dataset_log)에 정보 추가하기
    insert_many_rows(conn, "dictionary_matching_sent_dataset_log", hit)
    insert_many_rows(conn, "dictionary_matching_sent_dataset_log", wrong)
    insert_many_rows(conn, "dictionary_matching_sent_dataset_log", mismatch)

    '''
    이거 사전 업데이트하는거는 3, 4, 5 프로세스 종료된 이후 run_pipeline에서 구현되어야 함. 아니면 아래에 함수로 할까?
    '''
    # # 오탐사항 사전제외하기 (단순히 *_info_dictionary 테이블의 deletion_counts=+1 로 갈음함)
    # for wrong_log in wrong:
    #     if wrong_log[3] in table_dict:
    #         if table_dict[wrong_log[3]]['insertion_counts'] > table_dict[wrong_log[3]]['deletion_counts']:
    #             table_dict[wrong_log[3]]['deletion_counts'] += 1

    # # 사전 전부 삭제(DB 연산시간 줄이기 위해 모두 삭제 후 업데이트 내용(table_dict의 value)을 한번에 업로드하기 위함)
    # delete_all_rows(conn, dict_table_name, True)

    # # 업데이트 내용 사전에 전부 추가
    

    # 3. 사전매칭검증 종료시간
    end_time = datetime.now()

    # 3. 사전매칭검증 소요시간
    duration = end_time - start_time

    return {
        'metric': metric,
        'hit': hit,
        'wrong': wrong,
        'mismatch': mismatch,
        'fetched_dictionary_size': fetched_dictionary_size + 0.00000001,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration
    }