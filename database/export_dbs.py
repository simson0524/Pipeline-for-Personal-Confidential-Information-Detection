# database/export_dbs.py

from database.db_logics import *


def find_specific_experiment_name_all_logs(conn, experiment_name):
    # 아래는 모두 각 테이블의 experiment_name 컬럼의 값이 "experiment_name"인 로그들입니다. 
    
    # 실험의 총 개요 결과
    experiment = select_specific_row(
        conn=conn, 
        table_name='experiment', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )

    # 2. 모델학습검증
    model_train_performance = select_specific_row(
        conn=conn, 
        table_name='model_train_performance', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    model_train_sent_dataset_log = select_specific_row(
        conn=conn, 
        table_name='model_train_sent_dataset_log', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    
    # 3. 사전매칭검증
    dictionary_matching_performance = select_specific_row(
        conn=conn, 
        table_name='dictionary_matching_performance', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    dictionary_matching_sent_dataset_log = select_specific_row(
        conn=conn, 
        table_name='dictionary_matching_sent_dataset_log', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )

    # 4. NER/REGEX매칭검증
    ner_regex_matching_performance = select_specific_row(
        conn=conn, 
        table_name='ner_regex_matching_performance', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    ner_regex_matching_sent_dataset_log = select_specific_row(
        conn=conn, 
        table_name='ner_regex_matching_sent_dataset_log', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    
    # 5. 모델검증
    model_validation_performance = select_specific_row(
        conn=conn, 
        table_name='model_validation_performance', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    model_validation_sent_dataset_log = select_specific_row(
        conn=conn, 
        table_name='model_validation_sent_dataset_log', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    
    # 6. 모델학습검증
    generation_augmented_performance = select_specific_row(
        conn=conn, 
        table_name='generation_augmented_performance', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    generation_augmented_sent_dataset_log = select_specific_row(
        conn=conn, 
        table_name='generation_augmented_sent_dataset_log', 
        select_column_name="experiment_name", 
        select_value=experiment_name,
        experiment_name=experiment_name
        )
    

def result_with_customed_query(conn, query):
    """커스텀한 쿼리를 바탕으로 결과를 내주는 함수

    Args:
        conn: database connection
        query: It must contain ';' on its tail.
    """
    cursor = conn.cursor()

    # execute customed query
    try:
        cursor.execute( query )
        result = cursor.fetchall()
    except Exception as e:
        print(f"Exception raised -> {e}")
        print("쿼리를 실행하지 못했습니다\n")

    return result