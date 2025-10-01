# database/db_logics.py

from psycopg2 import sql
import pandas as pd
import psycopg2
import os


def get_column_names(cursor, table_name):
    cursor.execute(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = '{table_name}'
        ORDER BY ordinal_position;
    """)
    return [row[0] for row in cursor.fetchall()]


def select_specific_row(conn, table_name, select_column_name, select_value, experiment_name=None, return_csv=True):
    """
    선택한 열에서 특정 value를 기준으로 특정 행을 조회합니다.
    - conn: psycopg2.connect()로 생성된 커넥션 객체
    - table_name: 조회할 테이블명
    - select_column_name: Primary Key 컬럼명
    - select_value: 조회할 행을 식별하는 PK 값
    @return: 조회된 행 데이터 (tuple) 또는 None (데이터가 없을 경우)
    """
    column_names = []
    result = []
    with conn.cursor() as cur:
        try:
            # SQL Injection을 방지하며 동적 쿼리 생성
            query = psycopg2.sql.SQL("SELECT * FROM {table} WHERE {pk_col} = %s").format(
                table=psycopg2.sql.Identifier(table_name),
                pk_col=psycopg2.sql.Identifier(select_column_name)
            )
            
            # 쿼리 실행
            cur.execute(query, (select_value,))
            
            # 결과 가져오기
            result = cur.fetchall()

            # 컬럼명 가져오기
            if cur.description:
                column_names = [desc[0] for desc in cur.description]


        except (Exception, psycopg2.Error) as error:
            print(f"데이터 조회 중 오류 발생: {error}")
            return None, None
    
    if return_csv:
        if not result:
            print(f"조회된 데이터가 없습니다\ntable_name : {table_name}\nselected_column : {select_column_name}\nselected_value : {select_value}")
            return None, None
        try:
            # 경로 설정
            output_dir = 'data/process_log'
            file_name = f"{experiment_name}_{table_name}_log.csv"
            full_path = os.path.join(output_dir, file_name)

            # csv로 만들기
            df = pd.DataFrame(result, columns=column_names)
            df.to_csv(full_path, index=False, encoding='utf-8-sig')
            print(f"✅ 데이터가 '{full_path}' 경로에 성공적으로 저장되었습니다.")
        except Exception as e:
            print(f"DataFrame 생성 또는 CSV 저장 중 오류 발생: {e}")
    
    return column_names, result


def fetch_table_as_dict(conn, table_name, key_column):
    nested_dict = {}
    with conn.cursor() as cur:
        try:
            # 1. 테이블 전체 데이터를 조회하는 쿼리 실행
            # psycopg2.sql을 사용하여 테이블 이름을 안전하게 처리
            query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
            cur.execute(query)

            # 2. 쿼리 결과로부터 컬럼 이름 목록 가져오기
            if not cur.description:
                print(f"'{table_name}' 테이블에 데이터가 없거나 테이블이 존재하지 않습니다.")
                return {}
            column_names = [desc[0] for desc in cur.description]
            
            # 3. 모든 행(row)을 가져와서 딕셔너리 생성
            for row in cur.fetchall():
                # zip을 이용해 {컬럼이름: 값} 형태의 딕셔너리로 변환
                row_dict = dict(zip(column_names, row))
                
                # key_column의 값을 최상위 딕셔너리의 Key로 사용
                dict_key = row_dict.get(key_column)
                if dict_key is not None:
                    nested_dict[dict_key] = row_dict
                else:
                    print(f"경고: 행에 Key로 사용할 '{key_column}' 컬럼이 없습니다.")

        except (Exception, psycopg2.Error) as error:
            print(f"'{table_name}' 테이블 조회 중 오류 발생: {error}")

    return nested_dict


def fetch_generation_candidates(conn, experiment_name):
        # GT와 Prediction이 다른 데이터를 조회하는 SQL 쿼리
    query = """
        SELECT *
        FROM model_validation_sent_dataset_log
        WHERE experiment_name = %s
          AND ground_truth <> prediction;
    """
    
    results = []
    try:
        # DB 작업을 위한 cursor 생성
        with conn.cursor() as cur:
            # cur.execute()로 쿼리 실행
            cur.execute(query, (experiment_name,))
            
            # cur.fetchall()을 사용해 모든 결과를 List(Tuple) 형태로 가져옴
            results = cur.fetchall()
            print(f"✅ '{experiment_name}' 실험의 불일치 데이터 조회를 완료했습니다.")
            
    except Exception as e:
        print(f"데이터 조회 중 오류가 발생했습니다: {e}")
        # 오류 발생 시 빈 리스트 반환
    
    return results


def fetch_matching_predictions(conn, table_name, experiment_name, dict_label):
    """
    지정된 테이블과 experiment에서 ground_truth와 prediction이 
    모두 'dict_label'과 일치하는 데이터를 List(Tuple)로 조회합니다.

    Args:
        conn: psycopg2 커넥션 객체
        table_name (str): 조회할 테이블 이름
        experiment_name (str): 조회할 실험 이름
        dict_label (str): 찾고자 하는 라벨 이름

    Returns:
        list: 각 행이 튜플로 구성된 리스트
    """
    
    # psycopg2.sql을 사용하여 SQL 구문을 안전하게 동적으로 생성
    query = sql.SQL("""
        SELECT *
        FROM {table}
        WHERE experiment_name = %s
          AND ground_truth = %s
          AND prediction = %s;
    """).format(
        table=sql.Identifier(table_name) # 테이블 이름은 Identifier로 감싸줌
    )
    
    results = []
    try:
        with conn.cursor() as cur:
            params = (experiment_name, dict_label, dict_label)
            cur.execute(query, params)
            results = cur.fetchall()
            print(f"✅ '{table_name}' 테이블의 '{experiment_name}'에서 GT/PRED가 '{dict_label}'인 데이터 조회를 완료했습니다.")
            
    except Exception as e:
        print(f"데이터 조회 중 오류가 발생했습니다: {e}")
    
    return results




def insert_many_rows(conn, table_name, data_list):
    if not data_list:
        print("삽입할 데이터가 없습니다.")
        return
    
    with conn.cursor() as cur:
        try:
            # 컬럼명 & 컬럼개수
            column_names = get_column_names(cur, table_name)
            column_counts = len(column_names)

            # 유효성 검사
            first_row_len = len(data_list[0])
            if column_counts != first_row_len:
                raise ValueError(f"테이블 '{table_name}'의 컬럼 개수({column_counts}개)와 "
                                 f"입력 데이터의 개수({first_row_len}개)가 일치하지 않습니다.")

            # 데이터 삽입 Query
            cols_str = ", ".join(column_names)
            query = f"INSERT INTO {table_name} ({cols_str}) VALUES %s"

            # 삽입
            psycopg2.extras.execute_values(cur, query, data_list)

            conn.commit()
            print(f"{len(data_list)}개의 행이 '{table_name}' 테이블에 성공적으로 삽입되었습니다.")

        except (Exception, psycopg2.Error) as error:
            conn.rollback()
            print(f"데이터 삽입 중 오류 발생: {error}")


def update_specific_row(conn, table_name, pk_column_name, pk_value, target_column, new_value):
    """
    PK를 기준으로 특정 행의 지정된 열 값을 수정합니다.
    - conn: psycopg2.connect()로 생성된 커넥션 객체
    - table_name: 수정할 테이블명 (str)
    - pk_column_name: Primary Key 컬럼명 (str)
    - pk_value: 수정할 행을 식별하는 PK 값
    - target_column: 수정할 대상 열의 이름 (str)
    - new_value: 새로 업데이트할 값
    """
    with conn.cursor() as cur:
        try:
            # 1. 동적 SQL 쿼리 생성
            # SQL Injection 방지를 위해 컬럼명은 AsIs로, 값은 %s 플레이스홀더로 처리
            query = psycopg2.sql.SQL("UPDATE {table} SET {column} = %s WHERE {pk_col} = %s").format(
                table=psycopg2.sql.Identifier(table_name),
                column=psycopg2.sql.Identifier(target_column),
                pk_col=psycopg2.sql.Identifier(pk_column_name)
            )
            
            # 2. 쿼리 실행
            cur.execute(query, (new_value, pk_value))
            
            # 3. 수정된 행이 있는지 확인
            if cur.rowcount == 0:
                print(f"'{table_name}' 테이블에 {pk_column_name} = {pk_value}인 행이 없어 업데이트하지 못했습니다.")
            else:
                conn.commit()
                print(f"'{table_name}' 테이블에서 {pk_column_name} = {pk_value}인 행의 "
                      f"'{target_column}' 값이 성공적으로 업데이트되었습니다.")

        except (Exception, psycopg2.Error) as error:
            conn.rollback()
            print(f"데이터 수정 중 오류 발생: {error}")    


def delete_specific_row(conn, table_name, delete_column_name, delete_value):
    with conn.cursor() as cur:
        try:
            # SQL Injection 방지를 위해 동적 쿼리 생성
            query = psycopg2.sql.SQL("DELETE FROM {table} WHERE {pk_col} = %s").format(
                table=psycopg2.sql.Identifier(table_name),
                pk_col=psycopg2.sql.Identifier(delete_column_name)
            )

            # 쿼리 실행
            cur.execute(query, (delete_value,))

            # 삭제된 행이 있는지 확인
            if cur.rowcount == 0:
                print(f"'{table_name}' 테이블에 {delete_column_name} = {delete_value}인 행이 없어 삭제하지 못했습니다.")
            else:
                conn.commit()
                print(f"'{table_name}' 테이블에서 {delete_column_name} = {delete_value}인 행({cur.rowcount})이 성공적으로 삭제되었습니다.")

        except (Exception, psycopg2.Error) as error:
            conn.rollback()
            print(f"데이터 삭제 중 오류 발생: {error}")


def delete_all_rows(conn, table_name, confirm=False):
    """
    테이블의 모든 행을 삭제합니다 (TRUNCATE). 
    !!! 매우 위험한 함수이므로, 호출하는 쪽에서 반드시 확인 절차를 거쳐야 합니다. !!!
    - conn: psycopg2.connect()로 생성된 커넥션 객체
    - table_name: 모든 데이터를 삭제할 테이블명 (str)
    - confirm: True로 설정해야만 함수가 실행됩니다 (필수 안전장치).
    """
    if not confirm:
        # 함수를 호출한 쪽에서 confirm=True를 명시적으로 넘기지 않으면 절대 실행되지 않음
        print(f"'{table_name}' 테이블 전체 삭제가 취소되었습니다. "
              "실행하려면 `confirm=True` 인자를 명시적으로 전달해야 합니다.")
        return

    with conn.cursor() as cur:
        try:
            query = psycopg2.sql.SQL("TRUNCATE TABLE {table} RESTART IDENTITY CASCADE").format(
                table=psycopg2.sql.Identifier(table_name)
            )
            cur.execute(query)
            conn.commit()
            print(f"'{table_name}' 테이블의 모든 데이터가 성공적으로 삭제되었습니다.")
        except (Exception, psycopg2.Error) as error:
            conn.rollback()
            print(f"테이블 데이터 전체 삭제 중 오류 발생: {error}")