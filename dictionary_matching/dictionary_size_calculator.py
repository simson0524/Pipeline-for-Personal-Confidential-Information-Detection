# dictionary_matching/dictionary_size_calculator.py

from database.db_logics import *

def dictionary_size_calculator(conn, config, dict_table_name):
    dictionaries = {}
    each_dict_size = {}

    valid_domain_ids = config['dictionary_init']['domain_ids']
    
    for domain_id in valid_domain_ids:
        # 데이터베이스 내용 미리 dict으로 추출하여 Hash로 비교
        table_dict = fetch_dictionary_table_as_dict(
            conn=conn,
            table_name=dict_table_name,
            key_column='span_token',
            domain_id=domain_id
        )
        dictionaries[domain_id] = table_dict

        # insertion == deletion이 같으면 최종적으로 deletion되었다는 의미
        fetched_dictionary_size = 0
        for value in table_dict.values():
            if value['insertion_counts'] > value['deletion_counts']:
                fetched_dictionary_size += 1
            else:
                print(value,'\n')

        if fetched_dictionary_size == 0:
            fetched_dictionary_size = 0.000001 # epsilon

        each_dict_size[domain_id] = fetched_dictionary_size

    return dictionaries, each_dict_size

