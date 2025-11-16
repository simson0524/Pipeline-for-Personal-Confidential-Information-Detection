import pandas as pd

domain_1_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet/set_1_01_confscore.csv"
domain_2_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet/set_1_02_confscore.csv"
domain_3_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet/set_1_03_confscore.csv"
domain_4_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet/set_1_04_confscore.csv"
domain_5_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet/set_1_05_confscore.csv"
domain_paths = [domain_1_path, domain_2_path, domain_3_path, domain_4_path, domain_5_path]

# TODO 병합해서 하나의 csv만들기

# 저장될 파일 이름
output_filename = "set_1_all_confscore.csv"

# 각 CSV 파일을 DataFrame으로 읽어 리스트에 저장
df_list = [pd.read_csv(path) for path in domain_paths]

# 리스트에 있는 모든 DataFrame을 하나로 병합
combined_df = pd.concat(df_list, ignore_index=True)

# 병합된 DataFrame을 새로운 CSV 파일로 저장
combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"모든 CSV 파일이 성공적으로 '{output_filename}' 파일로 병합되었습니다.")
print("병합된 데이터 정보:")
print(combined_df.info())