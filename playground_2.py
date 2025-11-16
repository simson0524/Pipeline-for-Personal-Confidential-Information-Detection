import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ===================================================================
# 1. 샘플 데이터 생성 (실제 데이터로 이 부분을 교체하세요)
# ===================================================================

# 실제 CSV 파일이 있다면 아래 코드를 사용하세요.
CSV_FILE_PATH = '/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/process_log/test_42_01_model_train_sent_dataset_log_log.csv'
df = pd.read_csv(CSV_FILE_PATH)
df = df[df['validated_epoch'] == 5]

# 샘플 점수 딕셔너리 생성 (domain_id -> span_token -> score)
# 점수는 각 prediction 라벨별로 분포가 다르도록 임의로 생성
score_dict = {}

domain_1_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/08_contract_data/confidential_confscore.csv"
# domain_2_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet_confidential/fold_2_02_confscore.csv"
# domain_3_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet_confidential/fold_2_03_confscore.csv"
# domain_4_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet_confidential/fold_2_04_confscore.csv"
# domain_5_path = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/answer_sheet_confidential/fold_2_05_confscore.csv"
domain_paths = [domain_1_path]

for i, domain_path in enumerate(domain_paths):
    dictionary = {}
    domain_df = pd.read_csv(domain_path)

    for idx, row in tqdm(domain_df.iterrows(), desc=f"domain {i+1} 작업중"):
        if row['x_in'] == 0:
            continue

        dictionary[row['단어']] = float(row['conf_score'])
    
    score_dict[7] = dictionary


# ===================================================================
# 2. DataFrame에 점수(score) 정보 추가
# ===================================================================

print("DataFrame에 점수 정보를 매핑합니다...")

def get_score(row, score_map):
    try:
        return score_map[row['domain_id']][row['span_token']]
    except KeyError:
        return None

# apply 함수를 사용하여 'score' 컬럼을 한 번에 생성
df['score'] = df.apply(get_score, args=(score_dict,), axis=1)

# ❗ [핵심 수정] 점수가 없는(NaN) 행을 여기서 미리 제거합니다.
df.dropna(subset=['score'], inplace=True)

print("점수 매핑 완료. 최종 DataFrame:")
print(df)


# ===================================================================
# 3. 히스토그램 (Histogram) 생성 및 저장 (Y축 250, 범례 수정)
# ===================================================================
print("\n범주화된 카운트 플롯(히스토그램)을 생성합니다...")

# prediction 컬럼의 값을 플롯에 표시될 이름으로 변경
label_map = {'일반정보': '0 (NORMAL)', '기밀정보': '1 (CONFIDENTIAL)'}
df['prediction'] = df['prediction'].map(label_map)


# ---
# 👇 [수정됨] 범례 제목을 "Prediction Label"로 지정하기 위해
# DataFrame의 컬럼명 자체를 변경합니다.
df.rename(columns={'prediction': 'Prediction Label'}, inplace=True)
# ---


plt.figure(figsize=(12, 8))

# 0.05 단위로 점수를 범주화합니다.
bin_width_standard = 0.5  

sns.histplot(
    data=df,
    x='score',
    hue='Prediction Label', # <-- 방금 바꾼 컬럼명을 hue로 사용
    multiple="dodge",
    binwidth=bin_width_standard,
    shrink=0.8
)

plt.ylim(0, 500)

plt.title('test_42_model_validation - Score Distribution by Prediction', fontsize=16)
plt.xlabel('ConfScore (Binned)', fontsize=12)
plt.ylabel('Count (개수)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, axis='y')

# ---
# 👇 [수정됨] 이 라인을 삭제합니다.
# 이 라인이 seaborn이 자동으로 만든 범례를 덮어쓰고 있었습니다.
# 이제 'hue'에 사용된 컬럼명("Prediction Label")이 자동으로 범례의 제목이 됩니다.
# plt.legend(title='Prediction Label') 
# ---

# 플롯 저장
output_filename = 'test_42_model_validation_histogram_ylim250_legend_fixed.png'
plt.savefig(output_filename, dpi=300)

print(f"범례가 수정된 히스토그램이 '{output_filename}' 파일로 저장되었습니다.")


# ===================================================================
# 4. 점수(score)를 5단위로 범주화하고 CSV로 분할 저장
# ===================================================================
print("\n[Section 4] DataFrame을 점수(score) 기준으로 분할하여 CSV로 저장합니다...")

# CSV를 저장할 디렉토리를 생성합니다.
output_csv_dir = 'score_binned_csvs_confidential'
os.makedirs(output_csv_dir, exist_ok=True)
print(f"'{output_csv_dir}' 디렉토리에 CSV 파일을 저장합니다...")

# ---
# 1. 점수(score)의 최소/최대값을 기준으로 5단위 bin을 생성합니다.
#    df['score']는 Section 2에서 이미 NaN 값이 제거되었습니다.
# ---
bin_width = 5
min_val = df['score'].min()
max_val = df['score'].max()

# (예: min -8 -> -10, max 12 -> 15)
start_bin = np.floor(min_val / bin_width) * bin_width
end_bin = np.ceil(max_val / bin_width) * bin_width

# (예: [-10, -5, 0, 5, 10, 15])
# np.arange는 end 값을 포함하지 않으므로, bin_width를 한 번 더 더해줍니다.
# bins가 1개만 생성되는 경우(예: 모든 점수가 0~5 사이)를 대비해 dtype=float 지정
bins = np.arange(start_bin, end_bin + bin_width, bin_width, dtype=float)

# 만약 bins가 하나도 없거나(데이터가 비어서) 1개뿐이면(모든 데이터가 한 범위)
# [min, max]로 강제 설정합니다.
if len(bins) <= 1:
   bins = [start_bin, end_bin]

print(f"점수 범위를 {bin_width} 단위로 분할합니다. (기준: {bins})")

# ---
# 2. 'score_bin'이라는 새 컬럼에 각 행이 속한 범주를 저장합니다.
#    include_lowest=True : 첫 번째 범주가 (예: [-10, -5])가 되도록 보장
# ---
try:
    df['score_bin'] = pd.cut(
        df['score'], 
        bins=bins, 
        right=True,        # (0, 5] (0 < score <= 5)
        include_lowest=True  # 첫 번째 범주(left-most bin)의 왼쪽 경계 포함
    )
except ValueError as e:
    print(f"Score 범주화 중 오류 발생: {e}")
    print("bins가 유효하지 않을 수 있습니다. 2개 이상의 bin 경계가 필요합니다.")
    # 오류가 나도 중단하지 않고 다음으로 넘어가되, 빈 그룹만 생성됩니다.
    pass


# ---
# 3. 'score_bin'을 기준으로 DataFrame을 그룹화합니다.
# ---
# 'score_bin' 컬럼이 생성되지 않았으면 빈 그룹을 반환합니다.
if 'score_bin' in df.columns:
    grouped = df.groupby('score_bin')
else:
    grouped = [] # 빈 리스트로 만들어 루프를 건너뛰게 함

# ---
# 4. 각 그룹을 별도의 CSV 파일로 저장합니다.
# ---
saved_count = 0
for bin_name, group_df in grouped:
    # bin_name은 (0, 5] 같은 Interval 객체입니다.
    # 파일명으로 사용하기 좋은 문자열로 변환합니다.
    
    # 소수점이 포함될 경우 'p'로 변경 (예: 2.5 -> 2p5, -10 -> neg10)
    left_str = str(bin_name.left).replace('.', 'p').replace('-', 'neg')
    right_str = str(bin_name.right).replace('.', 'p').replace('-', 'neg')
    
    # 괄호와 공백 제거 (예: (neg10, neg5] -> neg10_to_neg5)
    filename = f"score_range_{left_str}_to_{right_str}.csv"
    output_path = os.path.join(output_csv_dir, filename)
    
    # 비어있지 않은 그룹만 저장
    if not group_df.empty:
        print(f"  -> {filename} 저장 중... ({len(group_df)} 행)")
        try:
            # utf-8-sig: 엑셀에서 한글이 깨지지 않도록 BOM 추가
            group_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            saved_count += 1
        except Exception as e:
            print(f"   !-> '{filename}' 저장 실패: {e}")

if saved_count > 0:
    print(f"총 {saved_count}개의 CSV 파일 저장이 완료되었습니다.")
else:
    print("저장할 데이터가 없거나 범주화에 실패했습니다.")