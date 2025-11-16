from database.create_dbs import get_connection
import yaml
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os

config_file_path = "run_config.yaml"
with open(config_file_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 분석하고 싶은 실험의 이름을 지정하세요.
TARGET_EXPERIMENT_NAME = "test_42_01" 

# 그래프를 저장할 디렉토리 이름
OUTPUT_DIR = "/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/visualizations"
# ===================================================================


def fetch_performance_data(experiment_name: str) -> pd.DataFrame:
    """
    데이터베이스에서 특정 실험의 성능 지표를 가져와 Pandas DataFrame으로 반환합니다.
    """
    query = """
        SELECT 
            performed_epoch,
            train_loss,
            valid_loss,
            confusion_matrix
        FROM 
            model_train_performance
        WHERE 
            experiment_name = %(exp_name)s
        ORDER BY 
            performed_epoch;
    """
    
    conn = get_connection(config)
    try:
        print("데이터베이스에 연결 중...")
        
        print(f"'{experiment_name}' 실험 데이터 가져오는 중...")
        df = pd.read_sql_query(query, conn, params={"exp_name": experiment_name})
        
        print("데이터 로드 완료.")
        return df

    except (Exception, psycopg2.Error) as error:
        print("데이터베이스 연결 또는 쿼리 실행 중 오류 발생:", error)
        return pd.DataFrame() # 오류 발생 시 빈 DataFrame 반환

    finally:
        if conn is not None:
            conn.close()
            print("데이터베이스 연결 종료.")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    불러온 데이터의 타입을 변환하고 Confusion Matrix를 개별 컬럼으로 분해합니다.
    (0 -> NORMAL, 1 -> CONFIDENTIAL 라벨 매핑 적용)
    """
    if df.empty:
        return df
    
    print("데이터 처리 중...")
    
    # Loss 값들을 텍스트에서 숫자(float) 타입으로 변환
    df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')
    df['valid_loss'] = pd.to_numeric(df['valid_loss'], errors='coerce')

    # Confusion Matrix (JSONB) 데이터를 개별 컬럼으로 확장
    if 'confusion_matrix' not in df.columns or df['confusion_matrix'].isnull().all():
        print("Warning: 'confusion_matrix' 컬럼이 없거나 데이터가 비어있습니다.")
        return df

    # +++ 변경: 라벨 이름 리스트 정의 (CM의 [0], [1] 순서와 일치)
    labels = ["NORMAL", "CONFIDENTIAL"]
    num_labels = len(labels)

    # +++ 변경: 0, 1 대신 NORMAL, CONFIDENTIAL을 컬럼명에 사용
    for pred_idx in range(num_labels):
        for true_idx in range(num_labels):
            # (예: "NORMAL")
            pred_name = labels[pred_idx] 
            # (예: "CONFIDENTIAL")
            true_name = labels[true_idx] 
            
            # (예: "cm_predNORMAL_trueCONFIDENTIAL")
            col_name = f'cm_pred{pred_name}_true{true_name}'
            
            # cm[예측][실제] 순서로 데이터 추출 (예: cm[0][1])
            df[col_name] = df['confusion_matrix'].apply(lambda cm: cm[pred_idx][true_idx])
            
    print("데이터 처리 완료.")
    return df


def plot_loss_curves(df: pd.DataFrame, experiment_name: str):
    """
    Train/Valid Loss 변화 그래프를 생성하고 파일로 저장합니다.
    """
    if df.empty or 'train_loss' not in df.columns:
        print("Loss 데이터를 찾을 수 없어 그래프를 생성할 수 없습니다.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df['performed_epoch'], df['train_loss'], marker='o', linestyle='-', label='Train Loss')
    ax.plot(df['performed_epoch'], df['valid_loss'], marker='s', linestyle='--', label='Validation Loss')
    
    ax.set_title(f'Train & Validation Loss Over Epochs\n(Experiment: {experiment_name})', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    max_epoch = df['performed_epoch'].max()
    if max_epoch > 0:
        ax.set_xticks(range(1, int(max_epoch) + 1))
    
    output_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_loss_curve.png")
    plt.savefig(output_path, dpi=300)
    print(f"Loss 그래프 저장 완료: {output_path}")
    plt.close(fig)


def plot_cm_trends_by_ground_truth_broken_axis(df: pd.DataFrame, experiment_name: str):
    """
    '끊어진 Y축'을 사용하여 Ground Truth 라벨별 예측값 변화 추이 그래프를 생성합니다.
    값의 차이가 클 때 변화를 극적으로 보여주는 데 효과적입니다.
    """
    # +++ 수정: 라벨 이름을 여기서 직접 정의
    eng_labels = ["NORMAL", "CONFIDENTIAL"]
    num_labels = len(eng_labels)

    # +++ 수정: cm 컬럼이 아닌, 처리된 컬럼(예: cm_predNORMAL_trueNORMAL)이 있는지 확인
    if df.empty or 'cm_predNORMAL_trueNORMAL' not in df.columns:
        print("Confusion Matrix 데이터를 찾을 수 없어 그래프를 생성할 수 없습니다.")
        print(f"필요한 컬럼 예시: cm_predNORMAL_trueNORMAL (현재 컬럼: {df.columns.tolist()})")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    
    for true_label in eng_labels:
        # 2개의 서브플롯(위, 아래) 생성. x축은 공유.
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.1)  # 두 플롯 사이의 간격 줄이기

        # 각 서브플롯에 동일한 데이터를 플로팅
        for pred_label in eng_labels:
            # +++ 변경 없음: 이 부분은 이미 NORMAL/CONFIDENTIAL을 사용하도록 잘 작성되어 있었습니다.
            col_name = f'cm_pred{pred_label}_true{true_label}' 
            if col_name not in df.columns:
                print(f"Warning: 컬럼 '{col_name}'을(를) 찾을 수 없습니다. 그래프에서 제외됩니다.")
                continue

            label_text = f'Predicted as {pred_label}'
            
            # 올바르게 예측한 경우
            if pred_label == true_label:
                line_style = {'marker': '*', 'linestyle': '-', 'linewidth': 2.5, 'zorder': 10}
                label_text += ' (Correct)'
            # 틀리게 예측한 경우
            else:
                line_style = {'marker': 'o', 'linestyle': '--', 'alpha': 0.8}
            
            # 두 축에 모두 플로팅
            ax_top.plot(df['performed_epoch'], df[col_name], label=label_text, **line_style)
            ax_bottom.plot(df['performed_epoch'], df[col_name], label=label_text, **line_style)

        # ==========================================================
        # Y축 범위 설정 (가장 중요한 부분)
        # ==========================================================
        # TODO: 데이터에 맞게 이 y축 범위 값을 직접 수정하세요.
        # (이 값은 라벨과 상관없이 데이터 분포에 따라 조절해야 합니다.)
        ylim_top = (42500, 44000) 
        ylim_bottom = (0, 2500)
        # ==========================================================

        ax_top.set_ylim(ylim_top)
        ax_bottom.set_ylim(ylim_bottom)

        # ... (끊어진 축 시각적 표현 코드는 동일) ...
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(axis='x', which='both', bottom=False)
        
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        # ...

        # 레이블 및 제목 설정
        fig.suptitle(f'Prediction Trends for Ground Truth Label: {true_label}\n(Experiment: {experiment_name})', fontsize=16)
        ax_bottom.set_xlabel('Epoch', fontsize=12)
        fig.supylabel('Sample Count', fontsize=12, x=0.06)
        
        handles, labels = ax_top.get_legend_handles_labels()
        ax_top.legend(handles, labels, title="Prediction", fontsize=10)
        
        max_epoch = df['performed_epoch'].max()
        if max_epoch > 0:
            plt.xticks(range(1, int(max_epoch) + 1))
        
        output_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_cm_trends_gt_{true_label}_broken.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

    # +++ 수정: num_labels를 eng_labels.length로 사용
    print(f"Ground Truth 라벨별 '끊어진 Y축' 그래프 {len(eng_labels)}개 저장 완료.")

def main():
    """
    메인 실행 함수
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"'{OUTPUT_DIR}' 디렉토리 생성 완료.")

    # 1. 데이터 가져오기
    perf_df = fetch_performance_data(TARGET_EXPERIMENT_NAME)

    if not perf_df.empty:
        # 2. 데이터 전처리
        processed_df = process_data(perf_df)
        
        # 3. 그래프 생성 및 저장
        plot_loss_curves(processed_df, TARGET_EXPERIMENT_NAME)
        plot_cm_trends_by_ground_truth_broken_axis(processed_df, TARGET_EXPERIMENT_NAME)
    else:
        print(f"'{TARGET_EXPERIMENT_NAME}'에 대한 데이터를 찾지 못했습니다. 스크립트를 종료합니다.")


if __name__ == '__main__':
    main()