# labeling_tools/metric_viewer.py

import subprocess
import time
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from label_studio_sdk.client import LabelStudio
import requests
import psycopg2
import json
import ast  


class ConfusionMatrixPipeline:
    def __init__(self, API_KEY, db_config, experiment_name, label_studio_url="http://localhost:8080"):
        self.api_key = API_KEY
        self.db_config = db_config
        self.label_studio_url = label_studio_url
        self.ls = None
        self.project = None
        self.processed_task_ids = set()
        self.uploaded_task_ids = []  # 업로드된 task id 저장용
        self.experiment_name = experiment_name

    # ------------------------
    # 1️⃣ Label Studio 서버 실행
    # ------------------------
    def start_label_studio(self):
    # 서버 이미 실행 중인지 확인
        try:
            r = requests.get(f"{self.label_studio_url}/api/version", timeout=3)
            if r.status_code == 200:
                print(f"✅ Label Studio 이미 실행 중 ({self.label_studio_url})")
                return
        except requests.exceptions.RequestException:
            pass

        # 서버 새로 실행 (기본 포트가 사용 중이면 8081로 변경 가능)
        new_port = 8080
        while True:
            try:
                r = requests.get(f"http://localhost:{new_port}/api/version", timeout=1)
                # 서버가 있으면 다음 포트로
                new_port += 1
            except requests.exceptions.RequestException:
                break

        print(f"🚀 Label Studio 서버 실행 중... 포트={new_port}")
        subprocess.run(["/home/student1/venv_1/bin/label-studio", "start", "--port", str(new_port)], shell=True)
        self.label_studio_url = f"http://localhost:{new_port}"

        # 서버 준비 대기
        timeout = 60
        for i in range(timeout):
            try:
                r = requests.get(f"{self.label_studio_url}/api/version")
                if r.status_code == 200:
                    print(f"✅ Label Studio 서버 준비 완료! ({self.label_studio_url})")
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError("Label Studio 서버가 준비되지 않았습니다.")

    # ------------------------
    # 2️⃣ SDK 연결
    # ------------------------
    def connect_sdk(self):
        self.ls = LabelStudio(base_url=self.label_studio_url, api_key=self.api_key)
        rc('font', family='Malgun Gothic')

  
    # ------------------------
    # 3️⃣ DB 연결 (새로운 DB 버전)
    # ------------------------
    def get_connection(self):
        return psycopg2.connect(**self.db_config)

    def load_matrices_by_experiment(self, config, experiment_name, table_name, column_name, is_confidential=False):
        """
        DB에서 마지막 행을 불러와서 confusion matrix로 반환.
        
        :param table_name: 테이블 이름
        :param column_name: 리스트 형태로 저장된 confusion matrix 컬럼
        :param is_confidential: 2x2인지 3x3인지 여부
        :return: numpy array (2x2 또는 3x3)
        """
        conn = self.get_connection()
        try:
            if table_name == 'model_train_performance':
                query = f"""
                    SELECT confusion_matrix
                    FROM "{table_name}"
                    WHERE experiment_name = %s AND performed_epoch = %s
                """
                df = pd.read_sql(query, conn, params=(experiment_name, config['exp']['num_epochs']))
            else:
                query = f"""
                    SELECT confusion_matrix
                    FROM "{table_name}"
                    WHERE experiment_name = %s
                """
                df = pd.read_sql(query, conn, params=(experiment_name, ))
        finally:
            conn.close()

        if df.empty:
            raise ValueError(f"{table_name} 테이블에서 데이터를 가져올 수 없습니다.")

        # # 시리즈(Series)에서 첫 번째 값(실제 데이터)을 추출합니다.
        # row_value = df['confusion_matrix'].iloc[0]

        # # 만약 값이 문자열이면 json.loads로 리스트로 변환
        # if isinstance(row_value, str):
        #     matrix_list = json.loads(row_value)
        # else:
        #     matrix_list = row_value  # 이미 리스트이면 그대로 사용

        # # 최종적으로 numpy 배열로 변환하여 반환
        # return np.array(matrix_list)

        matrices = []
        for row_value in df['confusion_matrix']:
            if isinstance(row_value, str):
                matrix_list = json.loads(row_value)
            else:
                matrix_list = row_value  # 이미 리스트/딕셔너리 형태인 경우
            
            matrices.append(np.array(matrix_list))

        return matrices


        # # 문자열을 실제 리스트로 변환
        # if isinstance(row_value, str):
        #     values = ast.literal_eval(row_value)
        # else:
        #     values = row_value  # 이미 리스트 형태이면 그대로 사용

        # # shape 결정
        # if is_confidential:
        #     return np.array(values, dtype=float).reshape(2, 2)
        # else:
        #     return np.array(values, dtype=float).reshape(3, 3)

    # ------------------------
    # 4️⃣ confusion matrix 생성 및 저장
    # ------------------------
    def generate_confusion_matrix_png(self, config, experiment_name, tables, labels, is_pii=True, output_file="matrix.png"):
        """
        개인정보(3x3)와 기밀정보(2x2)를 분리해서 PNG 생성
        :param tables: 테이블 리스트
        :param labels: 라벨 리스트
        :param is_pii: True → 개인정보(3x3), False → 기밀정보(2x2)
        """
        matrices = [
            matrix  # 3. 최종적으로 리스트에 담길 개별 matrix
            for t in tables  # 1. 기존처럼 각 테이블을 순회하고
            for matrix in self.load_matrices_by_experiment(config=config, table_name=t, experiment_name=experiment_name, column_name=labels)  # 2. 함수가 반환한 리스트를 다시 순회
        ]

        print(matrices) ## 로그용

        fig, axes = plt.subplots(1, len(matrices), figsize=(10*len(matrices), 10))

        if len(matrices) == 1:
            axes = [axes]  # axes가 단일 객체일 경우 리스트로 변환

        fold = config['exp']['k_fold']

        plot_title_list = []

        for i in range(fold):
            title = f"model_train_performance_fold_{i+1}"
            plot_title_list.append(title)

        plot_title_list += tables[1:]

        for i, mat in enumerate(matrices):
            total = mat.sum()
            ax = axes[i]
            cmap = "Blues"
            ax.imshow(mat, cmap=cmap)
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = int(mat[r, c])
                    perc = (val / total * 100) if total > 0 else 0
                    ax.text(c, r, f"{val}\n({perc:.1f}%)", ha="center", va="center", color="black")
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_xticklabels(['hit', 'wrong', 'mismatch']) if (i == len(matrices)-2 or i == len(matrices)-3) else ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            title_prefix = "Personal" if is_pii else "Confidential"
            ax.set_title(f"{title_prefix} {plot_title_list[i]}")
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"✅ PNG 생성 완료 → {output_file}")
        return output_file

    # ------------------------
    # 5️⃣ 프로젝트 생성 및 태스크 업로드
    # ------------------------
    def setup_project_and_upload_task(self, experiment_name, png_file):
        with open(png_file, "rb") as f:
            img_bytes = f.read()
        img_base64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        task = [{"data": {"image": img_base64}}]

        projects = self.ls.projects.list()
        self.project = next((p for p in projects if p.title == experiment_name), None)

        if self.project is None:
            self.project = self.ls.projects.create(
                title=experiment_name,
                label_config="""
                <View>
                  <Image name="image" value="$image"/>
                  <Header value="파이프라인을 계속하시겠습니까?"/>
                  <Choices name="next_step" toName="image" choice="single">
                    <Choice value="Yes"/>
                    <Choice value="No"/>
                  </Choices>
                </View>
                """
            )
            print(f"🎯 프로젝트 생성 완료. ID={self.project.id}")
        else:
            print(f"🎯 프로젝트 이미 존재. ID={self.project.id}")

        # ✅ 리턴값 확인
        self.ls.projects.import_tasks(id=self.project.id, request=task)
        #print("🔍 import_tasks response:", response)

         # 업로드 후 프로젝트의 전체 태스크 다시 조회
        tasks = list(self.ls.tasks.list(project=self.project.id))
        self.uploaded_task_ids = [t.id for t in tasks][-len(task):]  # 방금 올린 마지막 N개
        print("✅ 업로드된 태스크 ID:", self.uploaded_task_ids)

       
    def wait_for_task_label(self, task_id, check_interval=5, timeout=3600):
        """
        특정 task_id가 라벨링될 때까지 기다렸다가 Yes/No로 boolean 반환
        - 라벨링 완료 → True (Yes) / False (No)
        - timeout 지나도 라벨링 안 되면 None 반환
        """
        import time

        start_time = time.time()

        while True:
            task = self.ls.tasks.get(task_id)
            annotations = task.annotations or []

            
            for ann in annotations:
                results = getattr(ann, "result", []) or getattr(ann, "results", [])

                for r in results:
                    val = r.get("value", {})
                    choices = val.get("choices", [])
                    text = val.get("text", [])
                    labels = val.get("labels", [])
                    for v in choices + text + labels:
                        if str(v).strip().lower() == "yes":
                            return True
                        elif str(v).strip().lower() == "no":
                            return False
                            

            

            # timeout 체크
            if time.time() - start_time > timeout:
                print(f"⚠️ Task {task_id} 라벨링 대기 timeout ({timeout}초)")
                return None

            time.sleep(check_interval)


    # ------------------------
    # 8️⃣ 전체 실행
    # ------------------------
    def run(self, config, experiment_name, is_pii=True):
        self.start_label_studio()
        self.connect_sdk()

        if is_pii:
            labels = ["0", "1", "2"]
            tables = ['model_train_performance','dictionary_matching_performance', 'ner_regex_matching_performance', 'model_validation_performance']
            output_file = "pii_matrix.png"
        else:
            labels = ["0", "1"]
            tables = ['model_train_performance','dictionary_matching_performance', 'ner_regex_matching_performance', 'model_validation_performance']
            output_file = "conf_matrix.png"

        png_file = self.generate_confusion_matrix_png(
            config=config,
            experiment_name=experiment_name,
            tables=tables,
            labels=labels,
            is_pii=is_pii,
            output_file=output_file
        )
        self.setup_project_and_upload_task(experiment_name, png_file)

        if self.uploaded_task_ids:
            last_task_id = self.uploaded_task_ids[-1]
            result = self.wait_for_task_label(last_task_id)
            print(f"🔍 Task {last_task_id} 결과: {result}")
            return result
        else:
            print("⚠️ 업로드된 태스크가 없습니다.")
            return None


###########파이프라인 실행##################
def metric_viewer(config, experiment_name, API_KEY, is_pii=True):
    db_config = {
        "host": config['db']['host'],
        "port": config['db']['port'],
        "dbname": config['db']['dbname'],
        "user": config['db']['user'],
        "password": config['db']['password']
    }

    if is_pii:
        # 개인정보 프로젝트 실행
        pipeline_pii = ConfusionMatrixPipeline(API_KEY=API_KEY, db_config=db_config, experiment_name=experiment_name)
        print("===== 개인정보 프로젝트 실행 =====")
        result = pipeline_pii.run(config, experiment_name, is_pii=True)  # 개인정보만
        print("✅ 개인정보 프로젝트 결과:", result)
    else:
        # 기밀정보 프로젝트 실행
        pipeline_conf = ConfusionMatrixPipeline(API_KEY=API_KEY, db_config=db_config, experiment_name=experiment_name)
        print("===== 기밀정보 프로젝트 실행 =====")
        result = pipeline_conf.run(config, experiment_name, is_pii=False)  # 기밀정보만
        print("✅ 기밀정보 프로젝트 결과:", result)

    return result