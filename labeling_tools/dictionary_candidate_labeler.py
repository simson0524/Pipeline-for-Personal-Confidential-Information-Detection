# labeling_tools/dictionary_candidate_labeler.py

from label_studio_sdk.client import LabelStudio
import pandas as pd
import subprocess
import requests
import time

class DictionaryCandidateLabeler:
    def __init__(self, base_url: str, api_key: str):
        self.label_studio_url = base_url
        self.api_key = api_key
        self.ls = None
        self.project = None
        self.processed_task_ids = set()

    # ------------------------
    # Label Studio 서버 연결
    # ------------------------
    def start_label_studio(self):
        try:
            r = requests.get(f"{self.label_studio_url}/api/version", timeout=3)
            if r.status_code == 200:
                print(f"✅ Label Studio 이미 실행 중 ({self.label_studio_url})")
                return
        except requests.exceptions.RequestException:
            pass

        new_port = 8080
        while True:
            try:
                r = requests.get(f"http://localhost:{new_port}/api/version", timeout=1)
                new_port += 1
            except requests.exceptions.RequestException:
                break

        print(f"🚀 Label Studio 서버 실행 중... 포트={new_port}")
        subprocess.Popen(["label-studio", "start", "--port", str(new_port)], shell=True)
        self.label_studio_url = f"http://localhost:{new_port}"

        timeout = 60
        for _ in range(timeout):
            try:
                r = requests.get(f"{self.label_studio_url}/api/version")
                if r.status_code == 200:
                    print(f"✅ Label Studio 서버 준비 완료! ({self.label_studio_url})")
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError("Label Studio 서버가 준비되지 않았습니다.")

    def connect_sdk(self):
        self.ls = LabelStudio(base_url=self.label_studio_url, api_key=self.api_key)
        #self.ls =  Client(url=self.label_studio_url, api_key=self.api_key)
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            r = requests.get(f"{self.label_studio_url}/api/version", headers=headers)
            r.raise_for_status()
            print(f"✅ SDK 연결 성공, Label Studio 버전: {r.json().get('version')}")
        except Exception as e:
            raise RuntimeError(f"SDK 연결 실패: {e}")

    # ------------------------
    # 프로젝트 생성
    # ------------------------
    def create_project(self, is_pii: bool, total_count: int):
        info_type = "개인정보" if is_pii else "기밀정보"
        label_config = f"""
        <View>
            <Header value="현재 진행도: $idx / {total_count} ($percent%)"/>
            <View style="margin-top: 15px"/>
            <Header value="Q. 아래 하이라이트 된 단어를 {info_type} 사전에 등재하시겠습니까?"/>
            <View style="margin-top: 30px"/>
            <Text name="sentence" value="$sentence"/>
            <View style="margin-top: 30px"/>
            
            <Text name="span_token" value="span_token: $span_token"/>
            <View style="margin-top: 15px"/>
            <Text name="z_score" value="z_score: $z_score"/>
            <View style="margin-top: 15px"/>
            <Text name="domain_id" value="domain_id: $domain_id"/>
            <View style="margin-top: 20px"/>
            <Choices name="choice" toName="sentence" choice="single">
                <Choice value="예"/>
                <Choice value="아니오"/>
            </Choices>
        </View>
        """

        self.project = self.ls.projects.create(
            title=f"Dictionary_Candidate_Labeling_{info_type}",
            label_config=label_config
        )
        print(f"✅ 프로젝트 생성 완료: {self.project.id} ({info_type})")
        return self.project

    # ------------------------
    # Task 준비 및 업로드
    # ------------------------
    
    def prepare_tasks(self, candidate_list):
        tasks = []
        total = len(candidate_list)
        for i, row in enumerate(candidate_list, start=1):
            percent = round(i / total * 100, 2)
            sentence = row["sentence"]
            span_token = row["span_token"]

            # sentence 안에 span_token 위치 찾기
            start_positions = []
            start = 0
            while True:
                idx = sentence.find(span_token, start)
                if idx == -1:
                    break
                start_positions.append(idx)
                start = idx + len(span_token)

            # Task 생성
            for start_idx in start_positions:
                end_idx = start_idx + len(span_token)
                tasks.append({
                    "data": {
                        "sentence": sentence,
                        "span_token": span_token,
                        "z_score": row["z_score"],
                        "domain_id": row["domain_id"],
                        "idx": i,
                        "percent": percent
                    },
                    "predictions": [
                        {
                            "result": [
                                {
                                    "from_name": "choice",  # label config Choices name
                                    "to_name": "sentence",
                                    "type": "labels",
                                    "value": {
                                        "start": start_idx,
                                        "end": end_idx,
                                        "text": span_token,
                                        "labels": []  # 빈 리스트, 단순 하이라이트
                                    }
                                }
                            ]
                        }
                    ]
                })
        return tasks



    def upload_tasks(self, tasks):
        if not tasks:
            print("⚠️ 업로드할 Task가 없습니다.")
            return
        self.ls.projects.import_tasks(id=self.project.id, request=tasks)
        print(f"✅ {len(tasks)}개의 Task 업로드 완료!")

    # ------------------------
    # 결과 Fetch
    # ------------------------
    def fetch_results(self):
        if not self.project:
            raise ValueError("프로젝트가 존재하지 않습니다.")

        upload_to_dictionary = []

        print("📌 라벨링 결과 fetch 시작...")

        while True:
            tasks = list(self.ls.tasks.list(project=self.project.id, page=1))
            new_task_processed = False

            for task in tasks:
                task_id = task.id
                if task_id in self.processed_task_ids:
                    continue

                for ann in (task.annotations or []):
                    for r in ann.get("result", []):
                        val = r.get("value", {})
                        choice = val.get("choices", [None])[0]

                        if choice == "예":
                            upload_to_dictionary.append({
                                "sentence": task.data["sentence"],
                                "span_token": task.data["span_token"],
                                "z_score": task.data["z_score"],
                                "domain_id": task.data["domain_id"],
                            })
                            print(f"✅ Task {task_id} → 예 선택됨 ({task.data['span_token']})")

                        self.processed_task_ids.add(task_id)
                        new_task_processed = True

            if not new_task_processed:
                break

            print("⌛ 새로운 결과 대기 중...")
            time.sleep(2)

        print("✅ 모든 Task 처리 완료")
        return upload_to_dictionary

    # ------------------------
    # End-to-End 실행
    # ------------------------
    def run_pipeline(self, is_pii: bool, candidate_list):
        self.start_label_studio()
        self.connect_sdk()
        self.create_project(is_pii, len(candidate_list))
        tasks = self.prepare_tasks(candidate_list)
        self.upload_tasks(tasks)
        print("📌 웹에서 라벨링 완료 후 fetch_results() 호출하여 결과 확인 가능")
        return self.project



def dictionary_candidate_labeler(config, upload_candidate_to_dictionary):
    LABEL_STUDIO_URL = "http://localhost:8080"
    API_KEY = config['api_key']['label_studio']
    is_pii = config['exp']['is_pii']

    manager = DictionaryCandidateLabeler(LABEL_STUDIO_URL, API_KEY)
    project = manager.run_pipeline(is_pii=is_pii, candidate_list=upload_candidate_to_dictionary)

    input("✅ 웹에서 라벨링 완료 후 엔터를 누르세요...")
    upload_to_dictionary = manager.fetch_results()

    print("\n📦 [최종 업로드 후보 결과]")
    # for item in upload_to_dictionary:
    #     print(item)
    print(upload_to_dictionary)

    return upload_to_dictionary