"""별도 관리

Returns:
    _type_: _description_
"""


import os
import json
from typing import List, Dict, Any, Set
import psycopg2
from psycopg2.extras import RealDictCursor

from ner_regex_matching.regex_logics.regex_main import run_regex_detection
from ner_regex_matching.ner_logics.ner_main import run_ner_detection
from tqdm import tqdm  

class AnnotationPipeline:
    def __init__(self,
                 input_folder: str,
                 output_base: str,
                 db_config: Dict[str, Any],
                 default_domain_id: str = "06"):
        # 경로 설정
        self.input_folder = input_folder
        self.output_base = output_base
        self.out_all = os.path.join(output_base, "all")
        self.out_pii = os.path.join(output_base, "pii")
        self.out_conf = os.path.join(output_base, "confidential")
        for d in (self.out_all, self.out_pii, self.out_conf):
            os.makedirs(d, exist_ok=True)

        # DB 연결
        self.db_config = db_config
        self.conn = self.connect_postgresql(db_config)
        self.default_domain_id = default_domain_id

        # 전체 dictionary 한 번만 불러오기
        self.personal_dict_cache = self.fetch_dictionary("personal_info_dictionary", self.default_domain_id)
        self.conf_dict_cache = self.fetch_dictionary("confidential_info_dictionary", self.default_domain_id)

    def connect_postgresql(self, config: Dict[str, Any]):
        try:
            conn = psycopg2.connect(
                host=config["host"],
                port=config["port"],
                dbname=config["dbname"],
                user=config["user"],
                password=config["password"]
            )
            print("✅ PostgreSQL connected")
            return conn
        except Exception as e:
            print("❌ PostgreSQL connection failed:", e)
            return None

    def extract_domain_id(self, sent_id: str) -> str:
        parts = sent_id.split("_")
        if len(parts) >= 2 and parts[0].lower().startswith("sample"):
            return parts[1]
        return self.default_domain_id

    def fetch_dictionary(self, table_name: str, domain_id: str) -> List[Dict[str, Any]]:
        if self.conn is None:
            return []
        try:
            cur = self.conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(f"SELECT span_token, z_score FROM {table_name} WHERE domain_id=%s", (domain_id,))
            rows = cur.fetchall()
            cur.close()
            return [{"span_token": r.get("span_token"), "z_score": r.get("z_score", 1.0)} for r in rows]
        except Exception as e:
            print(f"⚠️ DB fetch error for {table_name}, domain {domain_id}:", e)
            return []

    @staticmethod
    def find_token_positions(sentence: str, token: str) -> List[Dict[str, int]]:
        res = []
        start = 0
        while True:
            idx = sentence.find(token, start)
            if idx == -1:
                break
            res.append({"start": idx, "end": idx + len(token)})
            start = idx + len(token)
        return res

    def dictionary_match(self, sentence: str, domain_id: str) -> List[Dict[str, Any]]:
        annotations = []
        # personal
        for entry in self.personal_dict_cache:
            token = entry["span_token"]
            if not token:
                continue
            for pos in self.find_token_positions(sentence, token):
                annotations.append({
                    "start": pos["start"],
                    "end": pos["end"],
                    "label": "개인",
                    "score": str(entry.get("z_score", 1.0)),
                    "span_text": sentence[pos["start"]:pos["end"]]
                })
        # confidential
        for entry in self.conf_dict_cache:
            token = entry["span_token"]
            if not token:
                continue
            for pos in self.find_token_positions(sentence, token):
                annotations.append({
                    "start": pos["start"],
                    "end": pos["end"],
                    "label": "기밀",
                    "score": str(entry.get("z_score", 1.0)),
                    "span_text": sentence[pos["start"]:pos["end"]]
                })
        return annotations

    def detection_to_annotations(self, sentence: str, detections: List[Dict[str, Any]], existing_spans: Set[str]):
        anns = []
        for d in detections:
            token = d.get("단어")
            if not token or token in existing_spans:
                continue
            positions = self.find_token_positions(sentence, token)
            for pos in positions:
                anns.append({
                    "start": pos["start"],
                    "end": pos["end"],
                    "label": d.get("구분", "기타"),
                    "score": str(d.get("score", 0.8)),
                    "span_text": sentence[pos["start"]:pos["end"]]
                })
        return anns

    def process_sentence(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        sentence_text = entry.get("sentence", "")
        sentence_id = entry.get("id", "")
        sequence = entry.get("sequence", 0)
        filename = entry.get("file_name", "Generated_0000")
        caseField = entry.get("caseField", "1")
        detailField = entry.get("detailField", "6")

        domain_id = self.extract_domain_id(sentence_id)

        # 1) dictionary
        annotations = self.dictionary_match(sentence_text, domain_id)

        # 2) regex
        try:
            regex_results = run_regex_detection(sentence_text) or []
        except Exception as e:
            print("⚠️ run_regex_detection error:", e)
            regex_results = []

        regex_annotations = self.detection_to_annotations(sentence_text, regex_results, {ann["span_text"] for ann in annotations})
        annotations.extend(regex_annotations)

        # 3) ner
        try:
            ner_results = run_ner_detection(sentence_text) or []
        except Exception as e:
            print("⚠️ run_ner_detection error:", e)
            ner_results = []

        ner_annotations = self.detection_to_annotations(sentence_text, ner_results, {ann["span_text"] for ann in annotations})
        annotations.extend(ner_annotations)

        # ------------------------------
        # 위치 겹침 기반 중복 제거
        # ------------------------------
        annotations_sorted = sorted(annotations, key=lambda x: x['start'])
        unique_annotations = []
        occupied = set()
        for ann in annotations_sorted:
            overlap = False
            for i in range(ann['start'], ann['end']):
                if i in occupied:
                    overlap = True
                    break
            if not overlap:
                unique_annotations.append(ann)
                for i in range(ann['start'], ann['end']):
                    occupied.add(i)

        return {
            "data": [{
                "sentence": sentence_text,
                "id": sentence_id,
                "filename": filename,
                "caseField": caseField,
                "detailField": detailField,
                "sequence": sequence
            }],
            "annotations": [{
                "id": sentence_id,
                "annotations": unique_annotations
            }]
        }

    def save_json(self, result_json: Dict[str, Any]):
        sentence_id = result_json["data"][0]["id"]
        with open(os.path.join(self.out_all, f"{sentence_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        anns = result_json["annotations"][0].get("annotations", [])
        labels = {a.get("label") for a in anns}
        if any(l in ("개인", "준식별") for l in labels):
            with open(os.path.join(self.out_pii, f"{sentence_id}.json"), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
        if "기밀" in labels:
            with open(os.path.join(self.out_conf, f"{sentence_id}.json"), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"Saved: {sentence_id}")

    def process_folder(self):
        all_files = [f for f in sorted(os.listdir(self.input_folder)) if f.endswith(".json")]
        print(f"총 {len(all_files)}개 파일 처리 시작...")
        for file_name in tqdm(all_files, desc="파일 처리 중", unit="파일"):
            file_path = os.path.join(self.input_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in tqdm(entries, desc=f"{file_name} 문장 처리", unit="문장", leave=False):
                result_json = self.process_sentence(entry)
                self.save_json(result_json)
        if self.conn:
            self.conn.close()
        print("🎉 모든 파일 처리 완료!")

# ------------------------------
# 테스트용 문장 실행
# ------------------------------
# def test_single_sentence(pipeline: AnnotationPipeline, sentence: str, sent_id: str = "sample_test_000001"):
#     test_entry = {
#         "sentence": sentence,
#         "id": sent_id,
#         "sequence": "0001",
#         "file_name": "test_file",
#         "caseField": "1",
#         "detailField": "6"
#     }
#     result = pipeline.process_sentence(test_entry)
#     print(json.dumps(result, ensure_ascii=False, indent=2))
#     print("\n✅ 중복 제거 후 annotation 개수:", len(result["annotations"][0]["annotations"]))

# if __name__ == "__main__":
#     pipeline = AnnotationPipeline(
#         input_folder="/home/student1/STT/stt_split_sentence_fixed3",
#         output_base="/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/06.Call_Records",
#         db_config={
#             "host": "127.0.0.1",
#             "port": 55432,
#             "dbname": "postgres",
#             "user": "student1",
#             "password": "onestone"
#         }
#     )

#     #test_sentence = "5일 5일 5랑 11일 15일 21일 15일 정도 15일로요 네 15일로 해드리고요"
#     test_sentence = "그래서 고객님이 지금 여기 흥국화재 거를 79세까지 보장을 받는 거 이유가 증권을 드리도록 나눠버리니까 여기서 기본 보험료 사업비 같은 거를 빼다 보니까 오히려 79세까지밖에 보장을 못 받는 거죠."

#     test_single_sentence(pipeline, test_sentence, sent_id="sample_06_000_000459")

# 폴더 실행
if __name__ == "__main__":
    pipeline = AnnotationPipeline(
        input_folder="/home/student1/STT/stt_split_sentence_fixed3",
        output_base="/home/student1/Pipeline-for-Personal-Confidential-Information-Detection/data/06.Call_Records",
        db_config={
            "host": "127.0.0.1",
            "port": 55432,
            "dbname": "postgres",
            "user": "student1",
            "password": "onestone"
        }
    )
    pipeline.process_folder()
