# ner_regex_matching/regex_logics/regex_main.py

import json
from pathlib import Path
from typing import List, Dict
from ner_regex_matching.regex_logics.detectors.name_detector import NameDetector
from ner_regex_matching.regex_logics.detectors.address_detector import AddressDetector
from ner_regex_matching.regex_logics.detectors.birth_age_detector import BirthAgeDetector
from ner_regex_matching.regex_logics.detectors.email_detector import EmailDetector
from ner_regex_matching.regex_logics.detectors.personal_id_detector import JuminDetector
from ner_regex_matching.regex_logics.detectors.phone_num_detector import PhoneDetector
from ner_regex_matching.regex_logics.detectors.card_num_detector import CardNumDetector
from ner_regex_matching.regex_logics.Dict.address_dict import sido_list, sigungu_list, dong_list
from ner_regex_matching.regex_logics.Dict.name_dict import sn1, nn1, nn2, name
from ner_regex_matching.regex_logics.Dict.stopwords_dict import stopwords
import pandas as pd
import os

# 사용할 디텍터 리스트 (필요에 따라 주석 해제 및 추가)
detectors = [
    #NameDetector(sn1, nn1, nn2, name,stopwords=stopwords),
    AddressDetector(sido_list, sigungu_list, dong_list),
    BirthAgeDetector(),
    EmailDetector(),
    JuminDetector(),
    PhoneDetector(),
    CardNumDetector()
    # 
    # ,
]

# detector_label_map.py
DETECTOR_TYPE_MAP = {
    "인물": {"개인/기밀": "개인", "식별/준식별": "식별"},
    "도시": {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "카드번호" : {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "도, 주": {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "군, 면, 동": {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "도로명": {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "건물명": {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "주소숫자": {"개인/기밀": "개인", "식별/준식별": "준식별"},
    "나이": {"개인/기밀": "개인", "식별/준식별": "식별"},
    "이메일주소": {"개인/기밀": "개인", "식별/준식별": "식별"},
    "주민번호": {"개인/기밀": "개인", "식별/준식별": "식별"},
    "전화번호": {"개인/기밀": "개인", "식별/준식별": "식별"},

    # 필요 시 디텍터 추가
}


def run_regex_detection(text: str) -> List[Dict]:
    """
    주어진 텍스트에서 모든 디텍터를 돌며 PII를 탐지하고 결과 리스트 반환
    """
    results = []
    detectors = [
    AddressDetector(sido_list, sigungu_list, dong_list),
    EmailDetector(),
    JuminDetector(),
    PhoneDetector(),
    CardNumDetector()
    ]


    for detector in detectors:
        matches = detector.detect(text) # detect 하는 부분
        for m in matches: 
            # detector.score()에 넘길 'match' 문자열이 없으면 추출해서 넣음
            if "match" not in m: # m안에 match 키가 없다면
                m["match"] = text[m["start"]:m["end"]]  # start부터 end까지 토큰을 match:토큰 형식으로 넣어라
            if isinstance(detector, AddressDetector):
            # 이미 내부 라벨이 들어있으니 건드리지 않음
                pass
            
            if "score" not in m or m["score"] is None:
    
                if hasattr(detector, "score"):
                    m["score"] = detector.score(m["match"])
                else:
                    m["score"] = 0.0
            label = m["label"]
            gubun = DETECTOR_TYPE_MAP.get(label,{}).get("개인/기밀","Unknown")

            results_item = {
                "단어": m["match"],
                "부서명": None,
                "문서명": None,
                "단어유형": label,
                "구분": gubun
            }

            results.append(results_item)

    return results
                


# ✅ 변환 함수: 원하는 JSON 포맷으로 가공
def convert_to_target_format(entry: Dict, results: List[Dict], filename: str, case_field: str, detail_field: str) -> Dict:
    sent_id = entry["id"]
    sentence = entry["sentence"]
    
    # sequence는 숫자로 변환
    sequence = entry.get("sequence")
    if isinstance(sequence, str) and sequence.isdigit():
        sequence = int(sequence)
    elif not isinstance(sequence, int):
        sequence = 0

    return {
        "data": [
            {
                "sentence": sentence,
                "id": sent_id,
                "filename": filename,
                "caseField": case_field,
                "detailField": detail_field,
                "sequence": sequence
            }
        ],
        "annotations": [
            {
                "id": sent_id,
                "annotations": [
                    {
                        "start": r["start"],
                        "end": r["end"],
                        "label": r["label"],
                        "score": r["score"]
                    } for r in results
                ]
            }
        ]
    }




def process_sentence_split_json(input_folder: Path, output_folder: Path,case_field=None, detail_field=None):
    output_folder.mkdir(parents=True, exist_ok=True)
    all_rows = [] 

    for file_path in input_folder.rglob("*.json"):
        print(f"📄 처리 중: {file_path.name}")
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            output_list = []

            for entry in data:
                sentence = entry["sentence"]
                results = run_pii_detection(sentence)

                for r in results:
                    label_type = r["label"]
                    label_info = DETECTOR_TYPE_MAP.get(label_type, {"개인/기밀": "", "식별/준식별": ""})
                    all_rows.append({
                        "도메인": file_path.parent.name,
                        "단어": r["match"],
                        "개인/기밀": label_info["개인/기밀"],
                        "식별/준식별": label_info["식별/준식별"],
                        "정보 유형": r["label"],
                        "score": r.get("score", None),
                        #"score" : 1.0,
                        #"저장 경로": str(file_path.name)
                    })

                formatted = convert_to_target_format(
                    entry,
                    results,
                    filename=str(file_path),
                    case_field=case_field,
                    detail_field=detail_field
                )
                output_list.append(formatted)

            output_path = output_folder / file_path.name
            output_path.write_text(json.dumps(output_list, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✔ 저장 완료: {file_path.name}")

        except Exception as e:
            print(f"❌ 에러 발생 - {file_path.name}: {e}")
    return all_rows  
    


# 법률 도메인 pii detector--------------------------------------------------------------------------------------



# if __name__ == "__main__":
#     print("🚀 PII Detector 시작됨") 
    
#     # 각 도메인별 입력·출력·필드 매핑
#     domain_config = {
#         "민사": {"input": "regex_based_doc_parsing/data_/sentence_split_json/1.민사/set1",
#                "output": "regex_based_doc_parsing/data_/pii_detection_output/민사/set1",
#                "case_field": "1", "detail_field": "1"},
#         "가사": {"input": "regex_based_doc_parsing/data_/sentence_split_json/2.가사/set1",
#                "output": "regex_based_doc_parsing/data_/pii_detection_output/가사/set1",
#                "case_field": "1", "detail_field": "2"},
#         "특허": {"input": "regex_based_doc_parsing/data_/sentence_split_json/3.특허/set1",
#                "output": "regex_based_doc_parsing/data_/pii_detection_output/특허/set1",
#                "case_field": "1", "detail_field": "3"},
#         "행정": {"input": "regex_based_doc_parsing/data_/sentence_split_json/4.행정/set1",
#                "output": "regex_based_doc_parsing/data_/pii_detection_output/행정/set1",
#                "case_field": "2", "detail_field": "4"},
#         "형사": {"input": "regex_based_doc_parsing/data_/sentence_split_json/5.형사/set1",
#                "output": "regex_based_doc_parsing/data_/pii_detection_output/형사/set1",
#                "case_field": "3", "detail_field": "5"},
#     }

#     base_path = Path("C:/Users/megan/onestone/BOAZ_Data_preprocess_logics/regex_based_doc_parsing/data_/")
#     output_j_csv = base_path / "output_j.csv"
#     output_p_csv = base_path / "output_p.csv"

#     # CSV 파일 초기화 (맨 처음에만 헤더 포함)
#     if output_j_csv.exists():
#         os.remove(output_j_csv)
#     if output_p_csv.exists():
#         os.remove(output_p_csv)

#     write_header_j = True
#     write_header_p = True

#     all_rows = []
#     for domain, cfg in domain_config.items():
#         print(f"📂 {domain} 처리 시작")
#         case_field = cfg["case_field"]
#         detail_field = cfg["detail_field"]

#         input_path = Path(cfg["input"])
#         output_path = Path(cfg["output"])

#         rows = process_sentence_split_json(input_path, output_path,
#                                            case_field=case_field, detail_field=detail_field)

#         if rows:
#             all_rows.extend(rows)
#         else:
#             print(f"⚠️ {domain}에서 탐지된 결과 없음")

#     if not all_rows:
#         print("⚠️ 모든 도메인에서 결과가 없습니다.")
#     else:
#         df = pd.DataFrame(all_rows)
#         df_j = df[df["식별/준식별"] == "준식별"]
#         df_p = df[df["식별/준식별"] != "준식별"]

#         base_path = Path("C:/Users/megan/onestone/BOAZ_Data_preprocess_logics/regex_based_doc_parsing/data_/")
#         output_j_csv = base_path / "output_j.csv"
#         output_p_csv = base_path / "output_p.csv"

#         # 혹시 기존 파일 삭제
#         if output_j_csv.exists():
#             os.remove(output_j_csv)
#         if output_p_csv.exists():
#             os.remove(output_p_csv)

#         df_j.to_csv(output_j_csv, index=False, encoding="utf-8-sig")
#         df_p.to_csv(output_p_csv, index=False, encoding="utf-8-sig")

#         print(f"✅ 모든 도메인의 결과를 누적하여 저장 완료: {len(df_j)} 준식별 rows, {len(df_p)} 식별 rows")
    

# openai pii detector --------------------------------------------------------------

# if __name__ == "__main__":
#     print("🚀 OpenAI PII Detector 시작됨") 

#     # 입력/출력 경로 설정
#     input_path = Path(r"C:\Users\megan\onestone\BOAZ_Data_preprocess_logics\regex_based_doc_parsing\data_\sentence_split_json\openai")
#     output_path = Path(r"C:\Users\megan\onestone\BOAZ_Data_preprocess_logics\regex_based_doc_parsing\data_\pii_detection_output\openai")

#     case_field = "0"
#     #detail_field = "OpenAI"

#     all_rows = process_sentence_split_json(input_path, output_path, case_field=case_field)
    

    # if not all_rows:
    #     print("⚠️ OpenAI 폴더에서 PII가 탐지되지 않음")
    # else:
    #     df = pd.DataFrame(all_rows)
    #     df_j = df[df["식별/준식별"] == "준식별"]
    #     df_p = df[df["식별/준식별"] != "준식별"]

    #     base_path = Path(r"C:\Users\megan\onestone\BOAZ_Data_preprocess_logics\regex_based_doc_parsing\data_")
    #     output_j_csv = base_path / "output_openai_j.csv"
    #     output_p_csv = base_path / "output_openai_p.csv"

    #     # 기존 파일 삭제
    #     if output_j_csv.exists():
    #         os.remove(output_j_csv)
    #     if output_p_csv.exists():
    #         os.remove(output_p_csv)

    #     df_j.to_csv(output_j_csv, index=False, encoding="utf-8-sig")
    #     df_p.to_csv(output_p_csv, index=False, encoding="utf-8-sig")

    #     print(f"✅ OpenAI PII 탐지 완료: {len(df_j)} 준식별, {len(df_p)} 식별 rows 저장됨")


if __name__ == "__main__":
    print("🚀 run_pii_detection 단일 테스트 실행")

    test_sentences = [
        "홍길동은 서울특별시 강남구 테헤란로 123에 거주하고 있으며, 전화번호는 010-1234-5678이다.",
        "김영희의 이메일은 younghee@example.com이고, 주민번호는 900101-2345678이다.",
        "카드번호 1234-5678-9876-5432는 유효하지 않습니다."
    ]

    for idx, text in enumerate(test_sentences, 1):
        print(f"\n📌 테스트 문장 {idx}: {text}")
        results = run_pii_detection(text)
        for r in results:
            print(
                f"매치: {r['match']}, "
                f"라벨: {r['label']}, "
                f"위치: ({r['start']}, {r['end']}), "
                f"score: {r['score']:.2f}"
            )
