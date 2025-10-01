# ner_regex_matching/ner_logics/ner_main.py

from ner_regex_matching.ner_logics.ner_module import run_ner
from ner_regex_matching.ner_logics.pii_list import extract_entities


def run_ner_detection(sentence):
    # (1) NER 수행 (JSON 저장 없이 메모리 반환)
    ner_results = run_ner(sentence)  # 파일별로 sentence에 대해 ner실행 + BI결합

    # (2) 후처리 → DataFrame 변환
    ner_dictionary = extract_entities(ner_results)  #결합된 BI 목록중 (이름 나이 날짜만 남기기)

    return ner_dictionary