# ner_regex_matching/ner_logics/pii_list.py


def extract_entities(
    ner_results,
    personal_types=None,
    confidential_types=None,
    identifier_types=None,
    quasi_identifier_types=None
):
    """
    NER 결과(list[dict])를 받아
    results_item 포맷의 리스트[dict] 반환
    """
    personal_types      = personal_types or {"PS_NAME","DT_YEAR","DT_MONTH","DT_DAY","DT_WEEK","QT_AGE"}
    confidential_types  = confidential_types or set()
    identifier_types    = identifier_types or set()
    quasi_identifier_types = quasi_identifier_types or set()
    interest_types = personal_types | confidential_types | identifier_types | quasi_identifier_types

    results = []
    for item in ner_results:
        for ent in item.get("entities") or []:
            etype = ent.get("entity_type", "")
            if etype not in interest_types:
                continue

            pc = "개인" if etype in personal_types else "기밀"
            # description 필드를 label로 매핑
            label = ent.get("description", "")

            results_item = {
                "단어": ent.get("token", ""),
                "부서명": None,      # 기본 None
                "문서명": None,      # 기본 None
                "단어유형": label,
                "구분": pc,
            }
            results.append(results_item)

    return results