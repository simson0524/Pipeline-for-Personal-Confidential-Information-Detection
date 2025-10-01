# ner_regex_matching/ner_logics/ner_module.py

from typing import Any, Dict, List, Tuple
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import time
import os


labels = [
#    '<PAD>',
    'B-AFA_ART_CRAFT',
    'B-AFA_DOCUMENT',
    'B-AFA_MUSIC',
    'B-AFA_PERFORMANCE',
    'B-AFA_VIDEO',
    'B-AFW_OTHER_PRODUCTS',
    'B-AFW_SERVICE_PRODUCTS',
    'B-AF_BUILDING',
    'B-AF_CULTURAL_ASSET',
    'B-AF_MUSICAL_INSTRUMENT',
    'B-AF_ROAD',
    'B-AF_TRANSPORT',
    'B-AF_WEAPON',
    'B-AM_AMPHIBIA',
    'B-AM_BIRD',
    'B-AM_FISH',
    'B-AM_INSECT',
    'B-AM_MAMMALIA',
    'B-AM_OTHERS',
    'B-AM_PART',
    'B-AM_REPTILIA',
    'B-AM_TYPE',
    'B-CV_ART',
    'B-CV_BUILDING_TYPE',
    'B-CV_CLOTHING',
    'B-CV_CULTURE',
    'B-CV_CURRENCY',
    'B-CV_DRINK',
    'B-CV_FOOD',
    'B-CV_FOOD_STYLE',
    'B-CV_FUNDS',
    'B-CV_LANGUAGE',
    'B-CV_LAW',
    'B-CV_OCCUPATION',
    'B-CV_POLICY',
    'B-CV_POSITION',
    'B-CV_PRIZE',
    'B-CV_RELATION',
    'B-CV_SPORTS',
    'B-CV_SPORTS_INST',
    'B-CV_SPORTS_POSITION',
    'B-CV_TAX',
    'B-CV_TRIBE',
    'B-DT_DAY',
    'B-DT_DURATION',
    'B-DT_DYNASTY',
    'B-DT_GEOAGE',
    'B-DT_MONTH',
    'B-DT_OTHERS',
    'B-DT_SEASON',
    'B-DT_WEEK',
    'B-DT_YEAR',
    'B-EV_ACTIVITY',
    'B-EV_FESTIVAL',
    'B-EV_OTHERS',
    'B-EV_SPORTS',
    'B-EV_WAR_REVOLUTION',
    'B-FD_ART',
    'B-FD_HUMANITIES',
    'B-FD_MEDICINE',
    'B-FD_OTHERS',
    'B-FD_SCIENCE',
    'B-FD_SOCIAL_SCIENCE',
    'B-LCG_BAY',
    'B-LCG_CONTINENT',
    'B-LCG_ISLAND',
    'B-LCG_MOUNTAIN',
    'B-LCG_OCEAN',
    'B-LCG_RIVER',
    'B-LCP_CAPITALCITY',
    'B-LCP_CITY',
    'B-LCP_COUNTRY',
    'B-LCP_COUNTY',
    'B-LCP_PROVINCE',
    'B-LC_OTHERS',
    'B-LC_SPACE',
    'B-MT_CHEMICAL',
    'B-MT_ELEMENT',
    'B-MT_METAL',
    'B-MT_ROCK',
    'B-OGG_ART',
    'B-OGG_ECONOMY',
    'B-OGG_EDUCATION',
    'B-OGG_FOOD',
    'B-OGG_HOTEL',
    'B-OGG_LAW',
    'B-OGG_LIBRARY',
    'B-OGG_MEDIA',
    'B-OGG_MEDICINE',
    'B-OGG_MILITARY',
    'B-OGG_OTHERS',
    'B-OGG_POLITICS',
    'B-OGG_RELIGION',
    'B-OGG_SCIENCE',
    'B-OGG_SPORTS',
    'B-PS_CHARACTER',
    'B-PS_NAME',
    'B-PS_PET',
    'B-PT_FLOWER',
    'B-PT_FRUIT',
    'B-PT_GRASS',
    'B-PT_OTHERS',
    'B-PT_PART',
    'B-PT_TREE',
    'B-PT_TYPE',
    'B-QT_ADDRESS',
    'B-QT_AGE',
    'B-QT_ALBUM',
    'B-QT_CHANNEL',
    'B-QT_COUNT',
    'B-QT_LENGTH',
    'B-QT_MAN_COUNT',
    'B-QT_ORDER',
    'B-QT_OTHERS',
    'B-QT_PERCENTAGE',
    'B-QT_PHONE',
    'B-QT_PRICE',
    'B-QT_SIZE',
    'B-QT_SPEED',
    'B-QT_SPORTS',
    'B-QT_TEMPERATURE',
    'B-QT_VOLUME',
    'B-QT_WEIGHT',
    'B-TI_DURATION',
    'B-TI_HOUR',
    'B-TI_MINUTE',
    'B-TI_OTHERS',
    'B-TI_SECOND',
    'B-TMIG_GENRE',
    'B-TMI_EMAIL',
    'B-TMI_HW',
    'B-TMI_MODEL',
    'B-TMI_PROJECT',
    'B-TMI_SERVICE',
    'B-TMI_SITE',
    'B-TMI_SW',
    'B-TMM_DISEASE',
    'B-TMM_DRUG',
    'B-TM_CELL_TISSUE_ORGAN',
    'B-TM_CLIMATE',
    'B-TM_COLOR',
    'B-TM_DIRECTION',
    'B-TM_SHAPE',
    'B-TM_SPORTS',
    'B-TR_ART',
    'B-TR_HUMANITIES',
    'B-TR_MEDICINE',
    'B-TR_OTHERS',
    'B-TR_SCIENCE',
    'B-TR_SOCIAL_SCIENCE',
    'I-AFA_ART_CRAFT',
    'I-AFA_DOCUMENT',
    'I-AFA_MUSIC',
    'I-AFA_PERFORMANCE',
    'I-AFA_VIDEO',
    'I-AFW_OTHER_PRODUCTS',
    'I-AFW_SERVICE_PRODUCTS',
    'I-AF_BUILDING',
    'I-AF_CULTURAL_ASSET',
    'I-AF_MUSICAL_INSTRUMENT',
    'I-AF_ROAD',
    'I-AF_TRANSPORT',
    'I-AF_WEAPON',
    'I-AM_AMPHIBIA',
    'I-AM_BIRD',
    'I-AM_FISH',
    'I-AM_INSECT',
    'I-AM_MAMMALIA',
    'I-AM_OTHERS',
    'I-AM_PART',
    'I-AM_REPTILIA',
    'I-AM_TYPE',
    'I-CV_ART',
    'I-CV_BUILDING_TYPE',
    'I-CV_CLOTHING',
    'I-CV_CULTURE',
    'I-CV_CURRENCY',
    'I-CV_DRINK',
    'I-CV_FOOD',
    'I-CV_FOOD_STYLE',
    'I-CV_FUNDS',
    'I-CV_LANGUAGE',
    'I-CV_LAW',
    'I-CV_OCCUPATION',
    'I-CV_POLICY',
    'I-CV_POSITION',
    'I-CV_PRIZE',
    'I-CV_RELATION',
    'I-CV_SPORTS',
    'I-CV_SPORTS_INST',
    'I-CV_SPORTS_POSITION',
    'I-CV_TAX',
    'I-CV_TRIBE',
    'I-DT_DAY',
    'I-DT_DURATION',
    'I-DT_DYNASTY',
    'I-DT_GEOAGE',
    'I-DT_MONTH',
    'I-DT_OTHERS',
    'I-DT_SEASON',
    'I-DT_WEEK',
    'I-DT_YEAR',
    'I-EV_ACTIVITY',
    'I-EV_FESTIVAL',
    'I-EV_OTHERS',
    'I-EV_SPORTS',
    'I-EV_WAR_REVOLUTION',
    'I-FD_ART',
    'I-FD_HUMANITIES',
    'I-FD_MEDICINE',
    'I-FD_OTHERS',
    'I-FD_SCIENCE',
    'I-FD_SOCIAL_SCIENCE',
    'I-LCG_BAY',
    'I-LCG_CONTINENT',
    'I-LCG_ISLAND',
    'I-LCG_MOUNTAIN',
    'I-LCG_OCEAN',
    'I-LCG_RIVER',
    'I-LCP_CAPITALCITY',
    'I-LCP_CITY',
    'I-LCP_COUNTRY',
    'I-LCP_COUNTY',
    'I-LCP_PROVINCE',
    'I-LC_OTHERS',
    'I-LC_SPACE',
    'I-MT_CHEMICAL',
    'I-MT_ELEMENT',
    'I-MT_METAL',
    'I-MT_ROCK',
    'I-OGG_ART',
    'I-OGG_ECONOMY',
    'I-OGG_EDUCATION',
    'I-OGG_FOOD',
    'I-OGG_HOTEL',
    'I-OGG_LAW',
    'I-OGG_LIBRARY',
    'I-OGG_MEDIA',
    'I-OGG_MEDICINE',
    'I-OGG_MILITARY',
    'I-OGG_OTHERS',
    'I-OGG_POLITICS',
    'I-OGG_RELIGION',
    'I-OGG_SCIENCE',
    'I-OGG_SPORTS',
    'I-PS_CHARACTER',
    'I-PS_NAME',
    'I-PS_PET',
    'I-PT_FLOWER',
    'I-PT_FRUIT',
    'I-PT_GRASS',
    'I-PT_OTHERS',
    'I-PT_PART',
    'I-PT_TREE',
    'I-PT_TYPE',
    'I-QT_ADDRESS',
    'I-QT_AGE',
    'I-QT_ALBUM',
    'I-QT_CHANNEL',
    'I-QT_COUNT',
    'I-QT_LENGTH',
    'I-QT_MAN_COUNT',
    'I-QT_ORDER',
    'I-QT_OTHERS',
    'I-QT_PERCENTAGE',
    'I-QT_PHONE',
    'I-QT_PRICE',
    'I-QT_SIZE',
    'I-QT_SPEED',
    'I-QT_SPORTS',
    'I-QT_TEMPERATURE',
    'I-QT_VOLUME',
    'I-QT_WEIGHT',
    'I-TI_DURATION',
    'I-TI_HOUR',
    'I-TI_MINUTE',
    'I-TI_OTHERS',
    'I-TI_SECOND',
    'I-TMIG_GENRE',
    'I-TMI_EMAIL',
    'I-TMI_HW',
    'I-TMI_MODEL',
    'I-TMI_PROJECT',
    'I-TMI_SERVICE',
    'I-TMI_SITE',
    'I-TMI_SW',
    'I-TMM_DISEASE',
    'I-TMM_DRUG',
    'I-TM_CELL_TISSUE_ORGAN',
    'I-TM_COLOR',
    'I-TM_DIRECTION',
    'I-TM_SHAPE',
    'I-TM_SPORTS',
    'I-TR_ART',
    'I-TR_HUMANITIES',
    'I-TR_MEDICINE',
    'I-TR_OTHERS',
    'I-TR_SCIENCE',
    'I-TR_SOCIAL_SCIENCE',
    'O',
#    '<START>',
#    '<STOP>'
]

# NER 클래스
ner_code = {
    'PS_NAME': '인물',
    'PS_CHARACTER': '캐랙터',
    'PS_PET': '반려동물',
    'FD_SCIENCE': '과학',
    'FD_SOCIAL_SCIENCE': '사회과학',
    'FD_MEDICINE': '의학',
    'FD_ART': '예술',
    'FD_HUMANITIES': '인문학',
    'FD_OTHERS': '기타',
    'TR_SCIENCE': '과학',
    'TR_SOCIAL_SCIENCE': '사회과학',
    'TR_MEDICINE': '의학',
    'TR_ART': '예술',
    'TR_HUMANITIES': '인문학',
    'TR_OTHERS': '기타',
    'AF_BUILDING': '건물',
    'AF_CULTURAL_ASSET': '문화재',
    'AF_ROAD': '도로, 철도',
    'AF_TRANSPORT': '교통수단, 놀이기구',
    'AF_MUSICAL_INSTRUMENT': '악기',
    'AF_WEAPON': '무기',
    'AF_ART_WORKS': '작품',
    'AF_WARES': '상품',
    'OGG_ECONOMY': '경제',
    'OGG_EDUCATION': '교육',
    'OGG_MILITARY': '군사',
    'OGG_MEDIA': '미디어 방송',
    'OGG_SPORTS': '스포츠',
    'OGG_ART': '예술',
    'OGG_MEDICINE': '의학 의료',
    'OGG_RELIGION': '종교',
    'OGG_SCIENCE': '과학',
    'OGG_LIBRARY': '도서관',
    'OGG_LAW': '법률',
    'OGG_POLITICS': '정부 행정',
    'OGG_FOOD': '음식',
    'OGG_HOTEL': '호텔',
    'OGG_OTHERS': '기타',
    'LCP_COUNTRY': '국가',
    'LCP_PROVINCE': '도, 주',
    'LCP_COUNTY': '군, 면, 동',
    'LCP_CITY': '도시',
    'LCP_CAPITALCITY': '수도',
    'LCG_RIVER': '강, 호수',
    'LCG_OCEAN': '바다',
    'LCG_BAY': '반도, 만',
    'LCG_MOUNTAIN': '산 ',
    'LCG_ISLAND': '섬',
    'LCG_CONTINENT': '대륙',
    'LC_SPACE': '천체',
    'LC_OTHERS': '기타',
    'CV_CULTURE': '문명, 혁명',
    'CV_TRIBE': '민족, 종족',
    'CV_LANGUAGE': '언어',
    'CV_POLICY': '제도, 정책',
    'CV_LAW': '법률',
    'CV_CURRENCY': '통화',
    'CV_TAX': '조세',
    'CV_FUNDS': '연금, 기금',
    'CV_ART': '예술분류',
    'CV_SPORTS': '스포츠, 레포츠, 레저',
    'CV_SPORTS_POSITION': '스포츠포지션',
    'CV_SPORTS_INST': '스포츠용품',
    'CV_PRIZE': '상, 훈장',
    'CV_RELATION': '가족관계',
    'CV_OCCUPATION': '직업',
    'CV_POSITION': '직위, 직책',
    'CV_FOOD': '음식 식재료',
    'CV_DRINK': '음료, 술',
    'CV_FOOD_STYLE': '음식유형',
    'CV_CLOTHING': '의복, 섬유',
    'CV_BUILDING_TYPE': '건축양식',
    'DT_DURATION': '기간',
    'DT_DAY': '날짜 절기',
    'DT_WEEK': '주',
    'DT_MONTH': '달',
    'DT_YEAR': '년',
    'DT_SEASON': '계절',
    'DT_GEOAGE': '지질시대',
    'DT_DYNASTY': '왕조시대',
    'DT_OTHERS': '기타',
    'TI_DURATION': '기간',
    'TI_HOUR': '시각',
    'TI_MINUTE': '분',
    'TI_SECOND': '초',
    'TI_OTHERS': '기타',
    'QT_AGE': '나이',
    'QT_SIZE': '면적',
    'QT_LENGTH': '길이',
    'QT_COUNT': '수량',
    'QT_MAN_COUNT': '인원',
    'QT_WEIGHT': '무게',
    'QT_PERCENTAGE': '비율',
    'QT_SPEED': '속도',
    'QT_TEMPERATURE': '온도',
    'QT_VOLUME': '부피',
    'QT_ORDER': '순서',
    'QT_PRICE': '금액',
    'QT_PHONE': '전화번호',
    'QT_SPORTS': '스포츠수량',
    'QT_CHANNEL': '미디어채널',
    'QT_ALBUM': '음악앨범수량',
    'QT_ADDRESS': '주소숫자',
    'QT_OTHERS': '기타',
    'EV_ACTIVITY': '사회운동, 선언',
    'EV_WAR_REVOLUTION': '전쟁, 혁명, 전투',
    'EV_SPORTS': '스포츠, 레저행사',
    'EV_FESTIVAL': '축제, 영화제, 콘서트, 행사',
    'EV_OTHERS': '기타',
    'AM_INSECT': '곤충',
    'AM_BIRD': '조류',
    'AM_FISH': '어류',
    'AM_MAMMALIA': '포유류',
    'AM_AMPHIBIA': '양서류',
    'AM_REPTILIA': '파출류',
    'AM_TYPE': '동물분류',
    'AM_PART': '부분명칭',
    'AM_OTHERS': '기타',
    'PT_FRUIT': '과일',
    'PT_FLOWER': '꽃',
    'PT_TREE': '나무',
    'PT_GRASS': '풀',
    'PT_TYPE': '식물',
    'PT_PART': '부분명칭',
    'PT_OTHERS': '기타',
    'MT_ELEMENT': '원소',
    'MT_METAL': '금속',
    'MT_ROCK': '암석',
    'MT_CHEMICAL': '화학물질',
    'TM_COLOR': '색깔',
    'TM_DIRECTION': '방향',
    'TM_CLIMATE': '기후',
    'TM_SHAPE': '모양',
    'TM_CELL_TISSUE_ORGAN': '세포, 조직, 기관',
    'TMM_DISEASE': '증상, 증세, 질병',
    'TMM_DRUG': '약',
    'TMI_HW': '하드웨어',
    'TMI_SW': '소프트웨어',
    'TMI_SITE': 'URL주소',
    'TMI_EMAIL': '이메일주소',
    'TMI_MODEL': '모델명 일련번호',
    'TMI_SERVICE': 'IT서비스용어',
    'TMI_PROJECT': '프로젝트',
    'TMIG_GENRE': '게임장르',
    'TM_SPORTS': '스포츠레저기술규칙'}
# ner_module_min.py  (간결 버전)

# --- 전제: 아래 두 개는 네가 기존에 선언한 것을 그대로 둔다고 가정 ---
# labels = [...]
# ner_code = {...}

# 라벨 매핑
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# HF 토큰 확인 및 클라이언트/토크나이저 준비 (한 번만)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("환경변수 HF_TOKEN 이 설정되어 있지 않습니다. os.environ['HF_TOKEN'] = 'hf_xxx' 로 설정하세요.")

client = InferenceClient(provider="auto", api_key=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("KPF/KPF-bert-ner")

# 청크 파라미터
MAX_LEN, STRIDE = 510, 50


def chunk_text_with_offsets(text: str) -> List[Tuple[str, int]]:
    """긴 문장을 토큰 길이 기준으로 겹치며 분할하고 (부분문장, 원문 시작오프셋)을 반환."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    chunks: List[Tuple[str, int]] = []
    start = 0
    while start < len(ids):
        end = min(len(ids), start + MAX_LEN)
        if end == 0:
            break
        char_s = offsets[start][0]
        char_e = offsets[end - 1][1]
        chunks.append((text[char_s:char_e], char_s))
        if end == len(ids):
            break
        start = end - STRIDE
    return chunks


def safe_inference(text: str, model: str = "KPF/KPF-bert-ner", max_retries: int = 3, wait_sec: int = 30) -> List[Dict[str, Any]]:
    """서버 500 오류에 대해 재시도하며 token_classification 호출."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.token_classification(text, model=model)
        except Exception as e:
            msg = str(e)
            if "500" in msg and attempt < max_retries:
                print(f"⚠️ 서버 오류 → {wait_sec}초 대기 후 재시도 ({attempt}/{max_retries})")
                time.sleep(wait_sec)
                continue
            print(f"❌ 추론 실패: {e}")
            return []
    return []


def merge_bi(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    BIO 토큰 엔티티 리스트를 결합.
    - 연속된 B-/I- (동일 entity_type) 토큰을 하나로 묶고 span/score 갱신
    - O 라벨은 그대로 유지
    """
    merged: List[Dict[str, Any]] = []
    buffer: Dict[str, Any] | None = None

    for ent in entities:
        bio = ent.get("bio_label", "O")
        ent_type = ent.get("entity_type", "O")

        if isinstance(bio, str) and bio.startswith("B-"):
            if buffer:
                merged.append(buffer)
            buffer = ent.copy()

        elif isinstance(bio, str) and bio.startswith("I-") and buffer and buffer.get("entity_type") == ent_type:
            # 토큰 이어붙이기(워드피스 접두어 제거)
            buffer["token"] = str(buffer.get("token", "")) + str(ent.get("token", "")).replace("##", "")
            # span 끝 확장
            span = buffer.get("span", [0, 0])
            span[1] = ent.get("span", span)[1]
            buffer["span"] = span
            # 점수 이동 평균
            prev = float(buffer.get("score", 0.0))
            cur = float(ent.get("score", 0.0))
            buffer["score"] = round((prev + cur) / 2, 3)

        else:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(ent)

    if buffer:
        merged.append(buffer)

    return merged


def run_ner(sentence: str) -> List[Dict[str, Any]]:
    """
    입력: 단일 문장 문자열
    출력: [{"sentence": str, "entities": [ {token,bio_label,entity_type,description,score,span}, ... ]}]
    """
    sent = (sentence or "").strip()
    if not sent:
        return [{"sentence": "", "entities": []}]

    # 청크 분할
    enc_len = len(tokenizer.encode(sent, add_special_tokens=False))
    chunks = [(sent, 0)] if enc_len <= MAX_LEN else chunk_text_with_offsets(sent)

    # 추론
    all_entities: List[Dict[str, Any]] = []
    for chunk_text, offset in chunks:
        toks = safe_inference(chunk_text)
        if not toks:
            print("⚠️ 추론 실패 청크 스킵")
            continue

        for ent in toks:
            # entity_group: "LABEL_3" 또는 이미 BIO 라벨 문자열일 수도 있음
            entity_group = ent.get("entity_group", "O")
            bio_lbl = "O"
            if isinstance(entity_group, str):
                try:
                    _, idx_str = entity_group.split("_")
                    idx = int(idx_str)
                    bio_lbl = id2label.get(idx, "O")
                except Exception:
                    bio_lbl = entity_group  # 이미 BIO 라벨일 수 있음

            ent_type = bio_lbl.split("-", 1)[-1] if "-" in bio_lbl else bio_lbl
            desc = ner_code.get(ent_type, ent_type)

            all_entities.append({
                "token": str(ent.get("word", "")).lstrip("##"),
                "bio_label": bio_lbl,
                "entity_type": ent_type,
                "description": desc,
                "score": round(float(ent.get("score", 0.0)), 3),
                "span": [int(ent.get("start", 0) + offset), int(ent.get("end", 0) + offset)],
            })

    # BIO 병합
    merged = merge_bi(all_entities)

    # 결과 리스트로 반환
    return merged