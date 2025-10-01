import re
from typing import List, Dict
from .base import BaseDetector
from ..Dict.name_dict import sn1, nn1, nn2, name
from ..Dict.stopwords_dict import stopwords



import re
from typing import List, Dict

class NameDetector:
    def __init__(self, sn1: List[str], nn1: List[str], nn2: List[str], name: List[str], stopwords: List[str] = None):
        self.sn1 = set(sn1)
        self.nn1 = set(nn1)
        self.nn2 = set(nn2)
        self.name = set(name)
        self.stopwords = set(stopwords) if stopwords else set()

        # 조사 리스트
        self.postpositions = ['은', '는', '이', '가', '도']

        # 정규식 패턴 준비
        # 이름 후보: 3글자(성+중간이름+마지막이름) or 2글자(성+name)
        # 성, 이름 조합 각각 가능해야 하므로 두 패턴으로 나눠서 OR 처리
        sn_pattern = '|'.join(map(re.escape, self.sn1))
        nn1_pattern = '|'.join(map(re.escape, self.nn1))
        nn2_pattern = '|'.join(map(re.escape, self.nn2))
        name_pattern = '|'.join(map(re.escape, self.name))

        # 3글자 이름 패턴
        pattern_3char = f"({sn_pattern})({nn1_pattern})({nn2_pattern})"
        # 2글자 이름 패턴
        pattern_2char = f"({sn_pattern})({name_pattern})"

        # 조사 또는 쉼표
        postp_pattern = f"[{''.join(self.postpositions)},]?"

        # 앞뒤 경계는 시작(^), 공백(\s), 쉼표(,), 혹은 문장 경계
        boundary = r"(^|[\s,])"
        boundary_end = r"(?=[\s,]|$)"

        # 최종 패턴 (이름 뒤에 조사 또는 쉼표가 올 수 있음)
        self.pattern = re.compile(
            rf"{boundary}(({pattern_3char}|{pattern_2char}){postp_pattern}){boundary_end}"
        )

    def detect(self, text: str) -> List[Dict]:
        results = []
        for match in self.pattern.finditer(text):
            full_match = match.group(2)  # 이름 + 조사 부분 포함

            # 조사나 쉼표 제거
            name_clean = re.sub(f"[{''.join(self.postpositions)},]$", "", full_match).strip()

            # stopwords 제외
            if name_clean in self.stopwords or len(name_clean) < 2:
                continue

            start = match.start(2)
            end = start + len(name_clean)

            results.append({
                "start": start,
                "end": end,
                "label": "인물",
                "match": name_clean
            })

        return results

    def score(self, match: str) -> float:
        name = match.strip()
        if len(name) < 2 or len(name) > 4 or name in self.stopwords:
            return None

        if name[0] not in self.sn1:
            return None

        # 완전 일치는 성 + 중간 + 끝 자모 조합 혹은 성 + 이름 리스트에 포함되는지 검사
        if (len(name) == 3 and
            name[1] in self.nn1 and
            name[2] in self.nn2):
            return 1.0

        if (len(name) == 2 and
            name[1:] in self.name):
            return 1.0

        # 블러 처리: 김*수
        if len(name) == 3 and name[1] == '*':
            s, _, n2 = name
            if s in self.sn1 and n2 in self.nn2:
                return 0.8

        # 블러 처리: 김민*
        if len(name) == 3 and name[2] == '*':
            if name[0] in self.sn1 and name[1] in self.nn1:
                return 0.8

        # 블러 처리: 김**
        if len(name) == 3 and name[0] in self.sn1 and name[1:] == '**':
            return 0.5

        # 블러 처리: **수
        if len(name) == 3 and name[2] in self.nn2 and name[:2] == '**':
            return 0.4

        # 마스킹된 이름 처리
        if any([
            re.fullmatch(r'[가-힣][0○△□]{2}', name),      # 김00, 김○○
            re.fullmatch(r'[가-힣]모', name),              # 김모
            re.fullmatch(r'[가-힣] 모', name),             # 김 모
            re.fullmatch(r'[가-힣]모씨', name),            # 김모씨
            re.fullmatch(r'[가-힣] 모씨', name),           # 김 모씨
            re.fullmatch(r'[가-힣][Xx*#]{2}', name),       # 김**, 김XX
        ]) and name[0] in self.sn1:
            return 0.5

        return None
    

