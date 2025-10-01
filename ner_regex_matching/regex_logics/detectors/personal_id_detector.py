import re
from typing import List, Dict

class JuminDetector:
    def __init__(self):
        self.full_pattern = re.compile(
            r'(?<![0-9a-zA-Z])'  # 앞에 숫자/알파벳 없거나 시작
            r'([0-9]{2})(0[1-9]|1[0-2])'       # 생년월
            r'(0[1-9]|[12][0-9]|3[01])'        # 생일
            r'-'                               # 하이픈
            r'([1-4]|[5-8])[0-9]{6}'           # 성별코드 및 나머지
            r'(?![0-9a-zA-Z])'  # 뒤에 숫자/알파벳 없거나 끝
        )

        self.masked_pattern = re.compile(
            r'(?<![0-9a-zA-Z])'
            r'([0-9]{2})(0[1-9]|1[0-2])'
            r'(0[1-9]|[12][0-9]|3[01])'
            r'-'
            r'('
            r'([1-4]|[5-8])[*xX#]{6}'  # 성별코드 + 6자리 마스킹
            r'|'
            r'[*]{7}'                  # 완전 마스킹 7자리
            r')'
            r'(?![0-9a-zA-Z])'
        )

    def detect(self, text: str) -> List[Dict]:
        results = []

        for pattern, score in [(self.full_pattern, 1.0), (self.masked_pattern, 0.8)]:
            for match in pattern.finditer(text):
                results.append({
                    'start': match.start(),
                    'end': match.end(),
                    'label': '주민번호',
                    'match': match.group(),
                    'score': score
                })
        return results