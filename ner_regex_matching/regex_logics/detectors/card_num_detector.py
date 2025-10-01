import re
from typing import List, Dict

class CardNumDetector:
    def __init__(self):
        # BIN 규칙 기반 카드번호 후보 (13~19자리, 하이픈/공백 허용)
        self.full_pattern = re.compile(
            r'(?<!\d)'
            r'((?:4[0-9]{12}(?:[0-9]{3})?)'       # Visa (13 또는 16자리)
            r'|(?:5[1-5][0-9]{14})'                # MasterCard (16자리)
            r'|(?:3[47][0-9]{13})'                 # Amex (15자리)
            r'|(?:6(?:011|5[0-9]{2})[0-9]{12}))'   # Discover (16자리)
            r'(?!\d)'
        )

        # 마스킹된 카드번호 (**** 포함)
        self.masked_pattern = re.compile(
            r'(?<!\d)(?:\d{4}|\*{4})(?:[- ]?(?:\d{4}|\*{4})){3}(?!\d)'
        )

    def detect(self, text: str) -> List[Dict]:
        results = []

        # 풀 카드번호 탐지 (BIN 규칙 기반)
        for match in self.full_pattern.finditer(text):
            card = match.group()
            if self.luhn_check(card):  # Luhn 검증
                results.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": "카드번호",
                    "match": card,
                    "score": 1.0
                })

        # 마스킹된 카드번호 탐지
        for match in self.masked_pattern.finditer(text):
            card = match.group()
            results.append({
                "start": match.start(),
                "end": match.end(),
                "label": "카드번호",
                "match": card,
                "score": 0.5
            })

        return results

# 신용카드 번호나 카드번호 형태의 유효성을 체크할 때 쓰이는 함수
    def luhn_check(self, card_number: str) -> bool: 
        # 하이픈, 공백 제거
        num = re.sub(r'[^0-9]', '', card_number)
        total = 0
        reverse_digits = num[::-1]
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0
