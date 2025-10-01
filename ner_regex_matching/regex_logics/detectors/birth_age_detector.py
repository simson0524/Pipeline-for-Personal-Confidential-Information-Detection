import re
from typing import List, Dict
from .base import BaseDetector  # base.py에 정의된 추상클래스

class BirthAgeDetector(BaseDetector):
    def detect(self, text: str) -> List[Dict]:
        """
        텍스트에서 생년월일 또는 나이 관련 표현을 찾고,
        {"start": 시작인덱스, "end": 끝인덱스, "label": "생년월일"} 형태로 리턴.
        """
        results = []

        # 정규식 패턴들
        patterns = [
            re.compile(r"(19\d{2}|20\d{2})([.\-/년\s]+)(0?[1-9]|1[0-2])([.\-/월\s]+)(0?[1-9]|[12][0-9]|3[01])([일\s]*)(출생|생)"),
            re.compile(r"(19\d{2}|20\d{2})([.\-/년\s]+)(0?[1-9]|1[0-2])([.\-/월\s]+)(0?[1-9]|[12][0-9]|3[01])([일\s]*)$"),
            re.compile(r"(19|20)\d{2}(년|년도)?\s*(생|출생)$"),
            re.compile(r"\b\d{2}년\s*(생|출생)$"),
            re.compile(r"(19|20)\d{2}(년|년도)?$"),
            re.compile(r"\d{2,3}(세|살)\s"),
        ]

        # 모든 패턴에 대해 텍스트를 검색하며 매치시 결과에 저장
        for pattern in patterns:
            for m in pattern.finditer(text):
                results.append({
                    "start": m.start(),
                    "end": m.end(),
                    "label": "나이"
                })

        return results

    def score(self, match: str) -> float:
        """
        매칭된 텍스트에 대해 점수를 계산하는 함수
        """
        value = match.strip()

        # 점수 계산 로직 (기존 score_birth 함수 내용)
        full_birth_pattern = re.compile(
            r"(19\d{2}|20\d{2})([.\-/년\s]+)(0?[1-9]|1[0-2])([.\-/월\s]+)(0?[1-9]|[12][0-9]|3[01])([일\s]*)(출생|생)"
        )
        ymd_pattern_only = re.compile(
            r"(19\d{2}|20\d{2})([.\-/년\s]+)(0?[1-9]|1[0-2])([.\-/월\s]+)(0?[1-9]|[12][0-9]|3[01])([일\s]*)$"
        )
        year_only_with_birth = re.compile(r"(19|20)\d{2}(년|년도)?\s*(생|출생)$")
        year_2digit_with_birth = re.compile(r"\b\d{2}년\s*(생|출생)$")
        year_only_plain = re.compile(r"(19|20)\d{2}(년|년도)?$")
        age_pattern = re.compile(r"^\d{2,3}(세|살)$")

        if full_birth_pattern.fullmatch(value):
            return 1.0
        elif year_only_with_birth.fullmatch(value):
            return 1.0
        elif year_2digit_with_birth.fullmatch(value):
            return 1.0
        elif ymd_pattern_only.fullmatch(value):
            return 0.4
        elif age_pattern.fullmatch(value):
            return 0.5
        elif year_only_plain.fullmatch(value):
            return 0.2
        else:
            return 0.2



