import re
from typing import List, Dict

class EmailDetector:
    def __init__(self):
        # 자주 쓰이는 도메인 확장자 리스트 (회사, 학교, 기관 등)
        self.allowed_domains = [
            "com", "net", "org", "co.kr", "ac.kr", "go.kr",
            "edu", "gov", "mil", "biz", "info", "name", "io",
            "kr", "xyz"
        ]

        # 도메인 확장자 정규식 조합 (예: com|net|org|co\.kr|ac\.kr ...)
        domain_pattern = '|'.join([re.escape(d) for d in self.allowed_domains])

        # 이메일 정규식 - username@domain.domain_extension
        # domain 부분은 영숫자, 하이픈, 점(.) 허용
        self.email_pattern = re.compile(
            rf"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.({domain_pattern}))"
        )

    def detect(self, text: str) -> List[Dict]:
        results = []
        for match in self.email_pattern.finditer(text):
            email = match.group(1)
            
            # 점수 계산: 마스킹 여부 체크
            if "*" in email or "…" in email:
                score = 0.5
            else:
                score = 1.0

            results.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "이메일주소",
                "match": email,
                "score": score
            })
        return results
