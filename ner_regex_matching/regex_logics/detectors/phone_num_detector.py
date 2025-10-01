import re

class PhoneDetector:
    def __init__(self):
        # 휴대폰 번호 패턴: 마스킹 포함
        self.mobile_pattern = re.compile(
            r'\b(010)[-\s]?(\d{4}|\*{4})[-\s]?(\d{4}|\*{4})\b'
        )

        # 지역번호 기반 유선 전화 패턴 (단어 경계 사용)
        self.area_codes = [
            '02', '031', '032', '033', '041', '042', '043', '044',
            '051', '052', '053', '054', '055', '061', '062', '063', '064'
        ]
        area_code_pattern = '|'.join(self.area_codes)

        # 지역번호-번호-번호 형식에 대해 단어 경계 포함
        self.landline_pattern = re.compile(
            rf'\b({area_code_pattern})[-\s]?(\d{{3,4}})[-\s]?(\d{{4}})\b'
        )

    def detect(self, text):
        results = []

        # 휴대폰 번호 탐지
        for match in self.mobile_pattern.finditer(text):
            phone_number = '-'.join(match.groups())
            start, end = match.span()
            results.append({
                'start': start,
                'end': end,
                'label': '전화번호',
                'match': phone_number,
                'score': 1.0
            })

        # 유선전화 번호 탐지
        for match in self.landline_pattern.finditer(text):
            phone_number = '-'.join(match.groups())
            start, end = match.span()
            results.append({
                'start': start,
                'end': end,
                'label': '전화번호',
                'match': phone_number,
                'score': 0.5
            })

        return results