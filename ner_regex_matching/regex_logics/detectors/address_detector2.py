import re
from typing import List, Dict
from .detectors.base import BaseDetector  # 추상클래스



class AddressDetector(BaseDetector):
    def __init__(self, sido_list: List[str], sigungu_list: List[str], dong_list: List[str]):
        self.sido_pattern = re.compile(f"({'|'.join(map(re.escape, sido_list))})")
        self.sigungu_pattern = re.compile(f"({'|'.join(map(re.escape, sigungu_list))})")
        self.dong_pattern = re.compile(f"({'|'.join(map(re.escape, dong_list))})")
        self.apt_pattern = re.compile(r"[가-힣]{2,20}(아파트|오피스텔|빌라|타워|센트럴|팰리스|스퀘어|리버|하우스|레지던스)")
        self.dong_num_pattern = re.compile(r"\d+동|\d+층")  # 동 또는 층
        self.ho_num_pattern = re.compile(r"\d+호")  # 호만

        # 전체 주소 블록 (시도~호)
        self.address_block_pattern = re.compile(
            f"({self.sido_pattern.pattern})?\\s*"
            f"({self.sigungu_pattern.pattern})?\\s*"
            f"({self.dong_pattern.pattern})\\s*"
            f"({self.apt_pattern.pattern})\\s*"
            f"({self.dong_num_pattern.pattern})?\\s*"
            f"({self.ho_num_pattern.pattern})?\\s*"
        )

    def detect(self, text: str) -> List[Dict]:

        
        results = []


        #1. 전체 주소 블록 탐지
        for match in self.address_block_pattern.finditer(text):
            matched_text = match.group().strip()
            start, end = match.start(), match.end()

            if matched_text:
                # 블록 점수 계산
                block_score = self.score(matched_text)
                
                label_map = {
                "sido": "도, 주",
                "sigungu": "도시",
                "dong": "군, 면, 동",
                "apartment": "건물명",
                "dong_num": "주소숫자",
                "ho_num": "주소숫자"
            }

                # 개별 요소 탐지 후 동일 점수 부여
                for pattern, eng_label in [
                    (self.sido_pattern, "sido"),
                    (self.sigungu_pattern, "sigungu"),
                    (self.dong_pattern, "dong"),
                    (self.apt_pattern, "apartment"),
                    #(self.ho_pattern, "ho"),
                ]:
                    for m in pattern.finditer(matched_text):
                        results.append({
                            "start": start + m.start(),
                            "end": start + m.end(),
                            "label": label_map[eng_label],
                            "match": m.group(),
                            "score": block_score
                        })
                # 동/층 + 호가 같이 있을 때만
                for dn in self.dong_num_pattern.finditer(matched_text):
                    # dn 이후에 ho_num이 붙어 있는 경우만
                    dn_end = dn.end()
                    next_text = matched_text[dn_end:]
                    ho_match = self.ho_num_pattern.match(next_text.strip())
                    if ho_match:
                        # 동/층 추가
                        results.append({
                            "start": start + dn.start(),
                            "end": start + dn.end(),
                            "label": label_map["dong_num"],
                            "match": dn.group(),
                            "score": block_score
                        })
                        # 호 추가
                        ho_start = dn_end + next_text.index(ho_match.group())
                        results.append({
                            "start": start + ho_start,
                            "end": start + ho_start + len(ho_match.group()),
                            "label": label_map["ho_num"],
                            "match": ho_match.group(),
                            "score": block_score
                        })

        return results

    


    def score(self, match: str) -> float:
        labels = []
        if self.sido_pattern.search(match):
            labels.append("sido")
        if self.sigungu_pattern.search(match):
            labels.append("sigungu")
        if self.dong_pattern.search(match):
            labels.append("dong")
        if self.apt_pattern.search(match):
            labels.append("apartment")
        # 동/층 패턴
        if self.dong_num_pattern.search(match) and self.ho_num_pattern.search(match):
            labels.append("dong_num")
            labels.append("ho_num")
        elif self.dong_num_pattern.search(match):
            labels.append("dong_num")

        return self.score_from_labels(labels)

    def score_from_labels(self, labels: List[str]) -> float:
        label_set = set(labels)
        # 관대하게 점수 주기
        if {"sido", "sigungu", "dong", "apartment", "dong_num", "ho_num"}.issubset(label_set):
            return 1.0
        elif {"sido", "sigungu", "dong", "apartment"}.issubset(label_set):
            return 0.8
        elif {"sigungu", "dong", "apartment", "dong_num", "ho_num"}.issubset(label_set):
            return 0.8
        elif {"sigungu", "dong", "apartment"}.issubset(label_set):
            return 0.6
        elif {"dong", "apartment"}.issubset(label_set):
            return 0.5
        elif {"sido", "sigungu", "dong"}.issubset(label_set):
            return 0.4
        elif {"sido", "sigungu"}.issubset(label_set):
            return 0.3
        elif {"sigungu", "dong"}.issubset(label_set):
            return 0.3
        elif "sido" in label_set:
            return 0.2
        elif "apartment" in label_set:
            return 0.2
        elif "sigungu" in label_set:
            return 0.2
        elif "dong" in label_set:
            return 0.2
        elif "ho" in label_set:
            return 0.1
        else:
            return 0.0