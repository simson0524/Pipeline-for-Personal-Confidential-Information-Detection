#regex_based_doc_parsing/pii_detector/detectors/base.py

from typing import List, Dict
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> List[Dict]:
        pass

    @abstractmethod
    def score(self, match: str) -> float:
        pass




# class PiiDetector:
#     def __init__(self, detectors: List[BaseDetector]):
#         self.detectors = detectors

#     def detect_all(self, data_list: List[Dict]) -> Dict:
#         data_output = []
#         annotations_output = []

#         for item in data_list:
#             sentence = item["sentence"]
#             item_id = item["id"]

#             data_output.append({
#                 "sentence": sentence,
#                 "id": item_id,
#                 "filename": item.get("filename", ""),
#                 "caseField": item.get("caseField", ""),
#                 "detailField": item.get("detailField", ""),
#                 "sequence": item.get("sequence", 0)
#             })

#             annos = []
#             for detector in self.detectors:
#                 for match_info in detector.detect(sentence):
#                     score = detector.score(sentence[match_info["start"]:match_info["end"]])
#                     annos.append({
#                         "start": match_info["start"],
#                         "end": match_info["end"],
#                         "label": match_info["label"],
#                         "score": score
#                     })

#             annotations_output.append({
#                 "id": item_id,
#                 "annotations": annos
#             })

#         return {
#             "data": data_output,
#             "annotations": annotations_output
#         }


# # 사용 예시
# detectors = [NameDetector(), EmailDetector()]  # 여러 detector 추가 가능
# pii_detector = PiiDetector(detectors)

# input_data = [
#     {
#         "sentence": "홍길동의 이메일은 hong@example.com입니다.",
#         "id": "sample_00_000_000001",
#         "filename": "ORIGIN_SOURCE_PATH",
#         "caseField": "1",
#         "detailField": "1",
#         "sequence": 0
#     }
# ]

# result = pii_detector.detect_all(input_data)
# print(result)
