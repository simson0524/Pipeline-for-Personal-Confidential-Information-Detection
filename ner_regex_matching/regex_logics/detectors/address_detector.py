import re

class AddressDetector:
    def __init__(self, sido_list, sigungu_list, dong_list):
        self.sido_list = sido_list
        self.sigungu_list = sigungu_list
        self.dong_list = dong_list

        # ✅ 리스트 기반 패턴
        self.sido_pattern = re.compile("|".join(map(re.escape, self.sido_list)))
        self.sigungu_pattern = re.compile("|".join(map(re.escape, self.sigungu_list)))
        self.dong_pattern = re.compile("|".join(map(re.escape, self.dong_list)))

        # ✅ 도로명/아파트/동호/번지 패턴
        self.road_pattern = re.compile(r"[가-힣0-9]+(?:로|길|대로)")
        self.apt_pattern = re.compile(r"(?:[가-힣0-9]+)?(?:아파트|빌딩|타워|오피스텔|기숙사|본관|별관)")
        self.dong_num_pattern = re.compile(r"\d+동")
        self.ho_num_pattern = re.compile(r"\d+호")
        self.addr_num_pattern = re.compile(r"\d+(?:-\d+)?(?:번지)?")

       
        self.address_block_pattern = re.compile(
        rf"({self.sido_pattern.pattern})?\s*"              # 1: 시도
        rf"({self.sigungu_pattern.pattern})?\s*"           # 2: 시군구1
        rf"({self.sigungu_pattern.pattern})?\s*"           # 3: 시군구2
        rf"({self.dong_pattern.pattern})?\s*"              # 4: 동/읍/면
        rf"({self.road_pattern.pattern})?\s*"              # 5: 도로명 (단독 캡처 허용)
        rf"({self.addr_num_pattern.pattern})?\s*"          # 6: 번지 (도로명 또는 동 있으면 캡처)
        rf"({self.apt_pattern.pattern})?\s*"               # 7: 건물명
        rf"({self.dong_num_pattern.pattern})?\s*"          # 8: 동번호 (건물명 없어도 허용)
        rf"({self.ho_num_pattern.pattern})?"               # 9: 호수 (건물명 없어도 허용)
        
)

    
    def detect(self, text: str):
        results = []
        label_map = {
            1: "도시",
            2: "도, 주",
            3: "도, 주",
            4: "군, 면, 동",
            5: "도로명",
            6: "건물명",
            7: "주소숫자",
            8: "주소숫자",
            9: "주소숫자",
        }

        for m in self.address_block_pattern.finditer(text):
            captured = [(i, m.group(i)) for i in range(1, 10) if m.group(i)]
            if not captured:
                continue

            # --- 조건별 플래그 ---
            has_sigungu = bool(m.group(2)) or bool(m.group(3))
            has_dong = bool(m.group(4))
            has_road = bool(m.group(5))
            has_addrnum = bool(m.group(6))
            has_building = bool(m.group(7))

            # ✅ 도로명은 반드시 동과 같이 등장해야만 유효
            if has_road and not (has_dong or has_sigungu):
                continue

            # ✅ 번지도 도로명 또는 동과 함께 있을 때만 유효
            if has_addrnum and not (has_road or has_dong):
                continue

            

             # ✅ 점수 계산 로직 (매칭 개수 기반)
            match_count = len(captured)
            if match_count >= 5: # 매칭 된 토큰 개수가 5개 이상 : 1.0점
                score = 1.0
            elif 4 <= match_count < 5: # 매칭 된 토큰 개수가 4개 : 0.6점
                score = 0.6
            elif 2 <= match_count < 4: # 매칭 된 토큰 개수가 2~3개 : 0.4점
                score = 0.4
            else:
                score = 0.2  # 매칭 된 토큰 개수가 1개 : 0.2점

            for idx, val in captured:
                results.append({
                    "start": m.start(idx),
                    "end": m.end(idx),
                    "match": val,
                    "label": label_map.get(idx, "기타"),
                    "score": score,
                })

        return results



sido_list = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도"
]
sigungu_list = ["성남시", "분당구", "수원시", "영통구", "용인시", "수지구", "강남구"]
dong_list = ["판교동", "정자동", "야탑동", "서현동", "역삼동","덕천동","오포읍"]

detector = AddressDetector(sido_list, sigungu_list, dong_list)

test_addresses = [
    "경기도 성남시 분당구 판교동 123-45",
    "경기도 수원시 영통구 정자동 풍덕천로 77",
    "경기도 용인시 수지구 서현동 아파트 999동 1호",
    "서울특별시 강남구 역삼동 ",
    "경기도 용인시 수지구 서현동 풍덕천로 301-2번지",
    "경기도 성남시 분당구 판교동 123-45",
    "용인시 서현동 풍덕천로",
    "서울특별시",
    "서울특별시 강남구 테헤란로 152 강남빌딩 301호",
    "경기도 성남시 분당구 판교동 123-45 현대아파트 202동 1503호",
    "경기도 용인시 수지구 덕천동 22-24",
    "경기도 용인시 오포읍"
]

for addr in test_addresses:
    print(f"\n📍 테스트 주소: {addr}")
    results = detector.detect(addr)
    if results:
        for r in results:
            print(f" - [{r['label']}] {r['match']} ({r['start']}~{r['end']})| score={r['score']}")
    else:
        print("❌ 탐지 실패")