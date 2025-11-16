# generated_augmentation/generate_sentences.py

from datetime import datetime
from openai import OpenAI
import os
import re

### API KEY ###
JH_GPT_API_KEY = os.environ.get("JH_GPT_API_KEY")
WW_GPT_API_KEY = os.environ.get("WW_GPT_API_KEY")
HJ_GPT_API_KEY = os.environ.get("HJ_GPT_API_KEY")


def generate_n_sentences(n, span_token, gt_label, pred_label, is_pii=True, model_name="gpt-5-mini", api_key=JH_GPT_API_KEY):
    start_time = datetime.now()

    client = OpenAI(api_key=api_key)

    if is_pii:
        prompt = f"""
        -개인정보 : 단 하나만으로도 특정 개인을 지칭하거나 추정할 수 있는 정보(이름, 주소, 주민번호, 사번 등)

        -준식별자 : 하나로는 특정 개인을 지칭할 수 없지만 여러 준식별정보가 결합하는 경우나, 소량의 준식별정보와 주변에 개인정보가 분포하는 경우.
        특정 개인을 지칭하거나 추정이 가능한 정보
        (직업, 시군구 주소, 신체관련정보, 생일 등)
        (시+군+구+세부주소가 나오면 시,군,구 까지는 각각 준식별자로 처리해야 해)
        (생년월일일 경우에는 년,월,일이 각각 준식별자로 처리되어야해.)

        -일반정보 : 개인정보와 준식별정보가 아닌 모든 정보


        우리는 {span_token}(이)가 {gt_label}라고 생각해서 이렇게 라벨링을 해서 데이터셋으로 사용했거든,
        그런데 우리 모델이 {span_token}(이)라는 단어를 "{pred_label}로 예측을 했어.

        이는 {span_token}가 {gt_label}와 {pred_label}로 모두 사용될 수 있다고 추론할 수 있을 것 같아.

        {span_token}가 문맥상 {pred_label}로 사용될 수 있는 예시 문장 {n}개를 생성해줬으면 좋겠어.
        각 문장은 char단위로 150개 이상 250개 이하로 별도의 추론설명 없이 한글 맞춤법에 유의하여 주고, 반드시 아래 포맷에 정확히 맞춰 구성해주면 좋겠어.

        특수문자는 ,와 .을 제외하고는 사용하지 마. 
        
        ===== 문장들의 시작 =====
        === {pred_label} 문장 ===
        [문장 1을 여기에 생성해줘...]
        === {pred_label} 문장 ===
        [문장 2를 여기에 생성해줘...]
        ...
        === {pred_label} 문장 ===
        [문장 {n}을 여기에 생성해줘...]
        ===== 문장들의 끝 ====="""
    else:
        prompt = f"""
        -기밀정보 : 
        1. 조직/단체 운영 정보
        직위/역할과 결합된 내부 직원명 (예: “회장 김민수”, “간사 이영희”)
        내부 의사결정 라인(결재선, 승인자, 지도교수, 운영진 명단 등)

        2. 기술·보안 정보
        시스템 접근 정보(계정, 링크, 비밀번호 힌트)
        내부 공유 문서명, 파일명, 설계도, 절차 문서
        미공개 URL, 회의 링크, 공유 드라이브 주소

        3.재무/운영 기밀
        개인 귀속 재정 정보(계좌번호, 후원자 명단, 회비 납부 내역)
        외부 공개되지 않은 지원금, 예산 세부내역, 협력 기관
        단순한 수치(총액, 기간, 날짜)는 기밀 아님

        -일반정보 : 기밀정보가 아닌 모든 정보


        우리는 {span_token}(이)가 {gt_label}라고 생각해서 이렇게 라벨링을 해서 데이터셋으로 사용했거든,
        그런데 우리 모델이 {span_token}(이)라는 단어를 "{pred_label}로 예측을 했어.

        이는 {span_token}가 {gt_label}와 {pred_label}로 모두 사용될 수 있다고 추론할 수 있을 것 같아.

        {span_token}가 문맥상 {pred_label}로 사용될 수 있는 예시 문장 {n}개를 생성해줬으면 좋겠어.
        각 문장은 char단위로 150개 이상 250개 이하로 별도의 추론설명 없이 한글 맞춤법에 유의하여 주고, 반드시 아래 포맷에 정확히 맞춰 구성해주면 좋겠어.

        특수문자는 ,와 .을 제외하고는 사용하지 마.  
        
        ===== 문장들의 시작 =====
        === {pred_label} 문장 ===
        [문장 1을 여기에 생성해줘...]
        === {pred_label} 문장 ===
        [문장 2를 여기에 생성해줘...]
        ...
        === {pred_label} 문장 ===
        [문장 {n}을 여기에 생성해줘...]
        ===== 문장들의 끝 ====="""

    # Get response from OpenAI client
    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    # 로그용
    output = response.output_text
    print( output )

    # try:
    #     samples = output.split("\n===== 문장들의 시작 =====")
    #     pred_samples = samples[1].split(f"\n=== {pred_label} 문장 ===\n")
    #     pred_samples[-1] = pred_samples[-1].split("\n===== 문장들의 끝 =====")[0]
    #     gt_samples = samples[2].split(f"\n=== {gt_label} 문장 ===\n")
    #     gt_samples[-1] = gt_samples[-1].split("\n===== 문장들의 끝 =====")[0]
    # except Exception as e:
    #     print(f"[### Exception occured ###]\n{e}")
    #     pred_samples = []
    #     gt_samples = []

    try:
        # 1. pred_label 문장들 추출
        # '=== pred_label 문장 ===' 바로 뒤에 오는 내용을 추출합니다.
        pred_pattern = f"=== {pred_label} 문장 ===\s*\[?(.*?)\]?(?=\s*===|\s*=====)"
        pred_matches = re.findall(pred_pattern, output, re.DOTALL)
        # 추출된 문장들의 앞뒤 공백 및 줄바꿈을 제거합니다.
        pred_samples = [s.strip() for s in pred_matches]

    except Exception as e:
        print(f"[### Exception occured ###]\n{e}")
        pred_samples = []


    # 로그용
    print(f"\n\n[{pred_label} 문장]\n")
    for i, sent in enumerate(pred_samples):
        print(f"{i}번문장 -> {sent}")

    end_time = datetime.now()

    duration = end_time - start_time

    return pred_samples