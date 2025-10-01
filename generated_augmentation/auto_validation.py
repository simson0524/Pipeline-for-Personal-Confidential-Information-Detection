# generated_augmentation/auto_validation.py

from datetime import datetime
from openai import OpenAI
import os

### API KEY ###
JH_GPT_API_KEY = os.environ.get("JH_GPT_API_KEY")
WW_GPT_API_KEY = os.environ.get("WW_GPT_API_KEY")
HJ_GPT_API_KEY = os.environ.get("HJ_GPT_API_KEY")


def auto_validation(span_token, samples, target_label, is_pii=True, model_name="gpt-5-mini", api_key=JH_GPT_API_KEY):
    start_time = datetime.now()

    client = OpenAI(api_key = api_key)
    
    if is_pii:
        prompt=f"""
            개인정보 : 단 하나만으로도 특정 개인을 지칭하거나 추정할 수 있는 정보(이름, 주소, 주민번호 등)
            준식별정보 : 하나로는 특정 개인을 지칭할 수 없지만 여러 준식별정보가 결합하는 경우나, 소량의 준식별정보와 주변에 개인정보가 분포하는 경우 특정 개인을 지칭하거나 추정이 가능한 정보(직업, 시군구 주소, 신체관련정보, 생일 등)
            일반정보 : 개인정보와 준식별정보가 아닌 모든 정보


            지금부터 너에게 계속해서 문장들을 줄거야.
            해당 문장에서 '{span_token}'(이)라는 단어가 단 하나만으로도 특정 개인을 지칭하거나 추정할 수 있는 경우 '개인정보', '{span_token}'(이)라는 단어가 다른 단어들과 결합하여 특정 개인을 지칭하거나 추정할 수 있는 경우 '준식별정보', 모두 아닌 경우 '일반정보'로만 답변해주면 돼."""
    else:
        prompt=f"""
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


            지금부터 너에게 계속해서 문장들을 줄거야.
            해당 문장에서 "{span_token}"(이)라는 단어가 위 기밀정보 설명에 해당하는 경우 "기밀정보", 아닌 경우 "일반정보"로만 답변해주면 돼.
            """

    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    if samples:
        validation_results = []

        for _, sample in enumerate( samples ):
            if sample:
                iter_response = client.responses.create(
                    model=model_name,
                    previous_response_id=response.id,
                    input=[{"role": "user", "content": sample}]
                )
                validated_label = iter_response.output_text

                if validated_label == target_label:
                    validation_results.append( (True, validated_label) )
                else:
                    validation_results.append( (False, validated_label) )
            else:
                validation_results.append( (False, None) )
    
    end_time = datetime.now()

    duration = end_time - start_time

    return validation_results, samples, start_time, end_time, duration
    # [ (문장데이터 추가 가능여부(bool), True라면 validated label이 뭔지(str), generated_sent(str)), ]