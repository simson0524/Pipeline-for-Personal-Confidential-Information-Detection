# GPT 생성 문서 README

### 총 5개 도메인 & 문서/문장 개요
1. 인사부 - 75개 문서/8624개 문장
2. 생산부 - 75개 문서/10986개 문장
3. 관리부 - 75개 문서/9745개 문장
4. 영업부 - 75개 문서/10283개 문장
5. 기술부 - 75개 문서/9489개 문장

각 부서별로 다른 직원구성과 업무구성을 갖고있기 때문에 도메인 분리기준을 "부서"로 하였습니다.

### 파일 설명
디렉토리 내 위치하는 각 json파일은 1개 문장에 대하여 개인정보/기밀정보인 토큰(단어)들의 annotation 정보가 달려있습니다.(일반정보 제외)

아래는 각 json파일의 구조입니다.

├── data        : 실 데이터
│   ├── sentence    : 문장 원문
│   ├── id          : 문장 고유 ID
│   ├── filename    : 문장이 소속된 원본 문서명
│   ├── caseField   : 사용 X
│   ├── detailField : 사용 X
│   └── sequence    : 해당 문장이 문서 내에서 위치한 순서
│
└── annotations : 어노테이션
    ├── id          : 문장 고유 ID
    └── annotations : 어노테이션(여러개인 경우가 있어 List(dict)로 관리됩니다)
        ├── start     : span_text의 sentence에서 char단위 시작위치
        ├── end       : span_text의 sentence에서 char단위 끝나는 위치+1 -> sentence[start:end] == span_text
        ├── label     : span_text의 sentence내에서 정보 구분(개인 or 기밀 or 준식별자)
        ├── score     : label 확신도
        └── span_text : 어노테이션 기준이 되는 단어


### 요청 사항
저희는 위 데이터들 전체에 대하여 각 도메인(부서)별 모든 토큰(단어)에 대한 지표를 확인하고 싶습니다.

1개 workspace에 대하여 부서 기준 5개의 도메인으로 분리하여 계산해주시면 감사하겠습니다.