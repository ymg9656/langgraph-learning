# LangGraph 실습 프로젝트

30시간 분량의 LangGraph 학습 자료를 보면서 바로 실습할 수 있는 Jupyter 노트북 프로젝트입니다.

## 빠른 시작

```bash
# 1. uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치 (가상환경 자동 생성 + uv.lock 기반 버전 고정)
uv sync

# 3. 환경 검증
uv run python verify_setup.py

# 4. Jupyter 실행
uv run jupyter notebook
```

### API 키 설정 (Module 04부터 필요)

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
# 또는 .env 파일 생성
cp .env.example .env
# .env 파일에서 API 키 수정
```

Module 01~03은 FakeLLM을 사용하므로 API 키 없이 실습 가능합니다.

---

## 학습 경로

| 모듈 | 주제 | 노트북 수 | 난이도 | API 키 |
|------|------|----------|--------|--------|
| **Part 1: 기초** |
| Module 01 | AI 에이전트 개요 | 1 (퀴즈) | ★☆☆☆☆ | 불필요 |
| Module 02 | LangGraph 기초 (그래프, 노드, 엣지) | 3 | ★★☆☆☆ | 불필요 |
| **Part 2: 첫 에이전트** |
| Module 03 | 개발 환경 (FakeLLM, 시각화) | 2 | ★★☆☆☆ | 불필요 |
| Module 04 | 첫 에이전트 만들기 | 2 | ★★★☆☆ | 선택 |
| **Part 3: 프롬프트 & LLM** |
| Module 05 | 프롬프트 엔지니어링 | 3 | ★★☆☆☆ | 선택 |
| Module 06 | 구조화된 출력 (Pydantic) | 3 | ★★★☆☆ | 선택 |
| Module 07 | LLM 호출 최적화 | 3 | ★★★☆☆ | 선택 |
| **Part 4: 프로덕션** |
| Module 08 | 에러 처리와 회복 탄력성 | 3 | ★★★★☆ | 선택 |
| Module 09 | 외부 시스템 통합 | 3 | ★★★★☆ | 선택 |
| Module 10 | 리소스 최적화 | 3 | ★★★★☆ | 선택 |
| **Part 5: 고급** |
| Module 11 | 품질 보증 | 3 | ★★★★☆ | 선택 |
| Module 12 | 고급 패턴 | 4 | ★★★★★ | 선택 |

---

## 실습 방법

### 1. 학습 자료 읽기
`docs/` 디렉토리에서 해당 모듈의 .md 파일을 읽습니다.

### 2. 노트북 열기
해당 모듈 디렉토리의 .ipynb 파일을 Jupyter에서 엽니다.

### 3. TODO 채우기
`# TODO:` 마커가 있는 코드 셀을 완성합니다. 막히면 힌트를 참고하세요:
- **힌트 1**: 방향 제시
- **힌트 2**: 핵심 키워드
- **힌트 3**: 거의 정답

### 4. 검증 셀 실행
각 TODO 아래의 검증 셀을 실행하여 정답을 확인합니다.

### 5. 솔루션 확인
`solutions/` 디렉토리에서 완성된 노트북을 확인할 수 있습니다.

---

## 디렉토리 구조

```
practice/
├── README.md              # 이 파일
├── pyproject.toml        # 의존성 정의
├── uv.lock               # 의존성 버전 고정 (모든 환경 동일)
├── .env.example          # API 키 템플릿
├── verify_setup.py       # 환경 검증
├── common/               # 공유 유틸리티
│   ├── fake_llm.py       # FakeLLM (API 키 없이 실습)
│   └── test_helpers.py   # 그래프 시각화, 상태 추적 유틸
├── module_01_agent_overview/
│   ├── 01_quiz.ipynb
│   └── solutions/
├── module_02_langgraph_fundamentals/
│   ├── 01_first_graph.ipynb
│   ├── 02_conditional_graph.ipynb
│   ├── 03_sentiment_analyzer.ipynb
│   └── solutions/
├── ...
└── module_12_advanced_patterns/
    ├── 01_checkpointing.ipynb
    ├── 02_subgraphs.ipynb
    ├── 03_parallel_send.ipynb
    ├── 04_streaming.ipynb
    └── solutions/
```

---

## 기술 스택

- **Python** 3.10+ (3.12 권장)
- **LangGraph** 0.3+ (그래프 기반 워크플로우)
- **LangChain** (LLM 통합)
- **Anthropic Claude** (LLM 백엔드)
- **Pydantic** 2.0+ (구조화된 출력)
- **Jupyter** (대화형 실습 환경)

---

## 참고 자료

학습 자료는 `docs/` 디렉토리에 있습니다:
- `docs/README.md` - 전체 학습 가이드
- `docs/part1-foundations/` - Part 1: 기초
- `docs/part2-first-agent/` - Part 2: 첫 에이전트
- `docs/part3-prompt-and-llm/` - Part 3: 프롬프트 & LLM
- `docs/part4-production/` - Part 4: 프로덕션
- `docs/part5-advanced/` - Part 5: 고급
