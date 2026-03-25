# Module 03: Jupyter 노트북 기반 에이전트 개발 환경 구축

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

- Jupyter 노트북 환경에서 LangGraph 에이전트를 대화형으로 개발하고 테스트할 수 있다
- FakeLLM을 만들어 API 키 없이, 비용 없이 에이전트를 실행할 수 있다
- Mermaid 다이어그램으로 에이전트 그래프를 시각적으로 확인할 수 있다
- 개별 노드를 독립적으로 호출하여 디버깅할 수 있다
- Mock 객체로 외부 API를 대체하는 테스트 패턴을 적용할 수 있다

---

## 사전 지식

| 항목 | 수준 | 비고 |
|------|------|------|
| Python 기초 | 필수 | 함수, 클래스, 딕셔너리 사용법 |
| Module 01 | 필수 | LangGraph 기본 개념 (노드, 엣지, State) |
| Module 02 | 권장 | LangGraph 설치 및 첫 그래프 실행 경험 |
| Jupyter Notebook | 선택 | 사용 경험이 없어도 이 모듈에서 배울 수 있음 |

---

## 1. 개념 설명

### 1.1 왜 Jupyter 노트북으로 에이전트를 개발하나?

일반적인 에이전트 개발 과정을 생각해 봅시다:

```
[기존 방식]
코드 수정 → 터미널에서 python main.py 실행 → 전체 파이프라인 실행 → 로그 확인
                                                ↑ 오래 걸림        ↑ 읽기 어려움
```

```
[Jupyter 방식]
셀에서 코드 수정 → Shift+Enter로 실행 → 즉시 결과 확인 → 그래프 시각화
                    ↑ 수 초 이내         ↑ 인라인 출력   ↑ 다이어그램 렌더링
```

Jupyter 노트북이 에이전트 개발에 적합한 4가지 이유:

| 장점 | 설명 |
|------|------|
| **빠른 피드백 루프** | 코드 변경 후 해당 셀만 실행하면 됨. 전체 프로그램을 다시 시작할 필요 없음 |
| **시각적 디버깅** | 그래프 다이어그램, state 변화, LLM 응답을 인라인으로 바로 확인 |
| **재현 가능한 테스트** | FakeLLM과 fixture를 사용하면 동일 입력 → 동일 출력이 항상 보장됨 |
| **외부 의존성 제거** | 실제 LLM API, 데이터베이스, 외부 서비스 없이 로컬에서 완전히 동작 |

### 1.2 Jupyter 노트북이란?

Jupyter 노트북은 코드, 실행 결과, 설명 텍스트를 하나의 문서에 담는 대화형 개발 환경입니다. 코드를 "셀(cell)" 단위로 나누어 실행하고, 각 셀의 결과를 바로 아래에서 확인할 수 있습니다.

```
┌──────────────────────────────────┐
│ [Cell 1] Python 코드             │  ← 코드 셀
│ x = 10                           │
│ print(x * 2)                     │
├──────────────────────────────────┤
│ 출력: 20                         │  ← 실행 결과
├──────────────────────────────────┤
│ [Cell 2] 마크다운 설명           │  ← 마크다운 셀
│ ## 다음 단계                     │
│ 아래에서 그래프를 만듭니다.       │
├──────────────────────────────────┤
│ [Cell 3] Python 코드             │  ← 코드 셀
│ graph = build_graph()            │
│ graph.invoke(...)                │
├──────────────────────────────────┤
│ 출력: {'result': '분석 완료'}    │  ← 실행 결과
└──────────────────────────────────┘
```

### 1.3 FakeLLM이란?

FakeLLM은 실제 LLM(ChatGPT, Claude 등) 대신 사용하는 **가짜 LLM 객체**입니다. 미리 정해둔 규칙에 따라 응답을 반환합니다.

```
[실제 LLM]
질문 → 인터넷 → OpenAI/Anthropic 서버 → 응답 (비용 발생, API 키 필요)

[FakeLLM]
질문 → 패턴 매칭 → 미리 준비한 응답 반환 (무료, API 키 불필요)
```

왜 FakeLLM을 사용하나요?

- **비용 절감**: 개발 중에 수십~수백 번 실행해도 비용이 0원
- **API 키 불필요**: 환경 설정 없이 바로 시작 가능
- **일관된 결과**: 같은 입력에 항상 같은 출력 (디버깅이 쉬움)
- **오프라인 가능**: 인터넷 연결 없이도 개발 가능

---

## 2. 단계별 실습

### 2.1 Python 가상환경 생성 및 활성화

프로젝트별로 독립적인 Python 환경을 만들어야 패키지 충돌을 방지할 수 있습니다.

```bash
# 1. 프로젝트 디렉토리 생성
mkdir my-agent-notebook
cd my-agent-notebook

# 2. 가상환경 생성 (Python 3.11 이상 권장)
python -m venv .venv

# 3. 가상환경 활성화
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 활성화 확인 (가상환경 경로가 출력되어야 함)
which python
# 출력 예시: /Users/yourname/my-agent-notebook/.venv/bin/python
```

> **팁**: 터미널 프롬프트 앞에 `(.venv)`가 표시되면 가상환경이 활성화된 것입니다.

### 2.2 Jupyter + ipykernel 설치

```bash
# Jupyter 노트북과 커널 패키지 설치
pip install jupyter ipykernel ipython

# 설치 확인
jupyter --version
```

### 2.3 LangGraph 관련 패키지 설치

```bash
# LangGraph 핵심 패키지
pip install langgraph langchain-core

# 시각화 및 유틸리티
pip install ipython

# (선택) state diff 비교용
pip install deepdiff
```

### 2.4 Jupyter 커널 등록

가상환경을 Jupyter 커널로 등록하면, 노트북에서 이 환경의 패키지를 사용할 수 있습니다.

```bash
# 커널 등록
python -m ipykernel install --user \
    --name my-agent-dev \
    --display-name "My Agent Dev (Python 3.12)"

# 등록된 커널 목록 확인
jupyter kernelspec list
```

### 2.5 Jupyter 노트북 시작

```bash
# Jupyter 노트북 서버 실행
jupyter notebook

# 또는 JupyterLab (더 현대적인 인터페이스)
# pip install jupyterlab
# jupyter lab
```

브라우저가 자동으로 열립니다. 오른쪽 위 "New" 버튼을 클릭하고, 방금 등록한 커널 "My Agent Dev (Python 3.12)"을 선택하여 새 노트북을 만듭니다.

---

## 3. 실전 예제

### 3.1 FakeLLM 클래스 만들기

아래 코드를 노트북의 첫 번째 셀에 입력하고 실행합니다. 이것이 우리가 사용할 FakeLLM입니다.

```python
# ── Cell 1: FakeLLM 정의 ──

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from typing import Any, Optional


class FakeLLM(BaseChatModel):
    """패턴 매칭 기반 가짜 LLM.

    미리 정의한 패턴-응답 매핑에 따라 응답을 반환합니다.
    실제 LLM API를 호출하지 않으므로 비용이 발생하지 않습니다.
    """

    responses: dict[str, str] = {}
    default_response: str = "이것은 FakeLLM의 기본 응답입니다."

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def _generate(self, messages: list[BaseMessage], **kwargs) -> Any:
        """메시지 내용을 패턴 매칭하여 적절한 응답을 반환합니다."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        # 마지막 메시지의 내용을 확인
        last_message = messages[-1].content if messages else ""

        # 패턴 매칭: responses 딕셔너리에서 키워드 검색
        matched_response = self.default_response
        for pattern, response in self.responses.items():
            if pattern.lower() in last_message.lower():
                matched_response = response
                break

        generation = ChatGeneration(
            message=AIMessage(content=matched_response)
        )
        return ChatResult(generations=[generation])


# FakeLLM 인스턴스 생성
fake_llm = FakeLLM(
    responses={
        "분석": "분석 결과: 이 문서는 Python 웹 프레임워크에 관한 기술 문서입니다.",
        "요약": "요약: 본 문서는 FastAPI를 활용한 REST API 구축 방법을 설명합니다.",
        "번역": "Translation: This document explains how to build REST APIs.",
    },
    default_response="처리되었습니다. 추가 정보가 필요하면 알려주세요.",
)

# 테스트
response = fake_llm.invoke("이 문서를 분석해주세요")
print(f"응답: {response.content}")
# 출력: 응답: 분석 결과: 이 문서는 Python 웹 프레임워크에 관한 기술 문서입니다.
```

### 3.2 Fixture 파일로 테스트 데이터 관리

패턴-응답 매핑이 많아지면 코드에 직접 넣기 어렵습니다. JSON 파일로 분리하여 관리합시다.

```python
# ── Cell 2: Fixture 파일 생성 및 로드 ──

import json
from pathlib import Path

# fixture 디렉토리 생성
fixtures_dir = Path("fixtures")
fixtures_dir.mkdir(exist_ok=True)

# fixture 파일 작성
fixture_data = {
    "analyze": {
        "pattern": "분석",
        "response": "분석 결과: 주요 키워드는 [Python, FastAPI, REST]입니다. 기술 문서로 분류됩니다."
    },
    "summarize": {
        "pattern": "요약",
        "response": "요약: 본 문서는 3개 섹션으로 구성되며, 핵심 내용은 API 설계 원칙입니다."
    },
    "classify": {
        "pattern": "분류",
        "response": "분류 결과: 카테고리=기술문서, 난이도=중급, 대상=백엔드 개발자"
    }
}

fixture_path = fixtures_dir / "llm_responses.json"
with open(fixture_path, "w", encoding="utf-8") as f:
    json.dump(fixture_data, f, ensure_ascii=False, indent=2)

print(f"Fixture 파일 생성 완료: {fixture_path}")


def load_fixtures(fixture_file: str = "fixtures/llm_responses.json") -> dict[str, str]:
    """Fixture JSON 파일을 로드하여 패턴-응답 딕셔너리로 변환합니다."""
    with open(fixture_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        item["pattern"]: item["response"]
        for item in data.values()
    }


# fixture 기반 FakeLLM 생성
responses = load_fixtures()
fake_llm = FakeLLM(responses=responses)

# 테스트
result = fake_llm.invoke("이 텍스트를 분류해주세요")
print(f"응답: {result.content}")
# 출력: 응답: 분류 결과: 카테고리=기술문서, 난이도=중급, 대상=백엔드 개발자
```

### 3.3 노트북에서 LangGraph 그래프 만들고 실행하기

이제 FakeLLM을 활용하여 실제 LangGraph 그래프를 노트북에서 실행해 봅시다.

```python
# ── Cell 3: 간단한 3-노드 그래프 정의 ──

from typing import TypedDict
from langgraph.graph import StateGraph, END


# 1. State 정의
class TextAnalysisState(TypedDict):
    """텍스트 분석 에이전트의 상태"""
    input_text: str           # 입력 텍스트
    analysis: str | None      # 분석 결과
    summary: str | None       # 요약 결과
    current_step: str         # 현재 단계


# 2. 노드 함수 정의
def analyze_text(state: TextAnalysisState) -> dict:
    """텍스트를 분석하는 노드"""
    response = fake_llm.invoke(f"다음 텍스트를 분석해주세요: {state['input_text']}")
    return {
        "analysis": response.content,
        "current_step": "analyzed",
    }


def summarize_text(state: TextAnalysisState) -> dict:
    """분석 결과를 요약하는 노드"""
    response = fake_llm.invoke(f"다음 분석 결과를 요약해주세요: {state['analysis']}")
    return {
        "summary": response.content,
        "current_step": "summarized",
    }


def format_result(state: TextAnalysisState) -> dict:
    """최종 결과를 포맷하는 노드"""
    formatted = f"[분석 결과]\n{state['analysis']}\n\n[요약]\n{state['summary']}"
    return {
        "summary": formatted,
        "current_step": "completed",
    }


# 3. 그래프 조립
graph_builder = StateGraph(TextAnalysisState)

# 노드 추가
graph_builder.add_node("analyze", analyze_text)
graph_builder.add_node("summarize", summarize_text)
graph_builder.add_node("format", format_result)

# 엣지 연결
graph_builder.set_entry_point("analyze")
graph_builder.add_edge("analyze", "summarize")
graph_builder.add_edge("summarize", "format")
graph_builder.add_edge("format", END)

# 컴파일
graph = graph_builder.compile()

print("그래프 빌드 완료!")
```

```python
# ── Cell 4: 그래프 실행 ──

import json

# 초기 상태 설정
initial_state: TextAnalysisState = {
    "input_text": "FastAPI는 Python으로 만든 고성능 웹 프레임워크입니다.",
    "analysis": None,
    "summary": None,
    "current_step": "start",
}

# stream 모드로 실행 (각 노드의 출력을 순차적으로 확인)
print("=" * 60)
print("그래프 실행 시작")
print("=" * 60)

for event in graph.stream(initial_state):
    node_name = list(event.keys())[0]
    node_output = event[node_name]
    print(f"\n>> 노드: {node_name}")
    print(f"   현재 단계: {node_output.get('current_step', 'N/A')}")
    for key, value in node_output.items():
        if key != "current_step" and value is not None:
            print(f"   {key}: {str(value)[:100]}")
    print("-" * 60)

print("\n실행 완료!")
```

### 3.4 그래프 시각화: Mermaid 다이어그램

LangGraph는 그래프 구조를 Mermaid 다이어그램으로 출력할 수 있습니다. 이를 통해 에이전트의 흐름을 시각적으로 파악할 수 있습니다.

```python
# ── Cell 5: Mermaid 텍스트 출력 ──

# 방법 1: Mermaid 텍스트 직접 출력
mermaid_text = graph.get_graph().draw_mermaid()
print(mermaid_text)

# 출력 예시:
# %%{init: {'flowchart': {'curve': 'linear'}}}%%
# graph TD;
#     __start__([start]):::first
#     analyze(analyze)
#     summarize(summarize)
#     format(format)
#     __end__([end]):::last
#     __start__ --> analyze;
#     analyze --> summarize;
#     summarize --> format;
#     format --> __end__;
```

```python
# ── Cell 6: Mermaid 다이어그램 렌더링 ──

from IPython.display import display, Image

# 방법 2: PNG 이미지로 렌더링 (노트북에 인라인 표시)
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    display(Image(png_bytes))
    print("그래프 다이어그램이 위에 표시됩니다.")
except Exception as e:
    # mermaid-cli가 없는 경우 Mermaid Live Editor 링크 제공
    import base64
    encoded = base64.urlsafe_b64encode(mermaid_text.encode()).decode()
    print(f"PNG 렌더링 실패: {e}")
    print(f"\n아래 링크를 브라우저에서 열면 다이어그램을 볼 수 있습니다:")
    print(f"https://mermaid.live/edit#base64:{encoded}")
```

```python
# ── Cell 7: 노트북에서 Mermaid 마크다운으로 표시 ──

from IPython.display import display, Markdown

# 방법 3: Jupyter에서 마크다운으로 렌더링
# (JupyterLab에서는 Mermaid를 기본 지원합니다)
display(Markdown(f"```mermaid\n{mermaid_text}\n```"))
```

### 3.5 노드별 디버깅: 개별 노드 독립 호출

전체 그래프를 실행하지 않고, 특정 노드만 따로 호출하여 테스트할 수 있습니다. LangGraph의 노드는 본질적으로 `(state) -> dict` 형태의 일반 Python 함수이기 때문입니다.

```python
# ── Cell 8: 개별 노드 독립 테스트 ──

# analyze_text 노드만 따로 실행
test_state: TextAnalysisState = {
    "input_text": "머신러닝은 데이터에서 패턴을 학습하는 AI 기술입니다.",
    "analysis": None,
    "summary": None,
    "current_step": "start",
}

# 노드 함수를 직접 호출
result = analyze_text(test_state)
print("=== analyze_text 노드 출력 ===")
print(f"analysis: {result['analysis']}")
print(f"current_step: {result['current_step']}")

# summarize_text 노드를 테스트하려면 analysis가 채워진 state가 필요
test_state_2: TextAnalysisState = {
    "input_text": "머신러닝은 데이터에서 패턴을 학습하는 AI 기술입니다.",
    "analysis": "분석 결과: AI/ML 관련 기술 문서",
    "summary": None,
    "current_step": "analyzed",
}

result_2 = summarize_text(test_state_2)
print("\n=== summarize_text 노드 출력 ===")
print(f"summary: {result_2['summary']}")
```

### 3.6 State Diff 시각화 (Before/After 비교)

노드 실행 전후의 상태 변화를 한눈에 보여주는 유틸리티를 만들어 봅시다.

```python
# ── Cell 9: State Diff 유틸리티 ──

def show_state_diff(before: dict, node_output: dict, node_name: str = ""):
    """노드 실행 전후의 상태 변경을 시각적으로 출력합니다.

    Args:
        before: 노드 실행 전 state
        node_output: 노드가 반환한 dict (변경된 필드만 포함)
        node_name: 노드 이름 (표시용)
    """
    header = f"State Diff: {node_name}" if node_name else "State Diff"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")

    changes_found = False

    for key, new_value in node_output.items():
        old_value = before.get(key)
        if old_value != new_value:
            changes_found = True
            old_display = _truncate(str(old_value))
            new_display = _truncate(str(new_value))
            print(f"\n  [변경] {key}:")
            print(f"    이전: {old_display}")
            print(f"    이후: {new_display}")

    if not changes_found:
        print("\n  (변경사항 없음)")

    print(f"\n{'=' * 60}\n")


def _truncate(text: str, max_len: int = 80) -> str:
    """긴 텍스트를 잘라서 보기 좋게 표시합니다."""
    return text[:max_len] + "..." if len(text) > max_len else text


# 사용 예시
before_state: TextAnalysisState = {
    "input_text": "FastAPI는 Python 웹 프레임워크입니다.",
    "analysis": None,
    "summary": None,
    "current_step": "start",
}

node_result = analyze_text(before_state)
show_state_diff(before_state, node_result, "analyze_text")

# 출력 예시:
# ============================================================
#   State Diff: analyze_text
# ============================================================
#
#   [변경] analysis:
#     이전: None
#     이후: 분석 결과: 주요 키워드는 [Python, FastAPI, REST]입니다. ...
#
#   [변경] current_step:
#     이전: start
#     이후: analyzed
#
# ============================================================
```

### 3.7 Mock 객체로 외부 시스템 대체하기

에이전트가 외부 API(데이터베이스, REST API 등)를 호출할 때, Mock 객체로 대체하면 외부 시스템 없이 테스트할 수 있습니다.

```python
# ── Cell 10: unittest.mock 기초 ──

from unittest.mock import MagicMock, patch

# 예시: 외부 API를 호출하는 함수
def fetch_document_from_api(doc_id: str) -> dict:
    """외부 API에서 문서를 가져오는 함수 (원래는 HTTP 요청 필요)"""
    import httpx
    response = httpx.get(f"https://api.example.com/docs/{doc_id}")
    return response.json()


# Mock으로 대체: 실제 HTTP 요청 없이 테스트
mock_api = MagicMock()
mock_api.return_value = {
    "id": "doc-001",
    "title": "Python 가이드",
    "content": "Python은 배우기 쉬운 프로그래밍 언어입니다.",
    "author": "홍길동",
}

# Mock 함수 호출
result = mock_api("doc-001")
print(f"Mock API 응답: {result['title']}")
print(f"호출 횟수: {mock_api.call_count}")
print(f"호출 인자: {mock_api.call_args}")
```

```python
# ── Cell 11: 에이전트 노드에서 Mock 활용 ──

from functools import partial
from typing import TypedDict

# 외부 도구에 의존하는 노드 함수
def fetch_document(state: dict, api_client=None) -> dict:
    """문서를 가져오는 노드.

    api_client를 외부에서 주입받으므로, 테스트 시 Mock으로 교체 가능합니다.
    """
    doc = api_client(state["doc_id"])
    return {
        "document": doc,
        "current_step": "fetched",
    }


# 실제 사용 시: 진짜 API 클라이언트 주입
# real_fetch = partial(fetch_document, api_client=real_api_client)

# 테스트 시: Mock 주입
mock_client = MagicMock(return_value={
    "id": "doc-001",
    "title": "테스트 문서",
    "content": "이것은 테스트용 문서입니다.",
})

test_fetch = partial(fetch_document, api_client=mock_client)

# 노드 실행
result = test_fetch({"doc_id": "doc-001", "document": None, "current_step": "start"})
print(f"가져온 문서: {result['document']['title']}")

# Mock 호출 검증
mock_client.assert_called_once_with("doc-001")
print("Mock 검증 통과! (정확히 1번 호출됨)")
```

### 3.8 노트북 표준 템플릿

매번 노트북을 만들 때마다 같은 설정을 반복하지 않도록, 표준 템플릿을 정해 둡시다.

```python
# ── 표준 템플릿: Cell 1 - 환경 초기화 ──

# 필수 import
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from IPython.display import display, Image, Markdown
from functools import partial
from unittest.mock import MagicMock
import json

print("환경 초기화 완료!")
```

```python
# ── 표준 템플릿: Cell 2 - FakeLLM 설정 ──

class FakeLLM(BaseChatModel):
    """표준 FakeLLM - 모든 노트북에서 재사용"""
    responses: dict[str, str] = {}
    default_response: str = "처리 완료."

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def _generate(self, messages: list[BaseMessage], **kwargs):
        last_msg = messages[-1].content if messages else ""
        matched = self.default_response
        for pattern, response in self.responses.items():
            if pattern.lower() in last_msg.lower():
                matched = response
                break
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=matched))]
        )

# FakeLLM 인스턴스 (프로젝트에 맞게 responses를 수정하세요)
fake_llm = FakeLLM(responses={})
print("FakeLLM 준비 완료!")
```

```python
# ── 표준 템플릿: Cell 3 - 유틸리티 함수 ──

def show_graph(graph):
    """그래프를 시각화합니다."""
    try:
        png = graph.get_graph().draw_mermaid_png()
        display(Image(png))
    except Exception:
        mermaid = graph.get_graph().draw_mermaid()
        display(Markdown(f"```mermaid\n{mermaid}\n```"))

def run_and_trace(graph, initial_state: dict):
    """그래프를 실행하고 각 노드의 출력을 추적합니다."""
    print("=" * 60)
    events = []
    for event in graph.stream(initial_state):
        events.append(event)
        node_name = list(event.keys())[0]
        print(f"\n>> 노드 실행: {node_name}")
        for key, val in event[node_name].items():
            print(f"   {key}: {_truncate(str(val))}")
        print("-" * 40)
    print("=" * 60)
    print("실행 완료!")
    return events

def _truncate(s: str, n: int = 80) -> str:
    return s[:n] + "..." if len(s) > n else s

print("유틸리티 함수 준비 완료!")
```

---

## 4. 연습 문제

### 연습 4-1: FakeLLM 확장하기

아래 요구사항에 맞게 FakeLLM을 만들어 보세요:

1. "감정" 키워드가 포함된 질문에는 `"감정 분석: 긍정적(75%), 부정적(15%), 중립(10%)"` 응답
2. "번역" 키워드가 포함된 질문에는 `"Translation: Hello, this is a test."` 응답
3. "코드" 키워드가 포함된 질문에는 `"코드 리뷰: 변수명이 명확하고, 함수 구조가 좋습니다."` 응답
4. 위 어느 것에도 해당하지 않으면 `"잘 모르겠습니다. 다시 질문해 주세요."` 응답

```python
# 여기에 코드를 작성하세요
fake_llm = FakeLLM(
    responses={
        # TODO: 패턴-응답 매핑을 채우세요
    },
    default_response="TODO: 기본 응답을 설정하세요",
)

# 테스트
assert "긍정" in fake_llm.invoke("이 리뷰의 감정을 분석해줘").content
assert "Translation" in fake_llm.invoke("이 문장을 번역해줘").content
assert "코드 리뷰" in fake_llm.invoke("이 코드를 검토해줘").content
assert "잘 모르겠습니다" in fake_llm.invoke("날씨가 좋다").content
print("모든 테스트 통과!")
```

### 연습 4-2: 3-노드 그래프 만들고 시각화하기

다음 3개 노드를 가진 "텍스트 처리 에이전트"를 만들어 보세요:

1. **clean_text**: 입력 텍스트에서 공백 정리 (FakeLLM 불필요)
2. **classify_text**: FakeLLM으로 텍스트 분류
3. **generate_report**: FakeLLM으로 분류 결과 기반 보고서 생성

```python
# 여기에 코드를 작성하세요

# 1. State 정의
class TextProcessState(TypedDict):
    raw_text: str
    cleaned_text: str | None
    classification: str | None
    report: str | None
    current_step: str

# 2. 노드 함수 정의
def clean_text(state: TextProcessState) -> dict:
    # TODO: 구현하세요
    pass

def classify_text(state: TextProcessState) -> dict:
    # TODO: FakeLLM을 사용하여 구현하세요
    pass

def generate_report(state: TextProcessState) -> dict:
    # TODO: FakeLLM을 사용하여 구현하세요
    pass

# 3. 그래프 조립
# TODO: StateGraph를 만들고 노드와 엣지를 연결하세요

# 4. 실행 및 시각화
# TODO: 그래프를 시각화하고, 샘플 입력으로 실행하세요
```

### 연습 4-3: Mock으로 외부 API 대체하기

다음 시나리오를 구현하세요:

- `search_web` 노드: 웹 검색 API를 호출하여 결과를 가져옴 (Mock으로 대체)
- `extract_info` 노드: 검색 결과에서 핵심 정보 추출 (FakeLLM 사용)
- Mock을 사용하여 `search_web`이 3개의 검색 결과를 반환하도록 설정

```python
# 여기에 코드를 작성하세요

# 힌트: functools.partial로 의존성을 주입하세요
# search_fn = partial(search_web, search_client=mock_search)
```

---

## 5. 핵심 정리

| 개념 | 설명 |
|------|------|
| **Jupyter 노트북** | 코드를 셀 단위로 실행하고 결과를 즉시 확인하는 대화형 개발 환경 |
| **ipykernel** | Python 가상환경을 Jupyter 커널로 등록하여, 노트북에서 특정 환경의 패키지를 사용 |
| **FakeLLM** | 실제 LLM API 없이 패턴 매칭으로 응답을 반환하는 가짜 LLM 객체 |
| **Fixture** | 테스트용 데이터를 JSON 파일로 외부 관리. FakeLLM의 응답 데이터 소스 |
| **Mermaid 다이어그램** | LangGraph 그래프 구조를 시각적으로 표현하는 다이어그램 포맷 |
| **노드 독립 호출** | 전체 그래프를 실행하지 않고 개별 노드 함수를 직접 호출하여 테스트 |
| **State Diff** | 노드 실행 전후의 상태 변화를 비교하여 보여주는 디버깅 기법 |
| **Mock 객체** | 외부 시스템(API, DB)을 가짜 객체로 대체. `unittest.mock` 표준 라이브러리 사용 |
| **의존성 주입** | `functools.partial`로 노드 함수에 외부 도구(LLM, API 클라이언트)를 주입. 테스트 시 교체 용이 |
| **표준 템플릿** | 모든 노트북에서 공통으로 사용하는 초기화 셀 패턴. 환경 설정의 일관성 보장 |

---

## 6. 참고 자료

### 공식 문서

- **Jupyter 공식 문서**: [https://jupyter.org/documentation](https://jupyter.org/documentation)
  - Jupyter 노트북의 설치, 사용법, 설정에 대한 종합 가이드
- **IPython 커널 설치 가이드**: [https://ipython.readthedocs.io/en/stable/install/kernel_install.html](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
  - 가상환경을 Jupyter 커널로 등록하는 방법
- **LangGraph 시각화**: [https://langchain-ai.github.io/langgraph/how-tos/visualization/](https://langchain-ai.github.io/langgraph/how-tos/visualization/)
  - Mermaid 다이어그램과 PNG 렌더링 방법
- **Python unittest.mock**: [https://docs.python.org/3/library/unittest.mock.html](https://docs.python.org/3/library/unittest.mock.html)
  - Mock, MagicMock, patch 등 테스트 도구 공식 레퍼런스

### 추가 학습 자료

- **LangChain FakeListLLM**: [https://python.langchain.com/docs/modules/model_io/llms/fake_llm](https://python.langchain.com/docs/modules/model_io/llms/fake_llm)
  - LangChain에서 제공하는 기본 FakeLLM 구현
- **functools.partial 문서**: [https://docs.python.org/3/library/functools.html#functools.partial](https://docs.python.org/3/library/functools.html#functools.partial)
  - 의존성 주입에 사용하는 partial 함수 상세 설명
- **Mermaid 공식 사이트**: [https://mermaid.js.org/](https://mermaid.js.org/)
  - Mermaid 다이어그램 문법과 라이브 에디터
- **JupyterLab**: [https://jupyterlab.readthedocs.io/](https://jupyterlab.readthedocs.io/)
  - Jupyter 노트북의 차세대 인터페이스

---

## 다음 단계

이 모듈에서 Jupyter 기반 개발 환경을 구축했습니다. 다음 **Module 04: 나의 첫 LangGraph 에이전트 만들기**에서는 이 환경을 활용하여 완전한 에이전트를 처음부터 끝까지 만들어 봅니다.

다음 모듈에서 다루는 내용:
- 실전 에이전트 프로젝트 설계 (문서 요약 에이전트)
- 4개 노드로 구성된 완전한 그래프 구현
- 조건부 분기와 에러 처리
- 의존성 주입을 활용한 테스트 가능한 에이전트 설계
- FakeLLM 기반 E2E 테스트
