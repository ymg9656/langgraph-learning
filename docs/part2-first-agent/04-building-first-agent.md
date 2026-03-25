# Module 04: 나의 첫 LangGraph 에이전트 만들기

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

- LangGraph로 완전한 에이전트를 처음부터 설계하고 구현할 수 있다
- TypedDict로 에이전트 상태(State)를 정의하고 각 필드의 역할을 설명할 수 있다
- 4개 노드를 가진 그래프를 조립하고, 일반 엣지와 조건부 엣지를 사용할 수 있다
- functools.partial로 의존성을 주입하여 테스트 가능한 에이전트를 만들 수 있다
- FakeLLM으로 에이전트를 E2E 테스트할 수 있다

---

## 사전 지식

| 항목 | 수준 | 비고 |
|------|------|------|
| Module 01-02 | 필수 | LangGraph 기본 개념 (노드, 엣지, State) |
| Module 03 | 필수 | Jupyter 환경, FakeLLM, 그래프 시각화 |
| Python 함수/클래스 | 필수 | def, class, TypedDict, dict 다루기 |
| Python 타입 힌트 | 권장 | str, list, dict, Optional 등 |

---

## 1. 개념 설명

### 1.1 프로젝트 설계: "문서 요약 에이전트"

이번 모듈에서 만들 에이전트는 **Document Summarizer Agent(문서 요약 에이전트)**입니다. 문서를 입력받아 내용을 분석하고 요약하여 깔끔한 결과물을 출력합니다.

#### 요구사항 정의

| 항목 | 내용 |
|------|------|
| **에이전트 이름** | 문서 요약 에이전트 (Document Summarizer) |
| **입력** | 문서 텍스트 (문자열) |
| **출력** | 구조화된 요약 결과 (제목, 핵심 내용, 키워드, 난이도) |
| **노드 수** | 4개 |
| **LLM 호출** | 2회 (분석 1회 + 요약 1회) |
| **에러 처리** | 빈 문서 또는 분석 실패 시 에러 경로로 분기 |

#### 워크플로우 다이어그램

```
[시작]
  |
  v
[fetch_document] ─── 문서 로드 및 유효성 검사
  |
  ├─ (문서가 비어있음) ──> [END] (에러 경로)
  |
  v
[analyze_content] ─── LLM으로 핵심 내용 분석
  |
  ├─ (분석 실패) ──> [END] (에러 경로)
  |
  v
[generate_summary] ─── LLM으로 요약 생성
  |
  v
[format_output] ─── 결과 포맷팅 후 완료
  |
  v
[END]
```

### 1.2 왜 이렇게 설계하나?

에이전트 설계에는 몇 가지 원칙이 있습니다:

1. **단일 책임 원칙**: 각 노드는 하나의 작업만 수행
   - `fetch_document`: 문서를 가져오는 것만 담당
   - `analyze_content`: 분석만 담당
   - 하나의 노드가 "가져오기 + 분석 + 요약"을 모두 하지 않음

2. **의존성 주입**: LLM, 외부 API 등을 노드 함수 내부에서 직접 생성하지 않고, 외부에서 주입
   - 테스트 시 FakeLLM으로 쉽게 교체 가능
   - 실제 운영 시 진짜 LLM으로 교체 가능

3. **에러 경로 분리**: 정상 경로와 에러 경로를 명확히 분리
   - 어디서 실패했는지 즉시 파악 가능
   - 에러 시 불필요한 노드 실행을 건너뜀

---

## 2. 단계별 실습

### 2.1 State 설계

State는 에이전트의 "기억"입니다. 모든 노드가 이 State를 통해 데이터를 주고받습니다.

```python
# ── Cell 1: State 정의 ──

from typing import TypedDict


class DocumentSummaryState(TypedDict):
    """문서 요약 에이전트의 상태 정의.

    각 노드는 이 State에서 필요한 데이터를 읽고, 결과를 State에 기록합니다.
    """

    # ─── 입력 필드 ───
    source_text: str
    """원본 문서 텍스트. 사용자가 처음에 제공하는 입력값."""

    # ─── 중간 처리 필드 ───
    document: dict | None
    """로드된 문서 정보. fetch_document 노드가 채움.
    예: {"title": "...", "content": "...", "word_count": 150}
    """

    analysis: dict | None
    """분석 결과. analyze_content 노드가 채움.
    예: {"keywords": [...], "topic": "...", "difficulty": "..."}
    """

    # ─── 출력 필드 ───
    summary: str | None
    """생성된 요약문. generate_summary 노드가 채움."""

    final_output: str | None
    """최종 포맷팅된 결과. format_output 노드가 채움."""

    # ─── 제어 필드 ───
    current_step: str
    """현재 진행 단계. 디버깅 및 추적 용도.
    값: "start" | "fetched" | "analyzed" | "summarized" | "completed" | "error"
    """

    error: str | None
    """에러 메시지. 에러 발생 시 채워지며, 조건 분기에서 에러 경로로 라우팅."""


print("State 정의 완료!")
print("필드 목록:", list(DocumentSummaryState.__annotations__.keys()))
```

#### State 필드 역할 요약

```
[source_text] ──fetch──> [document] ──analyze──> [analysis] ──summarize──> [summary] ──format──> [final_output]
                                                                                          |
[current_step]: "start" → "fetched" → "analyzed" → "summarized" → "completed"
[error]: None (정상) 또는 "에러 메시지" (실패 시)
```

| 필드 | 타입 | 채우는 노드 | 읽는 노드 | 설명 |
|------|------|-------------|-----------|------|
| `source_text` | `str` | (사용자 입력) | fetch_document | 원본 텍스트 |
| `document` | `dict \| None` | fetch_document | analyze_content | 파싱된 문서 정보 |
| `analysis` | `dict \| None` | analyze_content | generate_summary | 분석 결과 |
| `summary` | `str \| None` | generate_summary | format_output | 요약문 |
| `final_output` | `str \| None` | format_output | (최종 출력) | 포맷된 결과 |
| `current_step` | `str` | 모든 노드 | 조건 분기 | 진행 상태 |
| `error` | `str \| None` | 에러 발생 노드 | 조건 분기 | 에러 메시지 |

### 2.2 노드 함수 구현

#### 노드 1: fetch_document (문서 로드)

```python
# ── Cell 2: fetch_document 노드 ──

def fetch_document(state: DocumentSummaryState) -> dict:
    """문서를 로드하고 기본 정보를 추출하는 노드.

    Args:
        state: 현재 에이전트 상태. source_text 필드를 읽음.

    Returns:
        업데이트할 필드들. document, current_step, (에러 시) error.
    """
    source_text = state["source_text"]

    # 유효성 검사: 빈 문서 체크
    if not source_text or not source_text.strip():
        return {
            "error": "문서가 비어있습니다. 텍스트를 입력해주세요.",
            "current_step": "error",
        }

    # 문서 정보 추출
    lines = source_text.strip().split("\n")
    title = lines[0] if lines else "제목 없음"
    content = "\n".join(lines[1:]).strip() if len(lines) > 1 else source_text
    word_count = len(source_text.split())

    document = {
        "title": title,
        "content": content,
        "word_count": word_count,
        "line_count": len(lines),
    }

    print(f"  [fetch_document] 문서 로드 완료: '{title}' ({word_count}단어)")

    return {
        "document": document,
        "current_step": "fetched",
    }


# 단독 테스트
test_result = fetch_document({
    "source_text": "Python 가이드\n파이썬은 간결하고 읽기 쉬운 언어입니다.",
    "document": None, "analysis": None, "summary": None,
    "final_output": None, "current_step": "start", "error": None,
})
print(f"결과: {test_result}")
```

#### 노드 2: analyze_content (핵심 내용 분석)

```python
# ── Cell 3: analyze_content 노드 ──

import json


def analyze_content(state: DocumentSummaryState, llm=None) -> dict:
    """LLM을 사용하여 문서의 핵심 내용을 분석하는 노드.

    Args:
        state: 현재 에이전트 상태. document 필드를 읽음.
        llm: LLM 인스턴스 (의존성 주입). None이면 에러.

    Returns:
        업데이트할 필드들. analysis, current_step, (에러 시) error.
    """
    document = state.get("document")

    if not document:
        return {
            "error": "문서가 로드되지 않았습니다.",
            "current_step": "error",
        }

    if llm is None:
        return {
            "error": "LLM이 설정되지 않았습니다.",
            "current_step": "error",
        }

    # LLM에 분석 요청
    prompt = (
        f"다음 문서를 분석해주세요.\n\n"
        f"제목: {document['title']}\n"
        f"내용: {document['content']}\n\n"
        f"JSON 형식으로 응답해주세요: "
        f'{{"keywords": [...], "topic": "...", "difficulty": "초급|중급|고급"}}'
    )

    try:
        response = llm.invoke(prompt)
        content = response.content

        # JSON 파싱 시도
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError:
            # JSON이 아닌 경우 텍스트 기반 분석 결과 구성
            analysis = {
                "keywords": [],
                "topic": content[:100],
                "difficulty": "중급",
                "raw_response": content,
            }

        print(f"  [analyze_content] 분석 완료: 주제='{analysis.get('topic', 'N/A')}'")

        return {
            "analysis": analysis,
            "current_step": "analyzed",
        }

    except Exception as e:
        return {
            "error": f"분석 중 오류 발생: {str(e)}",
            "current_step": "error",
        }
```

#### 노드 3: generate_summary (요약 생성)

```python
# ── Cell 4: generate_summary 노드 ──

def generate_summary(state: DocumentSummaryState, llm=None) -> dict:
    """LLM을 사용하여 문서 요약을 생성하는 노드.

    Args:
        state: 현재 에이전트 상태. document와 analysis 필드를 읽음.
        llm: LLM 인스턴스 (의존성 주입).

    Returns:
        업데이트할 필드들. summary, current_step.
    """
    document = state.get("document", {})
    analysis = state.get("analysis", {})

    if llm is None:
        return {
            "error": "LLM이 설정되지 않았습니다.",
            "current_step": "error",
        }

    prompt = (
        f"다음 문서를 요약해주세요.\n\n"
        f"제목: {document.get('title', 'N/A')}\n"
        f"내용: {document.get('content', 'N/A')}\n"
        f"분석 결과: {json.dumps(analysis, ensure_ascii=False)}\n\n"
        f"3줄 이내로 요약해주세요."
    )

    try:
        response = llm.invoke(prompt)
        summary = response.content

        print(f"  [generate_summary] 요약 생성 완료 ({len(summary)}자)")

        return {
            "summary": summary,
            "current_step": "summarized",
        }

    except Exception as e:
        return {
            "error": f"요약 생성 중 오류 발생: {str(e)}",
            "current_step": "error",
        }
```

#### 노드 4: format_output (결과 포맷팅)

```python
# ── Cell 5: format_output 노드 ──

def format_output(state: DocumentSummaryState) -> dict:
    """최종 결과를 보기 좋게 포맷하는 노드.

    Args:
        state: 현재 에이전트 상태. document, analysis, summary 필드를 읽음.

    Returns:
        업데이트할 필드들. final_output, current_step.
    """
    document = state.get("document", {})
    analysis = state.get("analysis", {})
    summary = state.get("summary", "요약 없음")

    # 키워드 목록 포맷
    keywords = analysis.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "추출된 키워드 없음"

    # 최종 출력 구성
    output_parts = [
        "=" * 50,
        "  문서 요약 결과",
        "=" * 50,
        "",
        f"  제목: {document.get('title', 'N/A')}",
        f"  단어 수: {document.get('word_count', 'N/A')}",
        f"  주제: {analysis.get('topic', 'N/A')}",
        f"  난이도: {analysis.get('difficulty', 'N/A')}",
        f"  키워드: {keywords_str}",
        "",
        "-" * 50,
        "  요약:",
        f"  {summary}",
        "-" * 50,
    ]

    final_output = "\n".join(output_parts)

    print(f"  [format_output] 포맷팅 완료")

    return {
        "final_output": final_output,
        "current_step": "completed",
    }
```

### 2.3 그래프 조립

이제 4개의 노드를 연결하여 완전한 그래프를 만듭니다.

```python
# ── Cell 6: 그래프 조립 ──

from langgraph.graph import StateGraph, END
from functools import partial


def build_summarizer_graph(llm=None):
    """문서 요약 에이전트 그래프를 빌드합니다.

    Args:
        llm: LLM 인스턴스. 테스트 시 FakeLLM, 운영 시 실제 LLM.

    Returns:
        컴파일된 LangGraph 그래프.
    """
    graph = StateGraph(DocumentSummaryState)

    # ─── 노드 등록 ───
    # fetch_document는 LLM이 필요 없으므로 그대로 등록
    graph.add_node("fetch_document", fetch_document)

    # analyze_content, generate_summary는 LLM이 필요하므로
    # functools.partial로 llm을 주입
    graph.add_node(
        "analyze_content",
        partial(analyze_content, llm=llm),
    )
    graph.add_node(
        "generate_summary",
        partial(generate_summary, llm=llm),
    )

    # format_output도 LLM 불필요
    graph.add_node("format_output", format_output)

    # ─── 진입점 설정 ───
    graph.set_entry_point("fetch_document")

    # ─── 엣지 연결 ───

    # 조건부 엣지 1: fetch_document 이후
    # - 에러가 있으면 END로 (에러 경로)
    # - 정상이면 analyze_content로
    graph.add_conditional_edges(
        "fetch_document",
        _route_after_fetch,
        {
            "analyze": "analyze_content",
            "error": END,
        },
    )

    # 조건부 엣지 2: analyze_content 이후
    # - 에러가 있으면 END로 (에러 경로)
    # - 정상이면 generate_summary로
    graph.add_conditional_edges(
        "analyze_content",
        _route_after_analyze,
        {
            "summarize": "generate_summary",
            "error": END,
        },
    )

    # 일반 엣지: generate_summary → format_output → END
    graph.add_edge("generate_summary", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# ─── 조건 분기 함수 ───

def _route_after_fetch(state: DocumentSummaryState) -> str:
    """fetch_document 이후 분기 결정.

    Returns:
        "analyze": 정상 → 분석 진행
        "error": 에러 → 종료
    """
    if state.get("error"):
        return "error"
    return "analyze"


def _route_after_analyze(state: DocumentSummaryState) -> str:
    """analyze_content 이후 분기 결정.

    Returns:
        "summarize": 정상 → 요약 진행
        "error": 에러 → 종료
    """
    if state.get("error"):
        return "error"
    return "summarize"


print("그래프 빌드 함수 정의 완료!")
```

> **핵심 포인트: functools.partial**
>
> `partial(analyze_content, llm=fake_llm)`은 `analyze_content` 함수에 `llm=fake_llm`을 미리 채운 새 함수를 만듭니다. 이렇게 하면:
> - **테스트 시**: `partial(analyze_content, llm=fake_llm)` - FakeLLM 주입
> - **운영 시**: `partial(analyze_content, llm=real_llm)` - 실제 LLM 주입
>
> 노드 함수 코드는 전혀 변경하지 않아도 됩니다.

### 2.4 FakeLLM 준비 및 그래프 빌드

```python
# ── Cell 7: FakeLLM 생성 및 그래프 빌드 ──

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeLLM(BaseChatModel):
    """패턴 매칭 기반 가짜 LLM."""
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


# 문서 요약 에이전트용 FakeLLM 설정
fake_llm = FakeLLM(
    responses={
        "분석": json.dumps({
            "keywords": ["Python", "FastAPI", "웹 프레임워크"],
            "topic": "Python 웹 개발",
            "difficulty": "중급",
        }, ensure_ascii=False),
        "요약": (
            "본 문서는 FastAPI를 활용한 Python 웹 개발 방법을 설명합니다. "
            "REST API 설계 원칙과 비동기 처리 패턴을 중심으로 다루며, "
            "초보 개발자도 쉽게 따라할 수 있는 단계별 가이드를 제공합니다."
        ),
    },
    default_response="요청을 처리했습니다.",
)

# 그래프 빌드
graph = build_summarizer_graph(llm=fake_llm)
print("그래프 빌드 완료!")
```

### 2.5 그래프 시각화

```python
# ── Cell 8: 그래프 시각화 ──

from IPython.display import display, Image, Markdown

try:
    png = graph.get_graph().draw_mermaid_png()
    display(Image(png))
except Exception:
    mermaid = graph.get_graph().draw_mermaid()
    display(Markdown(f"```mermaid\n{mermaid}\n```"))

print("위 다이어그램에서 에이전트의 흐름을 확인하세요.")
print("- fetch_document에서 analyze_content 또는 END로 분기")
print("- analyze_content에서 generate_summary 또는 END로 분기")
```

---

## 3. 실전 예제

### 3.1 정상 경로 실행

```python
# ── Cell 9: 정상 케이스 실행 ──

# 초기 상태
initial_state: DocumentSummaryState = {
    "source_text": (
        "FastAPI 시작하기\n"
        "FastAPI는 Python 3.7+로 API를 구축하기 위한 고성능 웹 프레임워크입니다. "
        "Starlette과 Pydantic을 기반으로 하며, 자동 문서 생성과 타입 검사를 제공합니다. "
        "비동기 처리를 기본 지원하여 높은 처리량을 달성할 수 있습니다."
    ),
    "document": None,
    "analysis": None,
    "summary": None,
    "final_output": None,
    "current_step": "start",
    "error": None,
}

# stream 모드로 실행하여 각 노드의 실행을 추적
print("=" * 60)
print("  문서 요약 에이전트 실행")
print("=" * 60)

events = []
for event in graph.stream(initial_state):
    events.append(event)
    node_name = list(event.keys())[0]
    print(f"\n>> 노드 실행 완료: {node_name}")

# 최종 결과 확인
final_state = {}
for event in events:
    for node_output in event.values():
        final_state.update(node_output)

if final_state.get("final_output"):
    print("\n" + final_state["final_output"])
elif final_state.get("error"):
    print(f"\n[에러] {final_state['error']}")
```

### 3.2 에러 경로 실행

```python
# ── Cell 10: 에러 케이스 - 빈 문서 ──

error_state: DocumentSummaryState = {
    "source_text": "",  # 빈 문서!
    "document": None,
    "analysis": None,
    "summary": None,
    "final_output": None,
    "current_step": "start",
    "error": None,
}

print("=" * 60)
print("  에러 케이스: 빈 문서 테스트")
print("=" * 60)

for event in graph.stream(error_state):
    node_name = list(event.keys())[0]
    node_output = event[node_name]
    print(f"\n>> 노드: {node_name}")
    if node_output.get("error"):
        print(f"   에러 발생: {node_output['error']}")
    print(f"   현재 단계: {node_output.get('current_step', 'N/A')}")

# 출력:
# >> 노드: fetch_document
#    에러 발생: 문서가 비어있습니다. 텍스트를 입력해주세요.
#    현재 단계: error
# (analyze_content, generate_summary, format_output은 실행되지 않음)
```

### 3.3 각 노드 독립 테스트

전체 그래프를 실행하지 않고 개별 노드를 직접 테스트합니다.

```python
# ── Cell 11: 개별 노드 테스트 ──

print("=" * 60)
print("  개별 노드 테스트")
print("=" * 60)

# --- 테스트 1: fetch_document ---
print("\n[테스트 1] fetch_document - 정상 입력")
result_1 = fetch_document({
    "source_text": "테스트 제목\n테스트 본문 내용입니다.",
    "document": None, "analysis": None, "summary": None,
    "final_output": None, "current_step": "start", "error": None,
})
assert result_1["current_step"] == "fetched"
assert result_1["document"]["title"] == "테스트 제목"
print(f"  통과! document.title = '{result_1['document']['title']}'")

print("\n[테스트 1-b] fetch_document - 빈 입력")
result_1b = fetch_document({
    "source_text": "   ",
    "document": None, "analysis": None, "summary": None,
    "final_output": None, "current_step": "start", "error": None,
})
assert result_1b.get("error") is not None
print(f"  통과! error = '{result_1b['error']}'")

# --- 테스트 2: analyze_content ---
print("\n[테스트 2] analyze_content - FakeLLM 사용")
analyze_fn = partial(analyze_content, llm=fake_llm)
result_2 = analyze_fn({
    "source_text": "", "document": {"title": "테스트", "content": "분석 대상 텍스트"},
    "analysis": None, "summary": None, "final_output": None,
    "current_step": "fetched", "error": None,
})
assert result_2["current_step"] == "analyzed"
assert result_2["analysis"] is not None
print(f"  통과! analysis.topic = '{result_2['analysis'].get('topic', 'N/A')}'")

# --- 테스트 3: generate_summary ---
print("\n[테스트 3] generate_summary - FakeLLM 사용")
summary_fn = partial(generate_summary, llm=fake_llm)
result_3 = summary_fn({
    "source_text": "", "document": {"title": "테스트", "content": "요약 대상"},
    "analysis": {"keywords": ["테스트"], "topic": "테스트 주제", "difficulty": "초급"},
    "summary": None, "final_output": None,
    "current_step": "analyzed", "error": None,
})
assert result_3["current_step"] == "summarized"
assert result_3["summary"] is not None
print(f"  통과! summary 길이 = {len(result_3['summary'])}자")

# --- 테스트 4: format_output ---
print("\n[테스트 4] format_output")
result_4 = format_output({
    "source_text": "", "document": {"title": "테스트", "word_count": 50},
    "analysis": {"keywords": ["A", "B"], "topic": "테스트", "difficulty": "초급"},
    "summary": "테스트 요약입니다.",
    "final_output": None, "current_step": "summarized", "error": None,
})
assert result_4["current_step"] == "completed"
assert "테스트 요약입니다." in result_4["final_output"]
print(f"  통과! final_output 길이 = {len(result_4['final_output'])}자")

print("\n" + "=" * 60)
print("  모든 노드 테스트 통과!")
print("=" * 60)
```

---

## 4. 연습 문제

### 연습 4-1: 전체 코드를 한 파일로 정리

지금까지 작성한 코드를 하나의 Python 파일(`summarizer_agent.py`)로 정리해 보세요. 아래 뼈대를 완성하세요.

```python
"""문서 요약 에이전트 - 완전한 구현.

사용법:
    from summarizer_agent import build_summarizer_graph, FakeLLM

    fake_llm = FakeLLM(responses={...})
    graph = build_summarizer_graph(llm=fake_llm)
    result = graph.invoke(initial_state)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from functools import partial
import json

# ─── FakeLLM ───
# TODO: Module 03에서 만든 FakeLLM 클래스를 여기에 넣으세요

# ─── State ───
# TODO: DocumentSummaryState를 정의하세요

# ─── 노드 함수 ───
# TODO: 4개 노드 함수를 넣으세요

# ─── 조건 분기 함수 ───
# TODO: _route_after_fetch, _route_after_analyze를 넣으세요

# ─── 그래프 빌더 ───
# TODO: build_summarizer_graph 함수를 넣으세요

# ─── 메인 실행 ───
if __name__ == "__main__":
    fake_llm = FakeLLM(responses={
        "분석": '{"keywords": ["Python"], "topic": "프로그래밍", "difficulty": "초급"}',
        "요약": "이 문서는 Python 프로그래밍에 대한 입문 가이드입니다.",
    })

    graph = build_summarizer_graph(llm=fake_llm)

    result = graph.invoke({
        "source_text": "Python 입문\nPython은 배우기 쉬운 언어입니다.",
        "document": None, "analysis": None, "summary": None,
        "final_output": None, "current_step": "start", "error": None,
    })

    if result.get("final_output"):
        print(result["final_output"])
    elif result.get("error"):
        print(f"에러: {result['error']}")
```

### 연습 4-2: "이메일 분류 에이전트" 직접 만들기

아래 요구사항에 맞는 에이전트를 직접 설계하고 구현하세요.

#### 요구사항

| 항목 | 내용 |
|------|------|
| **에이전트 이름** | 이메일 분류 에이전트 (Email Classifier) |
| **입력** | 이메일 텍스트 (제목 + 본문) |
| **출력** | 분류 결과 (카테고리, 긴급도, 추천 조치) |
| **노드 수** | 4개 이상 |

#### 필요한 노드

1. **parse_email**: 이메일 텍스트에서 제목, 본문, 발신자를 추출
2. **classify_email**: LLM으로 카테고리 분류 (업무, 스팸, 마케팅, 개인)
3. **assess_urgency**: LLM으로 긴급도 판단 (높음, 중간, 낮음)
4. **generate_action**: 분류 결과를 바탕으로 추천 조치 생성

#### State 설계 힌트

```python
class EmailClassifierState(TypedDict):
    raw_email: str              # 원본 이메일 텍스트
    parsed_email: dict | None   # 파싱된 이메일 정보
    category: str | None        # 분류 카테고리
    urgency: str | None         # 긴급도
    action: str | None          # 추천 조치
    final_report: str | None    # 최종 보고서
    current_step: str           # 현재 단계
    error: str | None           # 에러 메시지
```

#### FakeLLM 설정 힌트

```python
fake_llm = FakeLLM(responses={
    "분류": '{"category": "업무", "confidence": 0.92}',
    "긴급": '{"urgency": "높음", "reason": "마감일이 내일입니다"}',
    "조치": "1. 즉시 회신 필요\n2. 관련 팀에 공유\n3. 일정에 등록",
})
```

#### 도전 과제

- 조건부 분기 추가: "스팸" 카테고리인 경우 assess_urgency를 건너뛰고 바로 종료
- 모든 노드를 개별적으로 테스트하는 코드 작성
- 그래프를 시각화하고, 정상/에러/스팸 세 가지 경로를 각각 실행해 보기

```python
# 여기에 코드를 작성하세요

# 1. State 정의

# 2. 노드 함수 구현 (4개)

# 3. 조건 분기 함수

# 4. 그래프 빌더

# 5. FakeLLM 설정 및 빌드

# 6. 테스트 케이스 3가지 실행:
#    - 정상 업무 이메일
#    - 스팸 이메일 (assess_urgency 건너뛰기)
#    - 빈 이메일 (에러 경로)
```

---

## 5. 핵심 정리

### 에이전트 개발 체크리스트

| 단계 | 핵심 질문 | 이 모듈에서 배운 답 |
|------|----------|-------------------|
| **State 설계** | 어떤 데이터가 노드 사이를 오가는가? | TypedDict로 명시적 필드 정의. 입력/중간/출력/제어 필드 분류 |
| **노드 구현** | 각 노드는 무엇을 하는가? | `(state) -> dict` 함수. 한 노드 = 한 책임 |
| **의존성 주입** | LLM, API를 어떻게 교체하는가? | `functools.partial`로 외부 주입. 테스트 시 FakeLLM/Mock 교체 |
| **엣지 연결** | 노드 실행 순서는? | 일반 엣지(무조건 이동) + 조건부 엣지(분기 결정 함수) |
| **에러 처리** | 실패하면 어떻게 되는가? | `state["error"]` 설정 + 조건부 엣지로 에러 경로 분기 |
| **테스트** | 어떻게 검증하는가? | 개별 노드 독립 호출 + FakeLLM 기반 E2E 실행 |

### 핵심 패턴 요약

```python
# 패턴 1: 노드 함수 시그니처
def my_node(state: MyState, llm=None, tool=None) -> dict:
    """state를 읽고, 처리하고, 변경된 필드만 반환"""
    # state에서 데이터 읽기
    data = state["some_field"]
    # 처리
    result = llm.invoke(data)
    # 변경된 필드만 반환
    return {"output_field": result, "current_step": "done"}

# 패턴 2: 의존성 주입
from functools import partial
node_fn = partial(my_node, llm=fake_llm, tool=mock_tool)

# 패턴 3: 조건부 분기
def route_fn(state: MyState) -> str:
    if state.get("error"):
        return "error_path"
    return "next_path"

graph.add_conditional_edges("my_node", route_fn, {
    "next_path": "next_node",
    "error_path": END,
})

# 패턴 4: 그래프 빌드
def build_graph(llm=None):
    graph = StateGraph(MyState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", partial(node_b, llm=llm))
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    return graph.compile()
```

---

## 6. 참고 자료

### 공식 문서

- **LangGraph How-to Guides**: [https://langchain-ai.github.io/langgraph/how-tos/](https://langchain-ai.github.io/langgraph/how-tos/)
  - LangGraph의 모든 기능에 대한 실전 가이드. 그래프 정의, 조건부 분기, 스트리밍 등
- **functools.partial**: [https://docs.python.org/3/library/functools.html#functools.partial](https://docs.python.org/3/library/functools.html#functools.partial)
  - 의존성 주입에 사용하는 partial 함수의 공식 레퍼런스
- **Python TypedDict**: [https://docs.python.org/3/library/typing.html#typing.TypedDict](https://docs.python.org/3/library/typing.html#typing.TypedDict)
  - State 정의에 사용하는 TypedDict의 공식 문서

### 추가 학습 자료

- **LangGraph 시각화**: [https://langchain-ai.github.io/langgraph/how-tos/visualization/](https://langchain-ai.github.io/langgraph/how-tos/visualization/)
  - 그래프 다이어그램 렌더링 방법
- **LangGraph Conditional Edges**: [https://langchain-ai.github.io/langgraph/how-tos/branching/](https://langchain-ai.github.io/langgraph/how-tos/branching/)
  - 조건부 분기 패턴 상세 설명
- **Python unittest.mock**: [https://docs.python.org/3/library/unittest.mock.html](https://docs.python.org/3/library/unittest.mock.html)
  - Mock 객체 활용 레퍼런스
- **LangGraph State Management**: [https://langchain-ai.github.io/langgraph/concepts/low_level/#state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
  - State 설계 원칙과 고급 패턴

### 흔한 실수와 해결책 (FAQ)

| 문제 | 원인 | 해결 |
|------|------|------|
| `TypeError: 'NoneType' is not subscriptable` | 이전 노드가 채워야 할 필드가 None인 상태에서 접근 | `state.get("field")` 사용 또는 None 체크 추가 |
| 그래프 실행 시 무한 루프 | 조건부 분기 함수가 잘못된 키를 반환 | 분기 함수의 반환값이 `add_conditional_edges`의 매핑 딕셔너리 키와 일치하는지 확인 |
| `KeyError: 'field_name'` | State에 정의하지 않은 필드를 사용 | TypedDict에 모든 필드를 정의하고, 초기 상태에서 모든 필드에 값을 설정 |
| FakeLLM이 기본 응답만 반환 | 패턴 매칭 키워드가 프롬프트에 포함되지 않음 | FakeLLM의 `responses` 키가 프롬프트 텍스트에 실제로 포함되는지 확인 |
| `partial()`로 주입한 인자가 전달 안 됨 | 노드 함수의 매개변수 이름 불일치 | `partial(fn, llm=llm)` 에서 `llm`이 함수 시그니처의 매개변수명과 정확히 일치해야 함 |
| 에러 경로로 가지 않고 예외 발생 | 노드 내부에서 `raise`를 사용 | `raise` 대신 `return {"error": "메시지", "current_step": "error"}` 패턴 사용 |
| 그래프 시각화가 안 됨 | `draw_mermaid_png()` 실패 (외부 도구 미설치) | `draw_mermaid()` 텍스트 출력 사용 또는 Mermaid Live Editor 활용 |

---

## 다음 단계

이 모듈에서 첫 번째 완전한 LangGraph 에이전트를 만들었습니다. 축하합니다!

지금까지 배운 것:
- Module 01-02: LangGraph 기본 개념과 설치
- Module 03: Jupyter 개발 환경, FakeLLM, 시각화
- **Module 04: 완전한 에이전트 설계, 구현, 테스트** (이번 모듈)

다음 모듈에서 다루는 내용:
- **Module 05**: 에이전트 고급 패턴 - Annotated State, Reducer, 서브그래프
- **Module 06**: 에이전트 프로덕션 배포 - 체크포인트, 재시도, 모니터링
