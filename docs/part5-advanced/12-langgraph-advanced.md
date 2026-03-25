# Module 12: LangGraph 고급 패턴 - 프로덕션 수준의 에이전트 아키텍처

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. 체크포인팅으로 장애 발생 시 마지막 성공 지점부터 재개할 수 있다
2. 서브그래프로 복잡한 워크플로우를 모듈 단위로 분리할 수 있다
3. Send API로 병렬 실행(Fan-out/Fan-in)을 구현할 수 있다
4. 스트리밍으로 실시간 진행 상태를 전달할 수 있다
5. 동적 노드 구성과 커스텀 Reducer를 활용할 수 있다
6. 그래프를 시각화하고 디버깅할 수 있다

---

## 사전 지식

| 항목 | 필요 수준 | 참고 모듈 |
|------|----------|----------|
| LangGraph 기초 | StateGraph, 노드, 엣지, 조건부 분기 | Module 09 |
| Python async/await | 기본적인 비동기 문법 | - |
| LLM API 호출 | ChatAnthropic 또는 ChatOpenAI | Module 03 |
| TypedDict | Python 타입 힌트 | Module 09 |

---

## 1. 개념 설명

### 1.1 왜 고급 패턴이 필요한가?

기본 LangGraph 그래프로도 에이전트를 만들 수 있지만, **프로덕션 환경**에서는 추가적인 요구사항이 생깁니다:

```
[개발 환경]                        [프로덕션 환경]
  에이전트 실행 실패            →    처음부터 다시? (비용 + 시간 낭비)
  모든 노드가 한 그래프에        →    재사용 불가, 유지보수 어려움
  하나씩 순차 처리              →    느림 (병렬 처리 필요)
  실행 완료까지 상태 모름        →    사용자는 진행 상황을 알고 싶음
```

**고급 패턴이 해결하는 문제:**

| 문제 | 패턴 | 효과 |
|------|------|------|
| 장애 시 처음부터 재시작 | 체크포인팅 | 마지막 성공 지점에서 재개 |
| 그래프가 너무 복잡 | 서브그래프 | 모듈로 분리하여 재사용 |
| 직렬 처리로 느림 | 병렬 실행 (Send API) | 여러 작업 동시 처리 |
| 진행 상황 알 수 없음 | 스트리밍 | 실시간 상태 전달 |
| 설정에 따라 다른 동작 | 동적 노드 구성 | 유연한 그래프 빌드 |

### 1.2 이 모듈에서 다루는 패턴 전체 그림

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 고급 패턴                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ 1. Checkpoint │  │ 2. Subgraph  │  │ 3. Parallel      │   │
│  │  (영속성)      │  │  (모듈화)     │  │  (병렬 실행)      │   │
│  │              │  │              │  │                  │   │
│  │  MemorySaver │  │  부모 ↔ 자식  │  │  Send API        │   │
│  │  SqliteSaver │  │  상태 매핑    │  │  Fan-out/Fan-in  │   │
│  │  PostgresSaver│  │  재사용 가능  │  │  operator.add    │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ 4. Streaming  │  │ 5. Dynamic   │  │ 6. State/Reducer │   │
│  │  (실시간)      │  │  (동적 구성)  │  │  (상태 관리)      │   │
│  │              │  │              │  │                  │   │
│  │  astream     │  │  설정 기반    │  │  Annotated       │   │
│  │  토큰 스트림  │  │  Feature flag │  │  Custom Reducer  │   │
│  │  이벤트 감지  │  │  조건부 노드  │  │  add_messages    │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 7. 시각화 & 디버깅 — draw_mermaid, LangSmith         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 단계별 실습

### 2.1 체크포인팅 (Persistence)

#### 왜 필요한가?

에이전트가 5개 노드를 실행하는 중에 4번째 노드에서 장애가 발생하면:

```
체크포인팅 없이:
  [Node A] → [Node B] → [Node C] → [Node D] ← 장애 발생!
                                         ↓
  전체 재시작 → [Node A] → [Node B] → [Node C] → [Node D] → ...
  (Node A~C를 다시 실행 = 시간 + 비용 낭비)

체크포인팅 있으면:
  [Node A] → ✓저장 → [Node B] → ✓저장 → [Node C] → ✓저장 → [Node D] ← 장애!
                                                        ↓
  마지막 체크포인트에서 재개 → [Node D] → [Node E] → 완료!
  (Node A~C는 건너뜀 = 시간 + 비용 절약)
```

#### 체크포인터 종류 비교

| 체크포인터 | 저장 위치 | 재시작 시 복구 | 용도 |
|-----------|----------|-------------|------|
| MemorySaver | 메모리 (RAM) | 프로세스 재시작 시 유실 | 개발/테스트 |
| SqliteSaver | SQLite 파일 | 파일이 있으면 복구 가능 | 로컬 개발 |
| PostgresSaver | PostgreSQL | 영속적 복구 가능 | 프로덕션 |

#### Step 1: MemorySaver (테스트용)

```python
# checkpoint_memory.py
"""MemorySaver를 사용한 체크포인팅 기본 예제."""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class ProcessState(TypedDict):
    """처리 상태."""
    input_data: str
    step1_result: str | None
    step2_result: str | None
    step3_result: str | None
    current_step: str


def step1_node(state: ProcessState) -> dict:
    """1단계: 데이터 수집."""
    print("  [Step 1] 데이터 수집 중...")
    result = f"수집 완료: {state['input_data']}"
    return {"step1_result": result, "current_step": "step1"}


def step2_node(state: ProcessState) -> dict:
    """2단계: 데이터 분석."""
    print("  [Step 2] 데이터 분석 중...")
    result = f"분석 완료: {state['step1_result']}"
    return {"step2_result": result, "current_step": "step2"}


def step3_node(state: ProcessState) -> dict:
    """3단계: 결과 생성."""
    print("  [Step 3] 결과 생성 중...")
    result = f"최종 결과: {state['step2_result']}"
    return {"step3_result": result, "current_step": "step3"}


# 그래프 구성
graph = StateGraph(ProcessState)
graph.add_node("step1", step1_node)
graph.add_node("step2", step2_node)
graph.add_node("step3", step3_node)

graph.set_entry_point("step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)

# 체크포인터 연결
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# --- 실행 ---

# thread_id: 실행을 식별하는 고유 키 (같은 ID면 이전 상태에서 재개)
config = {"configurable": {"thread_id": "task-001"}}

result = app.invoke(
    {
        "input_data": "사용자 로그 데이터",
        "step1_result": None,
        "step2_result": None,
        "step3_result": None,
        "current_step": "start",
    },
    config=config,
)

print(f"\n최종 결과: {result['step3_result']}")

# 체크포인트 상태 확인
state_snapshot = app.get_state(config)
print(f"마지막 단계: {state_snapshot.values['current_step']}")
```

#### Step 2: SqliteSaver (로컬 영속화)

```python
# checkpoint_sqlite.py
"""SqliteSaver를 사용한 영속적 체크포인팅.

프로그램을 종료하고 다시 실행해도 마지막 체크포인트에서 재개됩니다.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


class JobState(TypedDict):
    """작업 상태."""
    job_id: str
    data: str
    processed: bool
    result: str | None
    current_step: str


def fetch_data_node(state: JobState) -> dict:
    """데이터를 가져옵니다."""
    print(f"  [Fetch] Job {state['job_id']}: 데이터 수집 중...")
    return {
        "data": f"Raw data for {state['job_id']}",
        "current_step": "fetch",
    }


def process_data_node(state: JobState) -> dict:
    """데이터를 처리합니다."""
    print(f"  [Process] Job {state['job_id']}: 데이터 처리 중...")
    return {
        "processed": True,
        "result": f"Processed: {state['data']}",
        "current_step": "process",
    }


def save_result_node(state: JobState) -> dict:
    """결과를 저장합니다."""
    print(f"  [Save] Job {state['job_id']}: 결과 저장 중...")
    return {"current_step": "save"}


# 그래프 구성
graph = StateGraph(JobState)
graph.add_node("fetch", fetch_data_node)
graph.add_node("process", process_data_node)
graph.add_node("save", save_result_node)

graph.set_entry_point("fetch")
graph.add_edge("fetch", "process")
graph.add_edge("process", "save")
graph.add_edge("save", END)

# SqliteSaver: 파일로 체크포인트 저장
# 프로그램 종료 후에도 체크포인트가 유지됩니다
saver = SqliteSaver.from_conn_string("./checkpoints.db")
app = graph.compile(checkpointer=saver)


def run_job(job_id: str):
    """작업을 실행하거나 재개합니다."""
    config = {"configurable": {"thread_id": f"job-{job_id}"}}

    # 기존 체크포인트 확인
    existing_state = app.get_state(config)

    if existing_state.values:
        print(f"\n기존 체크포인트 발견! (마지막 단계: {existing_state.values.get('current_step')})")
        print("마지막 체크포인트에서 재개합니다...")

        # 에러 상태가 있다면 초기화
        if existing_state.values.get("error"):
            app.update_state(config, {"error": None})

        # 재개 (마지막 체크포인트부터)
        result = app.invoke(None, config=config)
    else:
        print(f"\n새로운 작업 시작: {job_id}")
        result = app.invoke(
            {
                "job_id": job_id,
                "data": "",
                "processed": False,
                "result": None,
                "current_step": "start",
            },
            config=config,
        )

    print(f"완료: {result.get('result', 'N/A')}")
    return result


# 실행
if __name__ == "__main__":
    run_job("analysis-001")
```

#### Step 3: PostgresSaver (프로덕션용)

```python
# checkpoint_postgres.py
"""PostgresSaver를 사용한 프로덕션 체크포인팅.

사전 조건: PostgreSQL이 실행 중이어야 합니다.
pip install langgraph-checkpoint-postgres psycopg[binary]
"""
from langgraph.checkpoint.postgres import PostgresSaver

# PostgreSQL 연결
# 환경변수에서 URI를 읽는 것이 좋습니다
POSTGRES_URI = "postgresql://user:password@localhost:5432/checkpoints_db"

saver = PostgresSaver.from_conn_string(POSTGRES_URI)
saver.setup()  # 체크포인트 테이블 자동 생성

# 이후 사용법은 MemorySaver/SqliteSaver와 동일:
# app = graph.compile(checkpointer=saver)
# result = app.invoke(input_state, config={"configurable": {"thread_id": "..."}})
```

#### 체크포인터 팩토리 패턴 (환경별 자동 선택)

```python
# checkpointer_factory.py
"""환경에 따라 적절한 체크포인터를 생성하는 팩토리."""
import os
import logging

logger = logging.getLogger(__name__)


def create_checkpointer(
    checkpointer_type: str = "memory",
    sqlite_path: str = "./checkpoints.db",
    postgres_uri: str | None = None,
):
    """환경에 따라 적절한 체크포인터를 생성합니다.

    Args:
        checkpointer_type: "memory", "sqlite", 또는 "postgres".
        sqlite_path: SQLite 파일 경로 (sqlite 타입 시).
        postgres_uri: PostgreSQL 연결 URI (postgres 타입 시).

    Returns:
        LangGraph 체크포인터 인스턴스.

    Examples:
        >>> # 테스트용
        >>> saver = create_checkpointer("memory")

        >>> # 로컬 개발용
        >>> saver = create_checkpointer("sqlite", sqlite_path="./dev_checkpoints.db")

        >>> # 프로덕션용
        >>> saver = create_checkpointer("postgres", postgres_uri="postgresql://...")
    """
    if checkpointer_type == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("체크포인터: MemorySaver (인메모리)")
        return MemorySaver()

    elif checkpointer_type == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info("체크포인터: SqliteSaver (경로: %s)", sqlite_path)
        return SqliteSaver.from_conn_string(sqlite_path)

    elif checkpointer_type == "postgres":
        if not postgres_uri:
            postgres_uri = os.getenv("CHECKPOINT_POSTGRES_URI")
        if not postgres_uri:
            raise ValueError(
                "PostgresSaver에는 postgres_uri가 필요합니다. "
                "환경변수 CHECKPOINT_POSTGRES_URI를 설정하세요."
            )
        from langgraph.checkpoint.postgres import PostgresSaver
        logger.info("체크포인터: PostgresSaver")
        saver = PostgresSaver.from_conn_string(postgres_uri)
        saver.setup()
        return saver

    else:
        raise ValueError(f"알 수 없는 체크포인터 타입: {checkpointer_type}")
```

#### thread_id 설계 가이드

```
thread_id는 "실행 세션"을 식별하는 고유 키입니다.
같은 thread_id로 호출하면 → 이전 체크포인트에서 재개
새로운 thread_id로 호출하면 → 처음부터 새로 실행

thread_id 설계 패턴:

  작업 유형 + 고유 식별자
  ─────────────────────
  "analysis-user123-20260323"     ← 사용자 + 날짜
  "review-pr-456"                 ← PR 번호
  "translate-doc-abc123"          ← 문서 ID
  "pipeline-order-789-attempt-2"  ← 재시도 포함
```

---

### 2.2 서브그래프 (Subgraph Composition)

#### 왜 필요한가?

그래프가 커지면 관리하기 어려워집니다:

```
문제: 10개 이상 노드가 한 그래프에 → 이해/유지보수 어려움

  [A] → [B] → [C] → [D] → [E] → [F] → [G] → [H] → [I] → [J]
  └──────────── 10개 노드가 하나의 거대한 그래프에 ──────────────┘

해결: 서브그래프로 분리 → 각 모듈을 독립적으로 관리

  [메인 그래프]
  [A] → [B] → [서브그래프 1] → [서브그래프 2] → [J]
                    │                  │
               [C] → [D] → [E]   [F] → [G] → [H] → [I]
```

**서브그래프의 장점:**

| 장점 | 설명 |
|------|------|
| 모듈화 | 각 부분을 독립적으로 개발/테스트 가능 |
| 재사용 | 같은 서브그래프를 여러 메인 그래프에서 사용 |
| 상태 격리 | 서브그래프는 필요한 필드만 접근 |
| 가독성 | 전체 구조를 한눈에 파악 가능 |

#### Step 1: 서브그래프 정의

```python
# subgraph_example.py
"""서브그래프 패턴 예제: 문서 분석 파이프라인.

메인 그래프: 입력 → 전처리 → [분석 서브그래프] → 결과 정리 → 출력
분석 서브그래프: 키워드 추출 → 요약 생성 → 감성 분석
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# --- 서브그래프 상태 (분석에 필요한 필드만) ---

class AnalysisSubState(TypedDict):
    """분석 서브그래프의 격리된 상태.

    메인 그래프의 전체 상태가 아닌,
    분석에 필요한 필드만 포함합니다.
    """
    text: str                       # 분석할 텍스트
    keywords: list[str] | None      # 추출된 키워드
    summary: str | None             # 요약
    sentiment: str | None           # 감성 (positive/negative/neutral)


# --- 서브그래프 노드 ---

def extract_keywords_node(state: AnalysisSubState) -> dict:
    """키워드를 추출합니다."""
    text = state["text"]
    # 간단한 키워드 추출 (실제로는 LLM 호출)
    words = text.split()
    keywords = [w for w in words if len(w) > 3][:5]
    print(f"  [서브그래프] 키워드 추출: {keywords}")
    return {"keywords": keywords}


def summarize_node(state: AnalysisSubState) -> dict:
    """텍스트를 요약합니다."""
    text = state["text"]
    # 간단한 요약 (실제로는 LLM 호출)
    summary = text[:100] + "..." if len(text) > 100 else text
    print(f"  [서브그래프] 요약 생성 완료")
    return {"summary": summary}


def analyze_sentiment_node(state: AnalysisSubState) -> dict:
    """감성을 분석합니다."""
    # 간단한 감성 분석 (실제로는 LLM 호출)
    text = state["text"].lower()
    if any(w in text for w in ["좋", "훌륭", "만족", "good", "great"]):
        sentiment = "positive"
    elif any(w in text for w in ["나쁜", "불만", "실망", "bad", "poor"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"
    print(f"  [서브그래프] 감성 분석: {sentiment}")
    return {"sentiment": sentiment}


# --- 서브그래프 빌드 ---

def build_analysis_subgraph():
    """분석 서브그래프를 빌드합니다.

    키워드 추출 → 요약 생성 → 감성 분석
    """
    graph = StateGraph(AnalysisSubState)

    graph.add_node("extract_keywords", extract_keywords_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("analyze_sentiment", analyze_sentiment_node)

    graph.set_entry_point("extract_keywords")
    graph.add_edge("extract_keywords", "summarize")
    graph.add_edge("summarize", "analyze_sentiment")
    graph.add_edge("analyze_sentiment", END)

    return graph.compile()


# --- 메인 그래프 상태 ---

class MainState(TypedDict):
    """메인 그래프 상태 (전체 필드 포함)."""
    document_id: str
    raw_text: str
    cleaned_text: str | None
    # 서브그래프 결과가 여기에 매핑됨
    text: str                       # 서브그래프 입력용
    keywords: list[str] | None
    summary: str | None
    sentiment: str | None
    # 최종 결과
    report: str | None


# --- 메인 그래프 노드 ---

def preprocess_node(state: MainState) -> dict:
    """텍스트를 전처리합니다."""
    raw = state["raw_text"]
    cleaned = raw.strip().replace("\n", " ")
    print(f"[메인] 전처리 완료: {len(cleaned)}자")
    return {"cleaned_text": cleaned, "text": cleaned}


def generate_report_node(state: MainState) -> dict:
    """최종 리포트를 생성합니다."""
    report = (
        f"문서 분석 리포트 (ID: {state['document_id']})\n"
        f"{'='*50}\n"
        f"키워드: {', '.join(state.get('keywords') or [])}\n"
        f"요약: {state.get('summary', 'N/A')}\n"
        f"감성: {state.get('sentiment', 'N/A')}\n"
    )
    print(f"[메인] 리포트 생성 완료")
    return {"report": report}


# --- 메인 그래프 구성 ---

def build_main_graph():
    """서브그래프가 포함된 메인 그래프를 구성합니다.

    흐름:
        preprocess → [analysis_subgraph] → generate_report → END
    """
    graph = StateGraph(MainState)

    # 서브그래프를 노드로 등록
    analysis_subgraph = build_analysis_subgraph()
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("analyze", analysis_subgraph)  # 서브그래프를 노드처럼 등록!
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "analyze")
    graph.add_edge("analyze", "generate_report")
    graph.add_edge("generate_report", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# --- 실행 ---

if __name__ == "__main__":
    app = build_main_graph()

    result = app.invoke(
        {
            "document_id": "DOC-001",
            "raw_text": "이 제품은 정말 훌륭합니다. 성능이 좋고 디자인이 만족스럽습니다.",
            "cleaned_text": None,
            "text": "",
            "keywords": None,
            "summary": None,
            "sentiment": None,
            "report": None,
        },
        config={"configurable": {"thread_id": "doc-analysis-001"}},
    )

    print(f"\n{result['report']}")
```

#### 상태 매핑 다이어그램

```
메인 그래프 상태 (MainState)         서브그래프 상태 (AnalysisSubState)
┌──────────────────────┐            ┌──────────────────────┐
│ document_id          │            │                      │
│ raw_text             │            │                      │
│ cleaned_text         │            │                      │
│ text ────────────────│───입력────→│ text                 │
│ keywords ←───────────│───출력────│ keywords             │
│ summary ←────────────│───출력────│ summary              │
│ sentiment ←──────────│───출력────│ sentiment            │
│ report               │            │                      │
└──────────────────────┘            └──────────────────────┘

* 서브그래프는 자신의 상태 필드만 접근 가능
* 이름이 같은 필드를 통해 자동으로 매핑됨
```

---

### 2.3 병렬 실행 (Fan-out / Fan-in)

#### 핵심 개념

```
순차 실행 (기존):
  [문서 A 분석] → [문서 B 분석] → [문서 C 분석] → [결과 합치기]
  총 시간: 30초 + 30초 + 30초 = 90초

병렬 실행 (Send API):
  ┌→ [문서 A 분석] ─┐
  ├→ [문서 B 분석] ─┼→ [결과 합치기]
  └→ [문서 C 분석] ─┘
  총 시간: max(30초, 30초, 30초) = 30초

Fan-out: 하나의 노드에서 여러 병렬 노드로 작업 분배
Fan-in:  여러 병렬 노드의 결과를 하나로 수집
```

#### Send API + Reducer 구현

```python
# parallel_example.py
"""Send API를 사용한 병렬 실행 예제: 멀티 문서 분석."""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langchain_anthropic import ChatAnthropic


# --- 상태 정의 ---

class DocumentResult(TypedDict):
    """개별 문서 분석 결과."""
    doc_id: str
    summary: str
    word_count: int


class ParallelState(TypedDict):
    """병렬 처리 상태.

    핵심: Annotated[list, operator.add]
    - 각 병렬 노드가 반환한 리스트가 자동으로 합쳐집니다.
    - 예: 노드 A가 [결과1]을, 노드 B가 [결과2]를 반환하면
          최종 results = [결과1, 결과2]
    """
    documents: list[dict]                                    # 입력 문서 목록
    results: Annotated[list[DocumentResult], operator.add]   # 병렬 결과 수집
    final_report: str | None                                 # 최종 리포트


# --- 노드 정의 ---

def prepare_node(state: ParallelState) -> dict:
    """문서 목록을 확인하고 병렬 처리를 준비합니다."""
    docs = state.get("documents", [])
    print(f"[준비] {len(docs)}개 문서를 병렬 분석합니다.")
    return {}


def fan_out_documents(state: ParallelState) -> list[Send]:
    """각 문서에 대해 독립적인 분석 노드를 생성합니다 (Fan-out).

    Send API 핵심:
    - Send("노드이름", 상태)를 리스트로 반환
    - 각 Send는 독립적으로 병렬 실행됨
    - 결과는 Reducer(operator.add)로 자동 수집됨
    """
    documents = state.get("documents", [])

    if not documents:
        return [Send("generate_report", state)]

    sends = []
    for doc in documents:
        # 각 문서에 대해 독립적인 상태를 만들어 Send
        send_state = {
            **state,
            "_current_doc": doc,    # 현재 처리할 문서
            "results": [],          # 각 병렬 노드의 초기값 (빈 리스트)
        }
        sends.append(Send("analyze_document", send_state))

    return sends


def analyze_document_node(state: ParallelState) -> dict:
    """개별 문서를 분석합니다 (병렬 실행됨).

    반환하는 results 리스트는 Reducer(operator.add)에 의해
    다른 병렬 노드의 결과와 자동으로 합쳐집니다.
    """
    doc = state.get("_current_doc", {})
    doc_id = doc.get("id", "unknown")
    content = doc.get("content", "")

    print(f"  [분석] 문서 {doc_id} 분석 중...")

    # 실제로는 LLM을 호출하여 분석
    word_count = len(content.split())
    summary = content[:50] + "..." if len(content) > 50 else content

    return {
        "results": [
            DocumentResult(
                doc_id=doc_id,
                summary=summary,
                word_count=word_count,
            )
        ],
    }


def generate_report_node(state: ParallelState) -> dict:
    """모든 병렬 분석 결과를 합쳐서 최종 리포트를 생성합니다 (Fan-in)."""
    results = state.get("results", [])

    report_lines = [
        f"멀티 문서 분석 리포트",
        f"{'='*40}",
        f"총 분석 문서: {len(results)}개",
        f"",
    ]

    for result in results:
        report_lines.append(
            f"  [{result['doc_id']}] "
            f"{result['summary']} "
            f"({result['word_count']} words)"
        )

    total_words = sum(r["word_count"] for r in results)
    report_lines.append(f"\n총 단어 수: {total_words}")

    report = "\n".join(report_lines)
    print(f"\n[리포트] 생성 완료")
    return {"final_report": report}


# --- 그래프 구성 ---

def build_parallel_graph():
    """병렬 문서 분석 그래프.

    흐름:
        prepare → [fan-out: 문서별 analyze] → [fan-in] → generate_report → END

    핵심 패턴:
        1. fan_out_documents 함수가 Send 리스트를 반환
        2. 각 Send는 독립적으로 analyze_document 노드를 실행
        3. results 필드가 operator.add Reducer로 자동 합산
        4. 모든 병렬 노드 완료 후 generate_report 실행
    """
    graph = StateGraph(ParallelState)

    graph.add_node("prepare", prepare_node)
    graph.add_node("analyze_document", analyze_document_node)
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("prepare")

    # Fan-out: prepare 이후 각 문서별로 병렬 분석
    graph.add_conditional_edges(
        "prepare",
        fan_out_documents,
        ["analyze_document", "generate_report"],
    )

    # Fan-in: 모든 병렬 노드 완료 후 리포트 생성
    graph.add_edge("analyze_document", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# --- 실행 ---

if __name__ == "__main__":
    app = build_parallel_graph()

    result = app.invoke({
        "documents": [
            {"id": "DOC-001", "content": "Python은 간결하고 읽기 쉬운 프로그래밍 언어입니다."},
            {"id": "DOC-002", "content": "LangGraph는 LLM 에이전트 워크플로우를 구축하는 프레임워크입니다."},
            {"id": "DOC-003", "content": "체크포인팅은 장애 복구를 위한 핵심 기능으로 프로덕션에서 필수입니다."},
        ],
        "results": [],
        "final_report": None,
    })

    print(result["final_report"])
```

#### Map-Reduce 패턴 다이어그램

```
                    Fan-out (Map)                Fan-in (Reduce)
                    ┌──────────────┐
 ┌──────────┐      │ analyze(A)   │──────┐
 │ prepare  │──────│ analyze(B)   │──────┤      ┌──────────────┐
 │          │      │ analyze(C)   │──────├─────→│ generate     │
 └──────────┘      │ analyze(D)   │──────┤      │ report       │
                    └──────────────┘      │      └──────────────┘
                                          │
                   results: [A결과]        │
                   results: [B결과]        │
                   results: [C결과]   ─────┘
                   results: [D결과]
                          │
                   operator.add로 합산
                          ↓
                   results: [A결과, B결과, C결과, D결과]
```

---

### 2.4 스트리밍 (Streaming)

#### 왜 필요한가?

LLM 호출은 수십 초에서 수 분이 걸릴 수 있습니다. 스트리밍 없이는 사용자가 "지금 뭘 하고 있는지" 알 수 없습니다.

```
스트리밍 없이:                      스트리밍 사용:
  [요청] ──── 60초 대기 ──── [결과]    [요청]
                                      ├─ "Step 1: 데이터 수집 중..."
  사용자: "멈춘 건가...?"              ├─ "Step 2: LLM 분석 중..."
                                      ├─ "Step 2: 토큰 생성 중... 45%"
                                      ├─ "Step 3: 결과 정리 중..."
                                      └─ [결과]
```

#### LangGraph 스트리밍 모드 비교

| 모드 | 설명 | 용도 |
|------|------|------|
| `"values"` | 각 스텝 후 전체 상태 | 상태 변화 추적 |
| `"updates"` | 노드가 반환한 변경분만 | 경량 업데이트 |
| `"messages"` | LLM 토큰 + 메타데이터 | 실시간 텍스트 출력 |
| `"custom"` | 사용자 정의 데이터 | 노드 내부 진행 상태 |

#### 구현: 노드 단위 + 토큰 단위 스트리밍

```python
# streaming_example.py
"""LangGraph 스트리밍 예제."""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic


class StreamState(TypedDict):
    """스트리밍 예제 상태."""
    query: str
    research: str | None
    analysis: str | None
    final_answer: str | None


def research_node(state: StreamState) -> dict:
    """리서치를 수행합니다."""
    return {"research": f"리서치 결과: '{state['query']}'에 대한 관련 정보"}


def analyze_node(state: StreamState) -> dict:
    """LLM으로 분석을 수행합니다."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    prompt = f"다음 정보를 분석해주세요:\n{state['research']}"
    response = llm.invoke(prompt)
    return {"analysis": response.content}


def answer_node(state: StreamState) -> dict:
    """최종 답변을 생성합니다."""
    return {"final_answer": f"최종 답변:\n{state['analysis']}"}


# 그래프 구성
graph = StateGraph(StreamState)
graph.add_node("research", research_node)
graph.add_node("analyze", analyze_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("research")
graph.add_edge("research", "analyze")
graph.add_edge("analyze", "answer")
graph.add_edge("answer", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


# --- 스트리밍 방식 1: 노드 업데이트 스트리밍 ---

async def stream_updates():
    """노드 단위 업데이트를 스트리밍합니다."""
    print("=== 노드 업데이트 스트리밍 ===\n")

    config = {"configurable": {"thread_id": "stream-001"}}
    input_state = {
        "query": "Python 비동기 프로그래밍의 장단점",
        "research": None,
        "analysis": None,
        "final_answer": None,
    }

    async for event in app.astream(
        input_state,
        config=config,
        stream_mode="updates",  # 변경된 필드만 스트리밍
    ):
        # event: {노드이름: {변경된 필드들}}
        for node_name, updates in event.items():
            print(f"[{node_name}] 완료")
            for key, value in updates.items():
                preview = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                print(f"  {key}: {preview}")
            print()


# --- 스트리밍 방식 2: LLM 토큰 스트리밍 ---

async def stream_tokens():
    """LLM 토큰을 실시간으로 스트리밍합니다."""
    print("=== 토큰 스트리밍 ===\n")

    config = {"configurable": {"thread_id": "stream-002"}}
    input_state = {
        "query": "FastAPI vs Flask 비교",
        "research": None,
        "analysis": None,
        "final_answer": None,
    }

    async for event in app.astream(
        input_state,
        config=config,
        stream_mode="messages",  # LLM 토큰 단위 스트리밍
    ):
        # messages 모드: (message, metadata) 2-tuple
        message, metadata = event
        if hasattr(message, "content") and message.content:
            print(message.content, end="", flush=True)

    print()  # 줄바꿈


# --- 스트리밍 방식 3: 여러 모드 동시 사용 ---

async def stream_combined():
    """업데이트 + 메시지를 동시에 스트리밍합니다."""
    print("=== 복합 스트리밍 ===\n")

    config = {"configurable": {"thread_id": "stream-003"}}
    input_state = {
        "query": "Docker 컨테이너 보안 모범사례",
        "research": None,
        "analysis": None,
        "final_answer": None,
    }

    async for mode, event in app.astream(
        input_state,
        config=config,
        stream_mode=["updates", "messages"],  # 여러 모드 동시 사용
    ):
        if mode == "updates":
            for node_name in event:
                print(f"\n--- [{node_name}] 노드 완료 ---")
        elif mode == "messages":
            message, metadata = event
            if hasattr(message, "content") and message.content:
                print(message.content, end="", flush=True)


# 실행
if __name__ == "__main__":
    asyncio.run(stream_updates())
```

---

### 2.5 동적 노드 구성

#### 설정 기반 그래프 빌드

```python
# dynamic_graph.py
"""설정에 따라 그래프 구조가 변하는 동적 노드 구성."""
from typing import TypedDict
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END


@dataclass
class GraphConfig:
    """그래프 빌드 설정.

    Feature flag처럼 각 기능을 활성화/비활성화할 수 있습니다.
    """
    enable_cache: bool = True           # 캐시 조회 활성화
    enable_validation: bool = True      # 결과 검증 활성화
    enable_notification: bool = False   # 알림 발송 활성화
    model_name: str = "claude-sonnet-4-20250514"   # 사용할 LLM 모델


class DynamicState(TypedDict):
    """동적 그래프 상태."""
    query: str
    cached_result: str | None
    llm_result: str | None
    validated: bool
    notified: bool
    final_result: str | None


def check_cache_node(state: DynamicState) -> dict:
    """캐시에서 결과를 조회합니다."""
    # 실제로는 Redis 등에서 조회
    print("[Cache] 캐시 조회 중...")
    return {"cached_result": None}  # 캐시 미스


def should_use_cache(state: DynamicState) -> str:
    """캐시 히트 여부에 따라 분기합니다."""
    if state.get("cached_result"):
        return "format_result"
    return "call_llm"


def call_llm_node(state: DynamicState) -> dict:
    """LLM을 호출합니다."""
    print("[LLM] LLM 호출 중...")
    return {"llm_result": f"LLM 분석 결과: {state['query']}"}


def validate_node(state: DynamicState) -> dict:
    """결과를 검증합니다."""
    print("[Validate] 결과 검증 중...")
    return {"validated": True}


def notify_node(state: DynamicState) -> dict:
    """알림을 발송합니다."""
    print("[Notify] 알림 발송 중...")
    return {"notified": True}


def format_result_node(state: DynamicState) -> dict:
    """최종 결과를 포맷합니다."""
    result = state.get("cached_result") or state.get("llm_result") or "결과 없음"
    return {"final_result": result}


def build_dynamic_graph(config: GraphConfig):
    """설정에 따라 다른 구조의 그래프를 빌드합니다.

    Args:
        config: 그래프 빌드 설정.

    Examples:
        >>> # 모든 기능 활성화
        >>> app = build_dynamic_graph(GraphConfig(
        ...     enable_cache=True,
        ...     enable_validation=True,
        ...     enable_notification=True,
        ... ))

        >>> # 최소 구성 (LLM만)
        >>> app = build_dynamic_graph(GraphConfig(
        ...     enable_cache=False,
        ...     enable_validation=False,
        ...     enable_notification=False,
        ... ))
    """
    graph = StateGraph(DynamicState)

    # 항상 포함되는 노드
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("format_result", format_result_node)

    if config.enable_cache:
        # 캐시 조회 활성화
        graph.add_node("check_cache", check_cache_node)
        graph.set_entry_point("check_cache")
        graph.add_conditional_edges(
            "check_cache",
            should_use_cache,
            {"call_llm": "call_llm", "format_result": "format_result"},
        )
    else:
        # 캐시 비활성화 → 바로 LLM 호출
        graph.set_entry_point("call_llm")

    # LLM 이후 경로 결정
    if config.enable_validation:
        graph.add_node("validate", validate_node)
        graph.add_edge("call_llm", "validate")
        last_node = "validate"
    else:
        last_node = "call_llm"

    if config.enable_notification:
        graph.add_node("notify", notify_node)
        graph.add_edge(last_node, "notify")
        graph.add_edge("notify", "format_result")
    else:
        graph.add_edge(last_node, "format_result")

    graph.add_edge("format_result", END)

    return graph.compile()


# --- 실행 ---

if __name__ == "__main__":
    # 설정 1: 전체 기능 활성화
    print("=== 전체 기능 활성화 ===")
    app = build_dynamic_graph(GraphConfig(
        enable_cache=True,
        enable_validation=True,
        enable_notification=True,
    ))
    result = app.invoke({
        "query": "보안 취약점 분석",
        "cached_result": None,
        "llm_result": None,
        "validated": False,
        "notified": False,
        "final_result": None,
    })
    print(f"결과: {result['final_result']}\n")

    # 설정 2: 최소 구성
    print("=== 최소 구성 ===")
    app_minimal = build_dynamic_graph(GraphConfig(
        enable_cache=False,
        enable_validation=False,
        enable_notification=False,
    ))
    result = app_minimal.invoke({
        "query": "간단한 질문",
        "cached_result": None,
        "llm_result": None,
        "validated": False,
        "notified": False,
        "final_result": None,
    })
    print(f"결과: {result['final_result']}")
```

---

### 2.6 State Annotation & Reducer 고급

#### 커스텀 Reducer 이해하기

```
Reducer란?
  여러 노드가 같은 필드에 값을 쓸 때, 값을 어떻게 합칠지 결정하는 함수.

기본 동작 (Reducer 없음):
  Node A: {"count": 1}
  Node B: {"count": 2}
  결과: count = 2  (마지막 값으로 덮어씀)

operator.add Reducer:
  Node A: {"items": ["a"]}
  Node B: {"items": ["b"]}
  결과: items = ["a", "b"]  (리스트 합침)

커스텀 Reducer:
  원하는 로직을 직접 정의 가능
```

```python
# state_reducer.py
"""State Annotation과 커스텀 Reducer 예제."""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


# --- 커스텀 Reducer 함수들 ---

def merge_dicts(existing: dict, new: dict) -> dict:
    """두 딕셔너리를 병합합니다 (new가 우선)."""
    merged = {**existing, **new}
    return merged


def max_value(existing: float, new: float) -> float:
    """두 값 중 큰 값을 유지합니다."""
    return max(existing, new)


def append_unique(existing: list, new: list) -> list:
    """중복 없이 리스트에 추가합니다."""
    seen = set(existing)
    result = list(existing)
    for item in new:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


# --- 다양한 Reducer를 사용하는 상태 정의 ---

class AdvancedState(TypedDict):
    """다양한 Reducer 패턴이 적용된 상태."""

    # 기본: 덮어쓰기 (Reducer 없음)
    current_step: str

    # 리스트 합산: operator.add
    # 여러 노드가 반환한 리스트가 자동으로 합쳐짐
    logs: Annotated[list[str], operator.add]

    # 메시지 관리: add_messages
    # LangChain 메시지 리스트를 관리 (중복 제거, ID 기반 업데이트)
    messages: Annotated[list, add_messages]

    # 커스텀 Reducer: 최대값 유지
    best_score: Annotated[float, max_value]

    # 커스텀 Reducer: 중복 없는 리스트
    discovered_tags: Annotated[list[str], append_unique]

    # 커스텀 Reducer: 딕셔너리 병합
    metadata: Annotated[dict, merge_dicts]


# --- 노드 정의 (각 노드가 Reducer를 활용) ---

def step_a_node(state: AdvancedState) -> dict:
    """Step A: 초기 처리."""
    return {
        "current_step": "step_a",
        "logs": ["Step A 시작", "데이터 로드 완료"],  # operator.add
        "messages": [HumanMessage(content="분석을 시작합니다.")],
        "best_score": 0.7,
        "discovered_tags": ["python", "async"],
        "metadata": {"source": "user_input", "version": "1.0"},
    }


def step_b_node(state: AdvancedState) -> dict:
    """Step B: 분석 처리."""
    return {
        "current_step": "step_b",
        "logs": ["Step B: LLM 호출", "분석 완료"],  # [A로그들] + [B로그들]
        "messages": [AIMessage(content="분석 결과가 준비되었습니다.")],
        "best_score": 0.85,           # max(0.7, 0.85) = 0.85
        "discovered_tags": ["async", "langgraph", "agent"],  # 중복 "async" 제거
        "metadata": {"analyzed_at": "2026-03-23", "model": "claude"},
    }


def step_c_node(state: AdvancedState) -> dict:
    """Step C: 결과 정리."""
    # 현재 상태 확인 (Reducer가 적용된 결과)
    print(f"  logs: {state['logs']}")       # Step A + B 로그 모두 포함
    print(f"  score: {state['best_score']}") # 0.85 (최대값)
    print(f"  tags: {state['discovered_tags']}")  # 중복 제거됨
    print(f"  metadata: {state['metadata']}")     # 병합됨

    return {
        "current_step": "step_c",
        "logs": ["Step C: 최종 정리 완료"],
        "best_score": 0.6,             # max(0.85, 0.6) = 0.85 (기존 유지)
        "discovered_tags": ["production"],
        "metadata": {"finalized": True},
    }


# --- 그래프 구성 ---

graph = StateGraph(AdvancedState)
graph.add_node("step_a", step_a_node)
graph.add_node("step_b", step_b_node)
graph.add_node("step_c", step_c_node)

graph.set_entry_point("step_a")
graph.add_edge("step_a", "step_b")
graph.add_edge("step_b", "step_c")
graph.add_edge("step_c", END)

app = graph.compile()

# --- 실행 ---

if __name__ == "__main__":
    result = app.invoke({
        "current_step": "init",
        "logs": [],
        "messages": [],
        "best_score": 0.0,
        "discovered_tags": [],
        "metadata": {},
    })

    print(f"\n최종 상태:")
    print(f"  logs ({len(result['logs'])}개): {result['logs']}")
    print(f"  best_score: {result['best_score']}")
    print(f"  discovered_tags: {result['discovered_tags']}")
    print(f"  metadata: {result['metadata']}")
    print(f"  messages: {len(result['messages'])}개")
```

---

### 2.7 그래프 시각화와 디버깅

#### Mermaid 다이어그램 생성

```python
# visualization.py
"""LangGraph 그래프 시각화."""
from typing import TypedDict
from langgraph.graph import StateGraph, END


class VisualizationState(TypedDict):
    data: str
    result: str | None


def node_a(state): return {"result": "A 완료"}
def node_b(state): return {"result": "B 완료"}
def node_c(state): return {"result": "C 완료"}

def route(state):
    if state.get("data", "").startswith("urgent"):
        return "fast_path"
    return "normal_path"


graph = StateGraph(VisualizationState)
graph.add_node("intake", node_a)
graph.add_node("normal_process", node_b)
graph.add_node("fast_process", node_c)
graph.add_node("output", node_a)

graph.set_entry_point("intake")
graph.add_conditional_edges(
    "intake",
    route,
    {"normal_path": "normal_process", "fast_path": "fast_process"},
)
graph.add_edge("normal_process", "output")
graph.add_edge("fast_process", "output")
graph.add_edge("output", END)

app = graph.compile()

# --- Mermaid 다이어그램 출력 ---

# 방법 1: Mermaid 텍스트 출력
mermaid_text = app.get_graph().draw_mermaid()
print("=== Mermaid 다이어그램 ===")
print(mermaid_text)

# 방법 2: PNG 이미지로 저장 (graphviz 필요)
# pip install pygraphviz
# app.get_graph().draw_mermaid_png(output_file_path="graph.png")

# 방법 3: Mermaid Live Editor에서 시각화
# https://mermaid.live/ 에 위 텍스트를 붙여넣으면 그래프를 볼 수 있습니다
```

#### 실행 추적 (디버깅)

```python
# debugging.py
"""LangGraph 실행 추적과 디버깅."""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class DebugState(TypedDict):
    input: str
    step1: str | None
    step2: str | None
    output: str | None


def step1_node(state):
    return {"step1": f"처리됨: {state['input']}"}

def step2_node(state):
    return {"step2": f"분석됨: {state['step1']}"}

def output_node(state):
    return {"output": f"완료: {state['step2']}"}


graph = StateGraph(DebugState)
graph.add_node("step1", step1_node)
graph.add_node("step2", step2_node)
graph.add_node("output", output_node)
graph.set_entry_point("step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", "output")
graph.add_edge("output", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


async def debug_execution():
    """실행 과정을 단계별로 추적합니다."""
    config = {"configurable": {"thread_id": "debug-001"}}

    print("=== 실행 추적 (values 모드) ===\n")

    # stream_mode="values": 각 스텝 후 전체 상태 출력
    async for state_snapshot in app.astream(
        {
            "input": "테스트 데이터",
            "step1": None,
            "step2": None,
            "output": None,
        },
        config=config,
        stream_mode="values",
    ):
        # 어떤 필드가 변경되었는지 확인
        non_none = {k: v for k, v in state_snapshot.items() if v is not None}
        print(f"  상태: {non_none}")

    print("\n=== 체크포인트 히스토리 ===\n")

    # 실행 후 체크포인트 히스토리 조회
    for i, state in enumerate(app.get_state_history(config)):
        step = state.values.get("output") or state.values.get("step2") or state.values.get("step1") or "시작"
        next_nodes = state.next
        print(f"  체크포인트 #{i}: 상태='{step}', 다음 노드={next_nodes}")


if __name__ == "__main__":
    asyncio.run(debug_execution())
```

#### LangSmith 연동 (프로덕션 모니터링)

```python
# langsmith_setup.py
"""LangSmith 연동으로 실행 추적을 프로덕션 수준으로 관리합니다.

사전 준비:
1. https://smith.langchain.com/ 에서 계정 생성
2. API 키 발급
3. 환경변수 설정
"""
import os

# LangSmith 연동을 위한 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key-here"  # 실제 키로 교체
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# 이후 LangGraph 그래프를 실행하면 자동으로 LangSmith에 추적 데이터가 전송됩니다.
# LangSmith 대시보드에서 다음을 확인할 수 있습니다:
#   - 각 노드의 실행 시간
#   - LLM 호출 입출력
#   - 에러 발생 지점
#   - 토큰 사용량
#   - 비용 추적
```

---

## 3. 실전 예제: 멀티 문서 분석 에이전트

체크포인팅 + 서브그래프 + 병렬 실행을 결합한 실전 에이전트입니다.

```python
# multi_doc_agent.py
"""실전 예제: 체크포인팅 + 서브그래프 + 병렬 실행이 결합된 멀티 문서 분석 에이전트.

아키텍처:
  [메인 그래프]
  load_documents → [fan-out: 문서별 분석 서브그래프] → [fan-in] → synthesize → END

  [분석 서브그래프] (재사용 가능)
  extract_keywords → summarize → classify

  핵심 패턴:
  - 체크포인팅: 장애 시 마지막 성공 지점에서 재개
  - 서브그래프: 분석 로직을 모듈로 분리하여 재사용
  - 병렬 실행: 여러 문서를 동시에 분석
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver


# ============================
# 서브그래프: 단일 문서 분석
# ============================

class DocAnalysisSubState(TypedDict):
    """문서 분석 서브그래프 상태."""
    doc_id: str
    content: str
    keywords: list[str] | None
    summary: str | None
    category: str | None


def extract_keywords_sub(state: DocAnalysisSubState) -> dict:
    """키워드를 추출합니다."""
    content = state["content"]
    words = content.split()
    keywords = sorted(set(w for w in words if len(w) > 2), key=len, reverse=True)[:5]
    print(f"    [{state['doc_id']}] 키워드: {keywords}")
    return {"keywords": keywords}


def summarize_sub(state: DocAnalysisSubState) -> dict:
    """문서를 요약합니다."""
    content = state["content"]
    summary = content[:80] + "..." if len(content) > 80 else content
    print(f"    [{state['doc_id']}] 요약 완료")
    return {"summary": summary}


def classify_sub(state: DocAnalysisSubState) -> dict:
    """문서를 분류합니다."""
    content = state["content"].lower()
    if any(w in content for w in ["error", "bug", "오류", "에러"]):
        category = "bug_report"
    elif any(w in content for w in ["feature", "기능", "추가", "개선"]):
        category = "feature_request"
    else:
        category = "general"
    print(f"    [{state['doc_id']}] 카테고리: {category}")
    return {"category": category}


def build_doc_analysis_subgraph():
    """단일 문서 분석 서브그래프."""
    graph = StateGraph(DocAnalysisSubState)
    graph.add_node("extract_keywords", extract_keywords_sub)
    graph.add_node("summarize", summarize_sub)
    graph.add_node("classify", classify_sub)

    graph.set_entry_point("extract_keywords")
    graph.add_edge("extract_keywords", "summarize")
    graph.add_edge("summarize", "classify")
    graph.add_edge("classify", END)

    return graph.compile()


# ============================
# 메인 그래프: 멀티 문서 처리
# ============================

class AnalysisResult(TypedDict):
    """개별 분석 결과."""
    doc_id: str
    keywords: list[str]
    summary: str
    category: str


class MultiDocState(TypedDict):
    """메인 그래프 상태."""
    documents: list[dict]
    # 서브그래프 입출력 매핑
    doc_id: str
    content: str
    keywords: list[str] | None
    summary: str | None
    category: str | None
    # 병렬 결과 수집
    analysis_results: Annotated[list[AnalysisResult], operator.add]
    # 최종 결과
    synthesis: str | None


def load_documents_node(state: MultiDocState) -> dict:
    """문서를 로드합니다."""
    docs = state.get("documents", [])
    print(f"\n[Load] {len(docs)}개 문서 로드 완료")
    return {}


def fan_out_for_analysis(state: MultiDocState) -> list[Send]:
    """각 문서에 대해 분석 서브그래프를 병렬 실행합니다."""
    documents = state.get("documents", [])

    if not documents:
        return [Send("synthesize", state)]

    sends = []
    for doc in documents:
        send_state = {
            **state,
            "doc_id": doc["id"],
            "content": doc["content"],
            "keywords": None,
            "summary": None,
            "category": None,
            "analysis_results": [],
        }
        sends.append(Send("analyze_single_doc", send_state))

    print(f"[Fan-out] {len(sends)}개 문서를 병렬 분석합니다.")
    return sends


def analyze_single_doc_wrapper(state: MultiDocState) -> dict:
    """서브그래프를 호출하여 단일 문서를 분석합니다.

    서브그래프 결과를 analysis_results에 추가합니다.
    """
    subgraph = build_doc_analysis_subgraph()

    result = subgraph.invoke({
        "doc_id": state["doc_id"],
        "content": state["content"],
        "keywords": None,
        "summary": None,
        "category": None,
    })

    return {
        "analysis_results": [
            AnalysisResult(
                doc_id=result["doc_id"],
                keywords=result.get("keywords", []),
                summary=result.get("summary", ""),
                category=result.get("category", "general"),
            )
        ],
    }


def synthesize_node(state: MultiDocState) -> dict:
    """모든 분석 결과를 종합합니다."""
    results = state.get("analysis_results", [])

    lines = [
        "=" * 50,
        "멀티 문서 분석 종합 리포트",
        "=" * 50,
        f"분석 문서 수: {len(results)}",
        "",
    ]

    # 카테고리별 분류
    categories = {}
    for r in results:
        cat = r.get("category", "general")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["doc_id"])

    lines.append("카테고리별 분류:")
    for cat, doc_ids in categories.items():
        lines.append(f"  {cat}: {', '.join(doc_ids)}")

    lines.append("")
    lines.append("개별 분석 결과:")
    for r in results:
        lines.append(f"  [{r['doc_id']}]")
        lines.append(f"    키워드: {', '.join(r.get('keywords', []))}")
        lines.append(f"    요약: {r.get('summary', 'N/A')}")
        lines.append(f"    카테고리: {r.get('category', 'N/A')}")

    synthesis = "\n".join(lines)
    print(f"\n[Synthesize] 종합 리포트 생성 완료")
    return {"synthesis": synthesis}


# --- 메인 그래프 구성 ---

def build_multi_doc_graph():
    """체크포인팅 + 서브그래프 + 병렬 실행 결합 그래프."""
    graph = StateGraph(MultiDocState)

    graph.add_node("load_documents", load_documents_node)
    graph.add_node("analyze_single_doc", analyze_single_doc_wrapper)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("load_documents")

    # Fan-out: 문서별 병렬 분석
    graph.add_conditional_edges(
        "load_documents",
        fan_out_for_analysis,
        ["analyze_single_doc", "synthesize"],
    )

    # Fan-in: 모든 분석 완료 후 종합
    graph.add_edge("analyze_single_doc", "synthesize")
    graph.add_edge("synthesize", END)

    # 체크포인팅 적용
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# --- 실행 ---

if __name__ == "__main__":
    app = build_multi_doc_graph()

    documents = [
        {
            "id": "BUG-001",
            "content": "로그인 시 NullPointerException 에러가 발생합니다. user 객체가 null인 경우 처리가 누락되었습니다.",
        },
        {
            "id": "FEAT-002",
            "content": "대시보드에 실시간 모니터링 기능을 추가해주세요. 현재 새로고침해야 최신 데이터를 볼 수 있습니다.",
        },
        {
            "id": "DOC-003",
            "content": "API 엔드포인트 문서를 업데이트합니다. 새로운 인증 헤더 형식에 대한 설명을 추가합니다.",
        },
    ]

    config = {"configurable": {"thread_id": "multi-doc-analysis-001"}}

    result = app.invoke(
        {
            "documents": documents,
            "doc_id": "",
            "content": "",
            "keywords": None,
            "summary": None,
            "category": None,
            "analysis_results": [],
            "synthesis": None,
        },
        config=config,
    )

    print(f"\n{result['synthesis']}")
```

---

## 4. 연습 문제

### 연습 1: 체크포인팅 추가하기 (난이도: 하)

**과제**: 아래 그래프에 SqliteSaver 체크포인팅을 추가하고, 장애 복구를 테스트하세요.

```python
# exercise_checkpoint.py
from typing import TypedDict
from langgraph.graph import StateGraph, END


class PipelineState(TypedDict):
    data: str
    step1_done: bool
    step2_done: bool
    result: str | None


def step1(state):
    print("Step 1 실행")
    return {"step1_done": True}


def step2(state):
    print("Step 2 실행")
    # TODO: 여기서 일부러 에러를 발생시켜 테스트
    return {"step2_done": True}


def step3(state):
    print("Step 3 실행")
    return {"result": "완료!"}


graph = StateGraph(PipelineState)
graph.add_node("step1", step1)
graph.add_node("step2", step2)
graph.add_node("step3", step3)
graph.set_entry_point("step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)

# TODO: 체크포인터를 추가하세요
app = graph.compile()
```

**요구사항**:
1. SqliteSaver를 사용하여 체크포인팅을 추가하세요
2. thread_id를 설정하여 실행하세요
3. step2에서 의도적으로 에러를 발생시키세요
4. 동일한 thread_id로 재실행하여 step1을 건너뛰는지 확인하세요

---

### 연습 2: 서브그래프 + 병렬 실행 결합 (난이도: 상)

**과제**: 다음 요구사항을 구현하세요:

1. "번역 서브그래프"를 만드세요 (원문 → 번역 → 검수)
2. 메인 그래프에서 3개 문서를 병렬로 번역 서브그래프에 보내세요
3. 병렬 번역 결과를 모아서 최종 리포트를 생성하세요
4. 체크포인팅을 적용하세요

**힌트**:
- Send API + Annotated[list, operator.add] 사용
- 서브그래프는 `graph.compile()`로 컴파일 후 메인 그래프의 노드에서 `invoke()` 호출
- Section 3의 실전 예제 패턴을 참고하세요

---

## 5. 핵심 정리

### 패턴별 요약 표

| 패턴 | 핵심 API | 주요 용도 | 난이도 |
|------|---------|----------|--------|
| 체크포인팅 | `compile(checkpointer=saver)` | 장애 복구, 실행 이력 | 하 |
| 서브그래프 | `add_node("name", sub.compile())` | 모듈화, 재사용 | 중 |
| 병렬 실행 | `Send API + operator.add` | 동시 처리, 성능 향상 | 상 |
| 스트리밍 | `astream(stream_mode=...)` | 실시간 상태 전달 | 중 |
| 동적 구성 | 조건부 `add_node/add_edge` | 유연한 그래프 빌드 | 중 |
| Reducer | `Annotated[type, reducer_fn]` | 상태 합산/병합 | 중 |
| 시각화 | `draw_mermaid()` | 디버깅, 문서화 | 하 |

### 체크포인터 선택 가이드

```
개발/테스트 → MemorySaver (빠르고 간단)
로컬 개발   → SqliteSaver (파일 기반 영속성)
프로덕션    → PostgresSaver (공유 DB, 다중 인스턴스)
```

### 기억해야 할 핵심 원칙

| 원칙 | 설명 |
|------|------|
| thread_id 설계 | 작업 유형 + 고유 식별자 조합으로 의미 있는 ID 부여 |
| 서브그래프 상태 격리 | 필요한 필드만 포함, 메인 상태와 이름 기반 매핑 |
| Reducer 필수 | 병렬 실행 시 같은 필드에 쓰면 반드시 Reducer 지정 |
| 점진적 도입 | 체크포인팅 먼저, 그 다음 서브그래프, 마지막에 병렬 실행 |

---

## 6. 참고 자료

| 주제 | 링크 |
|------|------|
| LangGraph Persistence (체크포인팅) | https://langchain-ai.github.io/langgraph/concepts/persistence/ |
| LangGraph Subgraphs (서브그래프) | https://langchain-ai.github.io/langgraph/how-tos/subgraph/ |
| LangGraph Send API (병렬 실행) | https://langchain-ai.github.io/langgraph/how-tos/map-reduce/ |
| LangGraph Streaming (스트리밍) | https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/ |
| LangSmith (모니터링) | https://docs.smith.langchain.com/ |
| LangGraph State Reducers | https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers |
| LangGraph Visualization | https://langchain-ai.github.io/langgraph/how-tos/visualization/ |

---

## 다음 단계

축하합니다! LangGraph의 프로덕션 수준 고급 패턴을 모두 학습했습니다.

**이제 할 수 있는 것들:**
- 장애에 강한 에이전트 (체크포인팅)
- 모듈화된 에이전트 아키텍처 (서브그래프)
- 빠른 병렬 처리 (Send API)
- 실시간 사용자 경험 (스트리밍)
- 유연한 설정 기반 동작 (동적 구성)

**다음 학습 추천:**
- Module 11의 품질 보장 패턴과 결합하여 프로덕션 수준의 에이전트 완성
- LangSmith를 연동하여 실제 모니터링 환경 구축
- Docker + PostgresSaver로 배포 가능한 에이전트 환경 구성

**복습 체크리스트:**
- [ ] MemorySaver / SqliteSaver / PostgresSaver의 차이를 설명할 수 있다
- [ ] thread_id를 활용하여 실행을 추적하고 장애 복구를 할 수 있다
- [ ] 서브그래프를 정의하고 메인 그래프에서 호출할 수 있다
- [ ] Send API + Reducer로 병렬 실행을 구현할 수 있다
- [ ] astream()으로 실시간 스트리밍을 구현할 수 있다
- [ ] 설정에 따라 동적으로 그래프 구조를 변경할 수 있다
- [ ] draw_mermaid()로 그래프를 시각화할 수 있다
