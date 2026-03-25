# Module 11: LLM 출력 품질 보장 - 신뢰할 수 있는 에이전트 만들기

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. LLM 출력 검증이 왜 필수인지 이해한다
2. 6단계 품질 보장 레이어 구조를 설명할 수 있다
3. Confidence 게이팅으로 낮은 신뢰도 결과를 필터링할 수 있다
4. Self-Reflection 루프를 LangGraph로 구현할 수 있다
5. Golden Set(평가 데이터셋)을 구축하고 자동 평가를 실행할 수 있다
6. Human-in-the-Loop 패턴을 LangGraph에서 구현할 수 있다

---

## 사전 지식

| 항목 | 필요 수준 | 참고 모듈 |
|------|----------|----------|
| Python 기초 | 클래스, 함수, 딕셔너리 | - |
| Pydantic | BaseModel, Field, validator | Module 06 |
| LangGraph 기초 | StateGraph, 노드, 엣지 | Module 09 |
| LLM API 호출 | ChatAnthropic 또는 ChatOpenAI | Module 03 |

---

## 1. 개념 설명

### 1.1 LLM 출력을 왜 검증해야 하는가?

LLM은 강력하지만, 출력을 **항상 신뢰할 수는 없습니다**. 다음과 같은 문제가 발생할 수 있습니다:

```
[사용자 요청]                    [LLM 응답]
"JSON으로 분석 결과를      →    { "분석": "문제없음"  ← 필수 필드 누락!
 반환해주세요.                    }
 summary, confidence,
 affected_files 포함"
```

**주요 문제 유형:**

| 문제 | 설명 | 예시 |
|------|------|------|
| 환각 (Hallucination) | 존재하지 않는 정보를 만들어냄 | 없는 함수명을 참조한 코드 생성 |
| JSON 구조 오류 | 파싱 불가능한 형식 반환 | 닫는 괄호 누락, 잘못된 콤마 |
| 필수 필드 누락 | 요청한 필드를 빠뜨림 | confidence 필드 없이 결과 반환 |
| 타입 불일치 | 기대와 다른 데이터 타입 | 숫자 대신 문자열 "0.85" 반환 |
| 낮은 신뢰도 | LLM 스스로 확신이 없는 결과 | confidence: 0.2인 분석 결과를 그대로 사용 |

**검증 없이 LLM 출력을 사용하면?**

```
LLM 출력 (검증 없음) → 잘못된 데이터 → 후속 처리 실패 → 시스템 전체 오류
                        ↓
                    사용자에게 잘못된 정보 전달
```

### 1.2 품질 보장 레이어 6단계 (피라미드 구조)

품질 보장은 한 번에 하나씩, **레이어를 쌓아올리는 방식**으로 구현합니다:

```
                    ┌─────────────────────┐
                    │  Layer 6: 정량 평가   │  ← Golden Set + 메트릭
                    │  (Quantitative)      │
                  ┌─┴─────────────────────┴─┐
                  │  Layer 5: 인간 검증       │  ← Human-in-the-Loop
                  │  (Human Review)          │
                ┌─┴─────────────────────────┴─┐
                │  Layer 4: 외부 검증           │  ← AST / 빌드 / 린트
                │  (External Validation)       │
              ┌─┴───────────────────────────────┴─┐
              │  Layer 3: 자기 검증                 │  ← Self-Reflection
              │  (Self-Reflection)                 │
            ┌─┴─────────────────────────────────────┴─┐
            │  Layer 2: 의미적 게이팅                    │  ← Confidence 임계값
            │  (Semantic Gating)                       │
          ┌─┴───────────────────────────────────────────┴─┐
          │  Layer 1: 스키마 유효성                          │  ← Pydantic 검증
          │  (Schema Validation)                           │
        ┌─┴─────────────────────────────────────────────────┴─┐
        │  Layer 0: JSON 파싱                                  │  ← 기본 파싱
        │  (JSON Parsing)                                      │
        └───────────────────────────────────────────────────────┘
```

**각 레이어의 역할:**

| 레이어 | 검증 대상 | 질문 | 도구 |
|--------|----------|------|------|
| Layer 0 | 구조 | "유효한 JSON인가?" | json.loads() |
| Layer 1 | 스키마 | "필수 필드와 타입이 맞는가?" | Pydantic |
| Layer 2 | 신뢰도 | "LLM이 자신 있는 결과인가?" | Confidence 임계값 |
| Layer 3 | 논리 | "논리적으로 올바른가?" | LLM Self-Reflection |
| Layer 4 | 실행 가능성 | "실제로 동작하는가?" | AST, 빌드, 린트 |
| Layer 5 | 최종 승인 | "사람이 봐도 괜찮은가?" | Human-in-the-Loop |
| Layer 6 | 정량 품질 | "기대 수준을 충족하는가?" | Golden Set 평가 |

> **팁**: 모든 레이어를 한 번에 구현할 필요는 없습니다. Layer 0 → 1 → 2 순서로 점진적으로 추가하세요.

---

## 2. 단계별 실습

### 2.1 Layer 0 + Layer 1: JSON 파싱 + 스키마 검증

**목표**: LLM 출력을 JSON으로 파싱하고, Pydantic으로 구조를 검증합니다.

**Step 1: 출력 스키마 정의**

```python
# schemas.py
"""LLM 출력 검증용 Pydantic 스키마 정의."""
from pydantic import BaseModel, Field, field_validator


class AnalysisResult(BaseModel):
    """LLM 분석 결과 스키마.

    LLM이 반환해야 하는 필수 필드와 타입을 정의합니다.
    """
    summary: str = Field(
        min_length=1,
        description="분석 요약"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="분석 신뢰도 (0.0~1.0)"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="발견된 문제 목록"
    )
    recommendation: str = Field(
        min_length=1,
        description="권장 조치사항"
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """confidence를 소수점 2자리로 반올림합니다."""
        return round(v, 2)
```

**Step 2: JSON 파서 + 스키마 검증 결합**

```python
# validated_parser.py
"""JSON 파싱 + Pydantic 스키마 검증을 결합한 파서."""
import json
import re
import logging
from typing import Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def parse_json_from_llm(text: str) -> dict:
    """LLM 출력에서 JSON을 추출합니다.

    3단계 폴백 전략:
    1. 전체 텍스트를 JSON으로 파싱
    2. ```json ... ``` 마크다운 블록에서 추출
    3. 첫 번째 { ~ 마지막 } 사이를 추출
    """
    # 1단계: 직접 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2단계: 마크다운 JSON 블록 추출
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3단계: 중괄호 범위 추출
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"JSON 파싱 실패: {text[:200]}...")


def validate_with_schema(
    raw_dict: dict,
    schema_class: Type[BaseModel],
    strict: bool = False,
) -> dict:
    """파싱된 dict를 Pydantic 스키마로 검증합니다.

    Args:
        raw_dict: JSON 파싱 결과 딕셔너리.
        schema_class: 검증에 사용할 Pydantic 모델 클래스.
        strict: True이면 검증 실패 시 예외 발생,
                False이면 복구 시도 후 실패하면 원본 반환.

    Returns:
        검증된 딕셔너리.
    """
    try:
        validated = schema_class.model_validate(raw_dict)
        logger.info(
            "스키마 검증 통과: %s",
            schema_class.__name__,
        )
        return validated.model_dump()

    except ValidationError as exc:
        logger.warning(
            "스키마 검증 실패: %s, 오류: %s",
            schema_class.__name__,
            exc.errors()[:3],  # 처음 3개 오류만 로깅
        )

        if strict:
            raise

        # 느슨한 모드: 기본값으로 복구 시도
        try:
            validated = schema_class.model_validate(raw_dict, strict=False)
            logger.info("기본값으로 복구 성공")
            return validated.model_dump()
        except ValidationError:
            logger.error("복구 실패 — 원본 dict 반환")
            return raw_dict


def parse_and_validate(
    text: str,
    schema_class: Type[BaseModel],
    strict: bool = False,
) -> dict:
    """LLM 출력 텍스트를 파싱하고 스키마로 검증합니다.

    JSON 파싱(Layer 0)과 스키마 검증(Layer 1)을 한 번에 수행합니다.
    """
    raw = parse_json_from_llm(text)
    return validate_with_schema(raw, schema_class, strict=strict)
```

**Step 3: 테스트**

```python
# test_validated_parser.py
"""파서 + 스키마 검증 테스트."""
from schemas import AnalysisResult
from validated_parser import parse_and_validate


def test_valid_json():
    """정상적인 JSON이 올바르게 파싱되고 검증되는지 테스트."""
    llm_output = '''```json
    {
        "summary": "로그인 API에 인증 누락 취약점이 발견되었습니다.",
        "confidence": 0.85,
        "issues": ["토큰 검증 미수행", "rate limit 미적용"],
        "recommendation": "JWT 검증 미들웨어를 추가하세요."
    }
    ```'''

    result = parse_and_validate(llm_output, AnalysisResult)

    assert result["summary"] == "로그인 API에 인증 누락 취약점이 발견되었습니다."
    assert result["confidence"] == 0.85
    assert len(result["issues"]) == 2
    print("테스트 통과: 정상 JSON 파싱 + 스키마 검증 성공")


def test_missing_field_recovery():
    """필수 필드가 누락된 경우 기본값으로 복구되는지 테스트."""
    llm_output = '{"summary": "분석 완료", "confidence": 0.9, "recommendation": "조치 필요"}'

    # issues 필드 누락 → default_factory=list로 복구
    result = parse_and_validate(llm_output, AnalysisResult, strict=False)

    assert result["issues"] == []
    print("테스트 통과: 누락 필드 기본값 복구 성공")


def test_invalid_confidence_range():
    """confidence가 범위를 벗어나면 검증이 실패하는지 테스트."""
    llm_output = '{"summary": "분석", "confidence": 1.5, "recommendation": "조치"}'

    try:
        parse_and_validate(llm_output, AnalysisResult, strict=True)
        assert False, "ValidationError가 발생해야 합니다"
    except Exception:
        print("테스트 통과: 범위 초과 confidence 거부 성공")


if __name__ == "__main__":
    test_valid_json()
    test_missing_field_recovery()
    test_invalid_confidence_range()
    print("\n모든 테스트 통과!")
```

---

### 2.2 Layer 2: Confidence 게이팅

**목표**: LLM이 반환하는 신뢰도(confidence) 값으로 결과를 필터링합니다.

**핵심 개념: 임계값별 동작 분기**

```
confidence 값에 따른 동작 분기:

  0.0          0.4          0.7          1.0
   |───REJECT───|───RETRY────|───ACCEPT───|
   │            │            │            │
   │  "결과를    │  "다시     │  "결과를    │
   │   거부"     │   시도"    │   수락"     │
```

| 범위 | 동작 | 설명 |
|------|------|------|
| 0.7 이상 | ACCEPT | 결과를 그대로 사용 |
| 0.4 ~ 0.7 미만 | RETRY | LLM에게 다시 요청 (최대 2회) |
| 0.4 미만 | REJECT | 결과를 거부하고 에러 처리 |

**구현:**

```python
# confidence_gate.py
"""Confidence 기반 의사결정 게이팅."""
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GateAction(str, Enum):
    """게이팅 결과 동작."""
    ACCEPT = "accept"   # 결과 수락
    RETRY = "retry"     # 재시도 권장
    REJECT = "reject"   # 결과 거부


@dataclass
class GateResult:
    """게이팅 판단 결과."""
    action: GateAction
    confidence: float
    threshold_accept: float
    threshold_reject: float
    message: str


def confidence_gate(
    confidence: float,
    gate_name: str = "default",
    threshold_accept: float = 0.7,
    threshold_reject: float = 0.4,
) -> GateResult:
    """Confidence 값에 따라 ACCEPT / RETRY / REJECT를 결정합니다.

    Args:
        confidence: LLM이 반환한 신뢰도 (0.0 ~ 1.0).
        gate_name: 게이트 식별자 (로깅용).
        threshold_accept: 이 값 이상이면 ACCEPT.
        threshold_reject: 이 값 미만이면 REJECT. 그 사이는 RETRY.

    Returns:
        GateResult: 동작, 신뢰도, 임계값, 설명 메시지.

    Examples:
        >>> result = confidence_gate(0.85, "analysis")
        >>> result.action
        GateAction.ACCEPT

        >>> result = confidence_gate(0.5, "analysis")
        >>> result.action
        GateAction.RETRY
    """
    if confidence >= threshold_accept:
        action = GateAction.ACCEPT
        message = f"Confidence {confidence:.2f} >= {threshold_accept} → 수락"
    elif confidence >= threshold_reject:
        action = GateAction.RETRY
        message = f"Confidence {confidence:.2f} — 재시도 권장"
    else:
        action = GateAction.REJECT
        message = f"Confidence {confidence:.2f} < {threshold_reject} → 거부"

    logger.info("[%s] %s", gate_name, message)

    return GateResult(
        action=action,
        confidence=confidence,
        threshold_accept=threshold_accept,
        threshold_reject=threshold_reject,
        message=message,
    )
```

**LangGraph 조건부 엣지에서 사용:**

```python
# graph_with_confidence.py
"""Confidence 게이팅이 적용된 LangGraph 에이전트."""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

from confidence_gate import confidence_gate, GateAction
from schemas import AnalysisResult
from validated_parser import parse_and_validate


class AgentState(TypedDict):
    """에이전트 상태."""
    query: str                  # 사용자 입력
    analysis: dict | None       # LLM 분석 결과
    retry_count: int            # 재시도 횟수
    max_retries: int            # 최대 재시도 횟수
    error: str | None           # 에러 메시지
    final_result: str | None    # 최종 결과


# --- 노드 정의 ---

def analyze_node(state: AgentState) -> dict:
    """LLM으로 분석을 수행하는 노드."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    prompt = f"""다음 질문을 분석하고 JSON으로 응답하세요.

질문: {state["query"]}

반드시 아래 형식으로 응답하세요:
{{
    "summary": "분석 요약",
    "confidence": 0.0~1.0 사이 값,
    "issues": ["발견된 문제 목록"],
    "recommendation": "권장 조치사항"
}}"""

    response = llm.invoke(prompt)
    result = parse_and_validate(response.content, AnalysisResult)

    return {
        "analysis": result,
        "retry_count": state.get("retry_count", 0),
    }


def decide_after_analysis(state: AgentState) -> str:
    """Confidence 값에 따라 다음 경로를 결정합니다."""
    analysis = state.get("analysis")
    if analysis is None:
        return "handle_error"

    confidence = analysis.get("confidence", 0.0)
    gate = confidence_gate(confidence, "analysis")

    if gate.action == GateAction.ACCEPT:
        return "format_result"
    elif gate.action == GateAction.RETRY:
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        if retry_count < max_retries:
            return "analyze"  # 재시도
        return "format_result"  # 재시도 소진 → 그대로 사용
    else:  # REJECT
        return "handle_error"


def format_result_node(state: AgentState) -> dict:
    """분석 결과를 최종 형태로 포맷합니다."""
    analysis = state["analysis"]
    return {
        "final_result": (
            f"분석 요약: {analysis['summary']}\n"
            f"신뢰도: {analysis['confidence']}\n"
            f"권장사항: {analysis['recommendation']}"
        ),
    }


def handle_error_node(state: AgentState) -> dict:
    """에러를 처리합니다."""
    return {
        "error": "분석 결과의 신뢰도가 너무 낮아 사용할 수 없습니다.",
        "final_result": None,
    }


# --- 그래프 구성 ---

def build_graph():
    """Confidence 게이팅이 적용된 분석 그래프를 구성합니다."""
    graph = StateGraph(AgentState)

    graph.add_node("analyze", analyze_node)
    graph.add_node("format_result", format_result_node)
    graph.add_node("handle_error", handle_error_node)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges(
        "analyze",
        decide_after_analysis,
        {
            "analyze": "analyze",           # RETRY → 재시도
            "format_result": "format_result", # ACCEPT → 결과 포맷
            "handle_error": "handle_error",   # REJECT → 에러 처리
        },
    )
    graph.add_edge("format_result", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


# 실행
if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "query": "Python 로그인 API의 보안 취약점을 분석해주세요.",
        "analysis": None,
        "retry_count": 0,
        "max_retries": 2,
        "error": None,
        "final_result": None,
    })
    print(result["final_result"] or result["error"])
```

---

### 2.3 Layer 3: Self-Reflection 루프

**목표**: LLM이 자신의 출력을 스스로 검토하여 품질을 높입니다.

**핵심 개념:**

```
  ┌──────────────────────────────────────────┐
  │              Self-Reflection 루프          │
  │                                          │
  │   [generate] → [reflect] → [decision]    │
  │       ↑            │           │         │
  │       │        미승인 &        │         │
  │       │      count < max       승인       │
  │       └────────────┘           │         │
  │                                ↓         │
  │                           [다음 단계]     │
  └──────────────────────────────────────────┘

  * 최대 반복 횟수를 반드시 제한! (무한 루프 방지)
```

**전체 구현:**

```python
# self_reflection_agent.py
"""Self-Reflection 루프가 적용된 에이전트."""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

from validated_parser import parse_json_from_llm

# 최대 반복 횟수 (무한 루프 방지)
MAX_REFLECTION_ROUNDS = 2


class ReflectionState(TypedDict):
    """Self-Reflection 에이전트 상태."""
    topic: str                      # 작성 주제
    draft: str                      # 현재 초안
    reflection: dict | None         # 검토 결과
    reflection_count: int           # 검토 횟수
    final_output: str | None        # 최종 결과


# --- 노드 정의 ---

def generate_node(state: ReflectionState) -> dict:
    """초안을 생성하거나, 피드백을 반영하여 수정합니다."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    reflection = state.get("reflection")

    if reflection and not reflection.get("approved", True):
        # 이전 검토에서 수정 요청이 있는 경우
        feedback = reflection.get("feedback", "")
        prompt = f"""이전 초안을 아래 피드백을 반영하여 수정해주세요.

이전 초안:
{state["draft"]}

피드백:
{feedback}

수정된 내용만 반환하세요."""
    else:
        # 최초 생성
        prompt = f"""다음 주제에 대해 간결하고 정확한 기술 문서를 작성해주세요.

주제: {state["topic"]}

핵심 내용을 포함하여 작성하세요."""

    response = llm.invoke(prompt)
    return {"draft": response.content}


def reflect_node(state: ReflectionState) -> dict:
    """생성된 초안을 비판적으로 검토합니다."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    prompt = f"""당신은 엄격한 기술 리뷰어입니다. 아래 문서를 검토하세요.

문서:
{state["draft"]}

검토 관점:
1. 기술적으로 정확한가?
2. 중요한 내용이 누락되지 않았는가?
3. 설명이 명확한가?

반드시 JSON으로 응답하세요:
{{
    "approved": true 또는 false,
    "score": 1~10 점수,
    "feedback": "구체적인 피드백 (승인 시 빈 문자열)"
}}"""

    response = llm.invoke(prompt)
    result = parse_json_from_llm(response.content)

    return {
        "reflection": result,
        "reflection_count": state.get("reflection_count", 0) + 1,
    }


def decide_after_reflection(state: ReflectionState) -> str:
    """검토 결과에 따라 다음 경로를 결정합니다."""
    reflection = state.get("reflection", {})
    count = state.get("reflection_count", 0)

    # 승인되었거나 최대 반복 횟수에 도달하면 완료
    if reflection.get("approved", True) or count >= MAX_REFLECTION_ROUNDS:
        return "finalize"

    # 미승인이면 다시 생성
    return "generate"


def finalize_node(state: ReflectionState) -> dict:
    """최종 결과를 확정합니다."""
    count = state.get("reflection_count", 0)
    score = state.get("reflection", {}).get("score", "N/A")

    return {
        "final_output": (
            f"[검토 {count}회 완료, 점수: {score}]\n\n"
            f"{state['draft']}"
        ),
    }


# --- 그래프 구성 ---

def build_reflection_graph():
    """Self-Reflection 루프가 포함된 그래프를 구성합니다.

    그래프 구조:
        generate → reflect → (approved?) → finalize
                     ↑          ↓ (not approved & count < max)
                     └──────────┘
    """
    graph = StateGraph(ReflectionState)

    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "reflect")

    # 순환 엣지: reflect → generate (미승인 시) 또는 finalize (승인 시)
    graph.add_conditional_edges(
        "reflect",
        decide_after_reflection,
        {
            "generate": "generate",   # 수정 후 재검토
            "finalize": "finalize",   # 승인 → 완료
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# 실행
if __name__ == "__main__":
    app = build_reflection_graph()
    result = app.invoke({
        "topic": "Python에서 async/await의 동작 원리",
        "draft": "",
        "reflection": None,
        "reflection_count": 0,
        "final_output": None,
    })
    print(result["final_output"])
```

---

### 2.4 Layer 4: 코드 수정 Pre-Apply 검증 (AST / 빌드 / 린트)

**목표**: LLM이 생성한 코드를 실제 적용하기 전에 구문 오류를 검출합니다.

```python
# code_validator.py
"""LLM 생성 코드의 Pre-Apply 검증."""
import ast
import subprocess
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


def validate_python_syntax(code: str) -> dict:
    """Python 코드의 AST 구문 검증을 수행합니다.

    Args:
        code: 검증할 Python 코드 문자열.

    Returns:
        {"valid": bool, "error": str | None}
    """
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"구문 오류 (Line {e.lineno}): {e.msg}",
        }


def validate_with_lint(code: str, filename: str = "temp.py") -> dict:
    """Ruff 린터로 코드 품질을 검사합니다.

    Args:
        code: 검사할 Python 코드.
        filename: 임시 파일 이름.

    Returns:
        {"valid": bool, "warnings": list[str]}
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["ruff", "check", temp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        warnings = [
            line for line in result.stdout.strip().split("\n") if line
        ]
        return {
            "valid": result.returncode == 0,
            "warnings": warnings,
        }
    except FileNotFoundError:
        logger.warning("ruff가 설치되어 있지 않습니다. 린트 검사를 건너뜁니다.")
        return {"valid": True, "warnings": ["ruff 미설치 — 검사 생략"]}
    finally:
        os.unlink(temp_path)


def pre_apply_validation(code: str) -> dict:
    """코드를 적용하기 전에 전체 검증을 수행합니다.

    검증 순서:
    1. AST 구문 검증 (필수)
    2. 린트 검사 (권장)

    Returns:
        {
            "can_apply": bool,
            "syntax": {"valid": bool, "error": str | None},
            "lint": {"valid": bool, "warnings": list[str]},
        }
    """
    syntax_result = validate_python_syntax(code)

    if not syntax_result["valid"]:
        # 구문 오류가 있으면 린트 검사도 건너뜀
        return {
            "can_apply": False,
            "syntax": syntax_result,
            "lint": {"valid": False, "warnings": ["구문 오류로 인해 생략"]},
        }

    lint_result = validate_with_lint(code)

    return {
        "can_apply": syntax_result["valid"],
        "syntax": syntax_result,
        "lint": lint_result,
    }


# 사용 예시
if __name__ == "__main__":
    # 정상 코드
    good_code = '''
def hello(name: str) -> str:
    return f"Hello, {name}!"
'''
    print("정상 코드 검증:", pre_apply_validation(good_code))

    # 구문 오류가 있는 코드
    bad_code = '''
def hello(name: str) -> str:
    return f"Hello, {name}!"
    if True  # SyntaxError: 콜론 누락
'''
    print("오류 코드 검증:", pre_apply_validation(bad_code))
```

---

### 2.5 Layer 5: Human-in-the-Loop

**목표**: 중요한 결정 전에 사람의 승인을 받습니다.

```python
# human_in_the_loop.py
"""LangGraph Human-in-the-Loop 패턴."""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class ApprovalState(TypedDict):
    """승인 워크플로우 상태."""
    task_description: str
    generated_plan: str | None
    human_approved: bool | None
    final_action: str | None


def generate_plan_node(state: ApprovalState) -> dict:
    """실행 계획을 생성합니다."""
    # 실제로는 LLM을 호출하여 계획을 생성
    plan = f"계획: '{state['task_description']}'에 대해 다음 작업을 수행합니다..."
    return {"generated_plan": plan}


def execute_plan_node(state: ApprovalState) -> dict:
    """승인된 계획을 실행합니다."""
    return {"final_action": f"실행 완료: {state['generated_plan']}"}


def reject_plan_node(state: ApprovalState) -> dict:
    """거부된 계획을 처리합니다."""
    return {"final_action": "계획이 거부되어 실행하지 않았습니다."}


def check_approval(state: ApprovalState) -> str:
    """승인 여부에 따라 분기합니다."""
    if state.get("human_approved"):
        return "execute"
    return "reject"


def build_approval_graph():
    """Human-in-the-Loop 그래프를 구성합니다.

    핵심: interrupt_before를 사용하여 'execute' 노드 실행 전에
    그래프를 일시 중단하고, 사람의 승인을 기다립니다.

    그래프 흐름:
        generate_plan → [일시 중단 / 사람 승인 대기]
                           → check_approval
                               → execute (승인) → END
                               → reject (거부)  → END
    """
    graph = StateGraph(ApprovalState)

    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("check_approval_gate", lambda state: state)  # 패스스루 노드
    graph.add_node("execute", execute_plan_node)
    graph.add_node("reject", reject_plan_node)

    graph.set_entry_point("generate_plan")
    graph.add_edge("generate_plan", "check_approval_gate")

    graph.add_conditional_edges(
        "check_approval_gate",
        check_approval,
        {"execute": "execute", "reject": "reject"},
    )
    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    # interrupt_before: check_approval_gate 노드 실행 전에 중단
    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["check_approval_gate"],
    )


# 사용 예시
if __name__ == "__main__":
    app = build_approval_graph()
    config = {"configurable": {"thread_id": "approval-001"}}

    # Step 1: 계획 생성까지 실행 (interrupt_before로 중단됨)
    result = app.invoke(
        {
            "task_description": "프로덕션 DB 스키마 변경",
            "generated_plan": None,
            "human_approved": None,
            "final_action": None,
        },
        config=config,
    )
    print("생성된 계획:", result["generated_plan"])

    # Step 2: 사람이 검토 후 승인/거부 결정
    # (실제 시스템에서는 UI나 API로 입력을 받음)
    human_decision = True  # True: 승인, False: 거부

    # Step 3: 상태를 업데이트하고 재개
    app.update_state(
        config,
        {"human_approved": human_decision},
    )

    # Step 4: 중단된 지점부터 계속 실행
    result = app.invoke(None, config=config)
    print("최종 결과:", result["final_action"])
```

---

### 2.6 Layer 6: Golden Set (평가 데이터셋)

**목표**: LLM 출력 품질을 정량적으로 측정하는 데이터셋을 구축합니다.

**Golden Set이란?**

```
Golden Set = 입력 + 기대 출력 + 평가 기준의 모음

┌─────────────────────────────────────────────────────┐
│  Golden Case #001                                    │
│                                                      │
│  입력: "로그인 API에서 NullPointerException 발생"     │
│                                                      │
│  기대 출력:                                          │
│    - summary: "인증 토큰 null 체크 누락"              │
│    - confidence: 0.7 이상                            │
│    - issues: 1개 이상                                │
│                                                      │
│  평가 기준:                                          │
│    - summary가 비어있지 않은가? (weight: 1.0)         │
│    - confidence가 0.7 이상인가? (weight: 0.8)        │
│    - issues에 항목이 있는가? (weight: 0.5)           │
└─────────────────────────────────────────────────────┘
```

**Step 1: Golden Set JSON 파일 작성**

```json
{
    "case_id": "analysis_001",
    "description": "NullPointerException 버그 분석",
    "input": {
        "query": "UserService.java의 getUser() 메서드에서 NullPointerException이 발생합니다. user 객체가 null일 때 getName()을 호출합니다."
    },
    "expected_output": {
        "summary": "null 체크 누락으로 인한 NullPointerException",
        "confidence": 0.85,
        "issues": ["user 객체 null 체크 누락"],
        "recommendation": "user != null 조건 검사 추가"
    },
    "evaluation_criteria": [
        {
            "name": "summary_not_empty",
            "check_type": "field_exists",
            "field": "summary",
            "weight": 1.0
        },
        {
            "name": "confidence_above_threshold",
            "check_type": "value_range",
            "field": "confidence",
            "min_value": 0.5,
            "max_value": 1.0,
            "weight": 0.8
        },
        {
            "name": "has_issues",
            "check_type": "list_min_length",
            "field": "issues",
            "min_length": 1,
            "weight": 0.5
        },
        {
            "name": "recommendation_not_empty",
            "check_type": "field_exists",
            "field": "recommendation",
            "weight": 0.7
        }
    ]
}
```

**Step 2: 평가 러너 구현**

```python
# golden_set_runner.py
"""Golden Set 기반 LLM 출력 품질 자동 평가."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_single_case(case: dict, actual_output: dict) -> dict:
    """단일 Golden Case를 평가합니다.

    Args:
        case: Golden Set 케이스 (기대 출력 + 평가 기준 포함).
        actual_output: LLM이 실제로 반환한 출력.

    Returns:
        평가 결과 딕셔너리 (점수, 개별 체크 결과 포함).
    """
    criteria = case.get("evaluation_criteria", [])
    checks = []
    total_weight = 0.0
    weighted_score = 0.0

    for criterion in criteria:
        weight = criterion.get("weight", 1.0)
        total_weight += weight

        passed = _run_check(criterion, actual_output)
        checks.append({
            "name": criterion["name"],
            "passed": passed,
            "weight": weight,
        })

        if passed:
            weighted_score += weight

    score = weighted_score / total_weight if total_weight > 0 else 0.0

    return {
        "case_id": case["case_id"],
        "score": round(score, 3),
        "passed": score >= 0.8,  # 80% 이상이면 통과
        "checks": checks,
    }


def _run_check(criterion: dict, actual_output: dict) -> bool:
    """개별 평가 기준을 실행합니다."""
    check_type = criterion["check_type"]
    field = criterion["field"]
    value = actual_output.get(field)

    if check_type == "field_exists":
        return value is not None and value != ""

    elif check_type == "value_range":
        if value is None:
            return False
        min_val = criterion.get("min_value", float("-inf"))
        max_val = criterion.get("max_value", float("inf"))
        return min_val <= float(value) <= max_val

    elif check_type == "list_min_length":
        if not isinstance(value, list):
            return False
        return len(value) >= criterion.get("min_length", 1)

    elif check_type == "exact_match":
        return value == criterion.get("expected_value")

    elif check_type == "contains":
        return criterion.get("substring", "") in str(value)

    return False


def run_golden_set(
    golden_set_dir: str,
    llm_function,
) -> dict:
    """Golden Set 디렉토리의 모든 케이스를 평가합니다.

    Args:
        golden_set_dir: Golden Set JSON 파일들이 있는 디렉토리 경로.
        llm_function: 입력을 받아 dict를 반환하는 LLM 호출 함수.

    Returns:
        전체 평가 요약 (평균 점수, 통과/실패 수, 상세 결과).
    """
    cases_dir = Path(golden_set_dir)
    results = []

    for case_file in sorted(cases_dir.glob("*.json")):
        case = json.loads(case_file.read_text(encoding="utf-8"))

        # LLM 호출
        actual_output = llm_function(case["input"])

        # 평가
        result = evaluate_single_case(case, actual_output)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        logger.info(
            "[%s] %s — score=%.3f",
            status, case["case_id"], result["score"],
        )

    # 전체 요약
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / total if total > 0 else 0.0

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
        "average_score": round(avg_score, 3),
        "details": results,
    }

    print(f"\n{'='*50}")
    print(f"Golden Set 평가 결과")
    print(f"{'='*50}")
    print(f"총 케이스: {total}")
    print(f"통과: {passed} / 실패: {total - passed}")
    print(f"통과율: {summary['pass_rate']*100:.1f}%")
    print(f"평균 점수: {summary['average_score']:.3f}")
    print(f"{'='*50}")

    return summary
```

---

## 3. 실전 예제: 6-레이어 품질 보장이 적용된 "코드 리뷰 에이전트"

6개 레이어를 모두 결합한 실전 에이전트를 구현합니다.

```python
# code_review_agent.py
"""6-레이어 품질 보장이 적용된 코드 리뷰 에이전트.

Layer 0: JSON 파싱 (parse_json_from_llm)
Layer 1: 스키마 검증 (Pydantic CodeReviewResult)
Layer 2: Confidence 게이팅 (confidence_gate)
Layer 3: Self-Reflection (reflect_review 노드)
Layer 4: 외부 검증 (코드 구문 검사)
Layer 5: Human-in-the-Loop (interrupt_before)
"""
from typing import TypedDict
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

from validated_parser import parse_and_validate
from confidence_gate import confidence_gate, GateAction

MAX_REFLECTION_ROUNDS = 2


# --- Layer 1: 스키마 정의 ---

class ReviewIssue(BaseModel):
    """코드 리뷰에서 발견된 문제."""
    severity: str = Field(pattern=r"^(HIGH|MEDIUM|LOW)$")
    description: str = Field(min_length=1)
    line_number: int | None = None
    suggestion: str = ""


class CodeReviewResult(BaseModel):
    """코드 리뷰 결과 스키마."""
    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[ReviewIssue] = Field(default_factory=list)
    approved: bool
    overall_quality: str = Field(pattern=r"^(GOOD|ACCEPTABLE|NEEDS_WORK|POOR)$")

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)


# --- 상태 정의 ---

class CodeReviewState(TypedDict):
    code: str                           # 리뷰할 코드
    review: dict | None                 # 리뷰 결과
    reflection: dict | None             # Self-Reflection 결과
    reflection_count: int               # Reflection 횟수
    human_approved: bool | None         # 인간 승인 여부
    final_result: str | None            # 최종 결과
    error: str | None


# --- 노드 정의 ---

def review_code_node(state: CodeReviewState) -> dict:
    """코드를 리뷰합니다. (Layer 0 + 1 적용)"""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    prompt = f"""다음 코드를 리뷰하고 JSON으로 응답하세요.

코드:
```
{state["code"]}
```

JSON 응답 형식:
{{
    "summary": "리뷰 요약",
    "confidence": 0.0~1.0,
    "issues": [
        {{"severity": "HIGH|MEDIUM|LOW", "description": "문제 설명", "line_number": null, "suggestion": "수정 제안"}}
    ],
    "approved": true/false,
    "overall_quality": "GOOD|ACCEPTABLE|NEEDS_WORK|POOR"
}}"""

    response = llm.invoke(prompt)

    # Layer 0 (JSON 파싱) + Layer 1 (스키마 검증)
    review = parse_and_validate(response.content, CodeReviewResult)
    return {"review": review}


def confidence_check_node(state: CodeReviewState) -> str:
    """Confidence 게이팅을 적용합니다. (Layer 2)"""
    review = state.get("review", {})
    gate = confidence_gate(
        review.get("confidence", 0.0),
        "code_review",
    )

    if gate.action == GateAction.ACCEPT:
        return "reflect"
    elif gate.action == GateAction.RETRY:
        return "review_code"  # 재시도
    else:
        return "handle_error"


def reflect_review_node(state: CodeReviewState) -> dict:
    """리뷰 결과를 자기 검토합니다. (Layer 3)"""
    count = state.get("reflection_count", 0)

    if count >= MAX_REFLECTION_ROUNDS:
        return {"reflection": {"approved": True}, "reflection_count": count}

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    prompt = f"""당신은 시니어 코드 리뷰어입니다. 아래 리뷰 결과를 검토하세요.

리뷰 결과:
{state["review"]}

검토 관점:
1. 리뷰가 공정하고 정확한가?
2. 놓친 중요한 이슈가 있는가?
3. 제안이 실용적인가?

JSON으로 응답:
{{"approved": true/false, "feedback": "피드백"}}"""

    response = llm.invoke(prompt)
    from validated_parser import parse_json_from_llm
    reflection = parse_json_from_llm(response.content)

    return {
        "reflection": reflection,
        "reflection_count": count + 1,
    }


def decide_after_reflection(state: CodeReviewState) -> str:
    """Reflection 결과에 따라 분기합니다."""
    reflection = state.get("reflection", {})
    count = state.get("reflection_count", 0)

    if reflection.get("approved", True) or count >= MAX_REFLECTION_ROUNDS:
        return "human_gate"
    return "review_code"  # 미승인 → 재생성


def human_gate_node(state: CodeReviewState) -> dict:
    """Human-in-the-Loop 게이트. (Layer 5)"""
    return state  # interrupt_before에 의해 여기서 중단됨


def check_human_decision(state: CodeReviewState) -> str:
    """인간의 최종 결정에 따라 분기합니다."""
    if state.get("human_approved", True):
        return "finalize"
    return "review_code"  # 인간이 거부하면 재시도


def finalize_node(state: CodeReviewState) -> dict:
    """최종 결과를 확정합니다."""
    review = state.get("review", {})
    return {
        "final_result": (
            f"코드 리뷰 완료\n"
            f"요약: {review.get('summary', 'N/A')}\n"
            f"품질: {review.get('overall_quality', 'N/A')}\n"
            f"신뢰도: {review.get('confidence', 'N/A')}\n"
            f"발견된 이슈: {len(review.get('issues', []))}건"
        ),
    }


def handle_error_node(state: CodeReviewState) -> dict:
    """에러를 처리합니다."""
    return {"error": "코드 리뷰 실패: 신뢰도가 너무 낮습니다."}


# --- 그래프 구성 ---

def build_code_review_graph():
    """6-레이어 품질 보장이 적용된 코드 리뷰 그래프.

    흐름:
        review_code → confidence_check
            → (ACCEPT) → reflect → (approved?) → human_gate
            →                         ↑    ↓(미승인)       → finalize
            →                         └────┘               → END
            → (RETRY) → review_code (재시도)
            → (REJECT) → handle_error → END
    """
    graph = StateGraph(CodeReviewState)

    graph.add_node("review_code", review_code_node)
    graph.add_node("reflect", reflect_review_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("handle_error", handle_error_node)

    graph.set_entry_point("review_code")

    graph.add_conditional_edges(
        "review_code",
        confidence_check_node,
        {
            "reflect": "reflect",
            "review_code": "review_code",
            "handle_error": "handle_error",
        },
    )

    graph.add_conditional_edges(
        "reflect",
        decide_after_reflection,
        {
            "human_gate": "human_gate",
            "review_code": "review_code",
        },
    )

    graph.add_conditional_edges(
        "human_gate",
        check_human_decision,
        {
            "finalize": "finalize",
            "review_code": "review_code",
        },
    )

    graph.add_edge("finalize", END)
    graph.add_edge("handle_error", END)

    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_gate"],
    )
```

---

## 4. 연습 문제

### 연습 1: Self-Reflection 루프 추가하기 (난이도: 중)

**과제**: Module 09에서 만든 에이전트 (또는 아래 간단한 에이전트)에 Self-Reflection 루프를 추가하세요.

**시작 코드** (Self-Reflection 없음):

```python
# exercise_start.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic


class TranslationState(TypedDict):
    source_text: str          # 원문
    translated: str | None    # 번역 결과
    final_output: str | None


def translate_node(state: TranslationState) -> dict:
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    prompt = f"다음 한국어를 영어로 번역하세요:\n\n{state['source_text']}"
    response = llm.invoke(prompt)
    return {"translated": response.content}


def finalize_node(state: TranslationState) -> dict:
    return {"final_output": state["translated"]}


graph = StateGraph(TranslationState)
graph.add_node("translate", translate_node)
graph.add_node("finalize", finalize_node)
graph.set_entry_point("translate")
graph.add_edge("translate", "finalize")
graph.add_edge("finalize", END)
app = graph.compile()
```

**요구사항**:
1. `reflect_translation` 노드를 추가하세요 (번역 품질 검토)
2. 미승인 시 `translate` 노드로 돌아가는 순환 구조를 구현하세요
3. 최대 반복 횟수를 2회로 제한하세요
4. `reflection_count` 필드를 상태에 추가하세요

**힌트**: Section 2.3의 Self-Reflection 패턴을 참고하세요.

---

### 연습 2: Confidence 게이팅 + Golden Set 평가 (난이도: 상)

**과제**:
1. 아래 에이전트에 Confidence 게이팅을 추가하세요 (임계값: 0.6)
2. Golden Set JSON 파일 3개를 작성하세요
3. `run_golden_set()` 함수로 평가를 실행하세요

```python
# 힌트: 다음 구조를 참고하세요
#
# golden_sets/
# ├── case_001.json
# ├── case_002.json
# └── case_003.json
#
# 각 JSON 파일에는 input, expected_output, evaluation_criteria를 포함하세요.
```

---

## 5. 핵심 정리

### 6단계 품질 보장 레이어 요약

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Layer 0  JSON 파싱        → json.loads() + 폴백 전략    │
│  Layer 1  스키마 검증       → Pydantic BaseModel          │
│  Layer 2  Confidence 게이팅 → 임계값 기반 ACCEPT/RETRY/REJECT │
│  Layer 3  Self-Reflection  → LLM이 자기 출력을 검토      │
│  Layer 4  외부 검증        → AST / 빌드 / 린트           │
│  Layer 5  Human-in-the-Loop → interrupt_before + 상태 업데이트 │
│  Layer 6  정량 평가        → Golden Set + 자동 메트릭     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 기억해야 할 핵심 원칙

| 원칙 | 설명 |
|------|------|
| 점진적 적용 | Layer 0 → 1 → 2 순서로 쌓아 올리기 |
| 무한 루프 방지 | Self-Reflection에 반드시 max_rounds 설정 |
| 임계값 조정 | 초기엔 낮게 설정 → 데이터 수집 후 최적화 |
| 정량적 측정 | Golden Set으로 "좋아졌는지/나빠졌는지" 객관적 판단 |
| 인간 개입 지점 | 위험도 높은 작업(DB 변경, 프로덕션 배포) 전에 배치 |

---

## 6. 참고 자료

| 주제 | 링크 |
|------|------|
| LangGraph Human-in-the-Loop | https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/ |
| LangGraph 순환 그래프 (Cycles) | https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/ |
| Pydantic Validation | https://docs.pydantic.dev/latest/concepts/validators/ |
| LLM Evaluation (Anthropic) | https://docs.anthropic.com/en/docs/build-with-claude/develop-tests |
| LangGraph Conditional Edges | https://langchain-ai.github.io/langgraph/how-tos/branching/ |
| JSON 파싱 전략 | https://python.langchain.com/docs/how_to/output_parser_json/ |

---

## 다음 단계

축하합니다! LLM 출력 품질 보장의 6단계 레이어를 모두 학습했습니다.

**다음 모듈에서 배울 내용:**
- **Module 12: LangGraph 고급 패턴** — 체크포인팅으로 장애 복구, 서브그래프로 모듈 분리, 병렬 실행으로 성능 향상
- 지금까지 만든 에이전트에 프로덕션 수준의 안정성을 추가합니다

**복습 체크리스트:**
- [ ] Pydantic으로 LLM 출력 스키마를 정의할 수 있다
- [ ] Confidence 게이팅으로 3단계 분기를 구현할 수 있다
- [ ] Self-Reflection 루프를 LangGraph 순환 그래프로 구현할 수 있다
- [ ] Golden Set을 작성하고 자동 평가를 실행할 수 있다
- [ ] Human-in-the-Loop 패턴을 interrupt_before로 구현할 수 있다
