# Module 08: 에러 처리와 회복 탄력성

> 실패해도 멈추지 않는 에이전트 만들기

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. AI 에이전트에서 발생하는 에러 유형을 분류하고, 재시도 가능 여부를 판단할 수 있다
2. Exponential Backoff + Jitter를 이해하고 직접 구현할 수 있다
3. LangGraph의 RetryPolicy를 노드에 적용할 수 있다
4. Circuit Breaker 패턴을 이해하고 에이전트에 적용할 수 있다
5. 우아한 실패 처리(Graceful Degradation)로 DLQ 직행을 방지할 수 있다

---

## 사전 지식

| 주제 | 수준 | 설명 |
|------|------|------|
| Python 기초 | 필수 | try/except, 데코레이터, 클래스 |
| LangGraph 기초 | 필수 | Module 05~07에서 다룬 StateGraph, 노드, 엣지 |
| HTTP 기초 | 권장 | 상태 코드(200, 429, 500 등)의 의미 |
| asyncio | 선택 | 비동기 프로그래밍 기초 (없어도 학습 가능) |

---

## 1. 개념 설명

### 1.1 왜 에러 처리가 중요한가?

AI 에이전트는 혼자서 동작하지 않습니다. LLM API를 호출하고, 외부 서비스(Jira, Slack, DB 등)와 통신하며, 파일을 읽고 쓰는 등 다양한 외부 시스템에 의존합니다.

```
┌──────────────┐     ┌──────────┐     ┌──────────┐
│  AI 에이전트  │────>│  LLM API │     │  Jira    │
│              │────>│ (Claude) │     │  API     │
│  Node A      │     └──────────┘     └──────────┘
│  Node B      │────>┌──────────┐     ┌──────────┐
│  Node C      │     │  Slack   │     │ Database │
│              │────>│  API     │────>│          │
└──────────────┘     └──────────┘     └──────────┘
```

이 중 **하나라도 장애가 발생하면** 에이전트 전체가 멈출 수 있습니다. 하지만 모든 에러가 치명적인 것은 아닙니다. 일시적인 네트워크 오류는 잠시 기다렸다 재시도하면 해결되고, 서버 과부하는 시간이 지나면 회복됩니다.

> **핵심 질문**: "이 에러는 다시 시도하면 성공할 수 있는가?"

### 1.2 에이전트 시스템에서 발생하는 에러 유형

AI 에이전트에서 만나는 에러를 4가지 범주로 나눌 수 있습니다:

#### (1) LLM API 에러

LLM(Claude, GPT 등)을 호출할 때 발생하는 에러입니다.

| 에러 | HTTP 코드 | 원인 | 빈도 |
|------|-----------|------|------|
| Rate Limit | 429 | 분당 요청 수 초과 | 높음 |
| Server Error | 500, 503 | LLM 서버 일시 장애 | 보통 |
| Timeout | - | 응답 시간 초과 (긴 프롬프트) | 보통 |
| Authentication | 401 | API 키 만료/잘못됨 | 낮음 |
| Bad Request | 400 | 프롬프트가 너무 길거나 잘못된 형식 | 낮음 |

#### (2) 외부 API 에러

Jira, Slack, GitLab 등 외부 서비스 호출 시 발생합니다.

| 에러 | HTTP 코드 | 원인 | 빈도 |
|------|-----------|------|------|
| Rate Limit | 429 | API 호출 한도 초과 | 보통 |
| 서버 오류 | 500, 502, 503 | 외부 서비스 장애 | 낮음 |
| 권한 없음 | 403 | 토큰 권한 부족 | 낮음 |
| 리소스 없음 | 404 | 잘못된 ID/경로 | 보통 |

#### (3) 데이터 에러

LLM 응답이나 입력 데이터를 처리할 때 발생합니다.

| 에러 | 원인 | 예시 |
|------|------|------|
| JSON 파싱 실패 | LLM이 잘못된 JSON 반환 | `json.JSONDecodeError` |
| 입력 데이터 부재 | 필수 필드가 비어있음 | `affected_files`가 빈 리스트 |
| 스키마 불일치 | LLM 응답이 예상 구조와 다름 | 필수 키 누락 |

#### (4) 시스템 에러

실행 환경 자체의 문제입니다.

| 에러 | 원인 | 예시 |
|------|------|------|
| 메모리 부족 | 대용량 데이터 처리 | `MemoryError` |
| 디스크 풀 | Git clone 누적 | `OSError: No space left` |
| 네트워크 단절 | DNS 실패, 연결 불가 | `ConnectionError` |

### 1.3 Retriable vs Non-retriable 에러 분류

에러를 만났을 때 가장 먼저 해야 할 질문: **"다시 시도하면 성공할 가능성이 있는가?"**

```
에러 발생!
    │
    ▼
┌─────────────────────┐
│ 재시도하면 성공할까? │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
  [YES]       [NO]
 Retriable   Non-retriable
    │           │
    ▼           ▼
 재시도!     에러 기록 후
 (backoff)   다른 경로로
```

#### 상세 분류표

| 에러 유형 | 분류 | 재시도 | 이유 |
|----------|------|--------|------|
| LLM Rate Limit (429) | **Retriable** | O (최대 3회, 2/4/8초) | 시간이 지나면 한도 초기화 |
| LLM Timeout | **Retriable** | O (최대 2회, 5/10초) | 일시적 서버 부하 가능성 |
| LLM Server Error (5xx) | **Retriable** | O (최대 2회, 3/6초) | 서버 재시작/복구 가능 |
| HTTP 401 Unauthorized | **Non-retriable** | X | 인증 정보가 바뀌어야 해결 |
| HTTP 403 Forbidden | **Non-retriable** | X | 권한 설정 변경 필요 |
| HTTP 404 Not Found | **Non-retriable** | X | 존재하지 않는 리소스 |
| JSON 파싱 실패 | **Non-retriable** | X | 같은 입력 = 같은 실패 |
| 입력 데이터 부재 | **Non-retriable** | X | 데이터가 없으면 계속 없음 |
| Jira/GitLab 429, 5xx | **Conditional** | O (해당 코드만) | 서버 일시 문제 |
| 파일 시스템 오류 | **Non-retriable** | X | 구조적 문제 |

> **Conditional(조건부)**: HTTP 상태 코드에 따라 재시도 여부가 달라집니다. 429, 500, 502, 503, 504는 재시도하고, 400, 401, 403, 404는 재시도하지 않습니다.

---

## 2. 단계별 실습

### 2.1 Exponential Backoff + Jitter

#### 개념: 도서관 비유

도서관에서 인기 있는 책을 빌리려고 합니다. 사서에게 갔더니 "지금 바빠요"라고 합니다.

- **단순 재시도**: 1초 후 다시 가기 -> 또 바빠요 -> 1초 후 다시 가기 -> ...
  - 문제: 다른 사람들도 1초마다 와서 사서가 더 바빠집니다
- **Exponential Backoff**: 1초 후 -> 2초 후 -> 4초 후 -> 8초 후 ...
  - 점점 더 오래 기다려서 사서에게 여유를 줍니다
- **Jitter**: 1~2초 후 -> 2~4초 후 -> 4~8초 후 ...
  - 무작위 대기 시간으로, 동시에 몰리는 것을 방지합니다

```
단순 재시도 (모두 1초 간격):
사용자A: ─X──X──X──X──X──> (모두 같은 시점에 재시도)
사용자B: ─X──X──X──X──X──>
사용자C: ─X──X──X──X──X──>
                ↑ 서버 과부하!

Exponential Backoff + Jitter:
사용자A: ─X────X────────X─────────────────X──>
사용자B: ─X──────X──────────X───────────────────X──>
사용자C: ─X───X───────────X──────────────X──>
                ↑ 재시도가 분산되어 서버 부하 감소
```

#### Thundering Herd (몰려오는 짐승 떼) 문제

Jitter가 없으면, 동시에 실패한 100개의 요청이 **정확히 같은 시점에** 재시도합니다. 이를 "Thundering Herd"라고 합니다. Jitter(랜덤 지연)를 추가하면 재시도 시점이 분산되어 서버 부하를 줄입니다.

#### 구현: retry_with_backoff 함수

```python
import time
import random
import logging

logger = logging.getLogger(__name__)


def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: tuple = (Exception,),
):
    """Exponential Backoff + Jitter로 함수를 재시도합니다.

    Args:
        fn: 실행할 함수 (인자 없는 callable)
        max_retries: 최대 재시도 횟수
        base_delay: 첫 번째 재시도 전 기본 대기 시간(초)
        max_delay: 최대 대기 시간(초)
        backoff_factor: 대기 시간 증가 배수 (보통 2.0)
        jitter: True면 랜덤 지연 추가
        retriable_exceptions: 재시도할 예외 타입

    Returns:
        fn()의 반환값

    Raises:
        마지막 시도에서 발생한 예외
    """
    last_exception = None

    for attempt in range(max_retries + 1):  # 최초 시도 + 재시도
        try:
            return fn()
        except retriable_exceptions as exc:
            last_exception = exc

            if attempt >= max_retries:
                # 마지막 시도도 실패
                logger.error(
                    "모든 재시도 실패",
                    extra={
                        "attempts": attempt + 1,
                        "error": str(exc)[:200],
                    },
                )
                raise

            # 대기 시간 계산: base_delay * (backoff_factor ^ attempt)
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)

            # Jitter 추가: 0 ~ delay 사이의 랜덤 값
            if jitter:
                delay = random.uniform(0, delay)

            logger.warning(
                "재시도 대기 중",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_seconds": round(delay, 2),
                    "error_type": type(exc).__name__,
                },
            )
            time.sleep(delay)

    raise last_exception
```

#### 사용 예시

```python
import httpx

def call_llm_api(prompt: str) -> str:
    """LLM API를 호출하는 함수."""
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={"prompt": prompt},
        headers={"Authorization": "Bearer YOUR_KEY"},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["content"]


# Exponential Backoff로 LLM API 호출
result = retry_with_backoff(
    fn=lambda: call_llm_api("안녕하세요"),
    max_retries=3,
    base_delay=2.0,
    retriable_exceptions=(httpx.HTTPStatusError, httpx.TimeoutException),
)
```

### 2.2 LangGraph RetryPolicy

LangGraph는 노드에 **내장 RetryPolicy**를 설정할 수 있습니다. `add_node()`의 `retry=` 파라미터로 전달합니다.

#### 기본 사용법

```python
from langgraph.graph import StateGraph
from langgraph.pregel import RetryPolicy
from typing import TypedDict


class AgentState(TypedDict):
    query: str
    result: str
    error: str | None


# RetryPolicy 정의
llm_retry_policy = RetryPolicy(
    max_attempts=3,           # 최대 3번 시도
    initial_interval=2.0,     # 첫 대기: 2초
    backoff_factor=2.0,       # 배수: 2 -> 4 -> 8초
    max_interval=30.0,        # 최대 대기: 30초
    jitter=True,              # 랜덤 지연 추가
)

# 노드 함수 정의
def analyze_node(state: AgentState) -> dict:
    """LLM을 호출하여 분석하는 노드."""
    # 이 함수에서 예외가 발생하면 RetryPolicy에 따라 자동 재시도
    result = call_llm("분석해주세요: " + state["query"])
    return {"result": result}


# 그래프에 노드 추가 (retry 정책 적용)
graph = StateGraph(AgentState)
graph.add_node("analyze", analyze_node, retry=llm_retry_policy)  # retry= 파라미터!
```

#### RetryPolicy 파라미터 상세

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `max_attempts` | int | 3 | 최대 시도 횟수 (최초 포함) |
| `initial_interval` | float | 1.0 | 첫 재시도 전 대기 시간(초) |
| `backoff_factor` | float | 2.0 | 대기 시간 증가 배수 |
| `max_interval` | float | 128.0 | 최대 대기 시간(초) |
| `jitter` | bool | True | 랜덤 지연 추가 여부 |
| `retry_on` | callable | None | 예외를 받아 재시도 여부를 반환하는 함수 |

#### retry_on으로 특정 예외만 재시도

```python
import anthropic
import httpx


def should_retry(exc: Exception) -> bool:
    """재시도 가능한 예외인지 판별합니다."""
    # LLM Rate Limit -> 재시도
    if isinstance(exc, anthropic.RateLimitError):
        return True
    # LLM 서버 오류 -> 재시도
    if isinstance(exc, anthropic.InternalServerError):
        return True
    # 타임아웃 -> 재시도
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout)):
        return True
    # 그 외 -> 재시도하지 않음
    return False


llm_retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=2.0,
    backoff_factor=2.0,
    retry_on=should_retry,  # 이 함수로 재시도 여부 결정
)
```

#### 내장 RetryPolicy의 한계

RetryPolicy는 편리하지만, **재시도를 모두 소진하면 예외를 그대로 raise**합니다. 즉, 그래프 전체가 중단됩니다. 에러를 "부드럽게" 처리하려면(예: state에 에러 기록 후 다른 경로로 이동) 커스텀 데코레이터가 필요합니다.

### 2.3 커스텀 retry_on_llm_error 데코레이터

실전에서는 다음 두 가지를 모두 처리해야 합니다:
- **Retriable 에러** -> backoff 재시도
- **Non-retriable 에러** -> `state["error"]`에 기록하고 failure 경로로 이동

```python
import time
import logging
from functools import wraps

import anthropic
import httpx

logger = logging.getLogger(__name__)

# 재시도 가능한 예외 목록
RETRIABLE_EXCEPTIONS = (
    anthropic.RateLimitError,       # 429 Rate Limit
    anthropic.InternalServerError,  # 500 서버 오류
    anthropic.APIConnectionError,   # 네트워크 연결 실패
    httpx.ReadTimeout,              # 읽기 타임아웃
    httpx.ConnectTimeout,           # 연결 타임아웃
)


def retry_on_llm_error(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
):
    """LLM 호출 노드용 재시도 데코레이터.

    - Retriable 에러: exponential backoff로 재시도
    - Non-retriable 에러: state["error"]에 기록하고 반환
    - 모든 재시도 실패: state["error"]에 기록하고 반환 (DLQ 직행 방지)

    사용법:
        @retry_on_llm_error(max_retries=3, base_delay=2.0)
        def my_node(state: MyState) -> dict:
            result = call_llm(...)
            return {"result": result}
    """
    def decorator(node_fn):
        @wraps(node_fn)
        def wrapper(state, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # 노드 함수 실행
                    return node_fn(state, **kwargs)

                except RETRIABLE_EXCEPTIONS as exc:
                    # --- Retriable: 재시도 ---
                    last_exception = exc

                    if attempt < max_retries:
                        delay = min(
                            base_delay * (backoff_factor ** attempt),
                            max_delay,
                        )
                        logger.warning(
                            "LLM 호출 실패 (재시도 가능), 재시도 중...",
                            extra={
                                "node": node_fn.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay_seconds": delay,
                                "error_type": type(exc).__name__,
                            },
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "LLM 호출 - 모든 재시도 실패",
                            extra={
                                "node": node_fn.__name__,
                                "total_attempts": max_retries + 1,
                                "error_type": type(exc).__name__,
                            },
                        )

                except Exception as exc:
                    # --- Non-retriable: 즉시 에러 반환 ---
                    logger.error(
                        "LLM 호출 실패 (재시도 불가)",
                        extra={
                            "node": node_fn.__name__,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                    )
                    return {
                        "error": f"[{node_fn.__name__}] {type(exc).__name__}: {exc}",
                        "current_step": node_fn.__name__,
                    }

            # 모든 재시도를 소진한 경우: state["error"]에 기록 (DLQ 직행 방지)
            return {
                "error": (
                    f"[{node_fn.__name__}] {type(last_exception).__name__}: "
                    f"{last_exception} (after {max_retries + 1} attempts)"
                ),
                "current_step": node_fn.__name__,
            }

        return wrapper
    return decorator
```

#### 데코레이터 적용 예시

```python
# 적용 전: 에러 발생 시 raise -> 그래프 전체 중단 -> DLQ
def analyze_node(state: AgentState) -> dict:
    try:
        result = call_llm(state["query"])
        return {"result": result, "current_step": "analyze"}
    except Exception as exc:
        logger.error("분석 실패", error=str(exc))
        raise  # -> 그래프 중단!


# 적용 후: Retriable은 재시도, Non-retriable은 에러 경로로
@retry_on_llm_error(max_retries=3, base_delay=2.0)
def analyze_node(state: AgentState) -> dict:
    result = call_llm(state["query"])
    return {"result": result, "current_step": "analyze"}
    # Retriable 에러 -> 자동 재시도 (최대 3회)
    # Non-retriable 에러 -> state["error"] 설정 -> 조건부 엣지에서 failure로 라우팅
```

### 2.4 Circuit Breaker 패턴

#### 개념: 가정의 차단기 비유

집에서 전기를 너무 많이 쓰면 차단기(두꺼비집)가 내려갑니다. 이는 과부하로 인한 화재를 방지하기 위한 **안전 장치**입니다.

소프트웨어에서 Circuit Breaker도 같은 역할을 합니다:
- 외부 시스템(Jira, LLM 등)이 계속 실패하면 -> 요청을 **차단**
- 이유: 실패할 게 뻔한 요청에 시간(타임아웃 30초 x 재시도 3회 = 90초)을 낭비하지 않기 위해
- 일정 시간 후 -> 소량의 요청으로 복구 여부 **확인**

#### 상태 전이 다이어그램

```
                연속 실패 >= threshold (예: 5회)
   CLOSED ─────────────────────────────────> OPEN
  (정상)                                    (차단)
     ^                                        |
     |                                        | timeout 경과 (예: 60초)
     |                                        v
     |        테스트 성공                   HALF_OPEN
     └─────────────────────────────────── (테스트 중)
                                              |
                                              | 테스트 실패
                                              v
                                            OPEN
                                           (다시 차단)
```

**각 상태의 역할:**

| 상태 | 동작 | 비유 |
|------|------|------|
| **CLOSED** (정상) | 모든 요청을 통과시키고, 실패 횟수를 추적 | 차단기 올라감 (전기 사용 가능) |
| **OPEN** (차단) | 모든 요청을 즉시 거부 (타임아웃 대기 없음) | 차단기 내려감 (전기 차단) |
| **HALF_OPEN** (테스트) | 1건만 통과시켜 복구 확인 | 차단기 살짝 올려서 테스트 |

#### 구현: CircuitBreaker 클래스

```python
import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit Breaker의 3가지 상태."""
    CLOSED = "closed"       # 정상 - 요청 통과
    OPEN = "open"           # 차단 - 요청 거부
    HALF_OPEN = "half_open" # 테스트 - 제한적 통과


class CircuitOpenError(Exception):
    """Circuit Breaker가 OPEN 상태일 때 발생하는 예외."""
    def __init__(self, name: str, remaining_seconds: float):
        self.name = name
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit '{name}'이 OPEN 상태입니다. "
            f"{remaining_seconds:.1f}초 후 재시도하세요."
        )


@dataclass
class CircuitBreaker:
    """Circuit Breaker 구현.

    외부 시스템 호출을 감싸서, 연속 실패 시 자동으로 차단합니다.

    Args:
        name: 서킷 식별자 (예: "jira-api", "llm-api")
        failure_threshold: OPEN으로 전환되는 연속 실패 횟수 (기본 5)
        recovery_timeout: OPEN에서 HALF_OPEN으로 전환되는 대기 시간(초) (기본 60)
        expected_exceptions: 실패로 카운트할 예외 타입
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exceptions: tuple = (Exception,)

    # 내부 상태 (외부에서 초기화하지 않음)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """현재 상태를 반환합니다. OPEN -> HALF_OPEN 자동 전환 포함."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        f"Circuit '{self.name}': OPEN -> HALF_OPEN "
                        f"({elapsed:.1f}초 경과)"
                    )
            return self._state

    def _record_success(self) -> None:
        """성공을 기록합니다."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(f"Circuit '{self.name}': HALF_OPEN -> CLOSED (복구됨)")

    def _record_failure(self) -> None:
        """실패를 기록합니다."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit '{self.name}': -> OPEN "
                    f"(연속 {self._failure_count}회 실패)"
                )

    def call(self, fn, *args, **kwargs):
        """Circuit Breaker를 통해 함수를 호출합니다.

        Args:
            fn: 호출할 함수
            *args, **kwargs: 함수 인자

        Returns:
            fn()의 반환값

        Raises:
            CircuitOpenError: 서킷이 OPEN 상태일 때
        """
        current_state = self.state

        # OPEN 상태: 즉시 거부 (타임아웃 대기 없이 빠르게 실패)
        if current_state == CircuitState.OPEN:
            remaining = self.recovery_timeout - (
                time.monotonic() - self._last_failure_time
            )
            raise CircuitOpenError(self.name, max(0, remaining))

        # CLOSED 또는 HALF_OPEN: 요청 통과
        try:
            result = fn(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exceptions:
            self._record_failure()
            raise
```

#### 에이전트 노드에 Circuit Breaker 적용

```python
import httpx

# Jira API용 Circuit Breaker 생성
jira_circuit = CircuitBreaker(
    name="jira-api",
    failure_threshold=5,          # 5번 연속 실패하면 차단
    recovery_timeout=60.0,        # 60초 후 복구 시도
    expected_exceptions=(httpx.HTTPError, httpx.TimeoutException),
)


def create_jira_ticket_node(state: AgentState) -> dict:
    """Jira 티켓을 생성하는 노드."""
    try:
        # Circuit Breaker를 통해 Jira API 호출
        result = jira_circuit.call(
            _do_jira_request,
            url="https://your-jira.atlassian.net/rest/api/3/issue",
            payload={"fields": {"summary": state["title"]}},
        )
        return {"jira_key": result["key"]}

    except CircuitOpenError as exc:
        # 서킷 열림 -> 즉시 실패 (타임아웃 없이)
        logger.warning(f"Jira 서킷 열림: {exc.remaining_seconds:.0f}초 후 재시도")
        return {"error": str(exc)}

    except Exception as exc:
        # 기타 에러
        return {"error": f"Jira 호출 실패: {exc}"}


def _do_jira_request(url: str, payload: dict) -> dict:
    """실제 Jira API 호출."""
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
    return response.json()
```

---

## 3. 실전 예제

### 외부 API + LLM 호출이 있는 에이전트에 전체 에러 처리 적용

실전에서는 하나의 에이전트에 LLM 호출, 외부 API 호출, 데이터 처리가 모두 포함됩니다. 이 예제에서는 3-노드 에이전트에 지금까지 배운 모든 패턴을 적용합니다.

```
[fetch_data] ──> [analyze] ──> [create_ticket]
 (외부 API)       (LLM)         (외부 API)
     │                │              │
     ▼                ▼              ▼
  Circuit           Retry         Circuit
  Breaker          Decorator      Breaker
```

#### 전체 코드

```python
import operator
import httpx
import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ── 1. State 정의 ──

def _replace(existing, new):
    """기존 값을 새 값으로 교체하는 reducer."""
    return new


class TaskState(TypedDict):
    """에이전트의 공유 상태."""
    query: Annotated[str, _replace]
    fetched_data: Annotated[str | None, _replace]
    analysis: Annotated[str | None, _replace]
    ticket_key: Annotated[str | None, _replace]
    error: Annotated[str | None, _replace]
    current_step: Annotated[str, _replace]


# ── 2. Circuit Breaker 설정 ──

api_circuit = CircuitBreaker(
    name="external-api",
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exceptions=(httpx.HTTPError, httpx.TimeoutException),
)

ticket_circuit = CircuitBreaker(
    name="ticket-api",
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exceptions=(httpx.HTTPError, httpx.TimeoutException),
)


# ── 3. 노드 정의 ──

def fetch_data_node(state: TaskState) -> dict:
    """외부 API에서 데이터를 가져오는 노드. (Circuit Breaker 적용)"""
    try:
        data = api_circuit.call(
            _call_external_api, state["query"]
        )
        return {"fetched_data": data, "current_step": "fetch_data"}
    except CircuitOpenError as exc:
        return {"error": f"API 서킷 열림: {exc}", "current_step": "fetch_data"}
    except Exception as exc:
        return {"error": f"데이터 가져오기 실패: {exc}", "current_step": "fetch_data"}


@retry_on_llm_error(max_retries=3, base_delay=2.0)
def analyze_node(state: TaskState) -> dict:
    """LLM으로 분석하는 노드. (Retry 데코레이터 적용)"""
    if not state.get("fetched_data"):
        return {"error": "분석할 데이터 없음", "current_step": "analyze"}

    result = call_llm(f"다음을 분석하세요: {state['fetched_data']}")
    return {"analysis": result, "current_step": "analyze"}


def create_ticket_node(state: TaskState) -> dict:
    """외부 시스템에 티켓을 생성하는 노드. (Circuit Breaker 적용)"""
    try:
        key = ticket_circuit.call(
            _create_ticket, state["analysis"]
        )
        return {"ticket_key": key, "current_step": "create_ticket"}
    except CircuitOpenError as exc:
        return {"error": f"티켓 서킷 열림: {exc}", "current_step": "create_ticket"}
    except Exception as exc:
        return {"error": f"티켓 생성 실패: {exc}", "current_step": "create_ticket"}


def handle_error_node(state: TaskState) -> dict:
    """에러를 로깅하고 알림을 보내는 노드."""
    logger.error(f"에러 발생: {state.get('error')}")
    return {"current_step": "handle_error"}


# ── 4. 라우팅 함수 ──

def route_after_fetch(state: TaskState) -> str:
    """fetch 후 라우팅: 에러 -> handle_error, 정상 -> analyze"""
    if state.get("error"):
        return "handle_error"
    return "analyze"


def route_after_analyze(state: TaskState) -> str:
    """analyze 후 라우팅: 에러 -> handle_error, 정상 -> create_ticket"""
    if state.get("error"):
        return "handle_error"
    return "create_ticket"


def route_after_ticket(state: TaskState) -> str:
    """ticket 생성 후 라우팅: 에러 -> handle_error, 정상 -> 종료"""
    if state.get("error"):
        return "handle_error"
    return END


# ── 5. 그래프 구성 ──

graph = StateGraph(TaskState)

# 노드 등록
graph.add_node("fetch_data", fetch_data_node)
graph.add_node("analyze", analyze_node)
graph.add_node("create_ticket", create_ticket_node)
graph.add_node("handle_error", handle_error_node)

# 엣지 연결
graph.set_entry_point("fetch_data")
graph.add_conditional_edges("fetch_data", route_after_fetch)
graph.add_conditional_edges("analyze", route_after_analyze)
graph.add_conditional_edges("create_ticket", route_after_ticket)
graph.add_edge("handle_error", END)

# 컴파일
app = graph.compile()

# 실행
result = app.invoke({
    "query": "버그 리포트 분석",
    "current_step": "start",
})
```

#### 에러 처리 흐름 요약

```
정상 흐름:
  fetch_data ──> analyze ──> create_ticket ──> END

fetch 실패 (Circuit Breaker):
  fetch_data ──> handle_error ──> END
       │
       └─ CircuitOpenError 또는 HTTPError
          -> state["error"] 설정
          -> route_after_fetch가 "handle_error"로 라우팅

LLM 실패 (Retry 데코레이터):
  fetch_data ──> analyze ──> handle_error ──> END
                    │
                    ├─ Retriable: 2초 -> 4초 -> 8초 재시도
                    └─ 3회 모두 실패 or Non-retriable
                       -> state["error"] 설정
                       -> route_after_analyze가 "handle_error"로 라우팅

티켓 생성 실패 (Circuit Breaker):
  fetch_data ──> analyze ──> create_ticket ──> handle_error ──> END
```

---

## 4. 연습 문제

### 연습 1: retry_with_backoff 동작 확인

다음 코드를 실행하고, 재시도 로그를 관찰하세요.

```python
import random
import logging

logging.basicConfig(level=logging.WARNING)

call_count = 0

def flaky_api():
    """70% 확률로 실패하는 API (학습용)."""
    global call_count
    call_count += 1
    if random.random() < 0.7:
        raise ConnectionError(f"서버 연결 실패 (시도 #{call_count})")
    return f"성공! (시도 #{call_count})"

# TODO: retry_with_backoff를 사용하여 flaky_api를 호출하세요
# - max_retries: 5
# - base_delay: 1.0
# - retriable_exceptions: (ConnectionError,)
```

### 연습 2: Circuit Breaker 상태 전이 관찰

```python
# TODO: CircuitBreaker를 생성하고 (failure_threshold=3, recovery_timeout=5),
#       연속 3번 실패시키면 OPEN 상태가 되는지 확인하세요.
#       그 후 5초 대기 후 HALF_OPEN으로 전환되는지 확인하세요.

cb = CircuitBreaker(
    name="test-circuit",
    failure_threshold=3,
    recovery_timeout=5.0,
    expected_exceptions=(ValueError,),
)

def always_fail():
    raise ValueError("항상 실패")

# 3번 실패시키기
for i in range(3):
    try:
        cb.call(always_fail)
    except ValueError:
        print(f"실패 #{i+1}, 상태: {cb.state.value}")

# OPEN 상태 확인
print(f"현재 상태: {cb.state.value}")  # -> "open"

# 5초 대기 후 HALF_OPEN 확인
import time
time.sleep(5)
print(f"5초 후 상태: {cb.state.value}")  # -> "half_open"
```

### 연습 3: 3-노드 에이전트에 에러 처리 추가

다음 에이전트에 retry 데코레이터와 Circuit Breaker를 추가하세요.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END


class QuizState(TypedDict):
    topic: str
    questions: str | None
    score: str | None
    error: str | None


def generate_questions(state: QuizState) -> dict:
    """TODO: @retry_on_llm_error 데코레이터를 추가하세요."""
    questions = call_llm(f"{state['topic']}에 대한 퀴즈 3문제를 만들어주세요")
    return {"questions": questions}


def evaluate_answers(state: QuizState) -> dict:
    """TODO: @retry_on_llm_error 데코레이터를 추가하세요."""
    score = call_llm(f"다음 퀴즈를 채점하세요: {state['questions']}")
    return {"score": score}


def save_result(state: QuizState) -> dict:
    """TODO: Circuit Breaker를 적용하여 DB 저장을 보호하세요."""
    save_to_database(state["score"])
    return {}


# TODO: 라우팅 함수를 만들어서 에러 시 handle_error 노드로 이동하게 하세요
# TODO: handle_error 노드를 추가하세요
# TODO: 그래프를 완성하세요
```

---

## 5. 핵심 정리

### 한눈에 보는 에러 처리 전략

```
┌─────────────────────────────────────────────────────────┐
│                    에러 발생!                             │
│                       │                                  │
│            ┌──────────┴──────────┐                       │
│            │                     │                       │
│     Retriable?              Non-retriable?               │
│       (429, 5xx,              (401, 404,                 │
│        timeout)               파싱 실패)                  │
│            │                     │                       │
│            ▼                     ▼                       │
│     Retry with              state["error"]               │
│     Backoff + Jitter        설정 후 failure               │
│            │                경로로 이동                    │
│            │                                             │
│     연속 실패 많으면?                                     │
│            │                                             │
│            ▼                                             │
│     Circuit Breaker                                      │
│     OPEN (즉시 실패)                                     │
│            │                                             │
│     시간 경과 후                                         │
│            │                                             │
│            ▼                                             │
│     HALF_OPEN                                            │
│     (테스트 요청)                                        │
└─────────────────────────────────────────────────────────┘
```

### 패턴별 요약

| 패턴 | 사용 시점 | 핵심 키워드 |
|------|----------|------------|
| **Exponential Backoff** | 일시적 에러 재시도 | 대기 시간 2배씩 증가 |
| **Jitter** | 다수 클라이언트의 동시 재시도 방지 | 랜덤 지연 |
| **RetryPolicy** | LangGraph 노드 레벨 자동 재시도 | `retry=` 파라미터 |
| **retry_on_llm_error** | Retriable/Non-retriable 분기 | 커스텀 데코레이터 |
| **Circuit Breaker** | 외부 시스템 장애 시 빠른 실패 | CLOSED/OPEN/HALF_OPEN |
| **Graceful Degradation** | DLQ 직행 방지 | `state["error"]` + 조건부 엣지 |

### 기억할 3가지 원칙

1. **모든 에러를 같은 방식으로 처리하지 마세요**: Retriable과 Non-retriable을 구분하세요
2. **실패를 숨기지 말고, 관리하세요**: `state["error"]`에 기록하고 적절한 경로로 이동
3. **빠른 실패가 느린 실패보다 낫습니다**: Circuit Breaker로 불필요한 대기를 줄이세요

---

## 6. 참고 자료

| 자료 | 링크 | 설명 |
|------|------|------|
| LangGraph RetryPolicy | [langgraph.graph.graph.CompiledGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph) | LangGraph 내장 재시도 정책 API 레퍼런스 |
| Circuit Breaker 패턴 | [Azure Architecture Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker) | Microsoft의 Circuit Breaker 패턴 설명 |
| tenacity 라이브러리 | [tenacity.readthedocs.io](https://tenacity.readthedocs.io/en/latest/) | Python 표준 재시도 라이브러리 |
| Exponential Backoff | [Google Cloud - Exponential Backoff](https://cloud.google.com/iot/docs/how-tos/exponential-backoff) | Google의 Exponential Backoff 가이드 |
| Resilience Patterns | [Microsoft - Resiliency patterns](https://learn.microsoft.com/en-us/azure/architecture/framework/resiliency/reliability-patterns) | 회복 탄력성 패턴 모음 |

---

## 다음 단계

이번 모듈에서 에러 처리와 회복 탄력성의 기본 패턴을 배웠습니다. 다음 모듈에서는 에이전트가 의존하는 **외부 시스템(API, 메시지 큐, 데이터베이스)**과의 연동을 깊이 다룹니다.

**다음**: [Module 09: 외부 시스템 연동 - API, 메시지 큐, 데이터베이스](./09-external-system-integration.md)
