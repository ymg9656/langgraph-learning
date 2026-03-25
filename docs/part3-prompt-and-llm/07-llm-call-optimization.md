# Module 07: LLM 호출 최적화 - 비용, 속도, 안정성

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. LLM 비용 구조(입력/출력 토큰, 모델별 가격)를 이해한다
2. 태스크 복잡도에 따라 적절한 모델을 선택하는 ModelRouter를 구현할 수 있다
3. LangChain Callback으로 토큰 사용량을 추적할 수 있다
4. LLM 응답 캐싱으로 중복 호출을 방지할 수 있다
5. 타임아웃과 재시도로 안정적인 LLM 호출을 구현할 수 있다

---

## 사전 지식

- **Module 05**: 프롬프트 엔지니어링 (ChatPromptTemplate, 체인)
- **Module 06**: 구조화된 출력 (Pydantic, with_structured_output)
- **Python 기초**: 클래스, 데코레이터, 딕셔너리, 예외 처리

> **용어 정리**
> - **토큰(Token)**: LLM이 텍스트를 처리하는 최소 단위. "Hello world" = 약 2토큰, "안녕하세요" = 약 3~5토큰.
> - **입력 토큰**: LLM에게 보내는 텍스트의 토큰 수 (프롬프트 + 데이터)
> - **출력 토큰**: LLM이 생성하는 응답의 토큰 수
> - **Callback**: LLM 호출의 시작/종료/에러 시점에 실행되는 후크(hook) 함수
> - **TTL (Time To Live)**: 캐시 데이터의 유효 기간

---

## 1. 개념 설명

### 1.1 LLM 비용 구조 이해

LLM API는 **토큰 단위로 과금**됩니다. 중요한 점은 입력 토큰과 출력 토큰의 가격이 다르다는 것입니다.

#### 토큰이란?

```
영어: "Hello, how are you?" = 5 tokens
한국어: "안녕하세요, 잘 지내시나요?" = 약 12 tokens (한국어는 토큰이 더 많이 필요)

코드: "def calculate(x, y): return x + y" = 약 10 tokens
```

> **경험 법칙**: 영어 1단어 = 약 1.3토큰, 한국어 1글자 = 약 1~2토큰

#### 모델별 가격 비교표 (USD / 1M tokens 기준)

| 모델 | 입력 가격 | 출력 가격 | 특징 | 적합한 용도 |
|------|----------|----------|------|------------|
| Claude Haiku 3.5 | $1.00 | $5.00 | 빠르고 저렴 | 분류, 필터링, 간단한 분석 |
| Claude Sonnet 4 | $3.00 | $15.00 | 균형잡힌 성능 | 코드 생성, 상세 분석 |
| Claude Opus 4 | $15.00 | $75.00 | 최고 성능 | 복잡한 추론, 연구 |
| GPT-4o-mini | $0.15 | $0.60 | 매우 저렴 | 간단한 작업 |
| GPT-4o | $2.50 | $10.00 | 높은 성능 | 범용 |

> **핵심 인사이트**: 출력 토큰이 입력 토큰보다 3~5배 비쌉니다. 따라서 출력 길이를 제어하는 것이 비용 절감의 핵심입니다.

#### 비용 계산 예시

```python
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """LLM 호출 비용을 추정한다 (USD).

    Args:
        model: 모델 이름
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수

    Returns:
        추정 비용 (USD)
    """
    # 모델별 가격표 (USD per 1M tokens)
    pricing = {
        "claude-haiku":  {"input": 1.00, "output": 5.00},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
        "claude-opus":   {"input": 15.00, "output": 75.00},
        "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
        "gpt-4o":        {"input": 2.50, "output": 10.00},
    }

    price = pricing.get(model, pricing["claude-sonnet"])  # 기본값: Sonnet
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return round(cost, 6)

# 예시: Sonnet으로 코드 분석 (입력 2000토큰, 출력 500토큰)
cost = estimate_cost("claude-sonnet", input_tokens=2000, output_tokens=500)
print(f"비용: ${cost}")  # $0.0135

# 같은 작업을 Haiku로 하면?
cost_haiku = estimate_cost("claude-haiku", input_tokens=2000, output_tokens=500)
print(f"Haiku 비용: ${cost_haiku}")  # $0.0045 (67% 절감!)
```

### 1.2 왜 최적화가 필요한가?

모든 LLM 호출에 가장 비싼 모델을 사용하면:

```
에이전트 파이프라인: 분석 -> 생성 -> 검증 (3번 호출)
- 모두 Sonnet: $0.0135 x 3 = $0.0405/건
- 하루 1,000건 처리: $40.50/일 = $1,215/월

최적화 후 (분석은 Haiku, 생성은 Sonnet):
- Haiku + Sonnet + Haiku: $0.0045 + $0.0135 + $0.0045 = $0.0225/건
- 하루 1,000건 처리: $22.50/일 = $675/월 (44% 절감)
```

---

## 2. 단계별 실습

### 2.1 모델 라우팅 (Model Routing)

**핵심 아이디어**: 간단한 작업에는 저렴한 모델을, 복잡한 작업에는 고성능 모델을 사용합니다.

#### 태스크 복잡도별 모델 선택 기준표

| 태스크 유형 | 예시 | 권장 모델 (Tier) | 근거 |
|------------|------|-----------------|------|
| 분류/라우팅 | "이 이슈는 버그인가 기능요청인가?" | Fast (Haiku) | 간단한 판단, 짧은 출력 |
| 요약/추출 | "변경사항을 요약해줘" | Fast (Haiku) | 구조화된 입력 -> 짧은 출력 |
| 코드 분석 (소규모) | "이 함수의 버그를 찾아줘" | Fast (Haiku) | 단일 파일, 짧은 코드 |
| 코드 분석 (대규모) | "이 PR 전체를 분석해줘" | Smart (Sonnet) | 다중 파일, 복잡한 관계 |
| 코드 생성 | "버그를 수정하는 코드를 작성해줘" | Smart (Sonnet) | 정밀한 코드 생성 필요 |
| 테스트 케이스 생성 | "테스트 시나리오를 만들어줘" | Smart (Sonnet) | 창의적 + 구조적 생성 |

#### ModelRouter 클래스 구현

```python
import logging
from dataclasses import dataclass
from enum import Enum
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """모델 등급."""
    FAST = "fast"      # Haiku: 빠르고 저렴
    SMART = "smart"    # Sonnet: 정밀하고 고성능


@dataclass
class RoutingRule:
    """라우팅 규칙."""
    task_name: str              # 태스크 이름
    default_tier: ModelTier     # 기본 모델 등급
    upgrade_threshold: int | None = None  # 이 바이트 이상이면 Smart로 업그레이드


class ModelRouter:
    """태스크 복잡도 기반 LLM 모델 라우터.

    간단한 태스크에는 Fast(저렴) 모델을, 복잡한 태스크에는 Smart(고성능) 모델을 할당한다.

    Usage:
        router = ModelRouter()
        llm = router.get_model("classify")         # -> Haiku
        llm = router.get_model("generate_code")    # -> Sonnet
        llm = router.get_model("analyze", input_size=10000)  # -> 크기가 크면 Sonnet
    """

    def __init__(
        self,
        fast_model: str = "claude-haiku-4-5-20250514",
        smart_model: str = "claude-sonnet-4-5-20250514",
        temperature: float = 0.1,
        rules: list[RoutingRule] | None = None,
    ):
        self._fast_model = fast_model
        self._smart_model = smart_model
        self._temperature = temperature

        # 기본 라우팅 테이블
        default_rules = [
            RoutingRule("classify", ModelTier.FAST),
            RoutingRule("summarize", ModelTier.FAST),
            RoutingRule("analyze", ModelTier.FAST, upgrade_threshold=5120),
            RoutingRule("generate_code", ModelTier.SMART),
            RoutingRule("generate_test", ModelTier.SMART),
            RoutingRule("evaluate", ModelTier.SMART),
        ]
        self._rules = {r.task_name: r for r in (rules or default_rules)}
        self._cache: dict[str, ChatAnthropic] = {}

    def get_model(self, task_name: str, input_size: int = 0) -> ChatAnthropic:
        """태스크에 적합한 LLM 인스턴스를 반환한다.

        Args:
            task_name: 태스크 이름 (라우팅 테이블에서 조회)
            input_size: 입력 데이터 크기 (바이트). 동적 업그레이드 판단용.

        Returns:
            ChatAnthropic 인스턴스
        """
        rule = self._rules.get(task_name)

        if rule is None:
            logger.warning("No routing rule for '%s', using smart model", task_name)
            tier = ModelTier.SMART
        else:
            tier = rule.default_tier
            # 입력 크기가 임계값 초과 시 Smart로 업그레이드
            if (
                tier == ModelTier.FAST
                and rule.upgrade_threshold
                and input_size > rule.upgrade_threshold
            ):
                logger.info(
                    "Upgrading %s: input_size=%d > threshold=%d",
                    task_name, input_size, rule.upgrade_threshold,
                )
                tier = ModelTier.SMART

        model_name = self._fast_model if tier == ModelTier.FAST else self._smart_model

        # 모델 인스턴스 캐싱 (동일 모델은 재사용)
        if model_name not in self._cache:
            self._cache[model_name] = ChatAnthropic(
                model=model_name,
                temperature=self._temperature,
            )

        logger.info("Routed: %s -> %s (%s)", task_name, model_name, tier.value)
        return self._cache[model_name]
```

#### 사용 예시

```python
router = ModelRouter()

# 간단한 분류 -> Haiku (저렴)
classify_llm = router.get_model("classify")
# 코드 생성 -> Sonnet (고성능)
generate_llm = router.get_model("generate_code")
# 작은 코드 분석 -> Haiku
analyze_llm = router.get_model("analyze", input_size=1000)
# 큰 코드 분석 -> Sonnet (자동 업그레이드)
analyze_llm_big = router.get_model("analyze", input_size=10000)
```

#### 비용 절감 추정

| 시나리오 | 전부 Sonnet | ModelRouter 적용 | 절감율 |
|---------|------------|-----------------|-------|
| 분류 (1,000건/일) | $13.50 | $4.50 (Haiku) | -67% |
| 코드 분석 (500건, 평균) | $6.75 | ~$4.00 (혼합) | -41% |
| 코드 생성 (200건/일) | $2.70 | $2.70 (Sonnet) | 0% |
| **합계** | **$22.95/일** | **$11.20/일** | **-51%** |

### 2.2 토큰 사용량 추적

LLM 비용을 관리하려면 먼저 **얼마나 사용하고 있는지** 알아야 합니다.

#### LangChain Callback 메커니즘

LangChain은 LLM 호출의 생명주기에 **Callback**을 걸 수 있습니다:

```
LLM 호출 시작  --->  on_llm_start()   : 시작 시간 기록
       |
       v
LLM 응답 수신  --->  on_llm_end()     : 토큰 수, 비용 계산
       |
       v
LLM 오류 발생  --->  on_llm_error()   : 실패 카운터 증가
```

#### 토큰 카운터 콜백 구현

```python
import time
import logging
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenCounterCallback(BaseCallbackHandler):
    """LLM 호출의 토큰 사용량과 비용을 추적하는 콜백.

    Usage:
        counter = TokenCounterCallback(agent_name="my-agent")
        llm = ChatAnthropic(model="...", callbacks=[counter])
        llm.invoke(...)
        print(counter.total_cost)
    """

    # 모델별 가격표 (USD per 1M tokens)
    PRICING = {
        "claude-haiku":  {"input": 1.00, "output": 5.00},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
        "default":       {"input": 3.00, "output": 15.00},
    }

    def __init__(self, agent_name: str = "unknown"):
        super().__init__()
        self.agent_name = agent_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self._start_time: float | None = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 호출 시작 시 호출됨."""
        self._start_time = time.monotonic()

    def on_llm_end(self, response: LLMResult, **kwargs):
        """LLM 호출 완료 시 호출됨."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0

        # 토큰 사용량 추출 (LLM 프로바이더에 따라 위치가 다를 수 있음)
        usage = {}
        if response.llm_output:
            usage = response.llm_output.get("usage", {})

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # 비용 계산
        model = response.llm_output.get("model_name", "default") if response.llm_output else "default"
        # 모델명에서 가격 키 추출 ("claude-haiku-4-5-..." -> "claude-haiku")
        price_key = "default"
        for key in self.PRICING:
            if key in model:
                price_key = key
                break

        price = self.PRICING[price_key]
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

        # 누적
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_count += 1

        logger.info(
            "[%s] LLM call #%d: input=%d, output=%d, cost=$%.6f, elapsed=%.2fs",
            self.agent_name, self.call_count, input_tokens, output_tokens, cost, elapsed,
        )

    def on_llm_error(self, error, **kwargs):
        """LLM 호출 실패 시 호출됨."""
        logger.error("[%s] LLM call failed: %s", self.agent_name, str(error))

    def get_summary(self) -> dict:
        """사용량 요약을 반환한다."""
        return {
            "agent": self.agent_name,
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }
```

#### 사용법

```python
# 콜백 생성
counter = TokenCounterCallback(agent_name="code-reviewer")

# LLM 생성 시 콜백 주입
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250514",
    temperature=0.1,
    callbacks=[counter],  # 콜백 등록
)

# 체인 실행 (자동으로 콜백이 호출됨)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"language": "python", "code": "..."})

# 사용량 확인
print(counter.get_summary())
# {"agent": "code-reviewer", "calls": 1, "input_tokens": 1523,
#  "output_tokens": 487, "total_cost_usd": 0.011844}
```

#### Prometheus 메트릭 연동 개념

프로덕션 환경에서는 Prometheus + Grafana로 메트릭을 시각화합니다:

```python
# prometheus_client 패키지 사용 (개념 설명)
from prometheus_client import Counter, Histogram

# 메트릭 정의 (애플리케이션 시작 시 1회)
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["agent", "model", "direction"],  # direction: input/output
)
llm_cost_total = Counter(
    "llm_cost_dollars_total",
    "Total LLM cost in USD",
    ["agent", "model"],
)
llm_duration = Histogram(
    "llm_call_duration_seconds",
    "LLM call duration",
    ["agent", "model"],
)

# TokenCounterCallback의 on_llm_end에서 메트릭 기록
# llm_tokens_total.labels(agent="my-agent", model="sonnet", direction="input").inc(1523)
# llm_cost_total.labels(agent="my-agent", model="sonnet").inc(0.011844)
```

### 2.3 LLM 응답 캐싱

동일한 입력에 대해 LLM을 반복 호출하는 것은 비용 낭비입니다. 캐싱으로 이를 방지합니다.

#### 캐싱이 필요한 경우

| 상황 | 캐싱 적합도 | 이유 |
|------|-----------|------|
| 동일 코드를 여러 번 분석 | HIGH | 같은 입력 = 같은 결과 |
| 재처리/재시도 | HIGH | 이전 결과 재사용 가능 |
| 코드 생성 (비결정적) | LOW | 매번 다른 결과가 나올 수 있음 |
| 사용자 대화 | LOW | 맥락이 매번 다름 |

#### 해시 기반 캐시 키 설계

```python
import hashlib
import json


def compute_cache_key(task_name: str, model: str, inputs: dict) -> str:
    """LLM 호출의 캐시 키를 생성한다.

    동일한 태스크/모델/입력이면 같은 키를 반환한다.
    입력 딕셔너리의 키 순서가 달라도 같은 키가 생성된다.
    """
    key_data = {
        "task": task_name,
        "model": model,
        "inputs": inputs,
    }
    serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# 같은 입력 -> 같은 키
key1 = compute_cache_key("analyze", "sonnet", {"code": "def f(): pass", "lang": "python"})
key2 = compute_cache_key("analyze", "sonnet", {"lang": "python", "code": "def f(): pass"})
assert key1 == key2  # 키 순서가 달라도 같은 해시
```

#### 간단한 인메모리 캐시 구현

```python
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMCache:
    """LLM 응답 캐시 (메모리 기반).

    TTL(Time To Live) 기반으로 캐시를 관리하며, 히트율을 추적한다.

    Usage:
        cache = LLMCache(default_ttl=3600)  # 1시간 TTL

        # 캐시 조회
        cached = cache.get("analyze", "sonnet", {"code": "..."})
        if cached:
            return cached  # 캐시 히트!

        # LLM 호출
        result = chain.invoke(...)

        # 캐시 저장
        cache.set("analyze", "sonnet", {"code": "..."}, result)
    """

    def __init__(self, default_ttl: int = 3600, max_entries: int = 500):
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)
        self._hits = 0
        self._misses = 0

    def get(self, task: str, model: str, inputs: dict) -> Any | None:
        """캐시에서 결과를 조회한다. 없거나 만료되면 None."""
        key = compute_cache_key(task, model, inputs)

        if key in self._store:
            value, expires_at = self._store[key]
            if time.time() < expires_at:
                self._hits += 1
                logger.info("Cache HIT: %s (hit_rate=%.1f%%)", task, self.hit_rate * 100)
                return value
            else:
                del self._store[key]  # 만료된 항목 삭제

        self._misses += 1
        return None

    def set(self, task: str, model: str, inputs: dict, result: Any, ttl: int | None = None):
        """결과를 캐시에 저장한다."""
        key = compute_cache_key(task, model, inputs)
        ttl = ttl or self._default_ttl

        # 용량 초과 시 가장 오래된 항목 제거
        if len(self._store) >= self._max_entries:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]

        self._store[key] = (result, time.time() + ttl)
        logger.info("Cache SET: %s, ttl=%ds, entries=%d", task, ttl, len(self._store))

    @property
    def hit_rate(self) -> float:
        """캐시 히트율 (0.0 ~ 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

### 2.4 타임아웃과 재시도

LLM API 호출은 네트워크를 통해 이루어지므로 실패할 수 있습니다.

#### LLM API 호출 실패 유형

| HTTP 코드 | 의미 | 재시도 가능? | 대응 |
|-----------|------|------------|------|
| 429 | Rate Limit Exceeded | O | 대기 후 재시도 |
| 500 | Internal Server Error | O | 즉시 재시도 |
| 503 | Service Unavailable | O | 대기 후 재시도 |
| Timeout | 응답 시간 초과 | O | 타임아웃 늘리거나 재시도 |
| 400 | Bad Request | X | 입력 수정 필요 |
| 401 | Unauthorized | X | API 키 확인 |

#### 기본 재시도 래퍼 구현

```python
import time
import logging
from typing import TypeVar, Callable

logger = logging.getLogger(__name__)
T = TypeVar("T")


class LLMCallError(Exception):
    """LLM 호출 실패 (재시도 소진 후)."""
    def __init__(self, task: str, attempts: int, last_error: Exception):
        super().__init__(f"LLM call failed after {attempts} attempts: [{task}] {last_error}")
        self.task = task
        self.attempts = attempts
        self.last_error = last_error


def resilient_llm_call(
    fn: Callable[[], T],
    task_name: str,
    *,
    max_retries: int = 2,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> T:
    """LLM 호출을 재시도 + 지수 백오프로 보호한다.

    Args:
        fn: LLM 호출 함수 (인자 없는 callable)
        task_name: 태스크 이름 (로깅용)
        max_retries: 최대 재시도 횟수
        initial_delay: 첫 재시도 대기 시간 (초)
        backoff_factor: 대기 시간 배율 (지수 백오프)

    Returns:
        fn()의 반환값

    Raises:
        LLMCallError: 모든 재시도 실패 시
    """
    max_attempts = max_retries + 1
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            start = time.monotonic()
            result = fn()
            elapsed = time.monotonic() - start

            logger.info(
                "[%s] LLM call succeeded: attempt=%d, elapsed=%.2fs",
                task_name, attempt, elapsed,
            )
            return result

        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break

            logger.warning(
                "[%s] LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                task_name, attempt, max_attempts, delay, str(exc),
            )
            time.sleep(delay)
            delay *= backoff_factor  # 1초 -> 2초 -> 4초 (지수 백오프)

    raise LLMCallError(task_name, max_attempts, last_error)
```

#### 사용법

```python
# Before: 실패 시 즉시 에러
result = chain.invoke({"code": "..."})

# After: 2번 재시도, 지수 백오프
result = resilient_llm_call(
    fn=lambda: chain.invoke({"code": "..."}),
    task_name="analyze_code",
    max_retries=2,        # 최대 2번 재시도 (총 3번 시도)
    initial_delay=1.0,    # 첫 재시도: 1초 대기
    backoff_factor=2.0,   # 두 번째 재시도: 2초 대기
)
```

**지수 백오프 시각화:**
```
시도 1: 즉시 실행 -> 실패
  (1초 대기)
시도 2: 재시도 -> 실패
  (2초 대기)
시도 3: 재시도 -> 성공! (또는 LLMCallError 발생)
```

### 2.5 동적 max_tokens

입력 길이에 따라 `max_tokens`를 조정하면 불필요한 출력을 방지하고 비용을 절감할 수 있습니다.

```python
def calculate_max_tokens(input_text: str, task_type: str) -> int:
    """입력 길이와 태스크 유형에 따라 max_tokens를 계산한다.

    Args:
        input_text: 입력 텍스트
        task_type: 태스크 유형 ("classify", "analyze", "generate")

    Returns:
        적절한 max_tokens 값
    """
    input_length = len(input_text)

    # 태스크별 출력 비율
    ratios = {
        "classify": 0.05,     # 분류: 입력의 5% 정도 출력
        "summarize": 0.2,     # 요약: 입력의 20%
        "analyze": 0.3,       # 분석: 입력의 30%
        "generate": 0.5,      # 생성: 입력의 50%
    }

    ratio = ratios.get(task_type, 0.3)
    estimated = int(input_length * ratio)

    # 최소/최대 제한
    min_tokens = 100
    max_tokens = 4096
    return max(min_tokens, min(estimated, max_tokens))

# 사용 예시
short_code = "def f(): pass"
long_code = "..." * 5000  # 매우 긴 코드

print(calculate_max_tokens(short_code, "classify"))   # 100 (최소값)
print(calculate_max_tokens(long_code, "generate"))     # 4096 (최대값)
```

---

## 3. 실전 예제

### 3.1 비용 효율적인 "코드 분석 + 수정 생성" 파이프라인

모든 최적화 기법을 조합한 완전한 파이프라인을 만들어 봅시다:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) 라우터, 캐시, 콜백 초기화
router = ModelRouter()
cache = LLMCache(default_ttl=3600)
counter = TokenCounterCallback(agent_name="code-pipeline")


def analyze_and_fix(code: str, language: str) -> dict:
    """코드를 분석하고 수정을 생성하는 파이프라인.

    Step 1: 코드 분석 (Haiku - 저렴) - 캐싱 적용
    Step 2: 수정 생성 (Sonnet - 고성능)
    """
    inputs = {"code": code, "language": language}

    # --- Step 1: 코드 분석 (Fast tier + 캐싱) ---
    cached_analysis = cache.get("analyze", "haiku", inputs)
    if cached_analysis:
        analysis = cached_analysis
    else:
        analyze_llm = router.get_model("analyze", input_size=len(code))
        analyze_prompt = ChatPromptTemplate.from_messages([
            ("system", "코드의 버그를 분석하세요. JSON으로 응답하세요."),
            ("human", "언어: {language}\n코드:\n{code}"),
        ])
        chain = analyze_prompt | analyze_llm | StrOutputParser()
        analysis = resilient_llm_call(
            fn=lambda: chain.invoke(inputs),
            task_name="analyze",
        )
        cache.set("analyze", "haiku", inputs, analysis)

    # --- Step 2: 수정 생성 (Smart tier) ---
    fix_llm = router.get_model("generate_code")
    fix_prompt = ChatPromptTemplate.from_messages([
        ("system", "분석 결과를 바탕으로 코드를 수정하세요."),
        ("human", "분석 결과:\n{analysis}\n\n원본 코드:\n{code}"),
    ])
    fix_chain = fix_prompt | fix_llm | StrOutputParser()
    fix = resilient_llm_call(
        fn=lambda: fix_chain.invoke({"analysis": analysis, "code": code}),
        task_name="generate_code",
    )

    # --- 결과 ---
    print(f"\n=== 비용 요약 ===")
    print(f"총 호출: {counter.call_count}회")
    print(f"총 비용: ${counter.total_cost:.6f}")
    print(f"캐시 히트율: {cache.hit_rate:.1%}")

    return {"analysis": analysis, "fix": fix}
```

---

## 4. 연습 문제

### 연습 1: ModelRouter를 적용한 2-노드 에이전트 만들기

**과제**: 다음 2단계 파이프라인을 ModelRouter로 최적화하세요.

```python
# 현재: 모든 단계에 동일한 Sonnet 모델 사용
llm = ChatAnthropic(model="claude-sonnet-4-5-20250514")

# Step 1: 이메일 분류 (문의/불만/칭찬/기타)
classify_chain = classify_prompt | llm | StrOutputParser()

# Step 2: 자동 응답 생성
respond_chain = respond_prompt | llm | StrOutputParser()
```

**요구사항:**
1. ModelRouter를 생성하고 적절한 RoutingRule을 정의하세요
2. Step 1(분류)은 Fast 모델, Step 2(응답 생성)는 Smart 모델 사용
3. TokenCounterCallback을 추가하여 각 단계의 비용을 확인하세요
4. Step 1에 LLMCache를 적용하세요 (같은 이메일은 캐싱)

### 연습 2: 재시도 래퍼 테스트

**과제**: `resilient_llm_call`의 동작을 테스트하세요.

```python
# 테스트용: 처음 2번은 실패, 3번째에 성공하는 함수
call_count = 0

def flaky_function():
    global call_count
    call_count += 1
    if call_count < 3:
        raise Exception(f"Simulated failure #{call_count}")
    return "Success!"

# resilient_llm_call로 감싸서 호출하세요
# max_retries=3으로 설정하면 성공해야 합니다
```

---

## 5. 핵심 정리

| 개념 | 핵심 내용 |
|------|----------|
| **토큰 비용** | 출력 토큰이 입력의 3~5배 비쌈. 출력 길이 제어가 핵심 |
| **ModelRouter** | 간단한 작업 = Fast(Haiku), 복잡한 작업 = Smart(Sonnet). 입력 크기로 동적 업그레이드 |
| **Callback** | `on_llm_start/end/error`로 토큰, 비용, 시간 추적 |
| **캐싱** | 동일 입력 반복 호출 방지. 해시 기반 키 + TTL 관리 |
| **재시도** | 지수 백오프로 429/500/timeout 대응. `resilient_llm_call()` |
| **max_tokens** | 입력 길이와 태스크 유형에 따라 동적 조정 |
| **비용 모니터링** | 일/월 예산 설정 + 초과 시 알림 |

---

## 6. 참고 자료

- **Anthropic 모델 가격**: https://docs.anthropic.com/en/docs/about-claude/models
  - Claude 모델별 가격, 컨텍스트 윈도우, 성능 비교.

- **LangChain Callbacks 가이드**: https://python.langchain.com/docs/concepts/callbacks/
  - BaseCallbackHandler, AsyncCallbackHandler 등 콜백 메커니즘 상세 설명.

- **LangChain Caching 가이드**: https://python.langchain.com/docs/how_to/llm_caching/
  - InMemoryCache, SQLiteCache 등 LangChain 내장 캐싱 사용법.

- **OpenAI Tokenizer**: https://platform.openai.com/tokenizer
  - 텍스트를 입력하면 토큰 수를 확인할 수 있는 도구 (Claude도 유사한 토큰화).

---

## 다음 단계

LLM 호출 최적화를 마스터했다면, 다음 모듈에서는 여러 LLM 호출을 **그래프 기반 워크플로우로 연결**하는 방법을 배웁니다.

**Module 08: LangGraph 기초** 에서 다루는 내용:
- 노드와 엣지로 에이전트 워크플로우 구성하기
- StateGraph로 상태 관리하기
- 조건부 분기와 루프 구현하기
