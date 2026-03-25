# Module 09: 외부 시스템 연동

> API, 메시지 큐, 데이터베이스와 안정적으로 통신하는 에이전트

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. 에이전트가 의존하는 외부 시스템의 유형과 특성을 이해한다
2. 동기 vs 비동기 I/O의 차이를 이해하고, asyncio 기본 패턴을 사용할 수 있다
3. 커넥션 풀링을 적용하여 HTTP 클라이언트를 효율적으로 관리할 수 있다
4. Rate Limit 처리를 구현하여 외부 API 호출을 안전하게 관리할 수 있다
5. Health Check 엔드포인트를 설계하여 시스템 상태를 모니터링할 수 있다
6. tenacity 라이브러리로 표준화된 재시도를 구현할 수 있다

---

## 사전 지식

| 주제 | 수준 | 설명 |
|------|------|------|
| Python 기초 | 필수 | 클래스, 데코레이터, 컨텍스트 매니저(`with`) |
| HTTP 기초 | 필수 | GET/POST, 상태 코드, 헤더 |
| Module 08 | 권장 | 에러 처리와 회복 탄력성 기본 개념 |
| asyncio | 선택 | async/await 기초 (이 모듈에서 설명) |

---

## 1. 개념 설명

### 1.1 에이전트의 외부 의존성

AI 에이전트는 단독으로 동작하지 않습니다. 다양한 외부 시스템과 통신하면서 작업을 수행합니다.

```
                    ┌─────────────────┐
                    │   AI 에이전트    │
                    │                 │
                    │  ┌───────────┐  │
                    │  │ LLM 호출  │  │
                    │  │ 노드      │  │
                    │  └───────────┘  │
                    └────┬──┬──┬──┬───┘
                         │  │  │  │
              ┌──────────┘  │  │  └──────────┐
              │             │  │              │
              ▼             ▼  ▼              ▼
        ┌──────────┐  ┌─────┐ ┌─────┐  ┌──────────┐
        │ 메시지 큐 │  │REST │ │ DB  │  │ CLI 도구 │
        │(Redis,   │  │ API │ │     │  │(Git,     │
        │ Kafka)   │  │     │ │     │  │ Docker)  │
        └──────────┘  └─────┘ └─────┘  └──────────┘
```

#### 외부 시스템 유형별 특성

| 시스템 유형 | 예시 | 통신 방식 | 장애 특성 |
|------------|------|----------|----------|
| **메시지 큐** | Redis Streams, Kafka | 비동기 메시지 소비/생산 | 연결 끊김 시 메시지 유실 가능 |
| **REST API** | Jira, GitLab, Slack | HTTP 요청/응답 | Rate Limit, 타임아웃 |
| **데이터베이스** | PostgreSQL, SQLite | SQL 쿼리 | 커넥션 풀 소진, 데드락 |
| **CLI 도구** | Git, Docker | subprocess 실행 | 프로세스 행, 디스크 풀 |

#### 현실적인 문제들

에이전트가 외부 시스템과 통신할 때 흔히 발생하는 문제를 살펴봅시다:

```python
# 문제 1: 매 호출마다 새 연결 생성 (TCP 핸드셰이크 오버헤드)
def call_jira():
    with httpx.Client(timeout=30) as client:  # 매번 새 TCP 연결!
        response = client.post(url, json=payload)
    return response.json()

# 문제 2: 동기 호출로 다른 작업 차단
def process_message():
    data = redis_client.xread(...)  # 여기서 멈춤 (blocking)
    result = call_llm(data)          # 여기서도 멈춤
    save_to_db(result)               # 여기서도 멈춤
    # -> 모든 작업이 순차적으로 기다림

# 문제 3: Rate Limit 무시
def call_api_100_times():
    for i in range(100):
        response = httpx.get(api_url)  # 429 받으면 어떻게?
```

### 1.2 동기 vs 비동기 I/O

#### 동기 호출의 문제: 줄 서서 기다리기

동기 코드에서는 하나의 I/O 작업이 완료될 때까지 **프로그램 전체가 대기**합니다.

```
동기 실행 (하나씩 순서대로):

시간 ──────────────────────────────────────>

Task 1: [API 호출 ████████████]
                                  [완료]
Task 2:                                   [API 호출 ████████████]
                                                                  [완료]
Task 3:                                                                   [API 호출 ████]
                                                                                        [완료]

총 소요: ═══════════════════════════════════════════════════════════════════════════════
         30초 + 30초 + 10초 = 70초
```

#### 비동기 호출: 동시에 기다리기

비동기 코드에서는 하나의 I/O가 대기 중일 때 **다른 작업을 실행**할 수 있습니다.

```
비동기 실행 (동시에 기다리기):

시간 ──────────────────────────>

Task 1: [API 호출 ████████████]  [완료]
Task 2: [API 호출 ████████████]  [완료]
Task 3: [API 호출 ████]          [완료]

총 소요: ══════════════════════
         max(30초, 30초, 10초) = 30초
```

> **비유**: 식당에서 주문하고 기다리는 것(동기)과, 주문 후 다른 볼일을 보다가 진동벨이 울리면 음식을 가져오는 것(비동기)의 차이입니다.

### 1.3 asyncio 기초

Python의 `asyncio`는 비동기 프로그래밍을 위한 표준 라이브러리입니다.

#### 핵심 키워드: async와 await

```python
import asyncio

# 동기 함수 (일반 함수)
def sync_hello():
    return "안녕하세요"

# 비동기 함수 (코루틴)
async def async_hello():
    return "안녕하세요"

# async 함수는 await로 호출
async def main():
    result = await async_hello()
    print(result)

# 실행
asyncio.run(main())
```

#### 비동기로 여러 작업 동시 실행

```python
import asyncio
import time


async def fetch_data(name: str, delay: float) -> str:
    """delay초 동안 대기 후 결과 반환 (API 호출 시뮬레이션)."""
    print(f"[{name}] 요청 시작...")
    await asyncio.sleep(delay)  # 비동기 대기 (다른 작업 실행 가능)
    print(f"[{name}] 응답 수신!")
    return f"{name}: 완료"


async def main():
    start = time.time()

    # 3개의 비동기 작업을 동시에 실행
    results = await asyncio.gather(
        fetch_data("Jira", 2.0),
        fetch_data("GitLab", 1.5),
        fetch_data("Slack", 1.0),
    )

    elapsed = time.time() - start
    print(f"\n결과: {results}")
    print(f"소요 시간: {elapsed:.1f}초")  # 약 2.0초 (가장 긴 작업 기준)


asyncio.run(main())
```

실행 결과:

```
[Jira] 요청 시작...
[GitLab] 요청 시작...
[Slack] 요청 시작...
[Slack] 응답 수신!
[GitLab] 응답 수신!
[Jira] 응답 수신!

결과: ['Jira: 완료', 'GitLab: 완료', 'Slack: 완료']
소요 시간: 2.0초
```

> 3개 작업을 동기로 실행하면 2.0 + 1.5 + 1.0 = 4.5초이지만, 비동기로는 2.0초만 소요됩니다.

---

## 2. 단계별 실습

### 2.1 redis.asyncio로 비동기 메시지 소비

메시지 큐(Redis Streams 등)에서 메시지를 소비할 때, 비동기 방식을 사용하면 메시지 처리 중에도 다음 메시지를 미리 가져올 수 있습니다.

#### 동기 vs 비동기 비교

```python
# === 동기 방식 (문제) ===
from redis import Redis

def sync_consume():
    client = Redis(host="localhost", port=6379)
    while True:
        # 블로킹: 메시지가 올 때까지 이 줄에서 멈춤
        response = client.xreadgroup(
            groupname="my-group",
            consumername="consumer-1",
            streams={"my-stream": ">"},
            count=1,
            block=1000,  # 최대 1초 대기
        )
        if response:
            process(response)  # 처리 중에는 다른 메시지 소비 불가


# === 비동기 방식 (개선) ===
from redis.asyncio import Redis as AsyncRedis, ConnectionPool


async def async_consume():
    # ConnectionPool: 커넥션 재사용 + 자동 재연결
    pool = ConnectionPool(
        host="localhost",
        port=6379,
        decode_responses=True,
        max_connections=10,        # 최대 커넥션 수
        retry_on_timeout=True,     # 타임아웃 시 자동 재연결
        health_check_interval=30,  # 30초마다 PING으로 상태 확인
    )
    client = AsyncRedis(connection_pool=pool)

    while True:
        try:
            # 비동기 대기: 대기 중 다른 작업 실행 가능
            response = await client.xreadgroup(
                groupname="my-group",
                consumername="consumer-1",
                streams={"my-stream": ">"},
                count=1,
                block=1000,
            )
            if response:
                await process(response)
        except Exception as err:
            logger.error(f"메시지 소비 오류: {err}")
            await asyncio.sleep(3)  # 3초 대기 후 재시도
```

### 2.2 커넥션 풀링

#### 매번 새 연결을 만드는 문제

HTTP 요청을 할 때마다 새 연결을 만들면:
1. TCP 3-way handshake (약 10~50ms)
2. TLS handshake (약 50~100ms, HTTPS의 경우)
3. 요청 전송 & 응답 수신
4. 연결 종료

```
매 호출마다 새 연결:

호출 1: [TCP 연결]──[TLS]──[요청/응답]──[연결 종료]
호출 2: [TCP 연결]──[TLS]──[요청/응답]──[연결 종료]  ← 같은 서버인데 또!
호출 3: [TCP 연결]──[TLS]──[요청/응답]──[연결 종료]  ← 또!
         ~~~~~~~~~~  ~~~~~
         매번 50ms+  매번 100ms+ = 오버헤드
```

```
커넥션 풀링 (연결 재사용):

초기:   [TCP 연결]──[TLS]──[연결 풀에 저장]
호출 1: [풀에서 가져오기]──[요청/응답]──[풀에 반환]
호출 2: [풀에서 가져오기]──[요청/응답]──[풀에 반환]  ← 기존 연결 재사용!
호출 3: [풀에서 가져오기]──[요청/응답]──[풀에 반환]  ← 재사용!
         ~~~~~~~~~~~~~~~~~
         거의 0ms
```

#### httpx.AsyncClient 재사용 패턴

```python
import httpx
import logging

logger = logging.getLogger(__name__)


class ManagedHttpClient:
    """라이프사이클이 관리되는 HTTP 클라이언트.

    - 커넥션 풀링으로 TCP/TLS 핸드셰이크 재사용
    - startup/shutdown으로 명시적 생명주기 관리
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_connections: int = 20,
        max_keepalive: int = 10,
    ):
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        """클라이언트를 초기화합니다. 에이전트 시작 시 1번 호출."""
        limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive,
        )
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(self._timeout),
        )
        logger.info(f"HTTP 클라이언트 초기화 (최대 {self._max_connections} 커넥션)")

    async def shutdown(self) -> None:
        """클라이언트를 종료합니다. 에이전트 종료 시 1번 호출."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("HTTP 클라이언트 종료")

    @property
    def client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트를 반환합니다."""
        if self._client is None or self._client.is_closed:
            raise RuntimeError("HTTP 클라이언트가 초기화되지 않았습니다. startup()을 먼저 호출하세요.")
        return self._client


# 사용 예시
http = ManagedHttpClient(timeout=30.0, max_connections=20)


async def main():
    await http.startup()

    try:
        # 같은 클라이언트를 재사용 (커넥션 풀링)
        resp1 = await http.client.get("https://api.example.com/data/1")
        resp2 = await http.client.get("https://api.example.com/data/2")
        resp3 = await http.client.post("https://api.example.com/data", json={...})
    finally:
        await http.shutdown()
```

#### 컨텍스트 매니저 패턴

`async with`를 사용하면 자동으로 리소스를 정리할 수 있습니다.

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def managed_client(timeout: float = 30.0, max_connections: int = 20):
    """HTTP 클라이언트를 컨텍스트 매니저로 관리합니다."""
    limits = httpx.Limits(max_connections=max_connections)
    client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(timeout))
    try:
        yield client
    finally:
        await client.aclose()


# 사용
async def fetch_data():
    async with managed_client() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
    # async with 블록을 벗어나면 자동으로 client.aclose() 호출
```

### 2.3 Rate Limit 핸들링

#### HTTP 429 상태코드의 의미

외부 API는 과도한 요청을 방지하기 위해 **Rate Limit**을 설정합니다. 한도를 초과하면 HTTP 429 (Too Many Requests) 응답을 반환합니다.

```
클라이언트                    API 서버
    │                            │
    │── 요청 1 ──────────────>   │  200 OK
    │── 요청 2 ──────────────>   │  200 OK
    │── 요청 3 ──────────────>   │  200 OK
    │── 요청 4 ──────────────>   │  429 Too Many Requests
    │                            │  Retry-After: 30
    │                            │
    │   (30초 대기)              │
    │                            │
    │── 요청 5 ──────────────>   │  200 OK
```

#### Retry-After 헤더 존중

429 응답에는 보통 `Retry-After` 헤더가 포함됩니다. 이 헤더는 "이 시간 후에 다시 시도하세요"라는 의미입니다.

```python
import asyncio
import httpx
import logging

logger = logging.getLogger(__name__)


async def call_api_with_rate_limit(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> httpx.Response:
    """Rate Limit을 처리하며 API를 호출합니다.

    429 응답 시 Retry-After 헤더를 존중하여 대기 후 재시도합니다.

    Args:
        client: httpx.AsyncClient 인스턴스
        method: HTTP 메서드 ("GET", "POST" 등)
        url: 요청 URL
        max_retries: 최대 재시도 횟수
        **kwargs: httpx 요청 추가 인자 (json, headers 등)

    Returns:
        httpx.Response

    Raises:
        httpx.HTTPStatusError: 재시도 실패 시
    """
    for attempt in range(max_retries + 1):
        response = await client.request(method, url, **kwargs)

        if response.status_code == 429:
            # Retry-After 헤더에서 대기 시간 추출
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                wait_seconds = float(retry_after)
            else:
                # 헤더가 없으면 Exponential Backoff
                wait_seconds = 2 ** attempt

            logger.warning(
                f"Rate Limit 초과 (429). {wait_seconds}초 후 재시도 "
                f"(시도 {attempt + 1}/{max_retries + 1})"
            )

            if attempt < max_retries:
                await asyncio.sleep(wait_seconds)
                continue

        # 429가 아니면 일반 응답 처리
        response.raise_for_status()
        return response

    # 모든 재시도 실패
    response.raise_for_status()
    return response  # 여기에 도달하지 않음
```

#### 토큰 버킷 알고리즘 개념

API 호출 **전에** Rate Limit을 예방하는 방법입니다.

```
토큰 버킷 비유:
- 양동이(버킷)에 토큰이 있음
- API 호출 시 토큰 1개를 소비
- 일정 속도로 토큰이 보충됨
- 토큰이 없으면 대기

┌─────────┐
│ ● ● ● ● │ ← 버킷 (최대 10개)
│ ● ● ● ● │
│ ● ●     │ ← 현재 6개
└─────────┘
     │
     │ 1초마다 2개씩 보충
     ▼
  API 호출 → 토큰 1개 소비
```

```python
import asyncio
import time


class TokenBucket:
    """토큰 버킷 기반 Rate Limiter.

    초당 허용 요청 수를 제한합니다.

    Args:
        rate: 초당 토큰 보충 속도 (예: 10 = 초당 10개)
        capacity: 버킷 최대 용량
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """토큰을 1개 획득합니다. 없으면 대기합니다."""
        async with self._lock:
            await self._refill()

            while self.tokens < 1:
                # 토큰이 보충될 때까지 대기
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                await self._refill()

            self.tokens -= 1

    async def _refill(self) -> None:
        """경과 시간에 따라 토큰을 보충합니다."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate,
        )
        self.last_refill = now


# 사용 예시: 초당 5개 요청으로 제한
rate_limiter = TokenBucket(rate=5, capacity=10)


async def safe_api_call(client: httpx.AsyncClient, url: str):
    """Rate Limit을 지키며 API를 호출합니다."""
    await rate_limiter.acquire()  # 토큰 대기
    return await client.get(url)  # 토큰 획득 후 호출
```

### 2.4 Health Check 엔드포인트

에이전트가 시작할 때, 의존하는 외부 시스템이 정상인지 확인해야 합니다. 또한 운영 중에도 `/health` 엔드포인트로 시스템 상태를 모니터링할 수 있어야 합니다.

```python
import asyncio
import httpx
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # 일부 시스템만 장애
    UNHEALTHY = "unhealthy"


@dataclass
class SystemHealth:
    """개별 시스템의 상태."""
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    error: str | None = None


async def check_redis_health(host: str, port: int) -> SystemHealth:
    """Redis/Valkey 연결 상태를 확인합니다."""
    try:
        from redis.asyncio import Redis
        client = Redis(host=host, port=port)
        start = asyncio.get_event_loop().time()
        await client.ping()
        latency = (asyncio.get_event_loop().time() - start) * 1000
        await client.aclose()
        return SystemHealth("redis", HealthStatus.HEALTHY, latency_ms=latency)
    except Exception as exc:
        return SystemHealth("redis", HealthStatus.UNHEALTHY, error=str(exc))


async def check_api_health(url: str, name: str) -> SystemHealth:
    """REST API의 상태를 확인합니다."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            start = asyncio.get_event_loop().time()
            response = await client.get(url)
            latency = (asyncio.get_event_loop().time() - start) * 1000
            if response.status_code < 500:
                return SystemHealth(name, HealthStatus.HEALTHY, latency_ms=latency)
            return SystemHealth(name, HealthStatus.UNHEALTHY, error=f"HTTP {response.status_code}")
    except Exception as exc:
        return SystemHealth(name, HealthStatus.UNHEALTHY, error=str(exc))


async def check_db_health(db_url: str) -> SystemHealth:
    """데이터베이스 연결 상태를 확인합니다."""
    try:
        # SQLAlchemy 비동기 엔진 사용 시
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            start = asyncio.get_event_loop().time()
            await conn.execute("SELECT 1")
            latency = (asyncio.get_event_loop().time() - start) * 1000
        await engine.dispose()
        return SystemHealth("database", HealthStatus.HEALTHY, latency_ms=latency)
    except Exception as exc:
        return SystemHealth("database", HealthStatus.UNHEALTHY, error=str(exc))


async def run_health_check() -> dict:
    """모든 시스템의 상태를 확인합니다."""
    checks = await asyncio.gather(
        check_redis_health("localhost", 6379),
        check_api_health("https://your-jira.atlassian.net/status", "jira"),
        check_api_health("https://gitlab.example.com/api/v4/version", "gitlab"),
        return_exceptions=True,
    )

    results = []
    for check in checks:
        if isinstance(check, Exception):
            results.append(SystemHealth("unknown", HealthStatus.UNHEALTHY, error=str(check)))
        else:
            results.append(check)

    # 전체 상태 판단
    unhealthy = [r for r in results if r.status == HealthStatus.UNHEALTHY]

    if len(unhealthy) == 0:
        overall = HealthStatus.HEALTHY
    elif len(unhealthy) < len(results):
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.UNHEALTHY

    return {
        "status": overall.value,
        "systems": [
            {
                "name": r.name,
                "status": r.status.value,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }
```

#### FastAPI Health Check 엔드포인트

```python
from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/health")
async def health_check():
    """시스템 상태를 확인하는 엔드포인트."""
    result = await run_health_check()

    # 상태에 따라 HTTP 코드 결정
    if result["status"] == "unhealthy":
        return Response(
            content=json.dumps(result),
            status_code=503,  # Service Unavailable
            media_type="application/json",
        )

    return result
```

응답 예시:

```json
{
  "status": "degraded",
  "systems": [
    {"name": "redis", "status": "healthy", "latency_ms": 1.2, "error": null},
    {"name": "jira", "status": "unhealthy", "latency_ms": null, "error": "Connection timeout"},
    {"name": "gitlab", "status": "healthy", "latency_ms": 45.3, "error": null}
  ]
}
```

### 2.5 tenacity 라이브러리로 표준화된 재시도

`tenacity`는 Python 재시도 로직의 표준 라이브러리입니다. Module 08에서 직접 구현한 재시도를 더 간결하게 작성할 수 있습니다.

#### 설치

```bash
pip install tenacity
```

#### 기본 사용법

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
import httpx

logger = logging.getLogger(__name__)


# === 기본: 3번 재시도, exponential backoff ===
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=30),
)
def simple_api_call():
    response = httpx.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()


# === 특정 예외만 재시도 ===
@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=30),
)
def network_call():
    response = httpx.get("https://api.example.com/data", timeout=10)
    response.raise_for_status()
    return response.json()


# === 재시도 전 로깅 ===
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,  # 최종 실패 시 원래 예외를 다시 발생
)
def logged_api_call():
    response = httpx.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()
```

#### HTTP 상태 코드별 재시도 필터링

```python
from tenacity import retry_if_exception_type

# 재시도 가능한 HTTP 상태 코드
RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# 재시도 불가한 상태 코드
NON_RETRIABLE_STATUS_CODES = {400, 401, 403, 404, 405, 422}


def is_retriable_http_error(exc: BaseException) -> bool:
    """HTTP 에러가 재시도 가능한지 판별합니다."""
    if isinstance(exc, httpx.TimeoutException):
        return True  # 타임아웃은 항상 재시도
    if isinstance(exc, httpx.ConnectError):
        return True  # 연결 오류는 항상 재시도
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRIABLE_STATUS_CODES
    return False


class RetriableHTTPError(retry_if_exception_type):
    """재시도 가능한 HTTP 에러만 필터링하는 tenacity 조건."""

    def __init__(self):
        super().__init__((httpx.HTTPError, httpx.TimeoutException))

    def __call__(self, retry_state) -> bool:
        if not super().__call__(retry_state):
            return False
        exc = retry_state.outcome.exception()
        return is_retriable_http_error(exc)


# 사용: 429, 5xx만 재시도하고, 401/404는 즉시 실패
@retry(
    retry=RetriableHTTPError(),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    reraise=True,
)
def smart_api_call(url: str, payload: dict) -> dict:
    """상태 코드에 따라 선택적으로 재시도하는 API 호출."""
    response = httpx.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()
```

#### tenacity 주요 옵션 요약

| 옵션 | 설명 | 예시 |
|------|------|------|
| `stop_after_attempt(n)` | n번 시도 후 중단 | `stop=stop_after_attempt(3)` |
| `stop_after_delay(n)` | n초 후 중단 | `stop=stop_after_delay(60)` |
| `wait_exponential(...)` | Exponential Backoff | `wait=wait_exponential(min=1, max=30)` |
| `wait_fixed(n)` | 고정 대기 시간 | `wait=wait_fixed(2)` |
| `retry_if_exception_type(...)` | 특정 예외만 재시도 | `retry=retry_if_exception_type(TimeoutError)` |
| `before_sleep_log(...)` | 재시도 전 로깅 | `before_sleep=before_sleep_log(logger, WARNING)` |
| `reraise=True` | 최종 실패 시 원래 예외 발생 | 기본값은 `RetryError` 발생 |

---

## 3. 실전 예제

### httpx.AsyncClient + Rate Limit + Health Check 통합 에이전트

```python
import asyncio
import httpx
import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ── State 정의 ──

def _replace(existing, new):
    return new


class AgentState(TypedDict):
    task_id: Annotated[str, _replace]
    data: Annotated[dict | None, _replace]
    analysis: Annotated[str | None, _replace]
    error: Annotated[str | None, _replace]
    current_step: Annotated[str, _replace]


# ── 공유 리소스 ──

rate_limiter = TokenBucket(rate=5, capacity=10)  # 초당 5요청
http_client: httpx.AsyncClient | None = None


async def init_resources():
    """에이전트 시작 시 공유 리소스를 초기화합니다."""
    global http_client
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20),
        timeout=httpx.Timeout(30.0),
    )
    # Health Check
    health = await run_health_check()
    if health["status"] == "unhealthy":
        raise RuntimeError(f"외부 시스템 장애: {health}")
    logger.info(f"시스템 상태: {health['status']}")


async def cleanup_resources():
    """에이전트 종료 시 리소스를 정리합니다."""
    global http_client
    if http_client:
        await http_client.aclose()


# ── 노드 정의 ──

async def fetch_task_data(state: AgentState) -> dict:
    """외부 API에서 작업 데이터를 가져옵니다."""
    try:
        await rate_limiter.acquire()
        response = await call_api_with_rate_limit(
            http_client, "GET",
            f"https://api.example.com/tasks/{state['task_id']}",
        )
        return {"data": response.json(), "current_step": "fetch"}
    except Exception as exc:
        return {"error": f"데이터 조회 실패: {exc}", "current_step": "fetch"}


async def analyze_data(state: AgentState) -> dict:
    """LLM으로 데이터를 분석합니다. (재시도 포함)"""
    if not state.get("data"):
        return {"error": "분석할 데이터 없음", "current_step": "analyze"}

    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def _call_llm(data):
        # LLM 호출 (예시)
        return f"분석 결과: {data.get('title', '제목 없음')}"

    try:
        result = await _call_llm(state["data"])
        return {"analysis": result, "current_step": "analyze"}
    except Exception as exc:
        return {"error": f"분석 실패: {exc}", "current_step": "analyze"}


async def save_result(state: AgentState) -> dict:
    """결과를 외부 시스템에 저장합니다."""
    try:
        await rate_limiter.acquire()
        response = await call_api_with_rate_limit(
            http_client, "POST",
            "https://api.example.com/results",
            json={"task_id": state["task_id"], "analysis": state["analysis"]},
        )
        return {"current_step": "save"}
    except Exception as exc:
        return {"error": f"저장 실패: {exc}", "current_step": "save"}


# ── 라우팅 ──

def route_on_error(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "next"


# ── 그래프 구성 ──

graph = StateGraph(AgentState)
graph.add_node("fetch", fetch_task_data)
graph.add_node("analyze", analyze_data)
graph.add_node("save", save_result)
graph.add_node("handle_error", lambda state: {"current_step": "error"})

graph.set_entry_point("fetch")
graph.add_conditional_edges("fetch", route_on_error, {"handle_error": "handle_error", "next": "analyze"})
graph.add_conditional_edges("analyze", route_on_error, {"handle_error": "handle_error", "next": "save"})
graph.add_edge("save", END)
graph.add_edge("handle_error", END)

app = graph.compile()
```

---

## 4. 연습 문제

### 연습 1: 비동기 API 호출

다음 동기 코드를 비동기로 변환하세요.

```python
# 현재: 동기 (3개 호출이 순차 실행 = 약 9초)
import httpx

def get_all_data():
    with httpx.Client() as client:
        users = client.get("https://api.example.com/users").json()      # 3초
        projects = client.get("https://api.example.com/projects").json() # 3초
        tickets = client.get("https://api.example.com/tickets").json()   # 3초
    return users, projects, tickets

# TODO: asyncio.gather를 사용하여 3개 호출을 동시에 실행하세요 (약 3초)
# async def get_all_data_async():
#     ...
```

### 연습 2: Rate Limiter 적용

```python
# TODO: TokenBucket을 사용하여 초당 3개 요청으로 제한하면서
#       10개의 API 호출을 하세요.

rate_limiter = TokenBucket(rate=3, capacity=5)

async def batch_api_calls():
    results = []
    for i in range(10):
        # TODO: rate_limiter.acquire()를 호출한 후 API 호출
        pass
    return results
```

### 연습 3: Health Check가 있는 에이전트

```python
# TODO: 에이전트 시작 시 Health Check를 수행하고,
#       모든 시스템이 healthy일 때만 작업을 시작하세요.
#       하나라도 unhealthy이면 30초 후 재확인하세요.

async def start_agent_with_health_check():
    for attempt in range(5):
        health = await run_health_check()
        if health["status"] == "healthy":
            print("모든 시스템 정상! 에이전트를 시작합니다.")
            # TODO: 에이전트 시작
            break
        else:
            print(f"시스템 장애 감지. {30}초 후 재확인... ({attempt + 1}/5)")
            # TODO: 대기 후 재확인
    else:
        print("시스템 복구 실패. 에이전트를 시작할 수 없습니다.")
```

---

## 5. 핵심 정리

### 한눈에 보는 외부 시스템 연동 패턴

| 패턴 | 문제 | 해결 | 도구 |
|------|------|------|------|
| **비동기 I/O** | 동기 호출로 전체 프로그램 차단 | async/await로 동시 실행 | `asyncio`, `redis.asyncio` |
| **커넥션 풀링** | 매번 새 TCP/TLS 연결 생성 | 연결을 재사용 | `httpx.AsyncClient`, `Limits` |
| **Rate Limit** | API 호출 한도 초과 (429) | Retry-After 존중, TokenBucket | `httpx`, 직접 구현 |
| **Health Check** | 장애 시스템에 요청 시도 | 시작 시 및 주기적 상태 확인 | FastAPI `/health` 엔드포인트 |
| **tenacity** | 재시도 로직 중복 구현 | 표준화된 데코레이터 | `@retry` 데코레이터 |

### 기억할 3가지 원칙

1. **연결은 재사용하세요**: 매번 새로 만들지 말고, 커넥션 풀을 사용하세요
2. **외부 시스템의 한도를 존중하세요**: Rate Limit을 초과하면 모두에게 피해가 갑니다
3. **시작 전에 확인하세요**: Health Check로 의존 시스템의 상태를 먼저 점검하세요

---

## 6. 참고 자료

| 자료 | 링크 | 설명 |
|------|------|------|
| httpx 공식 문서 | [python-httpx.org](https://www.python-httpx.org/) | Python 비동기 HTTP 클라이언트 |
| redis-py asyncio | [redis.readthedocs.io/asyncio](https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html) | Redis 비동기 사용 가이드 |
| tenacity | [tenacity.readthedocs.io](https://tenacity.readthedocs.io/en/latest/) | Python 재시도 라이브러리 |
| asyncio 공식 문서 | [docs.python.org/asyncio](https://docs.python.org/3/library/asyncio.html) | Python 비동기 프로그래밍 표준 라이브러리 |
| httpx Limits 설정 | [python-httpx.org/advanced/pool](https://www.python-httpx.org/advanced/#pool-limit-configuration) | 커넥션 풀 설정 가이드 |
| Token Bucket Algorithm | [Wikipedia](https://en.wikipedia.org/wiki/Token_bucket) | 토큰 버킷 알고리즘 설명 |

---

## 다음 단계

이번 모듈에서 외부 시스템과 안정적으로 통신하는 방법을 배웠습니다. 다음 모듈에서는 에이전트가 사용하는 **리소스(메모리, 디스크, LLM 토큰)**를 효율적으로 관리하는 방법을 다룹니다.

**다음**: [Module 10: 리소스 최적화 - 메모리, 디스크, 토큰 관리](./10-resource-optimization.md)
