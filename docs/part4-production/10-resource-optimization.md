# Module 10: 리소스 최적화

> 메모리, 디스크, 토큰을 효율적으로 관리하는 에이전트

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. 에이전트의 3대 리소스(디스크, 메모리, LLM 토큰) 문제를 식별할 수 있다
2. TTL 기반 디스크 관리와 Shallow Clone으로 디스크 사용량을 최적화할 수 있다
3. LangGraph Annotated Reducer 패턴으로 State 복사를 최소화할 수 있다
4. tiktoken으로 정확한 토큰 수를 계산하고, TokenBudget으로 토큰 예산을 관리할 수 있다
5. AST 기반 코드 압축으로 LLM 컨텍스트 윈도우를 효율적으로 활용할 수 있다
6. 노드 간 데이터 프루닝과 청크 분할로 대용량 데이터를 처리할 수 있다

---

## 사전 지식

| 주제 | 수준 | 설명 |
|------|------|------|
| Python 기초 | 필수 | dict, list, 클래스, 타입 힌트 |
| LangGraph 기초 | 필수 | StateGraph, TypedDict, 노드/엣지 |
| Module 08~09 | 권장 | 에러 처리, 외부 시스템 연동 |
| Git 기초 | 선택 | clone, fetch 개념 (디스크 관리 부분) |

---

## 1. 개념 설명

### 1.1 에이전트의 리소스 사용 문제

AI 에이전트는 실행할 때마다 상당한 리소스를 소비합니다. 한두 번 실행할 때는 문제가 없지만, **수십~수백 건을 연속 처리하면** 리소스 문제가 발생합니다.

#### 3대 리소스 문제

```
┌─────────────────────────────────────────────────────┐
│              에이전트 리소스 사용                      │
│                                                      │
│  1. 디스크 문제                                      │
│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│     │clone1│ │clone2│ │clone3│ │clone4│ ...         │
│     │500MB │ │1.2GB │ │800MB │ │1.5GB │            │
│     └──────┘ └──────┘ └──────┘ └──────┘            │
│     → 정리하지 않으면 디스크가 가득 참               │
│                                                      │
│  2. 메모리 문제                                      │
│     Node1: {**state, "result": ...}  ← 전체 복사    │
│     Node2: {**state, "data": ...}    ← 또 전체 복사  │
│     Node3: {**state, "fix": ...}     ← 또 전체 복사  │
│     → 노드가 많을수록 메모리 사용 증가               │
│                                                      │
│  3. LLM 토큰 문제                                    │
│     프롬프트: "다음 코드를 분석하세요:               │
│     [30KB의 코드 전체]"                              │
│     → 불필요한 코드까지 포함해서 토큰 낭비           │
│       (비용 증가 + 컨텍스트 윈도우 낭비)             │
└─────────────────────────────────────────────────────┘
```

| 리소스 | 현실적인 문제 | 결과 |
|--------|-------------|------|
| **디스크** | Git clone이 계속 쌓임, 임시 파일 미삭제 | 디스크 풀 -> 에이전트 정지 |
| **메모리** | 매 노드마다 State를 통째로 복사 | 메모리 2~3배 사용 |
| **LLM 토큰** | 30KB 코드를 그대로 프롬프트에 넣기 | 비용 증가 + 핵심 정보 희석 |

### 1.2 문자 수와 토큰 수는 다릅니다

LLM의 과금과 컨텍스트 윈도우는 **토큰** 단위입니다. 하지만 많은 개발자가 **문자 수**로 계산합니다. 이 둘은 같지 않습니다.

```
영어:  "Hello world"  = 11 문자 = 2 토큰
한글:  "안녕하세요"    =  5 문자 = 5~7 토큰 (한글 1자 ≈ 1~2 토큰)
코드:  "def hello():" = 13 문자 = 4 토큰

→ 문자 수 기반 절삭은 실제 토큰 예산을 정확히 제어할 수 없음!
```

| 절삭 방식 | 방법 | 문제점 |
|----------|------|--------|
| 문자 기반 | `text[:3000]` | 토큰 수를 예측할 수 없음 |
| 토큰 기반 | tiktoken으로 계산 후 절삭 | 정확하지만 의미를 고려하지 않음 |
| 시맨틱 기반 | AST로 중요한 부분만 추출 | 가장 효율적, 구현 복잡 |

---

## 2. 단계별 실습

### 2.1 디스크 관리: TTL 기반 자동 정리

Git clone은 프로젝트에 따라 수백 MB에서 수 GB까지 차지합니다. 에이전트가 작업을 끝낸 후에도 clone이 남아 있으면 디스크가 점점 차게 됩니다.

#### 개념: TTL(Time To Live) 기반 정리

```
TTL = 7일 (기본값)

Day 1: project-A clone (500MB) → 사용 시간 기록
Day 3: project-B clone (1GB)   → 사용 시간 기록
Day 5: project-A 다시 사용     → 사용 시간 갱신
Day 8: project-A → 아직 OK (Day 5에 사용했으므로 TTL 미초과)
        project-B → TTL 초과! (Day 3 이후 미사용) → 삭제
```

#### 구현: CloneManager

```python
import json
import shutil
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CloneMetadata:
    """clone의 메타데이터. 각 clone 디렉토리에 저장됩니다."""
    project_key: str
    last_used: float = field(default_factory=time.time)
    clone_size_bytes: int = 0
    clone_type: str = "full"  # "full" 또는 "shallow"

    def to_dict(self) -> dict:
        return {
            "project_key": self.project_key,
            "last_used": self.last_used,
            "clone_size_bytes": self.clone_size_bytes,
            "clone_type": self.clone_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CloneMetadata":
        return cls(**data)


class CloneManager:
    """Git clone의 생명주기를 관리합니다.

    기능:
      - TTL 기반 자동 정리 (기본 7일 미사용 시 삭제)
      - 최대 디스크 사용량 제한 (기본 10GB)
      - 사용 시간 추적

    사용법:
        manager = CloneManager("/data/repos", ttl_seconds=7*24*3600)
        manager.touch("my-project")       # 사용 시간 갱신
        manager.cleanup_expired()         # TTL 지난 clone 삭제
        manager.enforce_disk_limit()      # 디스크 한도 초과 시 정리
    """

    DEFAULT_TTL_SECONDS = 7 * 24 * 3600       # 7일
    DEFAULT_MAX_DISK_BYTES = 10 * 1024**3      # 10GB
    META_FILENAME = ".clone_meta.json"

    def __init__(
        self,
        base_dir: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_disk_bytes: int = DEFAULT_MAX_DISK_BYTES,
    ):
        self.base_dir = Path(base_dir)
        self.ttl_seconds = ttl_seconds
        self.max_disk_bytes = max_disk_bytes

    def touch(self, project_key: str) -> None:
        """clone 사용 시간을 갱신합니다. 작업 시작 시 호출하세요."""
        meta_path = self.base_dir / project_key / self.META_FILENAME
        meta = self._load_meta(meta_path, project_key)
        meta.last_used = time.time()
        meta.clone_size_bytes = self._get_dir_size(self.base_dir / project_key)
        self._save_meta(meta_path, meta)

    def cleanup_expired(self) -> list[str]:
        """TTL이 지난 clone을 삭제합니다.

        Returns:
            삭제된 project_key 목록
        """
        removed = []
        now = time.time()

        if not self.base_dir.exists():
            return removed

        for repo_dir in self.base_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            meta_path = repo_dir / self.META_FILENAME
            meta = self._load_meta(meta_path, repo_dir.name)

            age_hours = (now - meta.last_used) / 3600
            if now - meta.last_used > self.ttl_seconds:
                logger.info(
                    f"TTL 초과 clone 삭제: {repo_dir.name} "
                    f"({age_hours:.0f}시간 미사용)"
                )
                shutil.rmtree(repo_dir, ignore_errors=True)
                removed.append(repo_dir.name)

        return removed

    def enforce_disk_limit(self) -> list[str]:
        """디스크 제한 초과 시 가장 오래된 clone부터 삭제합니다.

        Returns:
            삭제된 project_key 목록
        """
        removed = []

        if not self.base_dir.exists():
            return removed

        # 모든 clone의 메타데이터 수집
        clones: list[tuple[CloneMetadata, Path]] = []
        for repo_dir in self.base_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            meta_path = repo_dir / self.META_FILENAME
            meta = self._load_meta(meta_path, repo_dir.name)
            meta.clone_size_bytes = self._get_dir_size(repo_dir)
            clones.append((meta, repo_dir))

        total_bytes = sum(m.clone_size_bytes for m, _ in clones)

        if total_bytes <= self.max_disk_bytes:
            return removed

        # 가장 오래된 것부터 삭제 (LRU: Least Recently Used)
        clones.sort(key=lambda x: x[0].last_used)

        for meta, repo_dir in clones:
            if total_bytes <= self.max_disk_bytes:
                break
            size_mb = meta.clone_size_bytes / (1024 * 1024)
            logger.info(f"디스크 한도 초과로 삭제: {meta.project_key} ({size_mb:.0f}MB)")
            total_bytes -= meta.clone_size_bytes
            shutil.rmtree(repo_dir, ignore_errors=True)
            removed.append(meta.project_key)

        return removed

    # ── 내부 메서드 ──

    def _load_meta(self, meta_path: Path, project_key: str) -> CloneMetadata:
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                return CloneMetadata.from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass
        return CloneMetadata(project_key=project_key)

    def _save_meta(self, meta_path: Path, meta: CloneMetadata) -> None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta.to_dict()))

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """디렉토리 전체 크기를 바이트 단위로 계산합니다."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return total
```

#### 사용 예시

```python
# 에이전트 초기화 시
manager = CloneManager(
    base_dir="/data/repos",
    ttl_seconds=7 * 24 * 3600,     # 7일
    max_disk_bytes=10 * 1024**3,    # 10GB
)

# 작업 시작 시
manager.touch("my-project")

# 주기적으로 정리 (매 시간 또는 에이전트 시작 시)
expired = manager.cleanup_expired()
if expired:
    print(f"TTL 만료로 삭제된 프로젝트: {expired}")

over_limit = manager.enforce_disk_limit()
if over_limit:
    print(f"디스크 한도 초과로 삭제된 프로젝트: {over_limit}")
```

### 2.2 Shallow Clone으로 디스크/시간 절약

대부분의 에이전트는 **최신 코드만** 필요합니다. Git의 전체 히스토리(모든 커밋, 모든 브랜치)를 가져올 필요가 없습니다.

```
Full Clone (기본):
  모든 커밋 히스토리 + 모든 브랜치
  → 500MB ~ 2GB (프로젝트에 따라)
  → 30초 ~ 120초 소요

Shallow Clone (--depth 1):
  최신 커밋 1개만
  → 50MB ~ 200MB (80~90% 절감)
  → 5초 ~ 15초 소요
```

#### Shallow Clone 명령어

```bash
# Full Clone (기본)
git clone https://github.com/user/repo.git

# Shallow Clone (최신 커밋 1개만)
git clone --depth 1 --single-branch https://github.com/user/repo.git
```

#### 코드에서 활용

```python
import subprocess
from pathlib import Path


def ensure_repo(
    project_key: str,
    git_url: str,
    branch: str = "main",
    base_dir: str = "/data/repos",
    shallow: bool = True,  # 기본값을 shallow로!
) -> Path:
    """리포지토리를 clone 또는 update합니다.

    Args:
        project_key: 프로젝트 식별자
        git_url: Git URL
        branch: 대상 브랜치
        base_dir: 기본 디렉토리
        shallow: True면 --depth 1로 shallow clone

    Returns:
        리포지토리 경로
    """
    repo_path = Path(base_dir) / project_key

    if (repo_path / ".git").exists():
        # 이미 있으면 fetch + reset
        fetch_args = ["git", "fetch", "origin", "--prune"]
        if shallow:
            fetch_args.extend(["--depth", "1"])
        subprocess.run(fetch_args, cwd=repo_path, check=True)
        subprocess.run(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=repo_path, check=True,
        )
    else:
        # 새로 clone
        repo_path.mkdir(parents=True, exist_ok=True)
        clone_args = ["git", "clone", "-b", branch]
        if shallow:
            clone_args.extend(["--depth", "1", "--single-branch"])
        clone_args.extend([git_url, str(repo_path)])
        subprocess.run(clone_args, check=True)

    return repo_path
```

#### Shallow Clone 효과 비교

| 항목 | Full Clone | Shallow Clone | 절감률 |
|------|-----------|---------------|--------|
| 일반적인 프로젝트 | 500MB ~ 2GB | 50MB ~ 200MB | 80~90% |
| Clone 소요 시간 | 30 ~ 120초 | 5 ~ 15초 | 75~85% |
| Fetch 시간 | 10 ~ 30초 | 3 ~ 8초 | 60~70% |

### 2.3 State 복사 최소화: LangGraph Annotated Reducer

#### 문제: `{**state, "field": value}` 패턴

LangGraph 에이전트에서 가장 흔한 패턴이지만, **비효율적**입니다.

```python
# 현재 패턴: 매 노드마다 전체 state를 복사
def my_node(state: MyState) -> MyState:
    result = do_something()
    return {
        **state,                    # ← 전체 state를 복사! (모든 필드)
        "result": result,           # ← 실제로 변경한 건 이 필드뿐
        "current_step": "my_node",
    }
```

**왜 문제인가?**

State에 대용량 필드(코드 30KB, diff 30KB 등)가 있으면, 노드를 거칠 때마다 이 데이터가 복사됩니다.

```
State 구조:
  - query: "분석해주세요" (50 bytes)
  - code_context: [30KB의 코드]      ← 대용량!
  - diff_text: [30KB의 diff]          ← 대용량!
  - result: "분석 결과" (500 bytes)

8개 노드를 거치면:
  30KB x 2필드 x 8노드 = 약 480KB의 불필요한 복사
```

#### 해결: Annotated Reducer 패턴

LangGraph는 `Annotated` 타입과 reducer 함수를 사용하면, **노드가 반환한 필드만 State에 병합**합니다.

```python
from typing import Annotated, TypedDict


def _replace(existing, new):
    """기존 값을 새 값으로 교체하는 reducer.

    LangGraph가 이 함수를 사용하여 state를 업데이트합니다.
    노드가 이 필드를 반환하면 -> 기존 값을 새 값으로 교체
    노드가 이 필드를 반환하지 않으면 -> 기존 값 유지 (복사 없음!)
    """
    return new


# ── Before: 일반 TypedDict ──
class OldState(TypedDict):
    query: str
    code_context: str | None     # 30KB
    diff_text: str | None        # 30KB
    result: str | None
    current_step: str
    error: str | None


# ── After: Annotated Reducer 적용 ──
class NewState(TypedDict):
    query: Annotated[str, _replace]
    code_context: Annotated[str | None, _replace]      # 30KB, 복사 안 됨!
    diff_text: Annotated[str | None, _replace]          # 30KB, 복사 안 됨!
    result: Annotated[str | None, _replace]
    current_step: Annotated[str, _replace]
    error: Annotated[str | None, _replace]
```

#### 노드 반환값 변경

```python
# ── Before: 전체 state 복사 ──
def analyze_node(state: OldState) -> OldState:
    result = do_analysis(state["code_context"])
    return {
        **state,                           # ← 전체 복사 (code_context 30KB 포함!)
        "result": result,
        "current_step": "analyze",
    }


# ── After: 변경 필드만 반환 ──
def analyze_node(state: NewState) -> dict:
    result = do_analysis(state["code_context"])
    return {
        "result": result,                  # ← 변경한 필드만!
        "current_step": "analyze",
    }
    # code_context(30KB)와 diff_text(30KB)는 복사되지 않음!
```

#### 리스트 필드에 항목 추가하기

`operator.add`를 reducer로 사용하면, 리스트에 항목을 **누적 추가**할 수 있습니다.

```python
import operator
from typing import Annotated, TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]  # 리스트 누적
    current_step: Annotated[str, _replace]


# Node 1
def greet_node(state: ChatState) -> dict:
    return {"messages": ["안녕하세요!"]}  # ["안녕하세요!"]

# Node 2
def ask_node(state: ChatState) -> dict:
    return {"messages": ["무엇을 도와드릴까요?"]}
    # -> 기존 messages + 새 messages = ["안녕하세요!", "무엇을 도와드릴까요?"]

# Node 3
def respond_node(state: ChatState) -> dict:
    return {"messages": ["네, 알겠습니다."]}
    # -> ["안녕하세요!", "무엇을 도와드릴까요?", "네, 알겠습니다."]
```

#### 기대 효과

| 에이전트 예시 | 노드 수 | 전체 복사 | Annotated Reducer | 절감 |
|-------------|---------|----------|-------------------|------|
| 분석 에이전트 (8 노드) | 8 | ~240KB | ~30KB | 87% |
| 테스트 에이전트 (13 노드) | 13 | ~390KB | ~45KB | 88% |
| 코드 수정 에이전트 (11 노드) | 11 | ~450KB | ~50KB | 89% |

### 2.4 토큰 최적화: tiktoken

#### 설치

```bash
pip install tiktoken
```

#### 기본 사용법

```python
import tiktoken

# 인코더 생성 (cl100k_base는 GPT-4, Claude와 호환되는 근사 인코딩)
enc = tiktoken.get_encoding("cl100k_base")

# 토큰 수 계산
text = "Hello, world!"
tokens = enc.encode(text)
print(f"텍스트: {text}")
print(f"토큰 수: {len(tokens)}")
print(f"토큰 목록: {tokens}")

# 한글 토큰 수 확인
korean_text = "안녕하세요, 세상!"
korean_tokens = enc.encode(korean_text)
print(f"\n텍스트: {korean_text}")
print(f"문자 수: {len(korean_text)}")
print(f"토큰 수: {len(korean_tokens)}")  # 문자 수보다 많음!

# 코드 토큰 수 확인
code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
code_tokens = enc.encode(code)
print(f"\n코드 문자 수: {len(code)}")
print(f"코드 토큰 수: {len(code_tokens)}")
```

#### TokenBudget 클래스 구현

토큰 예산을 관리하는 유틸리티입니다. "남은 토큰 예산 내에서 가능한 많은 데이터를 포함"하는 전략입니다.

```python
import tiktoken


class TokenBudget:
    """토큰 예산 관리.

    LLM에 보낼 프롬프트의 토큰 수를 정확하게 관리합니다.

    참고: Claude 모델은 tiktoken의 cl100k_base와 정확히 일치하지 않지만,
    실용적인 근사치로 충분합니다 (오차 약 10%).

    사용법:
        budget = TokenBudget(max_tokens=8000)
        budget.consume(system_prompt)           # 시스템 프롬프트 소비
        budget.consume(user_query)              # 사용자 질문 소비
        code = budget.truncate_to_budget(code)  # 남은 예산에 맞춰 코드 절삭
    """

    def __init__(self, model: str = "cl100k_base", max_tokens: int = 8000):
        self._enc = tiktoken.get_encoding(model)
        self.max_tokens = max_tokens
        self._used = 0

    def count(self, text: str) -> int:
        """텍스트의 토큰 수를 반환합니다."""
        return len(self._enc.encode(text))

    @property
    def remaining(self) -> int:
        """남은 토큰 예산."""
        return max(0, self.max_tokens - self._used)

    @property
    def used(self) -> int:
        """사용한 토큰 수."""
        return self._used

    def consume(self, text: str) -> str | None:
        """예산 내에서 텍스트를 소비합니다.

        Args:
            text: 소비할 텍스트

        Returns:
            예산 내이면 원본 텍스트, 초과 시 None
        """
        tokens = self.count(text)
        if tokens <= self.remaining:
            self._used += tokens
            return text
        return None

    def truncate_to_budget(self, text: str, reserve: int = 100) -> str:
        """남은 예산에 맞게 텍스트를 토큰 단위로 절삭합니다.

        Args:
            text: 절삭할 텍스트
            reserve: 여유분으로 남길 토큰 수 (응답 공간 등)

        Returns:
            예산에 맞게 절삭된 텍스트
        """
        available = self.remaining - reserve
        if available <= 0:
            return ""

        tokens = self._enc.encode(text)
        if len(tokens) <= available:
            self._used += len(tokens)
            return text

        # 토큰 단위로 절삭
        truncated = self._enc.decode(tokens[:available])
        self._used += available
        return truncated + "\n... [토큰 한도로 절삭됨]"
```

#### TokenBudget 사용 예시

```python
# 총 8000 토큰 예산으로 프롬프트 구성
budget = TokenBudget(max_tokens=8000)

# 1. 시스템 프롬프트 (약 200 토큰)
system_prompt = "당신은 코드 분석 전문가입니다. 다음 코드를 분석하세요."
budget.consume(system_prompt)
print(f"시스템 프롬프트 후 남은 토큰: {budget.remaining}")

# 2. 사용자 질문 (약 50 토큰)
user_query = "이 코드에서 버그를 찾아주세요."
budget.consume(user_query)
print(f"사용자 질문 후 남은 토큰: {budget.remaining}")

# 3. 코드 컨텍스트 (대용량 - 예산에 맞게 절삭)
large_code = open("big_file.py").read()  # 10000 토큰짜리 코드
truncated_code = budget.truncate_to_budget(large_code, reserve=500)
print(f"코드 추가 후 남은 토큰: {budget.remaining}")
print(f"코드가 절삭되었는가: {'절삭됨' if '절삭' in truncated_code else '전체 포함'}")
```

### 2.5 시맨틱 코드 압축: AST 기반 추출

코드를 LLM에 보낼 때, 전체 코드 대신 **구조적으로 중요한 부분만** 추출하면 토큰을 크게 절약할 수 있습니다.

#### 압축 레벨

```
Level 0: 전체 코드 (토큰 예산 충분할 때)
Level 1: import 제거 + 주석 제거
Level 2: 함수/메서드 시그니처 + 어노테이션만 (본문 생략)
Level 3: 클래스/함수 선언 목록만
```

#### Python AST를 활용한 코드 요약

```python
import ast
from dataclasses import dataclass


@dataclass
class CodeSummary:
    """코드 파일의 구조적 요약."""
    file_path: str
    classes: list[str]
    functions: list[str]
    full_token_count: int
    summary_token_count: int


class PythonCodeCompressor:
    """Python 코드의 구조적 압축.

    AST(Abstract Syntax Tree)를 사용하여 코드의 구조를 추출합니다.
    전체 코드 대신 클래스/함수 시그니처만 추출하면 토큰을 대폭 절약할 수 있습니다.
    """

    def extract_signatures(self, source_code: str) -> str:
        """Python 코드에서 클래스와 함수 시그니처를 추출합니다.

        Args:
            source_code: Python 소스 코드 문자열

        Returns:
            시그니처만 포함된 요약 문자열

        Example:
            입력:
                class Calculator:
                    def add(self, a: int, b: int) -> int:
                        '''두 수를 더합니다.'''
                        result = a + b
                        logger.info(f"Adding {a} + {b}")
                        return result

            출력:
                class Calculator:
                    def add(self, a: int, b: int) -> int:
                        '''두 수를 더합니다.'''
                        ...
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return source_code  # 파싱 실패 시 원본 반환

        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 클래스 선언
                decorators = self._get_decorators(node)
                lines.extend(decorators)
                lines.append(f"class {node.name}({self._get_bases(node)}):")

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 함수/메서드 시그니처
                decorators = self._get_decorators(node)
                lines.extend(decorators)
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                args = self._get_args(node)
                returns = self._get_return_annotation(node)
                sig = f"    {prefix} {node.name}({args}){returns}:"

                # docstring 포함
                docstring = ast.get_docstring(node)
                if docstring:
                    lines.append(sig)
                    lines.append(f"        '''{docstring}'''")
                    lines.append("        ...")
                else:
                    lines.append(sig)
                    lines.append("        ...")

        return "\n".join(lines)

    def _get_decorators(self, node) -> list[str]:
        """데코레이터 문자열 목록을 반환합니다."""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(f"    @{dec.id}")
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"    @{ast.dump(dec)}")
        return decorators

    def _get_bases(self, node: ast.ClassDef) -> str:
        """클래스의 부모 클래스 목록을 반환합니다."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr)
        return ", ".join(bases)

    def _get_args(self, node) -> str:
        """함수 인자를 문자열로 반환합니다."""
        args = []
        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg.arg}{annotation}")
        return ", ".join(args)

    def _get_return_annotation(self, node) -> str:
        """반환 타입 어노테이션을 반환합니다."""
        if node.returns:
            return f" -> {ast.unparse(node.returns)}"
        return ""
```

#### 압축 효과 시연

```python
original_code = '''
class DataProcessor:
    """데이터를 처리하는 클래스."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self._initialized = False

    def process(self, data: list[dict]) -> list[dict]:
        """데이터를 처리합니다."""
        results = []
        for item in data:
            validated = self._validate(item)
            if validated:
                transformed = self._transform(validated)
                results.append(transformed)
        self.logger.info(f"Processed {len(results)} items")
        return results

    def _validate(self, item: dict) -> dict | None:
        """항목을 검증합니다."""
        if "id" not in item:
            self.logger.warning("Missing id field")
            return None
        if "value" not in item:
            self.logger.warning("Missing value field")
            return None
        return item

    def _transform(self, item: dict) -> dict:
        """항목을 변환합니다."""
        return {
            "id": item["id"],
            "processed_value": item["value"] * 2,
            "timestamp": time.time(),
        }
'''

compressor = PythonCodeCompressor()
summary = compressor.extract_signatures(original_code)
print(summary)

# 출력:
# class DataProcessor():
#     def __init__(self, config: dict):
#         ...
#     def process(self, data: list[dict]) -> list[dict]:
#         '''데이터를 처리합니다.'''
#         ...
#     def _validate(self, item: dict) -> dict | None:
#         '''항목을 검증합니다.'''
#         ...
#     def _transform(self, item: dict) -> dict:
#         '''항목을 변환합니다.'''
#         ...

budget = TokenBudget(max_tokens=8000)
print(f"원본 토큰 수: {budget.count(original_code)}")  # ~200
print(f"요약 토큰 수: {budget.count(summary)}")         # ~60 (70% 절감)
```

### 2.6 노드 간 데이터 프루닝

후속 노드에서 더 이상 필요 없는 대용량 필드를 명시적으로 제거합니다.

```python
def cleanup_after_analysis(state: AnalysisState) -> dict:
    """분석 완료 후 대용량 원본 데이터를 정리합니다.

    analyze 노드가 끝나면 원본 diff_text(30KB)와
    raw_code(30KB)는 더 이상 필요 없습니다.
    결과(analysis_result)만 남기고 원본을 제거합니다.
    """
    return {
        "diff_text": None,      # 30KB 해제
        "raw_code": None,       # 30KB 해제
        "current_step": "cleanup",
    }

# 그래프에서 분석 노드 후에 정리 노드 추가
# analyze -> cleanup -> generate -> ...
graph.add_edge("analyze", "cleanup")
graph.add_edge("cleanup", "generate")
```

### 2.7 대용량 데이터 청크 분할

30KB짜리 diff를 한 번에 LLM에 보내면 토큰 한도를 초과할 수 있습니다. 파일 단위로 분할하여 처리합니다.

```python
def split_diff_by_file(diff_text: str) -> list[dict]:
    """Git diff를 파일 단위로 분할합니다.

    Args:
        diff_text: 전체 diff 문자열

    Returns:
        [{"file_path": "path/to/file.py", "diff": "--- a/...\n+++ b/...\n@@..."}]
    """
    chunks = []
    current_file = None
    current_lines = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            # 이전 파일 저장
            if current_file:
                chunks.append({
                    "file_path": current_file,
                    "diff": "\n".join(current_lines),
                })
            # 새 파일 시작
            parts = line.split(" b/")
            current_file = parts[-1] if len(parts) > 1 else "unknown"
            current_lines = [line]
        else:
            current_lines.append(line)

    # 마지막 파일
    if current_file:
        chunks.append({
            "file_path": current_file,
            "diff": "\n".join(current_lines),
        })

    return chunks


def analyze_diff_in_chunks(
    diff_text: str,
    budget: TokenBudget,
    call_llm,
) -> list[dict]:
    """diff를 파일 단위로 분할하여 LLM으로 분석합니다.

    Args:
        diff_text: 전체 diff 문자열
        budget: TokenBudget 인스턴스
        call_llm: LLM 호출 함수

    Returns:
        파일별 분석 결과 리스트
    """
    chunks = split_diff_by_file(diff_text)
    results = []

    for chunk in chunks:
        # 각 파일의 diff가 예산 내인지 확인
        chunk_text = chunk["diff"]

        if budget.remaining < 500:
            # 예산 부족 - 더 이상 분석 불가
            results.append({
                "file_path": chunk["file_path"],
                "analysis": "[토큰 예산 부족으로 건너뜀]",
            })
            continue

        # 예산에 맞게 절삭
        truncated = budget.truncate_to_budget(chunk_text, reserve=200)

        # LLM으로 분석
        analysis = call_llm(
            f"다음 파일의 변경사항을 분석하세요:\n\n"
            f"파일: {chunk['file_path']}\n"
            f"변경사항:\n{truncated}"
        )

        results.append({
            "file_path": chunk["file_path"],
            "analysis": analysis,
        })

    return results
```

---

## 3. 실전 예제

### TokenBudget + Annotated Reducer 적용한 에이전트

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END


def _replace(existing, new):
    return new


class OptimizedState(TypedDict):
    """최적화된 에이전트 State (Annotated Reducer 적용)."""
    query: Annotated[str, _replace]
    code_files: Annotated[list[dict] | None, _replace]  # 대용량 가능
    analysis: Annotated[str | None, _replace]
    suggestions: Annotated[list[str], operator.add]     # 누적 추가
    current_step: Annotated[str, _replace]
    error: Annotated[str | None, _replace]


# ── TokenBudget 활용 노드 ──

def prepare_context_node(state: OptimizedState) -> dict:
    """코드 파일들을 토큰 예산에 맞게 준비합니다."""
    budget = TokenBudget(max_tokens=6000)

    # 시스템 프롬프트 공간 확보
    budget.consume("시스템 프롬프트와 질문을 위한 공간 확보")

    code_files = state.get("code_files") or []
    compressor = PythonCodeCompressor()

    optimized_files = []
    for file_info in code_files:
        content = file_info["content"]
        path = file_info["path"]

        # 1. 원본이 예산 내이면 그대로 사용
        if budget.consume(content) is not None:
            optimized_files.append({"path": path, "content": content})
            continue

        # 2. 시그니처만 추출
        summary = compressor.extract_signatures(content)
        if budget.consume(summary) is not None:
            optimized_files.append({
                "path": path,
                "content": summary,
                "compressed": True,
            })
            continue

        # 3. 예산 부족 - 건너뜀
        optimized_files.append({
            "path": path,
            "content": f"# 토큰 예산 부족으로 생략 (원본 {budget.count(content)} 토큰)",
            "skipped": True,
        })

    return {
        "code_files": optimized_files,  # 최적화된 파일 목록으로 교체
        "current_step": "prepare_context",
    }


def analyze_node(state: OptimizedState) -> dict:
    """LLM으로 코드를 분석합니다."""
    code_context = "\n\n".join(
        f"# {f['path']}\n{f['content']}"
        for f in (state.get("code_files") or [])
    )

    result = call_llm(
        f"다음 코드를 분석하세요:\n\n"
        f"질문: {state['query']}\n\n"
        f"코드:\n{code_context}"
    )

    return {
        "analysis": result,
        "suggestions": ["코드 분석 완료"],  # 누적됨 (operator.add)
        "current_step": "analyze",
    }


def cleanup_node(state: OptimizedState) -> dict:
    """분석 완료 후 대용량 데이터를 정리합니다."""
    return {
        "code_files": None,     # 대용량 데이터 해제
        "suggestions": ["정리 완료"],
        "current_step": "cleanup",
    }


# ── 그래프 구성 ──

graph = StateGraph(OptimizedState)
graph.add_node("prepare_context", prepare_context_node)
graph.add_node("analyze", analyze_node)
graph.add_node("cleanup", cleanup_node)

graph.set_entry_point("prepare_context")
graph.add_edge("prepare_context", "analyze")
graph.add_edge("analyze", "cleanup")
graph.add_edge("cleanup", END)

app = graph.compile()

# 실행
result = app.invoke({
    "query": "이 코드에서 성능 문제를 찾아주세요",
    "code_files": [
        {"path": "main.py", "content": "...대용량 코드..."},
        {"path": "utils.py", "content": "...대용량 코드..."},
    ],
    "suggestions": [],
    "current_step": "start",
})

print(f"분석 결과: {result['analysis']}")
print(f"제안 사항: {result['suggestions']}")
# -> ["코드 분석 완료", "정리 완료"]  (operator.add로 누적됨)
```

---

## 4. 연습 문제

### 연습 1: TokenBudget 사용하기

```python
# TODO: TokenBudget을 사용하여 다음 작업을 수행하세요
# 1. 총 4000 토큰 예산으로 TokenBudget 생성
# 2. 시스템 프롬프트 소비: "당신은 코드 리뷰 전문가입니다."
# 3. 사용자 질문 소비: "다음 코드를 리뷰해주세요."
# 4. 남은 예산 확인 후, 대용량 코드를 예산에 맞게 절삭

budget = TokenBudget(max_tokens=4000)

# TODO: 코드를 작성하세요
# system_prompt = ...
# user_query = ...
# code = "..." * 1000  # 대용량 코드
# truncated_code = budget.truncate_to_budget(code, reserve=500)
# print(f"남은 예산: {budget.remaining}")
```

### 연습 2: Annotated Reducer로 리팩터링

다음 코드를 Annotated Reducer 패턴으로 리팩터링하세요.

```python
# ── Before: 전체 복사 패턴 ──

class TaskState(TypedDict):
    task_id: str
    data: dict | None
    result: str | None
    logs: list[str]
    error: str | None

def node_a(state: TaskState) -> TaskState:
    return {
        **state,  # ← 전체 복사!
        "data": {"key": "value"},
        "logs": state.get("logs", []) + ["Node A 완료"],
    }

def node_b(state: TaskState) -> TaskState:
    return {
        **state,  # ← 또 전체 복사!
        "result": "분석 완료",
        "logs": state.get("logs", []) + ["Node B 완료"],
    }

# TODO: Annotated Reducer 패턴으로 변환하세요
# 힌트:
# - 각 필드에 Annotated[Type, _replace] 적용
# - logs 필드는 Annotated[list[str], operator.add] 사용
# - 노드는 변경 필드만 반환
```

### 연습 3: 코드 압축 비교

```python
# TODO: PythonCodeCompressor를 사용하여 다음 코드를 압축하고,
#       원본과 압축본의 토큰 수를 비교하세요.

sample_code = '''
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class UserService:
    """사용자 관련 서비스."""

    def __init__(self, db_connection):
        self.db = db_connection
        self.cache = {}
        logger.info("UserService initialized")

    def get_user(self, user_id: int) -> Optional[dict]:
        """사용자 정보를 조회합니다."""
        if user_id in self.cache:
            return self.cache[user_id]
        user = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
        if user:
            self.cache[user_id] = user
        return user

    def create_user(self, name: str, email: str) -> dict:
        """새 사용자를 생성합니다."""
        user = {"name": name, "email": email}
        result = self.db.insert("users", user)
        logger.info(f"User created: {result}")
        return result

    def delete_user(self, user_id: int) -> bool:
        """사용자를 삭제합니다."""
        self.cache.pop(user_id, None)
        return self.db.delete("users", user_id)
'''

# compressor = PythonCodeCompressor()
# summary = compressor.extract_signatures(sample_code)
# budget = TokenBudget()
# print(f"원본 토큰: {budget.count(sample_code)}")
# print(f"압축 토큰: {budget.count(summary)}")
# print(f"절감률: {(1 - budget.count(summary) / budget.count(sample_code)) * 100:.0f}%")
```

---

## 5. 핵심 정리

### 최적화 전략 요약표

| 리소스 | 문제 | 해결 전략 | 핵심 도구 |
|--------|------|----------|----------|
| **디스크** | Git clone 누적 | TTL 기반 자동 정리 | `CloneManager` |
| **디스크** | 전체 히스토리 clone | Shallow clone (`--depth 1`) | Git CLI 옵션 |
| **메모리** | 매 노드마다 State 전체 복사 | Annotated Reducer 패턴 | `Annotated[T, reducer]` |
| **메모리** | 불필요한 대용량 필드 잔존 | 노드 간 데이터 프루닝 | cleanup 노드 |
| **토큰** | 문자 수 기반 부정확한 절삭 | tiktoken으로 정확한 토큰 계산 | `TokenBudget` |
| **토큰** | 코드 전체를 프롬프트에 포함 | AST 기반 시그니처 추출 | `PythonCodeCompressor` |
| **토큰** | 대용량 diff 한 번에 처리 | 파일 단위 청크 분할 | `split_diff_by_file` |

### Annotated Reducer 변환 Before/After

```python
# ── Before ──                          # ── After ──
class MyState(TypedDict):               class MyState(TypedDict):
    data: str | None                        data: Annotated[str | None, _replace]
    result: str | None                      result: Annotated[str | None, _replace]
    logs: list[str]                         logs: Annotated[list[str], operator.add]

def my_node(state):                     def my_node(state):
    return {                                return {
        **state,           # 전체 복사!        "result": "done",  # 변경분만!
        "result": "done",                      "logs": ["완료"],   # 누적 추가!
        "logs": state["logs"] + ["완료"],  }
    }
```

### 기억할 3가지 원칙

1. **측정하고 최적화하세요**: `TokenBudget.count()`로 실제 토큰 수를 확인한 후 최적화하세요
2. **필요한 것만 보내세요**: LLM에 전체 코드 대신 시그니처, 전체 diff 대신 관련 파일만 보내세요
3. **쓰고 나면 정리하세요**: clone은 TTL로, State는 프루닝으로, 토큰은 예산으로 관리하세요

---

## 6. 참고 자료

| 자료 | 링크 | 설명 |
|------|------|------|
| LangGraph State Reducers | [langgraph - Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) | Annotated Reducer 공식 문서 |
| tiktoken | [github.com/openai/tiktoken](https://github.com/openai/tiktoken) | OpenAI의 토큰 카운팅 라이브러리 |
| Python operator 모듈 | [docs.python.org/operator](https://docs.python.org/3/library/operator.html) | `operator.add` 등 내장 함수 |
| Python ast 모듈 | [docs.python.org/ast](https://docs.python.org/3/library/ast.html) | AST(추상 구문 트리) 파싱 |
| Git shallow clone | [git-scm.com/docs/git-clone](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt) | `--depth` 옵션 공식 문서 |
| LangGraph Annotated 예제 | [langgraph - How to define state](https://langchain-ai.github.io/langgraph/how-tos/define-state/) | State 정의 가이드 |

---

## 다음 단계

이번 모듈에서 에이전트의 3대 리소스(디스크, 메모리, LLM 토큰)를 효율적으로 관리하는 방법을 배웠습니다. Part 4의 핵심 주제인 에러 처리(Module 08), 외부 시스템 연동(Module 09), 리소스 최적화(Module 10)를 모두 다루었습니다.

이제 여러분은 프로덕션 수준의 안정적이고 효율적인 AI 에이전트를 만들 수 있는 기반을 갖추었습니다.

**추천 다음 학습**:
- LangGraph 체크포인트와 영속성 (장애 복구)
- LangGraph 서브그래프 (대규모 에이전트 모듈화)
- 모니터링과 옵저버빌리티 (LangSmith, 커스텀 메트릭)
