# Module 05: 프롬프트 엔지니어링 - LLM의 성능을 결정하는 핵심 기술

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. 프롬프트의 구조(역할, 맥락, 지시, 출력 형식)를 이해하고 설계할 수 있다
2. LangChain `ChatPromptTemplate`을 사용하여 재사용 가능한 프롬프트를 만들 수 있다
3. Zero-shot, One-shot, Few-shot 기법의 차이를 이해하고 적용할 수 있다
4. 프롬프트를 외부 YAML 파일로 분리하여 버전 관리할 수 있다
5. 도메인/언어별로 특화된 프롬프트를 설계할 수 있다

---

## 사전 지식

이 모듈을 시작하기 전에 다음 내용을 알고 있어야 합니다:

- **Python 기초**: 함수, 클래스, 딕셔너리, f-string 사용법
- **LLM 기본 개념**: LLM이 무엇이고, API를 통해 호출하는 방법 (Module 03 참고)
- **LangChain 기초**: LangChain이 무엇인지 기본 개념 (Module 04 참고)

> **용어 정리**
> - **프롬프트(Prompt)**: LLM에게 보내는 입력 텍스트. "질문"이라고 생각하면 됩니다.
> - **토큰(Token)**: LLM이 텍스트를 처리하는 최소 단위. 대략 한국어 1글자 = 1~2 토큰.
> - **System 메시지**: LLM에게 "역할"을 부여하는 특별한 메시지.
> - **Human 메시지**: 사용자가 LLM에게 보내는 실제 요청 메시지.

---

## 1. 개념 설명

### 1.1 프롬프트란? 왜 중요한가?

프롬프트는 LLM에게 보내는 **지시문**입니다. 같은 모델이라도 프롬프트에 따라 결과의 품질이 극적으로 달라집니다.

#### 동일한 모델, 다른 프롬프트 = 완전히 다른 결과

다음 두 프롬프트의 결과를 비교해 봅시다:

**나쁜 프롬프트:**
```
이 코드를 분석해줘.

def calculate_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
```

**결과**: "이 코드는 가격을 계산하는 함수입니다." (너무 일반적이고 유용하지 않음)

**좋은 프롬프트:**
```
당신은 시니어 Python 개발자입니다.
아래 함수의 버그, 성능 이슈, 개선점을 분석하세요.

## 분석 기준
- 예외 처리 여부
- 타입 안전성
- 엣지 케이스 (빈 리스트, None 등)

## 응답 형식
JSON으로 출력하세요:
{"bugs": [...], "improvements": [...], "severity": "HIGH|MEDIUM|LOW"}

## 코드
def calculate_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
```

**결과**: items가 None일 때 TypeError 발생, item.price가 음수인 경우 검증 누락 등 구체적인 분석 결과가 나옵니다.

> **핵심 포인트**: 프롬프트 엔지니어링은 "LLM에게 정확히 무엇을 원하는지 명확하게 전달하는 기술"입니다.

### 1.2 프롬프트의 4가지 구성 요소

좋은 프롬프트는 다음 4가지 요소를 포함합니다:

| 구성 요소 | 설명 | 예시 |
|-----------|------|------|
| **역할(Role)** | LLM에게 전문가 역할 부여 | "당신은 시니어 QA 엔지니어입니다" |
| **맥락(Context)** | 배경 정보, 도메인 지식 | "Spring Boot 기반 REST API 프로젝트입니다" |
| **지시(Instruction)** | 구체적으로 무엇을 해야 하는지 | "코드의 버그를 찾아 수정 방안을 제시하세요" |
| **출력 형식(Format)** | 원하는 응답 형태 | "JSON 형식으로, bugs와 fixes 키를 포함하세요" |

```
[역할] 당신은 시니어 소프트웨어 엔지니어입니다.

[맥락] 아래는 사용자 인증 모듈의 코드 변경사항(diff)입니다.
       이 프로젝트는 Java/Spring Boot 기반입니다.

[지시] 변경된 기능을 분석하고, 테스트가 필요한 시나리오를 식별하세요.

[형식] 다음 JSON 형식으로 응답하세요:
       { "summary": "...", "test_scenarios": [...] }
```

### 1.3 System 메시지 vs Human 메시지

LLM API는 보통 두 종류의 메시지를 구분합니다:

| 구분 | System 메시지 | Human 메시지 |
|------|-------------|-------------|
| **역할** | LLM의 행동 규칙 설정 | 실제 요청/데이터 전달 |
| **포함 내용** | 역할 정의, 규칙, 출력 형식 | 입력 데이터, 구체적 요청 |
| **변경 빈도** | 거의 변경되지 않음 | 매 요청마다 다름 |
| **비유** | 직원에게 "업무 매뉴얼"을 주는 것 | 직원에게 "오늘 할 일"을 주는 것 |

**System에 넣어야 할 것**: 역할, 규칙, 금지 사항, 출력 형식
**Human에 넣어야 할 것**: 입력 데이터, 변수, 구체적 요청문

```python
# System: 역할 + 규칙 (불변)
system_message = """
당신은 소프트웨어 테스트 전문가입니다.

## 평가 기준
- PASS: 소스코드가 기대 결과를 충족
- FAIL: 결함이 발견되어 기대 결과 미충족
- BLOCKED: 판단을 위한 정보 부족

## 응답 형식
반드시 유효한 JSON만 출력하세요.
"""

# Human: 데이터 + 요청 (매번 다름)
human_message = """
## TC 정보
- 제목: 로그인 API 정상 동작 검증
- 기대 결과: 200 OK, JWT 토큰 반환

## 관련 소스코드
{source_code}

위 정보를 기반으로 TC를 평가하세요.
"""
```

---

## 2. 단계별 실습

### 2.1 LangChain ChatPromptTemplate 사용법

LangChain의 `ChatPromptTemplate`은 프롬프트를 구조화하고 재사용할 수 있게 해주는 핵심 도구입니다.

#### Step 1: 기본 ChatPromptTemplate 만들기

```python
# 필요한 패키지 설치
# pip install langchain-core langchain-anthropic

from langchain_core.prompts import ChatPromptTemplate

# from_messages()로 System/Human 메시지를 분리하여 생성
prompt = ChatPromptTemplate.from_messages([
    (
        "system",   # 역할: LLM의 행동 규칙
        "당신은 코드 리뷰 전문가입니다.\n"
        "주어진 코드의 버그와 개선점을 분석하세요.\n"
        "반드시 JSON 형식으로 응답하세요."
    ),
    (
        "human",    # 역할: 사용자 요청 + 데이터
        "## 분석할 코드\n"
        "언어: {language}\n"
        "```\n{code}\n```\n\n"
        "위 코드를 분석해주세요."
    ),
])
```

> **주목**: `{language}`와 `{code}`는 **변수 자리표시자(placeholder)** 입니다. 나중에 실제 값으로 치환됩니다.

#### Step 2: 변수 치환하기

```python
# 변수에 실제 값을 넣어서 프롬프트 완성
formatted = prompt.format_messages(
    language="Python",
    code="def divide(a, b):\n    return a / b"
)

# 결과 확인
for msg in formatted:
    print(f"[{msg.type}] {msg.content[:80]}...")
# [system] 당신은 코드 리뷰 전문가입니다...
# [human] ## 분석할 코드\n언어: Python...
```

#### Step 3: 체인(Chain) 파이프라인 만들기

LangChain의 핵심 개념인 **파이프라인**을 사용하면, 프롬프트 -> LLM 호출 -> 결과 파싱을 한 줄로 연결할 수 있습니다.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

# LLM 인스턴스 생성
llm = ChatAnthropic(model="claude-sonnet-4-5-20250514", temperature=0.1)

# 파이프라인: 프롬프트 -> LLM -> 텍스트 파서
# "|" 연산자로 체인을 연결합니다 (UNIX 파이프와 유사)
chain = prompt | llm | StrOutputParser()

# 실행: 변수를 딕셔너리로 넣어서 호출
result = chain.invoke({
    "language": "Python",
    "code": "def divide(a, b):\n    return a / b"
})

print(result)
# {"bugs": [{"description": "ZeroDivisionError when b is 0", ...}], ...}
```

**파이프라인 동작 흐름:**

```
{language, code}                  <-- 입력 딕셔너리
      |
      v
[ChatPromptTemplate]  -- 변수 치환 --> System/Human 메시지 리스트
      |
      v
[ChatAnthropic]       -- API 호출 --> AIMessage(content="...")
      |
      v
[StrOutputParser]     -- 텍스트 추출 --> "결과 문자열"
```

### 2.2 Few-shot 학습

Few-shot 학습은 프롬프트에 **입출력 예시를 포함**시켜 LLM이 원하는 형식과 품질을 학습하게 하는 기법입니다.

#### Zero-shot vs One-shot vs Few-shot 비교

| 기법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Zero-shot** | 예시 없이 지시만 제공 | 간결, 토큰 절약 | 출력 형식 불안정 |
| **One-shot** | 예시 1개 제공 | 형식 이해 도움 | 예시가 편향될 수 있음 |
| **Few-shot** | 예시 2~5개 제공 | 안정적 출력, 높은 품질 | 토큰 사용 증가 |

#### 같은 작업에 다른 기법 적용 비교

**작업**: 이메일 내용을 분류하기 (카테고리: 문의, 불만, 칭찬, 기타)

**Zero-shot** (예시 없음):
```python
zero_shot = ChatPromptTemplate.from_messages([
    ("system", "이메일을 분류하세요. 카테고리: 문의, 불만, 칭찬, 기타"),
    ("human", "이메일 내용: {email}\n\nJSON으로 답하세요."),
])
```

**Few-shot** (예시 3개):
```python
few_shot = ChatPromptTemplate.from_messages([
    (
        "system",
        "이메일을 분류하세요. 카테고리: 문의, 불만, 칭찬, 기타\n"
        "반드시 JSON 형식으로만 응답하세요."
    ),
    # --- Few-shot 예시 시작 ---
    # 예시 1: 문의
    ("human", "이메일 내용: 환불 절차가 어떻게 되나요?"),
    ("assistant", '{{"category": "문의"}}'),
    # 예시 2: 불만
    ("human", "이메일 내용: 배송이 2주나 지연되어 정말 화가 납니다."),
    ("assistant", '{{"category": "불만"}}'),
    # 예시 3: 칭찬
    ("human", "이메일 내용: 친절한 상담 감사합니다. 문제 해결했어요!"),
    ("assistant", '{{"category": "칭찬"}}'),
    # --- Few-shot 예시 끝 ---
    # 실제 입력
    ("human", "이메일 내용: {email}"),
])
```

> **실험 결과**: Few-shot을 사용하면 JSON 형식 준수율이 약 70%에서 95% 이상으로 향상됩니다.

#### 좋은 Few-shot 예시 vs 나쁜 Few-shot 예시

**나쁜 예시** -- 너무 짧고 모호함:
```yaml
- input: "코드 변경"
  output: '{"summary": "변경됨"}'
```

**좋은 예시** -- 구체적이고 현실적:
```yaml
- input: |
    ## 변경 정보
    - 소스 브랜치: feature/user-auth
    - 언어/프레임워크: java / spring-boot

    ## Diff
    + @PostMapping("/api/v1/auth/login")
    + public ResponseEntity<LoginResponse> login(@RequestBody LoginRequest req) {
    +     User user = userService.authenticate(req.getEmail(), req.getPassword());
    +     String token = jwtService.generateToken(user);
    +     return ResponseEntity.ok(new LoginResponse(token));
    + }
  output: |
    {
      "summary": "사용자 로그인 API 엔드포인트 추가",
      "changed_features": [
        {
          "feature": "사용자 인증 - 로그인 API",
          "change_type": "added",
          "description": "POST /api/v1/auth/login. 이메일/비밀번호 인증 후 JWT 토큰 발급"
        }
      ],
      "test_scenarios": [
        {
          "scenario": "올바른 이메일/비밀번호로 로그인 시 JWT 토큰 반환 확인",
          "test_type": "functional",
          "priority": "HIGH"
        },
        {
          "scenario": "잘못된 비밀번호로 로그인 시 401 에러 반환 확인",
          "test_type": "edge_case",
          "priority": "HIGH"
        }
      ]
    }
```

**좋은 예시를 만드는 4가지 원칙:**

1. **현실적 입력**: 실제 프로젝트에서 나올 법한 데이터 사용
2. **모든 필드 포함**: 출력 JSON의 모든 필드를 빠짐없이 채움
3. **다양한 케이스**: 정상 케이스와 예외 케이스를 모두 포함
4. **구체적 내용**: "변경됨" 대신 "POST /api/v1/auth/login 엔드포인트 추가"

#### ChatPromptTemplate에 Few-shot 추가하기

```python
prompt_with_few_shots = ChatPromptTemplate.from_messages([
    # 1) System 메시지: 역할 + 규칙
    (
        "system",
        "당신은 시니어 소프트웨어 엔지니어입니다.\n"
        "코드 변경사항을 분석하고 테스트 시나리오를 식별하세요.\n"
        "반드시 JSON 형식으로 응답하세요."
    ),
    # 2) Few-shot 예시 쌍 (human -> assistant 반복)
    ("human", "## 변경 정보\n- 브랜치: feature/add-login\n- 언어: java\n\n## Diff\n+ @PostMapping(\"/login\") ..."),
    ("assistant", '{{"summary": "로그인 API 추가", "test_scenarios": [{{"scenario": "정상 로그인 검증", "priority": "HIGH"}}]}}'),
    ("human", "## 변경 정보\n- 브랜치: fix/null-check\n- 언어: python\n\n## Diff\n+ if user is None: return None"),
    ("assistant", '{{"summary": "null 체크 추가", "test_scenarios": [{{"scenario": "None 입력 시 안전 반환 검증", "priority": "MEDIUM"}}]}}'),
    # 3) 실제 Human 메시지 (변수 사용)
    (
        "human",
        "## 변경 정보\n- 브랜치: {branch}\n- 언어: {language}\n\n## Diff\n{diff}"
    ),
])
```

**메시지 순서 규칙:**
```
[System]    역할 + 규칙           -- 항상 첫 번째
[Human]     Few-shot 입력 1       -- 예시 쌍 시작
[Assistant] Few-shot 출력 1
[Human]     Few-shot 입력 2       -- 필요한 만큼 반복
[Assistant] Few-shot 출력 2
[Human]     실제 입력              -- 항상 마지막
```

### 2.3 프롬프트 외부 파일 관리

프롬프트를 Python 코드 안에 하드코딩하면 다음과 같은 문제가 생깁니다:
- 프롬프트를 수정할 때마다 코드를 배포해야 함
- 프롬프트 변경 이력 추적이 어려움
- A/B 테스트(두 프롬프트 비교)가 불가능

**해결책**: 프롬프트를 YAML 파일로 분리하고, 로더 유틸리티로 불러오는 방식.

#### Step 1: YAML 파일 형식 설계

```yaml
# prompts/code_review/v1.yaml
version: "v1"
description: "코드 변경사항 분석 프롬프트"
created_at: "2026-03-23"

system: |
  당신은 시니어 소프트웨어 엔지니어로, 코드 변경사항을 분석합니다.

  ## 핵심 규칙
  - 구체적이고 테스트 가능한 시나리오를 식별하세요
  - 정상 케이스와 예외 케이스를 모두 포함하세요
  - 반드시 JSON 형식으로 응답하세요
human: |
  ## 변경 정보
  - 소스 브랜치: {source_branch}
  - 언어/프레임워크: {language} / {framework}

  ## 변경된 파일
  {changed_files}

  ## Diff
  {diff_text}

  위 변경사항을 분석해주세요.

few_shots:
  - input: |
      ## 변경 정보
      - 소스 브랜치: feature/user-auth
      - 언어: java
      ## Diff
      + @PostMapping("/api/v1/auth/login") ...
    output: |
      {"summary": "로그인 API 추가", "test_scenarios": [...]}
```

#### Step 2: 버전 관리 전략

각 프롬프트 디렉토리에 `_meta.yaml` 파일을 두어 활성 버전을 관리합니다:

```yaml
# prompts/code_review/_meta.yaml
active_version: "v1"
history:
  - version: "v1"
    activated_at: "2026-03-23"
    note: "초기 버전"
```

**버전 관리 규칙:**
- 새 프롬프트는 `v2.yaml`로 만들고, 테스트 후 `_meta.yaml`의 `active_version`을 변경
- 기존 버전 파일은 절대 삭제하지 않음 (롤백 가능)
- 환경 변수 `PROMPT_VERSION_OVERRIDE`로 특정 버전 강제 가능

#### Step 3: 프롬프트 로더 유틸리티 구현

```python
# prompts/loader.py
import os
from pathlib import Path
from functools import lru_cache
import yaml
from langchain_core.prompts import ChatPromptTemplate

TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=32)  # 동일 프롬프트 반복 로드 방지
def load_prompt(
    agent: str,
    node: str,
    version: str | None = None,
) -> ChatPromptTemplate:
    """YAML 파일에서 프롬프트를 로드하여 ChatPromptTemplate을 반환한다.

    Args:
        agent: 에이전트 이름 (예: "code_review", "email_classifier")
        node: 노드/태스크 이름 (예: "analyze", "classify")
        version: 프롬프트 버전. None이면 _meta.yaml의 active_version 사용

    Returns:
        ChatPromptTemplate 인스턴스
    """
    prompt_dir = TEMPLATES_DIR / agent / node

    # 버전 결정 우선순위: 환경변수 > 인자 > _meta.yaml
    env_override = os.getenv("PROMPT_VERSION_OVERRIDE")
    if env_override:
        version = env_override
    elif version is None:
        meta_path = prompt_dir / "_meta.yaml"
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        version = meta["active_version"]

    # YAML 로드
    yaml_path = prompt_dir / f"{version}.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # 메시지 구성: system + (few-shot) + human
    messages = [("system", data["system"])]

    # few-shot 예제가 있으면 system과 human 사이에 삽입
    if "few_shots" in data:
        for example in data["few_shots"]:
            messages.append(("human", example["input"]))
            messages.append(("assistant", example["output"]))

    messages.append(("human", data["human"]))

    return ChatPromptTemplate.from_messages(messages)
```

#### 사용법

```python
# Before: 코드에 프롬프트 하드코딩
PROMPT = ChatPromptTemplate.from_messages([...])  # 긴 프롬프트 텍스트
chain = PROMPT | llm | parser

# After: YAML에서 로드
from prompts.loader import load_prompt

prompt = load_prompt("code_review", "analyze")
chain = prompt | llm | parser
```

---

## 3. 실전 예제

### 3.1 언어/도메인별 프롬프트 특화

같은 "코드 분석" 작업이라도, 프로그래밍 언어에 따라 프롬프트를 다르게 작성해야 최상의 결과를 얻을 수 있습니다.

#### Java 코드 분석 프롬프트

```yaml
# prompts/code_review/analyze/v1_java.yaml
system: |
  당신은 Java/Spring Boot 전문 시니어 개발자입니다.

  ## 분석 시 주의할 패턴
  - @Transactional 누락 여부
  - NullPointerException 가능성 (Optional 미사용)
  - @Valid/@Validated 어노테이션 누락
  - try-catch에서 Exception을 잡는 안티패턴
  - Spring Security 설정 미비

  ## 응답 규칙
  - 각 이슈에 심각도(HIGH/MEDIUM/LOW)를 부여하세요
  - 수정 코드 예시를 반드시 포함하세요
```

#### Python 코드 분석 프롬프트

```yaml
# prompts/code_review/analyze/v1_python.yaml
system: |
  당신은 Python 전문 시니어 개발자입니다.

  ## 분석 시 주의할 패턴
  - 타입 힌트 누락 (함수 파라미터, 반환값)
  - mutable default argument (def f(x=[]))
  - bare except (except: 대신 except Exception:)
  - async/await 미사용 (I/O 바운드 작업)
  - f-string 대신 .format() 또는 % 사용

  ## 응답 규칙
  - PEP 8 준수 여부를 확인하세요
  - mypy 호환성을 고려하세요
```

#### 언어에 따라 프롬프트 선택하기

```python
def get_review_prompt(language: str) -> ChatPromptTemplate:
    """언어에 맞는 코드 리뷰 프롬프트를 반환한다."""
    language_map = {
        "java": "v1_java",
        "python": "v1_python",
        "javascript": "v1_javascript",
    }
    version = language_map.get(language.lower(), "v1")  # 기본 범용 프롬프트
    return load_prompt("code_review", "analyze", version=version)
```

### 3.2 실전: "코드 리뷰 어시스턴트" 프롬프트 설계

전체 코드를 조합하여 완전한 코드 리뷰 어시스턴트를 만들어 봅시다:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

# 1) 프롬프트 설계 (Few-shot 포함)
review_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 시니어 코드 리뷰어입니다.
"
     "코드의 버그, 보안 이슈, 개선점을 분석하세요.

"
     "## 출력 형식
"
     "JSON으로 응답하세요:
"
     '{"bugs": [{"line": N, "issue": "...", "fix": "..."}], '
     '"security": [...], "improvements": [...], "score": 1-10}'),
    # Few-shot 예시
    ("human", "언어: python
```
def login(pw):
    if pw == 'admin': return True
```"),
    ("assistant",
     '{"bugs": [], '
     '"security": [{"line": 2, "issue": "하드코딩된 비밀번호", "fix": "환경변수 또는 해시 비교 사용"}], '
     '"improvements": [{"line": 1, "issue": "타입 힌트 없음", "fix": "def login(pw: str) -> bool:"}], '
     '"score": 3}'),
    # 실제 입력
    ("human", "언어: {language}
```
{code}
```"),
])

# 2) 체인 구성
llm = ChatAnthropic(model="claude-sonnet-4-5-20250514", temperature=0.1)
chain = review_prompt | llm | StrOutputParser()

# 3) 실행
result = chain.invoke({
    "language": "python",
    "code": "def get_user(id):
    conn = sqlite3.connect('db.sqlite')
"
            "    return conn.execute(f'SELECT * FROM users WHERE id={id}').fetchone()"
})
print(result)
# SQL Injection 취약점, 커넥션 미종료 등을 발견할 것입니다
```

---

## 4. 연습 문제

### 연습 1: 이메일 자동 응답 프롬프트 개선

**과제**: 아래의 Zero-shot 프롬프트를 Few-shot으로 개선하세요.

**시작 코드 (Zero-shot):**
```python
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "고객 이메일에 대한 자동 응답을 생성하세요. 정중하고 도움이 되는 톤을 사용하세요."),
    ("human", "고객 이메일: {email}

자동 응답을 작성하세요."),
])
```

**요구사항:**
1. 3가지 유형의 이메일에 대한 Few-shot 예시 추가 (문의, 불만, 환불 요청)
2. 각 예시의 응답은 150자 이내
3. 응답에 고객 이름을 포함하는 변수 `{customer_name}` 추가

**힌트**: Few-shot 예시에서는 변수를 사용하지 않고 고정 텍스트를 사용합니다. 변수는 마지막 Human 메시지에서만 사용합니다.

### 연습 2: YAML 프롬프트 파일 작성

**과제**: "회의록 요약기" 프롬프트를 YAML 파일로 작성하세요.

**요구사항:**
- System 메시지: 회의록 요약 전문가 역할
- Human 메시지: 회의 제목, 참석자, 회의 내용을 변수로 받기
- Few-shot 예시 1개 포함
- JSON 출력 형식: `{"summary": "...", "action_items": [...], "decisions": [...]}`

---

## 5. 핵심 정리

| 개념 | 핵심 내용 |
|------|----------|
| **프롬프트 구조** | 역할(Role) + 맥락(Context) + 지시(Instruction) + 형식(Format) |
| **System vs Human** | System = 규칙/역할 (불변), Human = 데이터/요청 (가변) |
| **ChatPromptTemplate** | `from_messages()`로 생성, `{variable}`로 변수 치환 |
| **체인 파이프라인** | `prompt \| llm \| parser` -- 파이프로 연결 |
| **Few-shot** | 예시 2~5개로 출력 품질/형식 안정화. Human/Assistant 쌍으로 삽입 |
| **외부 파일 관리** | YAML로 분리 + `_meta.yaml`로 버전 관리 + 로더 유틸리티 |
| **언어별 특화** | 같은 작업이라도 언어/프레임워크에 맞는 분석 기준 제공 |

---

## 6. 참고 자료

- **Anthropic 프롬프트 엔지니어링 가이드**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
  - Claude에 특화된 프롬프트 작성 가이드. 역할 부여, 출력 형식, XML 태그 활용법 등을 다룹니다.

- **LangChain ChatPromptTemplate 문서**: https://python.langchain.com/docs/concepts/prompt_templates/
  - `from_messages()`, 변수 치환, MessagesPlaceholder 등 상세 사용법.

- **OpenAI 프롬프트 베스트 프랙티스**: https://platform.openai.com/docs/guides/prompt-engineering
  - 모델에 무관한 일반적인 프롬프트 작성 원칙. 명확한 지시, 참조 텍스트 활용 등.

- **LangChain Few-shot 예제**: https://python.langchain.com/docs/how_to/few_shot_examples_chat/
  - ChatPromptTemplate에 Few-shot을 추가하는 구체적인 코드 예시.

---

## 다음 단계

프롬프트 엔지니어링을 마스터했다면, 다음 모듈에서는 LLM의 **응답을 프로그램이 사용할 수 있는 구조화된 데이터로 변환**하는 방법을 배웁니다.

**Module 06: 구조화된 출력** 에서 다루는 내용:
- LLM의 자유 텍스트 응답을 Pydantic 모델로 검증하기
- `with_structured_output()`으로 스키마 강제하기
- JSON 출력 파서 패턴과 폴백 전략
