# Module 06: 구조화된 출력 - LLM 응답을 데이터로 변환하기

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

1. LLM의 자유 텍스트 응답이 왜 프로그램에서 직접 사용하기 어려운지 이해한다
2. Pydantic 모델을 사용하여 출력 스키마를 정의할 수 있다
3. `with_structured_output()`으로 LLM 응답을 강제할 수 있다
4. JSON 출력 파서를 구현하고 폴백 전략을 적용할 수 있다
5. 실전에서 사용할 수 있는 출력 스키마를 설계할 수 있다

---

## 사전 지식

- **Python 기초**: 클래스, 데코레이터, 타입 힌트
- **Module 05**: 프롬프트 엔지니어링 기본 (ChatPromptTemplate, 체인)
- **JSON 기초**: JSON 형식과 Python dict의 관계

> **용어 정리**
> - **구조화된 출력(Structured Output)**: LLM이 정해진 스키마에 맞게 응답하도록 강제하는 것
> - **Pydantic**: Python의 데이터 검증 라이브러리. 타입 검사와 데이터 변환을 자동 수행
> - **스키마(Schema)**: 데이터의 구조를 정의하는 설계도 (어떤 필드가 있고, 각 필드의 타입은 무엇인지)
> - **파싱(Parsing)**: 텍스트를 프로그램이 사용할 수 있는 데이터 구조로 변환하는 것

---

## 1. 개념 설명

### 1.1 문제: LLM은 자유 텍스트를 반환한다

LLM에게 "버그를 분석해줘"라고 요청하면, 다양한 형태로 응답할 수 있습니다:

**응답 예시 A** (깔끔한 JSON):
```json
{"bugs": [{"line": 5, "issue": "null check missing"}], "severity": "HIGH"}
```

**응답 예시 B** (마크다운 코드블록에 감싸진 JSON):
```
분석 결과입니다:

\```json
{"bugs": [{"line": 5, "issue": "null check missing"}], "severity": "HIGH"}
\```

추가로 고려해야 할 사항은...
```

**응답 예시 C** (JSON이 아닌 자유 텍스트):
```
5번 줄에서 null 체크가 누락되어 있습니다.
심각도는 높음입니다.
```

프로그램에서 이 응답을 처리하려면 `result["bugs"][0]["line"]`처럼 접근해야 하는데, 응답 형태가 매번 다르면 프로그램이 깨집니다.

### 1.2 접근법 비교

| 접근법 | 방식 | 신뢰도 | 난이도 |
|--------|------|--------|--------|
| 텍스트 기반 JSON 스키마 | 프롬프트에 "이 형식으로 답하세요" 텍스트 추가 | 낮음 (LLM이 무시 가능) | 쉬움 |
| 정규표현식 파싱 | 응답에서 정규식으로 데이터 추출 | 낮음 (형식 변경에 취약) | 중간 |
| JSON 파서 + 폴백 | 여러 단계로 JSON 추출 시도 | 중간 (대부분 성공) | 중간 |
| **Pydantic + with_structured_output()** | **LLM이 스키마에 맞게 생성하도록 강제** | **높음 (구조 보장)** | **쉬움** |

> **결론**: `with_structured_output()`이 가장 안정적이고, JSON 파서는 폴백용으로 함께 사용합니다.

---

## 2. 단계별 실습

### 2.1 Pydantic 모델 기초

Pydantic은 Python의 **데이터 검증 라이브러리**입니다. 클래스를 정의하면, 데이터가 그 구조에 맞는지 자동으로 검증해줍니다.

#### 기본 사용법

```python
# pip install pydantic

from pydantic import BaseModel, Field

# BaseModel을 상속하여 데이터 스키마 정의
class BugReport(BaseModel):
    """버그 리포트 스키마."""
    line: int = Field(description="버그가 있는 라인 번호", ge=1)      # ge=1: 1 이상
    issue: str = Field(description="버그 설명", min_length=1)         # min_length: 빈 문자열 금지
    severity: str = Field(description="심각도", pattern=r"^(HIGH|MEDIUM|LOW)$")  # 정규식 패턴

# 올바른 데이터 -> 성공
bug = BugReport(line=5, issue="null check missing", severity="HIGH")
print(bug.model_dump())
# {"line": 5, "issue": "null check missing", "severity": "HIGH"}

# 잘못된 데이터 -> 에러 발생
try:
    bad_bug = BugReport(line=-1, issue="", severity="CRITICAL")
except Exception as e:
    print(e)
    # line: Input should be greater than or equal to 1
    # issue: String should have at least 1 character
    # severity: String should match pattern '^(HIGH|MEDIUM|LOW)$'
```

#### Field 옵션 정리

| 옵션 | 용도 | 예시 |
|------|------|------|
| `description` | 필드 설명 (LLM에게 힌트 제공) | `Field(description="버그 설명")` |
| `min_length` | 문자열 최소 길이 | `Field(min_length=1)` -- 빈 문자열 금지 |
| `max_length` | 문자열 최대 길이 | `Field(max_length=500)` |
| `ge` / `le` | 숫자 범위 (이상/이하) | `Field(ge=0, le=1.0)` -- 0~1 범위 |
| `pattern` | 정규식 패턴 매칭 | `Field(pattern=r"^(HIGH\|MEDIUM\|LOW)$")` |
| `default` | 기본값 | `Field(default="MEDIUM")` |
| `default_factory` | 기본값 팩토리 (리스트 등) | `Field(default_factory=list)` |

#### Enum으로 허용값 제한

```python
from enum import Enum

class Severity(str, Enum):
    """심각도를 3가지로 제한."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class TestStatus(str, Enum):
    """테스트 결과를 3가지로 제한."""
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"

class TestResult(BaseModel):
    status: TestStatus                             # PASS, FAIL, BLOCKED만 허용
    confidence: float = Field(ge=0.0, le=1.0)      # 0.0 ~ 1.0 범위
    reasoning: str = Field(min_length=1)            # 빈 문자열 금지
    severity: Severity = Field(default=Severity.MEDIUM)  # 기본값: MEDIUM
```

#### field_validator로 커스텀 검증

```python
from pydantic import BaseModel, Field, field_validator

class CodeAnalysis(BaseModel):
    affected_files: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    root_cause: str = Field(min_length=1)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """confidence를 소수점 2자리로 반올림."""
        return round(v, 2)

    @field_validator("affected_files")
    @classmethod
    def validate_file_paths(cls, v: list[str]) -> list[str]:
        """파일 경로가 비어있지 않은지 확인."""
        return [f for f in v if f.strip()]  # 빈 문자열 제거

# 사용
analysis = CodeAnalysis(
    affected_files=["src/main.py", "", "src/utils.py"],
    confidence=0.85678,
    root_cause="Missing null check"
)
print(analysis.confidence)       # 0.86 (반올림됨)
print(analysis.affected_files)   # ["src/main.py", "src/utils.py"] (빈 문자열 제거됨)
```

### 2.2 with_structured_output() 사용법

LangChain의 `with_structured_output()`은 LLM이 **Pydantic 모델에 맞는 출력을 생성하도록 강제**합니다.

#### 작동 원리

```
기존 방식:
  프롬프트("JSON 형식으로 답해줘") -> LLM -> 자유 텍스트 -> 파서로 JSON 추출 (실패 가능)

with_structured_output 방식:
  Pydantic 모델 정의 -> LLM이 스키마에 맞게 생성 -> Pydantic 인스턴스 반환 (구조 보장)
```

내부적으로는 **제약된 디코딩(Constrained Decoding)** 이라는 기술을 사용합니다. LLM이 토큰을 생성할 때, 스키마에 맞지 않는 토큰은 선택하지 못하도록 제한하는 방식입니다.

#### 사용법

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from enum import Enum

# 1) 출력 스키마 정의
class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Bug(BaseModel):
    line: int = Field(ge=1, description="버그가 있는 라인 번호")
    issue: str = Field(description="버그 설명")
    severity: Severity = Field(description="심각도")

class CodeReviewResult(BaseModel):
    """코드 리뷰 결과 스키마."""
    bugs: list[Bug] = Field(default_factory=list, description="발견된 버그 목록")
    score: int = Field(ge=1, le=10, description="코드 품질 점수 (1~10)")
    summary: str = Field(description="전체 리뷰 요약")

# 2) LLM에 structured output 적용
llm = ChatAnthropic(model="claude-sonnet-4-5-20250514", temperature=0.1)
structured_llm = llm.with_structured_output(CodeReviewResult)

# 3) 프롬프트 구성 (JSON 스키마 텍스트 불필요!)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 코드 리뷰 전문가입니다.\n주어진 코드의 버그를 분석하세요."),
    ("human", "언어: {language}\n코드:\n{code}"),
])

# 4) 체인 실행 (파서 불필요!)
chain = prompt | structured_llm
result = chain.invoke({
    "language": "python",
    "code": "def divide(a, b):\n    return a / b"
})

# result는 CodeReviewResult 인스턴스 (Pydantic 모델)
print(type(result))  # <class 'CodeReviewResult'>
print(result.bugs)   # [Bug(line=2, issue="ZeroDivisionError...", severity=<Severity.HIGH>)]
print(result.score)  # 4
```

> **주목**: `with_structured_output()`을 사용하면 프롬프트에 JSON 스키마를 텍스트로 적을 필요가 없습니다. LLM이 Pydantic 모델의 구조를 자동으로 이해합니다.

### 2.3 JSON 출력 파서 패턴

`with_structured_output()`이 지원되지 않는 모델을 사용하거나, 폴백이 필요한 경우를 위한 JSON 파서를 만들어 봅시다.

#### 기본 JSON 파서 (3단계 폴백)

LLM 응답에서 JSON을 추출하는 3단계 전략:

```python
import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON을 추출한다.

    3단계 폴백 전략:
    1) 직접 JSON 파싱 시도
    2) 마크다운 코드블록(```json ... ```) 내부 추출
    3) 첫 번째 { ... } 블록 탐색

    Args:
        text: LLM의 원본 응답 텍스트

    Returns:
        파싱된 딕셔너리

    Raises:
        ValueError: 모든 전략 실패 시
    """
    # 1단계: 직접 파싱 (응답 전체가 유효한 JSON인 경우)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2단계: 마크다운 코드블록 추출
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # 3단계: 중괄호 블록 탐색 (가장 바깥쪽 { ... })
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"JSON 파싱 실패. 원본 응답: {text[:200]}...")
```

#### ValidatedJsonOutputParser: 파싱 + Pydantic 검증 결합

JSON 파싱에 성공해도, 구조가 맞는지 추가로 검증해야 합니다:

```python
from pydantic import BaseModel, ValidationError


class ValidatedJsonOutputParser:
    """JSON 파싱 + Pydantic 스키마 검증을 결합한 파서.

    Usage:
        parser = ValidatedJsonOutputParser(schema=MyModel)
        result = parser.parse(llm_response_text)
    """

    def __init__(self, schema: type[BaseModel] | None = None, strict: bool = False):
        self.schema = schema
        self.strict = strict  # True: 검증 실패 시 예외, False: 원본 dict 반환

    def parse(self, text: str) -> dict:
        """텍스트를 파싱하고 스키마로 검증한다."""
        # 1) JSON 파싱
        raw = parse_json_response(text)

        # 2) 스키마 검증 (스키마가 지정된 경우)
        if self.schema is None:
            return raw

        try:
            validated = self.schema.model_validate(raw)
            logger.info("Schema validation passed: %s", self.schema.__name__)
            return validated.model_dump()
        except ValidationError as exc:
            logger.warning(
                "Schema validation failed: %s, errors: %s",
                self.schema.__name__,
                exc.errors()[:3],  # 처음 3개 오류만 로그
            )
            if self.strict:
                raise

            # 부분 복구: Pydantic의 기본값으로 누락 필드를 채워봄
            try:
                validated = self.schema.model_validate(raw, strict=False)
                return validated.model_dump()
            except ValidationError:
                logger.error("Recovery failed, returning raw dict")
                return raw
```

**사용법:**
```python
from langchain_core.output_parsers import StrOutputParser

# 방법 1: with_structured_output (권장)
structured_llm = llm.with_structured_output(CodeReviewResult)
chain = prompt | structured_llm

# 방법 2: JSON 파서 + Pydantic 검증 (폴백용)
parser = ValidatedJsonOutputParser(schema=CodeReviewResult)
chain = prompt | llm | StrOutputParser()
raw_text = chain.invoke({...})
result = parser.parse(raw_text)
```

### 2.4 스키마 설계 모범 사례

#### 필수 vs 선택 필드

```python
class AnalysisResult(BaseModel):
    # 필수 필드: 반드시 있어야 하는 핵심 데이터
    summary: str = Field(min_length=1, description="분석 요약")
    affected_files: list[str] = Field(min_length=1, description="영향받는 파일")

    # 선택 필드: 없을 수도 있는 부가 데이터
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="신뢰도")
    tags: list[str] = Field(default_factory=list, description="분류 태그")
    notes: str = Field(default="", description="추가 메모")
```

#### 기본값 전략

```python
class TestEvaluation(BaseModel):
    status: TestStatus                                    # 필수 (기본값 없음)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # 기본 0.5 (판단 불가 시)
    reasoning: str = Field(default="분석 정보 부족")          # 기본 메시지
    defects: list[str] = Field(default_factory=list)         # 기본 빈 리스트
```

> **원칙**: 핵심 필드는 필수로, 부가 필드는 합리적인 기본값과 함께 선택 필드로 설정합니다.

#### 중첩 모델

```python
class DefectInfo(BaseModel):
    """발견된 결함 정보."""
    description: str = Field(min_length=1)
    affected_file: str = Field(default="")
    severity: Severity

class TestEvaluation(BaseModel):
    """테스트 평가 결과."""
    status: TestStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    defects_found: list[DefectInfo] = Field(default_factory=list)  # 중첩 모델 리스트
```

---

## 3. 실전 예제

### 3.1 "버그 리포트 분석기" 출력 스키마 설계

완전한 버그 리포트 분석기를 만들어 봅시다:

```python
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


# --- 스키마 정의 ---
class BugCategory(str, Enum):
    LOGIC = "logic"           # 로직 오류
    NULL_REFERENCE = "null_reference"  # Null 참조
    SECURITY = "security"     # 보안 취약점
    PERFORMANCE = "performance"  # 성능 이슈
    OTHER = "other"

class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class SuggestedFix(BaseModel):
    description: str = Field(description="수정 방법 설명")
    code_snippet: str = Field(default="", description="수정 코드 예시")

class BugAnalysis(BaseModel):
    """단일 버그 분석 결과."""
    title: str = Field(min_length=1, description="버그 제목 (한 줄)")
    category: BugCategory = Field(description="버그 카테고리")
    priority: Priority = Field(description="우선순위")
    affected_lines: list[int] = Field(description="영향받는 라인 번호들")
    description: str = Field(min_length=10, description="상세 설명")
    suggested_fix: SuggestedFix = Field(description="수정 제안")

    @field_validator("affected_lines")
    @classmethod
    def validate_lines(cls, v: list[int]) -> list[int]:
        return [line for line in v if line > 0]

class BugReportAnalysis(BaseModel):
    """전체 버그 리포트 분석 결과."""
    total_bugs: int = Field(ge=0, description="발견된 총 버그 수")
    bugs: list[BugAnalysis] = Field(default_factory=list)
    overall_risk: Priority = Field(description="전체 위험도")
    summary: str = Field(min_length=1, description="분석 요약")

    @field_validator("total_bugs")
    @classmethod
    def match_bug_count(cls, v: int, info) -> int:
        bugs = info.data.get("bugs", [])
        if bugs and v != len(bugs):
            return len(bugs)
        return v


# --- 프롬프트 + 체인 ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 시니어 버그 분석가입니다.\n"
     "주어진 코드의 버그를 분석하고 수정 방안을 제시하세요."),
    ("human", "파일: {file_path}\n언어: {language}\n\n```\n{code}\n```"),
])

llm = ChatAnthropic(model="claude-sonnet-4-5-20250514", temperature=0.1)
structured_llm = llm.with_structured_output(BugReportAnalysis)
chain = prompt | structured_llm

# --- 실행 ---
result = chain.invoke({
    "file_path": "src/auth/login_service.py",
    "language": "python",
    "code": (
        "def authenticate(email, password):\n"
        "    user = db.query(f'SELECT * FROM users WHERE email={email}')\n"
        "    if user.password == password:\n"
        "        return create_token(user)\n"
        "    return None"
    ),
})

print(f"총 {result.total_bugs}개 버그 발견")
for bug in result.bugs:
    print(f"  [{bug.priority.value}] {bug.title} (카테고리: {bug.category.value})")
    print(f"    설명: {bug.description}")
    print(f"    수정: {bug.suggested_fix.description}")
```

---

## 4. 연습 문제

### 연습 1: "회의록 요약기" Pydantic 모델 설계

**과제**: 회의록을 요약하는 LLM의 출력 스키마를 Pydantic으로 설계하세요.

**요구사항:**
1. `MeetingSummary` 모델에 다음 필드 포함:
   - `title`: 회의 제목 (필수, 빈 문자열 금지)
   - `date`: 회의 날짜 (문자열, 필수)
   - `attendees`: 참석자 목록 (최소 1명)
   - `summary`: 요약 (최소 20자)
   - `action_items`: 액션 아이템 목록 (각 항목은 중첩 모델)
   - `decisions`: 결정 사항 목록

2. `ActionItem` 중첩 모델:
   - `task`: 할 일 설명 (필수)
   - `assignee`: 담당자 (필수)
   - `deadline`: 마감일 (선택, 기본값 "미정")
   - `priority`: HIGH/MEDIUM/LOW (Enum)

3. `with_structured_output()`을 사용하는 체인 구성

**힌트**: Enum과 Field의 옵션을 적극 활용하세요.

### 연습 2: JSON 파서 테스트

**과제**: `parse_json_response()` 함수를 다음 3가지 입력에 대해 테스트하세요.

```python
# 입력 1: 깔끔한 JSON
text1 = '{"status": "PASS", "score": 8}'

# 입력 2: 마크다운 코드블록
text2 = "분석 결과입니다:\\n```json\\n{\"status\": \"FAIL\", \"score\": 3}\\n```"

# 입력 3: 텍스트 속에 묻힌 JSON
text3 = '결과를 알려드립니다. {"status": "BLOCKED", "score": 5} 추가 분석이 필요합니다.'
```

각 입력에 대해 파싱 결과를 확인하고, 어떤 단계(1단계/2단계/3단계)에서 성공했는지 설명하세요.

---

## 5. 핵심 정리

| 개념 | 핵심 내용 |
|------|----------|
| **문제** | LLM은 자유 텍스트를 반환하지만, 프로그램은 구조화된 데이터가 필요 |
| **Pydantic** | `BaseModel` 상속으로 스키마 정의, `Field`로 제약 조건 설정 |
| **Enum** | 허용 값을 명시적으로 제한 (예: HIGH/MEDIUM/LOW) |
| **field_validator** | 커스텀 검증 로직 추가 (반올림, 빈 값 필터링 등) |
| **with_structured_output()** | LLM이 Pydantic 스키마에 맞게 응답하도록 강제. 가장 안정적 |
| **JSON 파서 폴백** | 3단계: 직접 파싱 -> 코드블록 추출 -> 중괄호 탐색 |
| **ValidatedJsonOutputParser** | JSON 파싱 + Pydantic 검증 결합. 폴백용 |
| **스키마 설계** | 필수 필드는 기본값 없이, 선택 필드는 합리적 기본값과 함께 |

---

## 6. 참고 자료

- **Pydantic 공식 문서**: https://docs.pydantic.dev/latest/
  - BaseModel, Field, Validator의 모든 옵션을 상세히 설명합니다.

- **LangChain Structured Output 가이드**: https://python.langchain.com/docs/how_to/structured_output/
  - `with_structured_output()`의 사용법과 다양한 LLM 프로바이더별 차이점.

- **LangChain Output Parsers**: https://python.langchain.com/docs/concepts/output_parsers/
  - StrOutputParser, JsonOutputParser 등 다양한 파서의 사용법.

- **Anthropic Structured Output**: https://docs.anthropic.com/en/docs/build-with-claude/structured-output
  - Claude의 네이티브 JSON Schema 지원에 대한 설명.

---

## 다음 단계

구조화된 출력을 마스터했다면, 다음 모듈에서는 LLM 호출의 **비용, 속도, 안정성을 최적화**하는 방법을 배웁니다.

**Module 07: LLM 호출 최적화** 에서 다루는 내용:
- 모델별 가격 비교와 태스크에 맞는 모델 선택
- 토큰 사용량 추적과 비용 모니터링
- 재시도, 타임아웃, 캐싱 전략
