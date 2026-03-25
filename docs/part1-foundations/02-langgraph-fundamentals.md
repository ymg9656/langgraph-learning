# Module 02: LangGraph 기초 - 그래프, 노드, 엣지, 상태

> **학습 시간**: 약 2.5시간
> **난이도**: ★★☆☆☆ (초급)
> **코드 실습**: 있음 (Python 코드 직접 작성)

---

## 학습 목표

이 모듈을 완료하면 다음을 할 수 있습니다:

- [ ] LangGraph의 역할과 장점을 설명할 수 있다
- [ ] TypedDict로 상태(State)를 정의할 수 있다
- [ ] 노드 함수를 작성하고 그래프에 추가할 수 있다
- [ ] 일반 엣지와 조건부 엣지를 사용할 수 있다
- [ ] StateGraph를 컴파일하고 실행할 수 있다
- [ ] Mermaid 다이어그램으로 그래프를 시각화할 수 있다

---

## 사전 지식

이 모듈을 시작하기 전에 다음을 확인하세요:

- [ ] Module 01 완료 (에이전트 개념 이해)
- [ ] Python 기초 (함수, 딕셔너리, 타입 힌트)
- [ ] 환경 설정 완료 (`pip install langgraph` 설치됨)
- [ ] 코드 에디터 준비 (VS Code 권장)

> **Python 딕셔너리가 낯설다면?**
> ```python
> # 딕셔너리 = 키:값 쌍의 모음
> person = {"name": "홍길동", "age": 30}
> print(person["name"])  # "홍길동"
> ```

---

## 1. 개념 설명

### 1.1 LangGraph란?

> **LangGraph**는 **코드로 워크플로우를 그래프로 표현하는 프레임워크**입니다.

Module 01에서 배운 노드, 엣지, 상태, 그래프 개념을 실제 Python 코드로
구현할 수 있게 해주는 도구입니다.

```
┌─────────────────────────────────────────────────────────┐
│                LangGraph의 역할                          │
│                                                         │
│   "이런 에이전트를 만들고 싶다"  (아이디어)                  │
│        ↓                                                │
│   ┌──────────────────────────────────────┐              │
│   │            LangGraph                  │              │
│   │                                      │              │
│   │  - 상태(State) 정의 도구              │              │
│   │  - 노드(Node) 등록 도구               │              │
│   │  - 엣지(Edge) 연결 도구               │              │
│   │  - 그래프 컴파일 & 실행 엔진          │              │
│   │  - 시각화 도구                        │              │
│   │                                      │              │
│   └──────────────────────────────────────┘              │
│        ↓                                                │
│   "실제로 동작하는 에이전트"  (코드)                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### LangGraph를 사용하는 이유

| 직접 구현 | LangGraph 사용 |
|----------|---------------|
| 상태 관리를 직접 코딩 | TypedDict로 간단히 정의 |
| 분기 로직을 if-else로 복잡하게 작성 | 조건부 엣지로 깔끔하게 표현 |
| 실행 흐름 추적이 어려움 | 자동 시각화 (Mermaid) |
| 에러 시 어디서 멈췄는지 파악 어려움 | 체크포인트로 자동 추적 |
| 테스트가 어려움 | 노드 단위 테스트 가능 |

---

### 1.2 핵심 개념 상세 설명

#### (1) StateGraph - 상태를 중심으로 동작하는 그래프

StateGraph는 LangGraph의 핵심 클래스입니다. 모든 에이전트는 StateGraph로 시작합니다.

```
┌─────────────────────────────────────────────────────────┐
│                   StateGraph 구조                        │
│                                                         │
│   StateGraph(상태 타입)                                   │
│        │                                                │
│        ├── add_node("이름", 함수)      ← 노드 추가       │
│        ├── add_edge("A", "B")         ← 엣지 추가       │
│        ├── set_entry_point("시작")     ← 시작점 설정     │
│        ├── add_conditional_edges(...) ← 조건부 엣지      │
│        │                                                │
│        └── compile()                  ← 실행 가능하게 컴파일│
│              │                                          │
│              └── invoke(초기상태)      ← 실행!           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### (2) TypedDict - 상태의 타입 정의

TypedDict는 Python의 내장 기능으로, "이 딕셔너리에는 이런 키와 타입이 있다"를
선언하는 방법입니다.

```python
from typing import TypedDict

# 일반 딕셔너리 (타입 정보 없음)
state = {"message": "안녕", "count": 0}

# TypedDict (타입 정보 있음) - LangGraph에서 사용
class MyState(TypedDict):
    message: str      # message는 문자열
    count: int        # count는 정수
```

> **왜 TypedDict를 사용하나요?**
> - 상태에 어떤 데이터가 있는지 한눈에 파악
> - 에디터에서 자동완성 지원
> - 타입 실수를 미리 발견

```
┌─────────────────────────────────────────────────────────┐
│              TypedDict 시각적 이해                        │
│                                                         │
│   class AgentState(TypedDict):                          │
│       messages: list[str]    ← 대화 내역 (문자열 목록)    │
│       current_step: str      ← 현재 단계 (문자열)        │
│       result: str            ← 최종 결과 (문자열)        │
│                                                         │
│   이것은 마치 "양식지"와 같습니다:                         │
│                                                         │
│   ┌──────────────────────────────────┐                  │
│   │ [양식지]                          │                  │
│   │                                  │                  │
│   │ messages:     [_____________]    │                  │
│   │ current_step: [_____________]    │                  │
│   │ result:       [_____________]    │                  │
│   │                                  │                  │
│   │ → 각 노드가 이 양식지를 읽고 채움  │                  │
│   └──────────────────────────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### (3) Node - 상태를 받아 변환하는 함수

노드는 **상태를 입력받아, 변경할 부분만 반환하는 함수**입니다.

```python
# 노드 함수의 기본 형태
def my_node(state: MyState) -> dict:
    # 1. 상태에서 필요한 데이터를 읽음
    current_message = state["message"]

    # 2. 처리 로직 수행
    new_message = current_message + " (처리됨)"

    # 3. 변경할 부분만 딕셔너리로 반환
    return {"message": new_message}
```

```
┌─────────────────────────────────────────────────────────┐
│                   노드의 동작 원리                        │
│                                                         │
│   상태 (입력)              노드 함수              상태 (갱신) │
│   ┌────────────┐     ┌───────────────┐     ┌────────────┐│
│   │message: "안녕"│──▶│  처리 로직     │──▶│message:     ││
│   │count: 0     │    │  (변환, 계산)  │    │ "안녕 (처리됨)"││
│   └────────────┘     └───────────────┘     │count: 0     ││
│                                            └────────────┘│
│                                                         │
│   ※ 노드가 반환하지 않은 필드(count)는 그대로 유지됩니다     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

> **핵심 규칙**: 노드 함수는 **변경할 필드만** 딕셔너리로 반환합니다.
> 반환하지 않은 필드는 이전 값이 그대로 유지됩니다.

#### (4) Edge - 노드 간 연결

엣지는 두 가지 종류가 있습니다:

**일반 엣지**: 항상 같은 다음 노드로 이동

```python
# A 노드 다음에는 항상 B 노드 실행
graph.add_edge("A", "B")
```

**조건부 엣지**: 조건에 따라 다른 노드로 이동

```python
# A 노드 후 routing_function의 결과에 따라 분기
graph.add_conditional_edges(
    "A",                        # 출발 노드
    routing_function,           # 분기 조건 함수
    {
        "positive": "B",        # 결과가 "positive"이면 B로
        "negative": "C",        # 결과가 "negative"이면 C로
    }
)
```

```
┌─────────────────────────────────────────────────────────┐
│              엣지 종류 비교                               │
│                                                         │
│   일반 엣지:                                             │
│                                                         │
│   [노드A] ─────────────────▶ [노드B]                    │
│            (항상 B로 이동)                                │
│                                                         │
│                                                         │
│   조건부 엣지:                                            │
│                                                         │
│                  ┌── "양성" ──▶ [노드B]                  │
│   [노드A] ──(판단)──┤                                    │
│                  └── "음성" ──▶ [노드C]                  │
│             (조건에 따라 분기)                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### (5) END - 종료 노드

`END`는 그래프의 실행을 종료하는 특별한 노드입니다.

```python
from langgraph.graph import StateGraph, END

# 마지막 노드 다음에 종료
graph.add_edge("last_node", END)
```

---

## 2. 단계별 실습

### 첫 번째 그래프 만들기: "인사 처리기"

가장 간단한 2노드 그래프를 만들어봅시다.

```
   목표: 이름을 입력하면 인사말을 생성하는 그래프

   [START] → [인사생성] → [포맷팅] → [END]
```

#### Step 1: State 정의

파일을 생성합니다: `01_first_graph.py`

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END


# Step 1: 상태(State) 정의
# - 그래프 전체에서 공유할 데이터의 구조를 선언합니다
class GreetingState(TypedDict):
    name: str           # 사용자 이름
    greeting: str       # 생성된 인사말
    formatted: str      # 최종 포맷된 결과
```

> **이해 포인트**: `GreetingState`는 "이 그래프에서 사용할 데이터는
> name, greeting, formatted 세 가지입니다"라고 선언하는 것입니다.

#### Step 2: 노드 함수 작성

```python
# Step 2: 노드 함수 작성
# - 각 노드는 state를 받아서, 변경할 부분만 딕셔너리로 반환합니다

def generate_greeting(state: GreetingState) -> dict:
    """인사말을 생성하는 노드"""
    name = state["name"]
    greeting = f"안녕하세요, {name}님!"
    print(f"  [인사생성 노드] '{name}' → '{greeting}'")
    return {"greeting": greeting}


def format_output(state: GreetingState) -> dict:
    """결과를 포맷팅하는 노드"""
    greeting = state["greeting"]
    formatted = f"=== {greeting} 좋은 하루 되세요! ==="
    print(f"  [포맷팅 노드] '{greeting}' → '{formatted}'")
    return {"formatted": formatted}
```

> **이해 포인트**: 각 함수는 `state`에서 필요한 데이터를 읽고,
> 처리 결과를 딕셔너리로 반환합니다. 반환된 값만 상태가 업데이트됩니다.

#### Step 3: StateGraph 생성, 노드/엣지 추가

```python
# Step 3: 그래프 생성 및 구성
# - StateGraph에 상태 타입을 전달하여 생성
# - 노드와 엣지를 추가하여 워크플로우를 정의

# 3-1: 그래프 생성 (상태 타입 지정)
graph = StateGraph(GreetingState)

# 3-2: 노드 추가 (이름, 함수)
graph.add_node("generate", generate_greeting)
graph.add_node("format", format_output)

# 3-3: 시작점 설정 (가장 먼저 실행할 노드)
graph.set_entry_point("generate")

# 3-4: 엣지 추가 (노드 간 연결)
graph.add_edge("generate", "format")    # generate 다음에 format 실행
graph.add_edge("format", END)           # format 다음에 종료
```

> **이해 포인트**: 마치 레고 블록을 조립하듯 노드를 추가하고 연결합니다.
> `set_entry_point`는 "여기서 시작해!"를 지정하는 것입니다.

#### Step 4: 컴파일 및 실행

```python
# Step 4: 컴파일 및 실행

# 4-1: 그래프 컴파일 (실행 가능한 형태로 변환)
app = graph.compile()

# 4-2: 초기 상태를 전달하여 실행
print("=== 그래프 실행 시작 ===")
result = app.invoke({
    "name": "홍길동",
    "greeting": "",
    "formatted": ""
})

# 4-3: 결과 확인
print("\n=== 실행 결과 ===")
print(f"최종 결과: {result['formatted']}")
```

#### Step 5: 결과 확인

전체 코드를 실행합니다:

```bash
python 01_first_graph.py
```

예상 출력:

```
=== 그래프 실행 시작 ===
  [인사생성 노드] '홍길동' → '안녕하세요, 홍길동님!'
  [포맷팅 노드] '안녕하세요, 홍길동님!' → '=== 안녕하세요, 홍길동님! 좋은 하루 되세요! ==='

=== 실행 결과 ===
최종 결과: === 안녕하세요, 홍길동님! 좋은 하루 되세요! ===
```

#### 전체 코드 (복사용)

```python
"""
01_first_graph.py - 첫 번째 LangGraph 그래프
목표: 이름을 입력하면 인사말을 생성하는 간단한 2노드 그래프
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


# 1. 상태 정의
class GreetingState(TypedDict):
    name: str
    greeting: str
    formatted: str


# 2. 노드 함수 작성
def generate_greeting(state: GreetingState) -> dict:
    """인사말을 생성하는 노드"""
    name = state["name"]
    greeting = f"안녕하세요, {name}님!"
    print(f"  [인사생성 노드] '{name}' → '{greeting}'")
    return {"greeting": greeting}


def format_output(state: GreetingState) -> dict:
    """결과를 포맷팅하는 노드"""
    greeting = state["greeting"]
    formatted = f"=== {greeting} 좋은 하루 되세요! ==="
    print(f"  [포맷팅 노드] '{greeting}' → '{formatted}'")
    return {"formatted": formatted}


# 3. 그래프 구성
graph = StateGraph(GreetingState)
graph.add_node("generate", generate_greeting)
graph.add_node("format", format_output)
graph.set_entry_point("generate")
graph.add_edge("generate", "format")
graph.add_edge("format", END)

# 4. 컴파일 및 실행
app = graph.compile()

print("=== 그래프 실행 시작 ===")
result = app.invoke({
    "name": "홍길동",
    "greeting": "",
    "formatted": ""
})

print("\n=== 실행 결과 ===")
print(f"최종 결과: {result['formatted']}")
```

---

### 그래프 실행 흐름 추적

위 코드가 실행될 때 내부에서 일어나는 일을 단계별로 살펴봅시다:

```
┌─────────────────────────────────────────────────────────┐
│                 실행 흐름 상세 추적                        │
│                                                         │
│  [1] invoke() 호출                                      │
│      초기 상태: {name: "홍길동", greeting: "", formatted: ""}│
│                                                         │
│  [2] entry_point → "generate" 노드 실행                  │
│      입력: {name: "홍길동", greeting: "", formatted: ""}   │
│      반환: {greeting: "안녕하세요, 홍길동님!"}              │
│      상태: {name: "홍길동",                               │
│             greeting: "안녕하세요, 홍길동님!",              │
│             formatted: ""}                               │
│                                                         │
│  [3] edge → "format" 노드 실행                           │
│      입력: {name: "홍길동",                               │
│             greeting: "안녕하세요, 홍길동님!",              │
│             formatted: ""}                               │
│      반환: {formatted: "=== 안녕하세요, ... ==="}          │
│      상태: {name: "홍길동",                               │
│             greeting: "안녕하세요, 홍길동님!",              │
│             formatted: "=== 안녕하세요, ... ==="}          │
│                                                         │
│  [4] edge → END (종료)                                   │
│      최종 상태 반환                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 조건부 엣지 (add_conditional_edges)

이제 조건에 따라 다른 경로로 분기하는 그래프를 만들어봅시다.

파일: `02_conditional_graph.py`

```
   목표: 점수에 따라 다른 메시지를 생성하는 그래프

                          ┌── 점수 >= 80 ──▶ [축하 메시지] ──┐
   [START] → [점수확인] ──┤                                   ├──▶ [END]
                          └── 점수 <  80 ──▶ [격려 메시지] ──┘
```

```python
"""
02_conditional_graph.py - 조건부 엣지 실습
목표: 점수에 따라 다른 메시지를 생성하는 그래프
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


# 1. 상태 정의
class ScoreState(TypedDict):
    student_name: str       # 학생 이름
    score: int              # 점수
    grade: str              # 등급 (pass/fail)
    message: str            # 결과 메시지


# 2. 노드 함수들
def check_score(state: ScoreState) -> dict:
    """점수를 확인하고 등급을 판정하는 노드"""
    score = state["score"]
    grade = "pass" if score >= 80 else "fail"
    print(f"  [점수확인] {state['student_name']}: {score}점 → {grade}")
    return {"grade": grade}


def congratulate(state: ScoreState) -> dict:
    """합격 축하 메시지를 생성하는 노드"""
    msg = f"축하합니다, {state['student_name']}님! {state['score']}점으로 합격입니다!"
    print(f"  [축하] {msg}")
    return {"message": msg}


def encourage(state: ScoreState) -> dict:
    """격려 메시지를 생성하는 노드"""
    msg = f"아쉽지만 {state['student_name']}님, {state['score']}점입니다. 다음에 화이팅!"
    print(f"  [격려] {msg}")
    return {"message": msg}


# 3. 라우팅 함수 (조건부 엣지에서 사용)
def route_by_grade(state: ScoreState) -> str:
    """등급에 따라 다음 노드를 결정하는 함수"""
    if state["grade"] == "pass":
        return "pass"
    else:
        return "fail"


# 4. 그래프 구성
graph = StateGraph(ScoreState)

# 노드 추가
graph.add_node("check", check_score)
graph.add_node("congratulate", congratulate)
graph.add_node("encourage", encourage)

# 시작점 설정
graph.set_entry_point("check")

# 조건부 엣지: check 노드 후 grade에 따라 분기
graph.add_conditional_edges(
    "check",            # 출발 노드
    route_by_grade,     # 라우팅 함수
    {
        "pass": "congratulate",     # "pass" → 축하 노드로
        "fail": "encourage",        # "fail" → 격려 노드로
    }
)

# 축하/격려 노드 → 종료
graph.add_edge("congratulate", END)
graph.add_edge("encourage", END)

# 5. 컴파일 및 실행
app = graph.compile()

# 테스트 1: 높은 점수
print("=== 테스트 1: 높은 점수 ===")
result1 = app.invoke({
    "student_name": "김철수",
    "score": 95,
    "grade": "",
    "message": ""
})
print(f"결과: {result1['message']}\n")

# 테스트 2: 낮은 점수
print("=== 테스트 2: 낮은 점수 ===")
result2 = app.invoke({
    "student_name": "이영희",
    "score": 65,
    "grade": "",
    "message": ""
})
print(f"결과: {result2['message']}")
```

실행 결과:

```
=== 테스트 1: 높은 점수 ===
  [점수확인] 김철수: 95점 → pass
  [축하] 축하합니다, 김철수님! 95점으로 합격입니다!
결과: 축하합니다, 김철수님! 95점으로 합격입니다!

=== 테스트 2: 낮은 점수 ===
  [점수확인] 이영희: 65점 → fail
  [격려] 아쉽지만 이영희님, 65점입니다. 다음에 화이팅!
결과: 아쉽지만 이영희님, 65점입니다. 다음에 화이팅!
```

#### 라우팅 함수의 동작 원리

```
┌─────────────────────────────────────────────────────────┐
│               라우팅 함수의 역할                           │
│                                                         │
│   check 노드 실행 후:                                    │
│   상태 = {student_name: "김철수", score: 95, grade: "pass"}│
│                                                         │
│        ↓                                                │
│   route_by_grade(state) 호출                             │
│        ↓                                                │
│   state["grade"] == "pass"  →  "pass" 반환               │
│        ↓                                                │
│   매핑: {"pass": "congratulate", "fail": "encourage"}    │
│        ↓                                                │
│   "pass" → "congratulate" 노드 실행                      │
│                                                         │
│                                                         │
│   ※ 라우팅 함수는 반드시 매핑의 키 중 하나를 반환해야 합니다 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### Mermaid 시각화

LangGraph는 그래프를 Mermaid 다이어그램으로 자동 변환할 수 있습니다.

```python
"""
03_visualize_graph.py - 그래프 시각화
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


class ScoreState(TypedDict):
    student_name: str
    score: int
    grade: str
    message: str


def check_score(state: ScoreState) -> dict:
    grade = "pass" if state["score"] >= 80 else "fail"
    return {"grade": grade}

def congratulate(state: ScoreState) -> dict:
    return {"message": "합격!"}

def encourage(state: ScoreState) -> dict:
    return {"message": "다음 기회에!"}

def route_by_grade(state: ScoreState) -> str:
    return "pass" if state["grade"] == "pass" else "fail"


graph = StateGraph(ScoreState)
graph.add_node("check", check_score)
graph.add_node("congratulate", congratulate)
graph.add_node("encourage", encourage)
graph.set_entry_point("check")
graph.add_conditional_edges("check", route_by_grade, {
    "pass": "congratulate",
    "fail": "encourage",
})
graph.add_edge("congratulate", END)
graph.add_edge("encourage", END)

app = graph.compile()

# Mermaid 다이어그램 출력
mermaid_code = app.get_graph().draw_mermaid()
print("=== Mermaid 다이어그램 ===")
print(mermaid_code)
```

출력되는 Mermaid 코드를 [Mermaid Live Editor](https://mermaid.live/)에
붙여넣으면 시각적인 다이어그램을 확인할 수 있습니다.

```
┌─────────────────────────────────────────────────────────┐
│                 시각화 활용 방법                           │
│                                                         │
│   1. 위 코드 실행하여 Mermaid 코드 복사                    │
│   2. https://mermaid.live/ 접속                          │
│   3. 왼쪽 에디터에 Mermaid 코드 붙여넣기                   │
│   4. 오른쪽에서 그래프 다이어그램 확인                       │
│                                                         │
│   ※ VS Code 사용자는 "Mermaid Preview" 확장을 설치하면     │
│     에디터에서 바로 확인할 수 있습니다                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 실전 예제

### 3-노드 "텍스트 분류기"

실전에 가까운 예제를 만들어봅시다. 입력된 텍스트의 주제를 분류하는 그래프입니다.

```
   전체 흐름:
   [START] → [입력처리] → [분류] → [결과포맷] → [END]
```

파일: `04_text_classifier.py`

```python
"""
04_text_classifier.py - 텍스트 분류기 (3노드 그래프)
목표: 입력 텍스트를 분석하여 카테고리를 분류하고 결과를 포맷팅
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


# ========================================
# 1. 상태 정의
# ========================================
class ClassifierState(TypedDict):
    raw_text: str           # 원본 입력 텍스트
    cleaned_text: str       # 전처리된 텍스트
    category: str           # 분류 결과
    confidence: float       # 신뢰도 (0.0 ~ 1.0)
    result_report: str      # 최종 결과 리포트


# ========================================
# 2. 카테고리 키워드 사전
# ========================================
# 실제 프로덕션에서는 LLM이 이 역할을 합니다.
# 여기서는 학습 목적으로 키워드 매칭을 사용합니다.
CATEGORY_KEYWORDS = {
    "기술": ["프로그래밍", "코드", "개발", "서버", "API", "데이터베이스",
             "Python", "JavaScript", "버그", "배포"],
    "비즈니스": ["매출", "수익", "마케팅", "전략", "투자", "시장",
                "고객", "영업", "경영", "계약"],
    "건강": ["운동", "식단", "다이어트", "병원", "건강", "수면",
            "스트레스", "영양", "체중", "의사"],
    "교육": ["학습", "강의", "시험", "학교", "교수", "수업",
            "과제", "학생", "교육", "공부"],
}


# ========================================
# 3. 노드 함수들
# ========================================
def preprocess(state: ClassifierState) -> dict:
    """입력 텍스트를 전처리하는 노드"""
    raw = state["raw_text"]

    # 공백 정리 및 소문자 변환 (한국어는 소문자 개념 없음)
    cleaned = raw.strip()
    # 불필요한 연속 공백 제거
    cleaned = " ".join(cleaned.split())

    print(f"  [전처리] 원본 길이: {len(raw)} → 정리 후: {len(cleaned)}")
    return {"cleaned_text": cleaned}


def classify(state: ClassifierState) -> dict:
    """텍스트를 카테고리로 분류하는 노드"""
    text = state["cleaned_text"]

    # 각 카테고리의 키워드 매칭 점수 계산
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        # 텍스트에 포함된 키워드 수 카운트
        match_count = sum(1 for kw in keywords if kw in text)
        scores[category] = match_count

    # 최고 점수 카테고리 선택
    if max(scores.values()) == 0:
        best_category = "기타"
        confidence = 0.0
    else:
        best_category = max(scores, key=scores.get)
        total_keywords = len(CATEGORY_KEYWORDS[best_category])
        confidence = round(scores[best_category] / total_keywords, 2)

    print(f"  [분류] 점수: {scores}")
    print(f"  [분류] 결과: {best_category} (신뢰도: {confidence})")
    return {
        "category": best_category,
        "confidence": confidence
    }


def format_result(state: ClassifierState) -> dict:
    """결과를 보기 좋게 포맷팅하는 노드"""
    report = (
        f"┌{'─' * 40}┐\n"
        f"│ 텍스트 분류 결과                        │\n"
        f"├{'─' * 40}┤\n"
        f"│ 입력: {state['cleaned_text'][:30]:30s}  │\n"
        f"│ 카테고리: {state['category']:28s}  │\n"
        f"│ 신뢰도: {state['confidence']:<30} │\n"
        f"└{'─' * 40}┘"
    )
    print(f"  [포맷] 리포트 생성 완료")
    return {"result_report": report}


# ========================================
# 4. 그래프 구성
# ========================================
graph = StateGraph(ClassifierState)

# 노드 추가
graph.add_node("preprocess", preprocess)
graph.add_node("classify", classify)
graph.add_node("format", format_result)

# 시작점 설정
graph.set_entry_point("preprocess")

# 엣지 연결: 전처리 → 분류 → 포맷 → 종료
graph.add_edge("preprocess", "classify")
graph.add_edge("classify", "format")
graph.add_edge("format", END)

# 컴파일
app = graph.compile()


# ========================================
# 5. 테스트 실행
# ========================================
test_texts = [
    "Python 프로그래밍으로 API 서버를 개발하고 배포했습니다",
    "올해 매출이 전년 대비 20% 증가하여 마케팅 전략을 수정합니다",
    "매일 30분 운동과 균형 잡힌 식단으로 건강을 관리하세요",
    "학생들의 학습 효과를 높이기 위해 새로운 교육 방법을 도입합니다",
]

for i, text in enumerate(test_texts, 1):
    print(f"\n{'='*50}")
    print(f"테스트 {i}: '{text[:40]}...'")
    print(f"{'='*50}")

    result = app.invoke({
        "raw_text": text,
        "cleaned_text": "",
        "category": "",
        "confidence": 0.0,
        "result_report": ""
    })

    print(f"\n{result['result_report']}")
```

실행:

```bash
python 04_text_classifier.py
```

---

## 4. 연습 문제

### 연습 1: 기본 그래프 수정

위의 "인사 처리기" (`01_first_graph.py`)를 수정하여 다음 기능을 추가해보세요:

1. 상태에 `language` 필드를 추가 (값: "ko" 또는 "en")
2. 새로운 노드 `detect_language`를 추가하여 이름에 한글이 포함되면 "ko", 아니면 "en"으로 설정
3. 실행 순서: `detect_language` → `generate` → `format` → END
4. `generate` 노드에서 language에 따라 한국어/영어 인사말 생성

**힌트**:
```python
# 한글 포함 여부 확인
def has_korean(text):
    for char in text:
        if '가' <= char <= '힣':
            return True
    return False
```

<details>
<summary>정답 코드 (클릭하여 펼치기)</summary>

```python
"""연습 1 정답: 다국어 인사 처리기"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


class GreetingState(TypedDict):
    name: str
    language: str
    greeting: str
    formatted: str


def has_korean(text: str) -> bool:
    for char in text:
        if '가' <= char <= '힣':
            return True
    return False


def detect_language(state: GreetingState) -> dict:
    lang = "ko" if has_korean(state["name"]) else "en"
    print(f"  [언어감지] '{state['name']}' → {lang}")
    return {"language": lang}


def generate_greeting(state: GreetingState) -> dict:
    name = state["name"]
    if state["language"] == "ko":
        greeting = f"안녕하세요, {name}님!"
    else:
        greeting = f"Hello, {name}!"
    print(f"  [인사생성] {greeting}")
    return {"greeting": greeting}


def format_output(state: GreetingState) -> dict:
    formatted = f">>> {state['greeting']} <<<"
    return {"formatted": formatted}


graph = StateGraph(GreetingState)
graph.add_node("detect", detect_language)
graph.add_node("generate", generate_greeting)
graph.add_node("format", format_output)
graph.set_entry_point("detect")
graph.add_edge("detect", "generate")
graph.add_edge("generate", "format")
graph.add_edge("format", END)

app = graph.compile()

# 테스트
for name in ["홍길동", "John"]:
    result = app.invoke({"name": name, "language": "", "greeting": "", "formatted": ""})
    print(f"결과: {result['formatted']}\n")
```

</details>

### 연습 2: 조건부 엣지 추가

다음 요구사항을 가진 그래프를 처음부터 만들어보세요:

**"주문 처리 그래프"**

상태:
- `item_name`: 상품명 (str)
- `quantity`: 수량 (int)
- `in_stock`: 재고 여부 (bool)
- `order_status`: 주문 상태 (str)
- `message`: 결과 메시지 (str)

노드:
- `check_stock`: 수량이 10 이하면 재고 있음(True), 10 초과면 재고 없음(False) 설정
- `process_order`: 주문 처리 (order_status = "완료")
- `backorder`: 입고 대기 처리 (order_status = "입고대기")

흐름:
```
[START] → [check_stock] ──(재고있음)──▶ [process_order] ──▶ [END]
                         └─(재고없음)──▶ [backorder] ──────▶ [END]
```

<details>
<summary>정답 코드 (클릭하여 펼치기)</summary>

```python
"""연습 2 정답: 주문 처리 그래프"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


class OrderState(TypedDict):
    item_name: str
    quantity: int
    in_stock: bool
    order_status: str
    message: str


def check_stock(state: OrderState) -> dict:
    in_stock = state["quantity"] <= 10
    print(f"  [재고확인] {state['item_name']} x {state['quantity']} → 재고: {'있음' if in_stock else '없음'}")
    return {"in_stock": in_stock}


def process_order(state: OrderState) -> dict:
    msg = f"{state['item_name']} {state['quantity']}개 주문이 완료되었습니다."
    print(f"  [주문처리] {msg}")
    return {"order_status": "완료", "message": msg}


def backorder(state: OrderState) -> dict:
    msg = f"{state['item_name']} {state['quantity']}개는 재고 부족으로 입고 대기 중입니다."
    print(f"  [입고대기] {msg}")
    return {"order_status": "입고대기", "message": msg}


def route_by_stock(state: OrderState) -> str:
    return "in_stock" if state["in_stock"] else "out_of_stock"


graph = StateGraph(OrderState)
graph.add_node("check_stock", check_stock)
graph.add_node("process_order", process_order)
graph.add_node("backorder", backorder)
graph.set_entry_point("check_stock")
graph.add_conditional_edges("check_stock", route_by_stock, {
    "in_stock": "process_order",
    "out_of_stock": "backorder",
})
graph.add_edge("process_order", END)
graph.add_edge("backorder", END)

app = graph.compile()

# 테스트
for item, qty in [("키보드", 5), ("모니터", 15)]:
    print(f"\n--- 주문: {item} x {qty} ---")
    result = app.invoke({
        "item_name": item,
        "quantity": qty,
        "in_stock": False,
        "order_status": "",
        "message": ""
    })
    print(f"결과: {result['message']}")
```

</details>

### 연습 3: 4-노드 "감정 분석기" 만들기 (도전!)

다음 요구사항으로 4노드 그래프를 만들어보세요:

**"텍스트 감정 분석기"**

상태:
- `raw_text`: 원본 텍스트
- `cleaned_text`: 전처리된 텍스트
- `sentiment`: 감정 ("긍정", "부정", "중립")
- `keywords`: 감지된 키워드 목록
- `report`: 최종 분석 리포트

노드:
1. `preprocess`: 텍스트 전처리 (공백 정리, 소문자 변환)
2. `analyze_sentiment`: 키워드 기반 감정 분석
   - 긍정 키워드: 좋다, 훌륭, 최고, 행복, 감사, 사랑, 만족, 추천, 완벽, 대박
   - 부정 키워드: 나쁘다, 최악, 불만, 화남, 실망, 싫다, 후회, 별로, 짜증, 불편
3. `generate_report`: 분석 리포트 생성
4. `add_recommendation`: 감정에 따른 추천 메시지 추가 (조건부 엣지 사용)

흐름:
```
[START] → [preprocess] → [analyze_sentiment] → [generate_report]
                                                       │
                                    ┌── 긍정 ──▶ [감사 메시지] ──┐
                                    ├── 부정 ──▶ [개선 안내] ────┤──▶ [END]
                                    └── 중립 ──▶ [기본 안내] ────┘
```

**힌트**: `add_recommendation` 대신 3개의 별도 노드(`thank`, `improve`, `default_msg`)를 만들고 조건부 엣지로 분기하세요.

<details>
<summary>정답 코드 (클릭하여 펼치기)</summary>

```python
"""연습 3 정답: 감정 분석기"""
from typing import TypedDict
from langgraph.graph import StateGraph, END


class SentimentState(TypedDict):
    raw_text: str
    cleaned_text: str
    sentiment: str
    keywords: list[str]
    report: str


POSITIVE_KW = ["좋다", "훌륭", "최고", "행복", "감사", "사랑", "만족", "추천", "완벽", "대박"]
NEGATIVE_KW = ["나쁘다", "최악", "불만", "화남", "실망", "싫다", "후회", "별로", "짜증", "불편"]


def preprocess(state: SentimentState) -> dict:
    cleaned = " ".join(state["raw_text"].strip().split())
    return {"cleaned_text": cleaned}


def analyze_sentiment(state: SentimentState) -> dict:
    text = state["cleaned_text"]
    pos = [kw for kw in POSITIVE_KW if kw in text]
    neg = [kw for kw in NEGATIVE_KW if kw in text]

    if len(pos) > len(neg):
        sentiment = "긍정"
    elif len(neg) > len(pos):
        sentiment = "부정"
    else:
        sentiment = "중립"

    keywords = pos + neg
    return {"sentiment": sentiment, "keywords": keywords}


def generate_report(state: SentimentState) -> dict:
    report = (
        f"[감정 분석 리포트]\n"
        f"  입력: {state['cleaned_text'][:50]}\n"
        f"  감정: {state['sentiment']}\n"
        f"  키워드: {', '.join(state['keywords']) if state['keywords'] else '없음'}"
    )
    return {"report": report}


def thank_message(state: SentimentState) -> dict:
    return {"report": state["report"] + "\n  >> 긍정적인 피드백 감사합니다!"}


def improve_message(state: SentimentState) -> dict:
    return {"report": state["report"] + "\n  >> 불편을 드려 죄송합니다. 개선하겠습니다."}


def default_message(state: SentimentState) -> dict:
    return {"report": state["report"] + "\n  >> 추가 의견이 있으시면 알려주세요."}


def route_sentiment(state: SentimentState) -> str:
    return state["sentiment"]


graph = StateGraph(SentimentState)
graph.add_node("preprocess", preprocess)
graph.add_node("analyze", analyze_sentiment)
graph.add_node("report", generate_report)
graph.add_node("thank", thank_message)
graph.add_node("improve", improve_message)
graph.add_node("default", default_message)

graph.set_entry_point("preprocess")
graph.add_edge("preprocess", "analyze")
graph.add_edge("analyze", "report")
graph.add_conditional_edges("report", route_sentiment, {
    "긍정": "thank",
    "부정": "improve",
    "중립": "default",
})
graph.add_edge("thank", END)
graph.add_edge("improve", END)
graph.add_edge("default", END)

app = graph.compile()

# 테스트
texts = [
    "이 서비스 정말 최고입니다! 만족하고 추천합니다",
    "너무 별로예요. 불만이고 실망입니다",
    "오늘 점심 메뉴는 김치찌개입니다",
]

for text in texts:
    print(f"\n{'='*50}")
    result = app.invoke({
        "raw_text": text, "cleaned_text": "", "sentiment": "",
        "keywords": [], "report": ""
    })
    print(result["report"])
```

</details>

---

## 5. 핵심 정리

### 이 모듈에서 배운 것

```
┌─────────────────────────────────────────────────────────┐
│                  Module 02 핵심 정리                      │
│                                                         │
│  1. LangGraph = 워크플로우를 그래프로 표현하는 프레임워크    │
│                                                         │
│  2. 핵심 요소:                                           │
│     - StateGraph: 그래프 객체 생성                        │
│     - TypedDict: 상태 타입 선언                           │
│     - Node: state → dict 형태의 함수                     │
│     - Edge: 노드 간 연결 (일반/조건부)                    │
│     - END: 그래프 종료 지점                               │
│                                                         │
│  3. 그래프 생성 5단계:                                    │
│     Step 1: TypedDict로 State 정의                       │
│     Step 2: 노드 함수 작성                                │
│     Step 3: StateGraph 생성 + 노드/엣지 추가              │
│     Step 4: compile()로 컴파일                            │
│     Step 5: invoke()로 실행                               │
│                                                         │
│  4. 조건부 엣지:                                         │
│     - add_conditional_edges(출발, 라우팅함수, 매핑)       │
│     - 라우팅 함수는 매핑의 키 중 하나를 반환                │
│                                                         │
│  5. 시각화:                                              │
│     - app.get_graph().draw_mermaid()                    │
│     - Mermaid Live Editor에서 확인                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 자주 하는 실수

| 실수 | 해결 방법 |
|------|----------|
| 노드 함수에서 전체 상태를 반환 | 변경할 필드만 딕셔너리로 반환 |
| set_entry_point를 빼먹음 | 반드시 시작 노드를 지정해야 함 |
| END를 연결하지 않음 | 마지막 노드에 END 엣지 필수 |
| 조건부 엣지 매핑에 없는 키 반환 | 라우팅 함수가 매핑 키 중 하나를 반환하는지 확인 |
| compile() 전에 invoke() 호출 | 반드시 compile() 후 invoke() |

### 코드 패턴 요약

```python
# === LangGraph 기본 패턴 ===
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. 상태 정의
class MyState(TypedDict):
    field1: str
    field2: int

# 2. 노드 함수
def my_node(state: MyState) -> dict:
    return {"field1": "새 값"}

# 3. 라우팅 함수 (조건부 엣지용)
def my_router(state: MyState) -> str:
    return "option_a" if state["field2"] > 0 else "option_b"

# 4. 그래프 구성
graph = StateGraph(MyState)
graph.add_node("node_name", my_node)
graph.set_entry_point("node_name")
graph.add_edge("node_name", END)
# 또는 조건부: graph.add_conditional_edges("node", my_router, {...})

# 5. 컴파일 및 실행
app = graph.compile()
result = app.invoke({"field1": "", "field2": 0})
```

---

## 6. 참고 자료

### 필수 읽기

| 자료 | 링크 | 설명 |
|------|------|------|
| LangGraph 빠른 시작 | [langgraph tutorials](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | 공식 튜토리얼로 시작하기 |
| LangGraph 핵심 개념 | [langgraph concepts](https://langchain-ai.github.io/langgraph/concepts/) | StateGraph, Node, Edge 공식 설명 |
| Python TypedDict | [python docs](https://docs.python.org/3/library/typing.html#typing.TypedDict) | TypedDict 공식 문서 |

### 추가 학습

| 자료 | 링크 | 설명 |
|------|------|------|
| LangGraph How-to Guides | [langgraph how-to](https://langchain-ai.github.io/langgraph/how-tos/) | 상황별 구현 가이드 |
| Mermaid Live Editor | [mermaid.live](https://mermaid.live/) | 그래프 시각화 온라인 도구 |
| LangGraph GitHub | [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 소스 코드 및 예제 |
| LangChain Academy - LangGraph | [academy.langchain.com](https://academy.langchain.com/) | 공식 교육 과정 (영문) |

### API 레퍼런스

| 클래스/함수 | 설명 |
|------------|------|
| `StateGraph(state_type)` | 상태 그래프 생성 |
| `graph.add_node(name, func)` | 노드 추가 |
| `graph.add_edge(from, to)` | 일반 엣지 추가 |
| `graph.add_conditional_edges(from, router, mapping)` | 조건부 엣지 추가 |
| `graph.set_entry_point(name)` | 시작 노드 설정 |
| `graph.compile()` | 그래프 컴파일 |
| `app.invoke(initial_state)` | 그래프 실행 |
| `app.get_graph().draw_mermaid()` | Mermaid 다이어그램 생성 |

---

## 다음 단계

축하합니다! Module 02를 완료했습니다.

이제 LangGraph의 기본 구조를 이해하고 직접 그래프를 만들 수 있습니다.
다음 모듈에서는 상태 관리를 더 깊이 배웁니다:

> **Module 03: State 관리 심화 - Reducer와 체크포인트** (예정)
>
> - Annotation과 Reducer 패턴
> - 리스트 상태에 값 누적하기
> - 체크포인트로 상태 저장/복구
> - 실전: 대화 히스토리 관리

기초가 탄탄해야 응용이 가능합니다. 연습 문제를 모두 풀어보셨나요?
