# AI Agent 개발 입문: LangGraph로 시작하는 에이전트 개발

> **비개발자와 주니어 개발자를 위한 실습 중심 AI 에이전트 개발 교육 과정**

---

## 과정 소개

이 교육 과정은 AI 에이전트 개발을 처음 접하는 분들을 위해 설계되었습니다.
LangGraph 프레임워크를 사용하여 단순한 그래프 구조부터 프로덕션 수준의 에이전트까지
단계적으로 학습합니다.

### 과정 목표

이 과정을 완료하면 다음을 할 수 있습니다:

1. AI 에이전트의 핵심 개념과 아키텍처를 이해하고 설명할 수 있다
2. LangGraph로 상태 기반 그래프를 설계하고 구현할 수 있다
3. LLM(대규모 언어 모델)을 에이전트에 연결하고 프롬프트를 설계할 수 있다
4. 도구(Tool)를 만들어 에이전트에 연결할 수 있다
5. 에러 처리, 테스트, 배포 등 프로덕션 고려사항을 적용할 수 있다
6. 멀티 에이전트 시스템을 설계하고 구현할 수 있다

### 대상 수강생

| 대상 | 설명 |
|------|------|
| 비개발자 | AI 에이전트에 관심 있고 프로그래밍 기초를 배우고 싶은 분 |
| 주니어 개발자 | Python 기초는 있지만 AI/LLM 개발 경험이 없는 분 |
| QA 엔지니어 | AI 기반 테스트 자동화에 관심 있는 분 |
| 기획자/PM | AI 에이전트 프로젝트를 이해하고 관리하고 싶은 분 |

---

## 사전 준비 사항

### 필수 환경

| 항목 | 최소 요구사항 | 권장사항 |
|------|-------------|---------|
| Python | 3.10 이상 | 3.12 |
| pip | 최신 버전 | 최신 버전 |
| OS | Windows 10, macOS 12, Ubuntu 20.04 | macOS 또는 Linux |
| 메모리 | 4GB RAM | 8GB RAM 이상 |
| 에디터 | 아무 텍스트 에디터 | VS Code |

### 필수 사전 지식

- **Python 기초**: 변수, 함수, 조건문, 반복문, 딕셔너리
- **pip 사용법**: `pip install 패키지명` 으로 패키지 설치 가능
- **가상환경(venv)**: `python -m venv .venv` 로 가상환경 생성/활성화 가능
- **터미널 기초**: cd, ls, mkdir 등 기본 명령어 사용 가능

> **참고**: Python이 처음이시라면 아래 무료 강좌를 먼저 수강하세요.
> - [점프 투 파이썬](https://wikidocs.net/book/1)
> - [Python 공식 튜토리얼 (한국어)](https://docs.python.org/ko/3/tutorial/)

---

## 학습 경로 (Learning Path)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         학습 경로 다이어그램                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Part 1: 기초 다지기 (Foundation)                                     │
│  ┌──────────┐    ┌──────────────┐                                    │
│  │ Module 01 │───▶│  Module 02   │                                    │
│  │ AI 에이전트│    │  LangGraph   │                                    │
│  │ 개요      │    │  기초        │                                    │
│  └──────────┘    └──────────────┘                                    │
│       │                  │                                            │
│       ▼                  ▼                                            │
│  Part 2: 첫 에이전트 만들기 (First Agent)                              │
│  ┌──────────┐    ┌──────────────┐                                    │
│  │ Module 03 │───▶│  Module 04   │                                    │
│  │ 개발 환경 │    │  첫 에이전트 │                                    │
│  │ 구축      │    │  만들기      │                                    │
│  └──────────┘    └──────────────┘                                    │
│       │                  │                                            │
│       ▼                  ▼                                            │
│  Part 3: 프롬프트와 LLM 심화 (Prompt & LLM)                           │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Module 05 │───▶│  Module 06   │───▶│  Module 07   │               │
│  │ 프롬프트  │    │  구조화된    │    │  LLM 호출    │               │
│  │ 엔지니어링│    │  출력        │    │  최적화      │               │
│  └──────────┘    └──────────────┘    └──────────────┘               │
│       │                                      │                       │
│       ▼                                      ▼                       │
│  Part 4: 프로덕션 (Production)                                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Module 08 │───▶│  Module 09   │───▶│  Module 10   │               │
│  │ 에러 처리 │    │  외부 시스템 │    │  리소스      │               │
│  │ & 회복    │    │  연동        │    │  최적화      │               │
│  └──────────┘    └──────────────┘    └──────────────┘               │
│       │                                      │                       │
│       ▼                                      ▼                       │
│  Part 5: 고급 패턴 (Advanced)                                         │
│  ┌──────────┐    ┌──────────────┐                                    │
│  │ Module 11 │───▶│  Module 12   │                                    │
│  │ 품질 보증 │    │  고급 패턴   │                                    │
│  └──────────┘    └──────────────┘                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 모듈 목록 (전체 교육 시간: 약 30시간)

### Part 1: 기초 다지기 (Foundation) - 약 5시간

| 모듈 | 제목 | 학습 시간 | 핵심 내용 |
|------|------|----------|----------|
| **01** | [AI 에이전트 개요와 핵심 개념](part1-foundations/01-ai-agent-overview.md) | 2시간 | 에이전트란?, LLM 기초, 아키텍처 패턴 |
| **02** | [LangGraph 기초 - 그래프, 노드, 엣지, 상태](part1-foundations/02-langgraph-fundamentals.md) | 3시간 | StateGraph, TypedDict, 노드, 엣지, 조건부 분기 |

### Part 2: 첫 에이전트 만들기 (First Agent) - 약 5시간

| 모듈 | 제목 | 학습 시간 | 핵심 내용 |
|------|------|----------|----------|
| **03** | [Jupyter 노트북 기반 에이전트 개발 환경 구축](part2-first-agent/03-jupyter-dev-environment.md) | 2시간 | Jupyter 환경, FakeLLM, 그래프 시각화 |
| **04** | [나의 첫 LangGraph 에이전트 만들기](part2-first-agent/04-building-first-agent.md) | 3시간 | 문서 요약 에이전트, 의존성 주입, FakeLLM E2E 테스트 |

### Part 3: 프롬프트와 LLM 심화 (Prompt & LLM) - 약 7.5시간

| 모듈 | 제목 | 학습 시간 | 핵심 내용 |
|------|------|----------|----------|
| **05** | [프롬프트 엔지니어링](part3-prompt-and-llm/05-prompt-engineering.md) | 2.5시간 | 시스템 프롬프트, Few-shot, YAML 프롬프트 |
| **06** | [구조화된 출력](part3-prompt-and-llm/06-structured-output.md) | 2.5시간 | Pydantic 모델, LLM 응답을 데이터로 변환 |
| **07** | [LLM 호출 최적화](part3-prompt-and-llm/07-llm-call-optimization.md) | 2.5시간 | 비용 계산, 모델 라우팅, 캐싱 전략 |

### Part 4: 프로덕션 (Production) - 약 7.5시간

| 모듈 | 제목 | 학습 시간 | 핵심 내용 |
|------|------|----------|----------|
| **08** | [에러 처리와 회복 탄력성](part4-production/08-error-handling-resilience.md) | 2.5시간 | 재시도, 서킷 브레이커, 회복 탄력적 에이전트 |
| **09** | [외부 시스템 연동](part4-production/09-external-system-integration.md) | 2.5시간 | 비동기 호출, Rate Limiter, 헬스 체크 |
| **10** | [리소스 최적화](part4-production/10-resource-optimization.md) | 2.5시간 | 토큰 예산, Annotated Reducer, 코드 압축 |

### Part 5: 고급 패턴 (Advanced) - 약 5시간

| 모듈 | 제목 | 학습 시간 | 핵심 내용 |
|------|------|----------|----------|
| **11** | [LLM 출력 품질 보장](part5-advanced/11-quality-assurance.md) | 2.5시간 | 검증 피라미드, Self-Reflection, Golden Set, Human-in-the-Loop |
| **12** | [LangGraph 고급 패턴](part5-advanced/12-langgraph-advanced.md) | 2.5시간 | 체크포인팅, 서브그래프, 병렬 처리, 스트리밍 |

---

## 환경 설정 가이드

### Step 1: Python 설치 확인

```bash
# Python 버전 확인 (3.10 이상이어야 합니다)
python --version
# 또는
python3 --version
```

출력 예시:
```
Python 3.12.0
```

### Step 2: 프로젝트 디렉토리 생성

```bash
# 학습용 디렉토리 생성
mkdir ai-agent-study
cd ai-agent-study
```

### Step 3: 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 가상환경 활성화 (Windows)
# .venv\Scripts\activate
```

> **확인**: 터미널 프롬프트 앞에 `(.venv)` 가 표시되면 성공입니다.

### Step 4: 패키지 설치

```bash
# 핵심 패키지 설치
pip install langgraph langchain-anthropic langchain-core pydantic

# 설치 확인
pip list | grep -E "(langgraph|langchain|pydantic)"
```

출력 예시:
```
langchain-anthropic    0.3.x
langchain-core         0.3.x
langgraph              0.3.x
pydantic               2.x.x
```

### Step 5: API 키 설정

```bash
# Anthropic API 키 설정 (macOS/Linux)
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows (PowerShell)
# $env:ANTHROPIC_API_KEY="your-api-key-here"
```

> **중요**: API 키는 절대 코드에 직접 작성하지 마세요!
> 환경변수 또는 `.env` 파일을 사용하세요.

### Step 6: 설치 검증

다음 Python 코드를 실행하여 설치를 확인합니다:

```python
# verify_setup.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class TestState(TypedDict):
    message: str

def hello(state: TestState) -> dict:
    return {"message": "설치 완료! 학습을 시작하세요!"}

graph = StateGraph(TestState)
graph.add_node("hello", hello)
graph.set_entry_point("hello")
graph.add_edge("hello", END)

app = graph.compile()
result = app.invoke({"message": ""})
print(result["message"])
```

```bash
python verify_setup.py
# 출력: 설치 완료! 학습을 시작하세요!
```

---

## 학습 자료 사용 방법

### 각 모듈의 구성

모든 모듈은 동일한 구조를 따릅니다:

```
┌─────────────────────────────────────────┐
│  학습 목표        ← 이번 모듈에서 배울 것   │
│  사전 지식        ← 필요한 선수 지식        │
│  1. 개념 설명     ← 이론과 다이어그램       │
│  2. 단계별 실습   ← 코드를 따라 치며 학습    │
│  3. 실전 예제     ← 완성된 동작 코드        │
│  4. 연습 문제     ← 스스로 풀어보기         │
│  5. 핵심 정리     ← 요약과 체크리스트       │
│  6. 참고 자료     ← 더 깊이 공부할 링크     │
│  다음 단계        ← 다음 모듈 안내          │
└─────────────────────────────────────────┘
```

### 권장 학습 방법

1. **순서대로 학습하세요**: Module 01부터 순서대로 진행합니다
2. **코드를 직접 타이핑하세요**: 복사-붙여넣기 대신 직접 입력하면 더 잘 기억됩니다
3. **연습 문제를 꼭 풀어보세요**: 이해도를 확인하는 가장 좋은 방법입니다
4. **모르는 부분은 참고 자료를 확인하세요**: 각 모듈 끝에 추가 학습 링크가 있습니다
5. **하루 1모듈씩 진행하세요**: 2~3시간씩 집중해서 학습하는 것이 효과적입니다

### 코드 실행 방법

```bash
# 각 모듈의 코드 파일을 실행하는 방법
python 파일이름.py

# 예시
python 01_first_graph.py
```

---

## 전체 과정 일정표 (권장)

```
┌─────────┬────────────────────────────────────┬──────────┐
│  주차    │  학습 내용                          │ 예상 시간 │
├─────────┼────────────────────────────────────┼──────────┤
│  1주차   │  Module 01 ~ 02 (기초 다지기)       │  5시간   │
│  2주차   │  Module 03 ~ 04 (첫 에이전트)       │  5시간   │
│  3주차   │  Module 05 ~ 07 (프롬프트 & LLM)   │  7.5시간 │
│  4주차   │  Module 08 ~ 10 (프로덕션)          │  7.5시간 │
│  5주차   │  Module 11 ~ 12 (고급 패턴)         │  5시간   │
├─────────┼────────────────────────────────────┼──────────┤
│  합계    │  12개 모듈                          │ 약 30시간 │
└─────────┴────────────────────────────────────┴──────────┘
```

---

## Part별 상세 개요

### Part 1: 기초 다지기 (Foundation)

AI 에이전트의 개념을 이해하고 LangGraph의 핵심 구성 요소를 학습합니다.
코드 없이 개념을 먼저 이해한 후, 간단한 그래프를 만들며 실습합니다.

**핵심 질문**: "AI 에이전트란 무엇이고, 어떻게 구조화하는가?"

### Part 2: 첫 에이전트 만들기 (First Agent)

Jupyter 개발 환경을 구축하고, LLM을 연동하여 실제로 동작하는 에이전트를 완성합니다.
이 Part를 마치면 스스로 간단한 에이전트를 만들 수 있게 됩니다.

**핵심 질문**: "에이전트가 실제로 어떻게 생각하고 행동하는가?"

### Part 3: 프롬프트와 LLM 심화 (Prompt & LLM)

프롬프트 엔지니어링, 구조화된 출력, LLM 호출 최적화 전략을 학습합니다.
같은 에이전트라도 프롬프트에 따라 성능이 크게 달라짐을 체험합니다.

**핵심 질문**: "에이전트가 더 정확하고 효율적으로 동작하게 하려면?"

### Part 4: 프로덕션 (Production)

에러 처리와 회복 탄력성, 외부 시스템 연동, 리소스 최적화를 학습합니다.
안정적으로 운영 가능한 에이전트 시스템을 만드는 방법을 익힙니다.

**핵심 질문**: "에이전트를 실제 서비스에 안전하게 배포하려면?"

### Part 5: 고급 패턴 (Advanced)

LLM 출력 품질 보장과 프로덕션 수준의 고급 아키텍처 패턴을 학습합니다.
체크포인팅, 서브그래프, 병렬 처리, 스트리밍 등 심화 기능을 다룹니다.

**핵심 질문**: "프로덕션 수준의 신뢰할 수 있는 에이전트를 어떻게 만드는가?"

---

## 자주 묻는 질문 (FAQ)

### Q: 프로그래밍을 해본 적이 없어도 수강할 수 있나요?

A: Python 기초 문법(변수, 함수, 조건문)은 알아야 합니다. 완전 초보라면 먼저
   [점프 투 파이썬](https://wikidocs.net/book/1)을 1~3장까지 학습한 후 시작하세요.

### Q: API 키가 꼭 필요한가요?

A: Part 1~2 (Module 01~04)는 FakeLLM을 사용하므로 API 키 없이 학습할 수 있습니다.
   Module 05부터는 Anthropic API 키가 필요합니다.
   [Anthropic Console](https://console.anthropic.com/)에서 발급받을 수 있습니다.

### Q: 예상 비용은 어느 정도인가요?

A: 전체 과정을 학습하는 데 약 $5~10 정도의 API 비용이 발생합니다.
   Anthropic은 신규 가입 시 무료 크레딧을 제공하기도 합니다.

### Q: 오프라인에서도 학습할 수 있나요?

A: Part 1~2의 Module 01~04는 오프라인으로 학습 가능합니다.
   LLM API를 사용하는 Module 05부터는 인터넷 연결이 필요합니다.

---

## 참고 자료 모음

### 공식 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 공식 문서](https://python.langchain.com/docs/)
- [Anthropic Claude 문서](https://docs.anthropic.com/)
- [Pydantic 공식 문서](https://docs.pydantic.dev/)

### 학습 자료
- [LangGraph 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangChain Academy](https://academy.langchain.com/)
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)

### 커뮤니티
- [LangChain Discord](https://discord.gg/langchain)
- [LangChain GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

---

## 버전 정보

| 항목 | 버전      |
|------|---------|
| 교육 과정 버전 | 1.0     |
| 최종 업데이트 | 2026-03 |
| LangGraph 기준 버전 | 0.3.x   |
| Python 기준 버전 | 3.12    |

---

> **학습을 시작하려면** [Module 01: AI 에이전트 개요와 핵심 개념](part1-foundations/01-ai-agent-overview.md)으로 이동하세요.
