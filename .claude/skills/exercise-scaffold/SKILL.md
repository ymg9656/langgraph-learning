---
name: exercise-scaffold
description: 코드 예제를 TODO 마커 기반 Jupyter 노트북(.ipynb)과 검증 셀로 변환합니다.
---

# Exercise Scaffold

학습 자료의 코드 예제를 실습용 Jupyter 노트북으로 변환합니다.
TODO 마커, 힌트, 검증 셀(assert)을 포함한 완전한 실습 환경을 생성합니다.

## 핵심 기능

- 코드 예제 → TODO 기반 실습 노트북 자동 변환
- 단계별 힌트 시스템 (3단계: 방향 → 키워드 → 거의 정답)
- assert 기반 검증 셀 자동 생성
- 솔루션 노트북 동시 생성
- FakeLLM / 실제 API 듀얼 모드 지원

## 워크플로우

### 1. 분석 결과 기반 노트북 구조 결정

모듈 분석 JSON에서 실습 후보 코드 블록을 읽고 노트북 셀 구조를 결정합니다.

### 2. 실습 노트북 생성

.ipynb JSON 구조로 노트북을 생성합니다.

```
셀 구조:
[markdown] # 제목 + 학습 목표
[markdown] 개념 설명
[code]     import & 설정
[markdown] ## 실습 1
[code]     # TODO: 코드 작성 (+ 힌트)
[code]     # 검증 셀 (assert)
...반복...
[markdown] ## 정리 + 다음 모듈
```

### 3. 검증 셀 생성

각 TODO에 대응하는 assert 기반 검증 코드를 생성합니다.

```python
# 검증 셀 예시
assert isinstance(graph, StateGraph), "graph는 StateGraph 인스턴스여야 합니다"
assert "node_a" in graph.nodes, "node_a가 그래프에 추가되어야 합니다"
print("✅ 실습 1 완료!")
```

### 4. 솔루션 노트북 생성

TODO가 모두 채워진 완성본 노트북을 solutions/ 디렉토리에 생성합니다.

### 5. 실행 검증

솔루션 노트북을 실행하여 에러가 없는지 확인합니다.

```bash
jupyter nbconvert --to notebook --execute solutions/{notebook}_solution.ipynb
```

## 에러 처리

| 에러 | 원인 | 해결 |
|------|------|------|
| JSON 파싱 에러 | .ipynb 구조 오류 | nbformat 라이브러리로 검증 |
| Import 에러 | 의존성 누락 | requirements.txt 확인 |
| Assert 실패 | 검증 조건 오류 | 솔루션 코드로 검증 후 조건 수정 |

## 사용 예시

### 예시 1: 단일 실습 생성

```
Module 02의 첫 번째 실습 "2노드 인사 그래프"를
01_first_graph.ipynb로 생성해주세요.
```

### 예시 2: 모듈 전체 실습 생성

```
Module 02의 모든 실습 노트북(3개)과 솔루션을 생성해주세요.
```
