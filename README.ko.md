## Agentic Context Engineering for Services (ACES)

**[주요 기능](#주요-기능-service-oriented)**

**[아키텍처](#아키텍처)**

**[구현 상세 및 차이점](#구현-상세-및-차이점)**

**[성능 평가](#성능-평가--자기-개선-분석)**

**[설치 및 사용법](#설치-및-사용법)**

**[라이선스](#라이선스-및-인용)**

![ace](./static/ace.png)

**Agentic Context Engineering (ACE)**는 모델의 가중치를 미세 조정(Fine-tuning)하는 대신 **컨텍스트(Context)**를 최적화하여 스스로 성장하는 LLM 프레임워크입니다. Zhang et al. (2025)에 의해 제안된 ACE는 컨텍스트를 **진화하는 플레이북(Evolving Playbook)**—전략, 코드 스니펫, 교훈이 담긴 동적 컬렉션—으로 취급함으로써 **간결성 편향(Brevity Bias)**과 **컨텍스트 붕괴(Context Collapse)** 문제를 해결합니다.

이 저장소는 이론적인 ACE 프레임워크를 비동기 학습, 멀티 모델 지원, 견고한 메모리 아키텍처를 갖춘 **서비스 가능한 수준의 풀스택 에이전트 서비스**로 구현한 프로젝트입니다.

![ACE Framework UI](./static/frontend.png)

## 주요 기능 (Service-Oriented)

알고리즘 이론에 초점을 맞춘 원본 논문과 달리, 이 구현체는 **실제 서비스 운영(Real-world serving)**을 목적으로 설계되었습니다.

### 1. 비동기 아키텍처 (Zero Latency Learning)

반응성을 극대화하기 위해 **추론(Inference)**과 **학습(Learning)** 과정을 분리했습니다:

- **Serving Graph**: 사용자 쿼리 처리, 지능형 라우팅, 검색(Retrieval)을 담당하여 즉각적인 응답을 제공합니다.
- **Learning Graph**: 백그라운드에서 비동기로 실행되며, 실행 궤적(Trajectory)을 분석하여 사용자를 기다리게 하지 않고 Playbook을 업데이트합니다.

### 2. 멀티 LLM Providers 지원

UI/API를 통해 체인의 각 부분에 대해 최신 SOTA 모델들을 동적으로 교체하여 사용할 수 있습니다:

- **OpenAI**
- **Anthropic**
- **Google**

### 3. 스마트 라우팅 및 모드 전환

- **Standard Mode (Async)**: 지능형 라우터가 쿼리를 "단순 질문(Simple)"(즉답)과 "복합 작업(Complex)"(ACE Playbook 필요)으로 분류합니다. 학습은 백그라운드에서 일어납니다.
- **Full Debug Mode (Sync)**: 디버깅과 시각화를 위해 전체 사이클(Retrieve → Generate → Evaluate → Reflect → Update)을 동기적으로 강제 실행합니다.

### 4. 견고한 메모리 및 스토리지 스택

- **Vector Store (Qdrant)**: Playbook 항목들의 의미론적 임베딩을 저장합니다.
- **Relational DB (SQLite)**: 메타데이터, 사용 통계(Helpful/Harmful 카운트), 타임스탬프를 관리합니다.
- **Session Memory (Redis)**: 슬라이딩 윈도우 방식을 적용하여 멀티턴 대화 기록을 관리합니다.

## 아키텍처

이 프로젝트는 **LangGraph**를 활용하여 세 가지의 독립적인 상태 그래프(State Graph)를 오케스트레이션합니다:

### A. Serving Graph

속도에 최적화된 그래프입니다.

1. **Router**: 쿼리를 분류합니다 (Simple vs. Complex).
2. **Simple Generator**: 가벼운 대화나 사실 확인을 처리합니다.
3. **Retriever**: 작업에 적합한 전략을 검색합니다.
4. **Generator**: Playbook을 참고하여 해결책을 생성합니다.

### B. Learning Graph

품질에 최적화된 그래프입니다. FastAPI 백그라운드 태스크로 비동기 실행됩니다.

1. **Evaluator**: **단위 테스트(코드 실행)**와 **LLM 논리**를 결합한 하이브리드 평가를 수행합니다.
2. **Reflector**: 성공과 실패의 근본 원인(Root Cause)을 진단합니다.
3. **Curator**: Insights를 모아 `ADD`(추가) 또는 `UPDATE`(갱신) 작업을 생성합니다.
4. **Update**: Playbook에 변경 사항을 적용합니다 (가지치기 및 중복 제거 포함).

### C. Full Graph

동기적 디버깅 및 개발을 위해 두 그래프를 결합한 형태입니다.

## 구현 상세 및 차이점

학술적 연구를 실제 배포 가능한 서비스로 연결하기 위해 다음과 같은 기능들을 강화했습니다.

| 기능           | 논문 (이론)                      | 본 구현체 (프로덕션)                                                      |
| :------------- | :------------------------------- | :------------------------------------------------------------------------ |
| **워크플로우** | 순차적 실행 (Generate → Reflect) | **비동기 분리**: Serving Graph (Fast) + Background Learning Graph         |
| **라우팅**     | 모든 쿼리를 동일하게 처리        | **의미론적 라우터**: "잡담"과 "전략적 작업"을 구분                        |
| **메모리**     | 추상적인 개념                    | **Redis**: 영구적인 세션 기록 관리                                        |
| **스토리지**   | 단일 소스                        | **하이브리드**: Qdrant (벡터) + SQLite (메타) + Redis (세션)              |
| **평가**       | LLM 피드백에만 의존              | **하이브리드 실행**: 샌드박스 코드 실행 + LLM 추론                        |
| **언어**       | 영어 단일 언어                   | **Canonical English Storage**: 다국어 입력 → 영어 로직 처리 → 현지화 출력 |

### 핵심 로직 및 고급 메커니즘

#### 1. 구체적인 가지치기(Pruning) 및 메모리 관리

논문에서는 "성장과 정제(Grow-and-refine)"를 추상적으로 언급했지만, 우리는 엄격한 **메모리 관리 전략**을 구현했습니다:

- **독성 감지(Poison Detection)**: 긍정적 사용(Helpful)보다 부정적 피드백(Harmful)이 많은 항목을 자동으로 제거합니다.
- **용량 제어(Capacity Control)**: `MAX_PLAYBOOK_SIZE`를 강제합니다. **효용성 점수(Utility Score)**와 **최신성(Recency, LRU)**을 기반으로 항목을 방출합니다.
- **의미론적 중복 제거**: Curator가 새로운 항목을 추가하기 전에 벡터 유사도를 검사하여 컨텍스트 오염을 방지합니다.

#### 2. 교차 언어(Cross-Lingual) RAG 아키텍처

지식의 파편화를 방지하기 위해 다음과 같이 설계했습니다:

- **입력**: 사용자는 어떤 언어(예: 한국어)로든 질문할 수 있습니다.
- **처리**: 내부 로직(검색, 반성, 큐레이션)은 **영어**로 수행되어 단일화된 지식 베이스를 유지합니다.
- **출력**: 최종 응답은 사용자의 언어로 생성됩니다.

#### 3. 구조화된 검색 최적화

Curator는 "방법(How-to)" 쿼리에 대한 검색 정확도를 극대화하기 위해 `Context-Action` 스키마를 강제합니다.

## 성능 평가 및 자기 개선 분석

우리는 두 가지 까다로운 벤치마크인 OpenAI HumanEval(코드 생성)과 HotpotQA(멀티홉 추론)를 사용하여 ACE 프레임워크의 효과를 검증했습니다. 아래 시각화 자료는 가중치 업데이트 없이 시스템이 시간이 지남에 따라 어떻게 자율적으로 성능을 향상시키는지 보여줍니다.

### 학습 곡선 및 지식 역학 (Learning Curve & Knowledge Dynamics)

대시보드는 에이전트의 실시간 진화 과정을 보여줍니다. 두 도메인 모두에서 에이전트는 성공적으로 지식을 축적하고 정의된 제약 조건 내에서 메모리를 관리합니다.

#### 추론 벤치마크 (HotpotQA)

![alt text](./evaluation/figures/hotpotqa_metrics.png)

#### 코딩 벤치마크 (HumanEval)

![alt text](./evaluation/figures/human_eval_metrics.png)

**주요 관찰 사항**:

- **자기 개선 (상단)**: 두 벤치마크 모두에서 누적 정확도(파란색)가 꾸준히 상승하는 추세를 보입니다. 특히 HumanEval에서는 에이전트가 일반적인 코딩 패턴을 학습함에 따라 초기 단계에서 빠른 적응력을 보여줍니다.
- **메모리 관리 (중단)**: Playbook 크기(초록색)가 설정된 제한(예: 50 또는 60개)에서 안정화됩니다. 평평한 그래프는 우리의 **Pruning & LRU 로직**이 낮은 효용의 항목을 적극적으로 제거하여 컨텍스트 오염을 방지하고 있음을 확인시켜 줍니다.
- **검색 효용성 (하단)**: 적중률(Hit Rate, 빨간색)이 성공률과 상관관계를 보이며, Router와 Retriever가 현재 작업에 적합한 전략을 효과적으로 가져오고 있음을 증명합니다.

### "Helpful" 컨텍스트의 영향

Playbook이 실제로 도움이 될까요? 검색된 컨텍스트가 "Helpful"로 표시된 경우와 "Low Utility"(중립/해로움)인 경우의 성공률 차이를 분석했습니다.

#### HotpotQA (Reasoning)

![alt text](./evaluation/figures/hotpotqa_metrics_impact.png)

#### HumanEval (Coding)

![alt text](./evaluation/figures/human_eval_metrics_impact.png)

## 설치 및 사용법

### 사전 요구 사항

시작하기 전에 다음이 설치되어 있는지 확인하세요:

- Python 3.12+
- uv (Fast Python 패키지 매니저)
- Redis Server (localhost:6379에서 실행 중이어야 함)

### 1. 설치

레포지토리를 복제하고 uv를 사용하여 의존성을 설치합니다.

```bash
uv sync
uv pip install -e .
```

### 2. 설정

먼저 환경 변수를 설정합니다.

1. 예제 설정 파일을 복사합니다:

```bash
# 예제 파일 이름 변경
mv config-example.ini config.ini
# 또는 config-example.ini를 그대로 사용해도 됩니다 (시스템이 자동 감지)
```

2. `config.ini`를 열고 API 키와 설정을 구성합니다.

⚠️ **임베딩 모델 관련 중요 사항**
`HUGGINGFACE_ACCESS_TOKEN` 상태를 변경하는 경우 (예: 오픈소스 모델에서 Gemma 모델로 전환), Dimension 불일치 오류를 방지하기 위해 반드시 Vector Store와 Database 폴더(`data/`)를 초기화(삭제)해야 합니다.

### 3. 실행

Backend(API)와 Frontend(UI)를 별도의 터미널에서 실행해야 합니다.

#### Windows

```bash
@REM Terminal 1: Backend 실행
start cmd /k "uv run python -m main"

@REM Terminal 2: Frontend 실행
start cmd /k "uv run streamlit run web/app.py"
```

#### Linux / macOS

```bash
# Terminal 1: Backend 실행
uv run python -m main

# Terminal 2: Frontend 실행
uv run streamlit run web/app.py
```

실행 후 다음 주소로 접속할 수 있습니다:

- Frontend (UI): http://localhost:8501
- Backend (API): http://localhost:8000

**준비 상태 확인 (Ready Check)**:
사이드바에 Knowledge Base 카운트가 표시되면(Vector Store가 성공적으로 로드됨을 의미) 애플리케이션을 사용할 준비가 된 것입니다.

![alt_text](./static/vector_store_loading.png)

## 라이선스 및 인용

이 프로젝트는 연구 논문 "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"를 기반으로 한 오픈 소스 구현체입니다.

### 라이선스

이 저장소는 **MIT License**에 따라 배포됩니다. 학술 및 상업적 목적으로 코드를 자유롭게 사용, 수정 및 배포할 수 있습니다.

### 인용

연구에 이 코드나 ACE 프레임워크를 사용하는 경우 원본 논문을 인용해 주세요:

```bibtex
@article{zhang2025ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and others},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025}
}
```
