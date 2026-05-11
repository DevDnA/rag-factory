<div align="center">

# slm-factory

**도메인 문서만 넣으면 AI 채팅 서비스가 만들어집니다.**

문서 넣고, 명령어 하나, 30초 만에 RAG 웹 채팅.<br>
프로덕션이 필요하면 도메인 어투를 학습한 경량 SLM까지.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-black?logo=ollama&logoColor=white)](https://ollama.com)

[문서 홈](https://devdna.github.io/slm-factory/) · [사용 가이드](https://devdna.github.io/slm-factory/guide.html) · [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) · [RAG 서비스 가이드](https://devdna.github.io/slm-factory/integration-guide.html)

</div>

<br>

## 왜 필요한가

범용 LLM은 우리 조직의 문서를 모릅니다. ChatGPT에 사내 규정을 물어보면 그럴듯한 거짓말을 합니다.

slm-factory는 **도메인 문서를 넣으면 그 문서만으로 답변하는 AI 채팅 서비스**를 만듭니다. Ollama 기반으로 모든 처리가 로컬에서 실행되어 사내 문서가 밖으로 나가지 않으며, 문서에 없는 내용은 "없다"고 답합니다.

## 빠른 시작

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory && ./setup.sh

slf init my-project
cp /path/to/documents/*.pdf my-project/documents/

slf rag
```

브라우저에서 **http://localhost:8000/chat** — 채팅이 시작됩니다.

> `setup.sh`가 Python 환경, 의존성, Ollama 모델을 한 번에 준비합니다. PDF, HWP, HWPX, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, TXT, MD 12개 포맷을 자동 감지합니다.

## 핵심 구조: 지식은 RAG, 스타일은 파인튜닝

slm-factory의 두 기능은 역할이 명확히 다릅니다.

```
지식(WHAT) → RAG가 담당: 문서를 검색하여 정확한 정보를 전달
스타일(HOW) → 파인튜닝이 담당: 도메인 어투, 근거 인용, 답변 형식을 학습
```

**파인튜닝은 지식 주입이 아닙니다.** 1B 모델은 문서 내용을 암기할 수는 있지만 일반화할 수 없습니다 — 학습 데이터와 동일한 질문에는 답하지만, 조금만 다르게 물어도 깨진 답변을 생성합니다(과적합). 파인튜닝의 역할은 "주어진 문서를 읽고, 도메인에 맞는 어투로, 근거를 인용하며 답변하는 스타일"을 학습시키는 것입니다. 지식은 항상 RAG가 제공하고, Student 모델은 그 지식을 **어떻게 전달할지**를 담당합니다.

## 두 가지 패턴

| 패턴 | 명령어 | 역할 | 적합한&nbsp;경우 |
|------|--------|------|-------------|
| **RAG&nbsp;+&nbsp;베이스&nbsp;모델** | `slf rag` | RAG&nbsp;검색&nbsp;→&nbsp;Teacher(qwen3.5:9b)&nbsp;답변 | 빠른&nbsp;검증 |
| **RAG&nbsp;+&nbsp;파인튜닝&nbsp;SLM** | `slf tune` | RAG&nbsp;검색&nbsp;→&nbsp;Student(1B)&nbsp;답변,&nbsp;9배&nbsp;빠름 | 프로덕션 |

```bash
slf rag                # RAG + Teacher 즉시 시작 (30초)
slf tune               # 스타일 학습 + RAG + 채팅 (30분)
```

> `slf rag`는 파인튜닝된 Student 모델이 Ollama에 있으면 자동으로 사용합니다. `slf rag`로 시작하고, 도메인 어투·근거 인용 등 응답 품질을 높이려면 `slf tune`으로 전환하세요. 두 패턴 모두 RAG 검색 근거를 기반으로 답변하여 할루시네이션을 억제합니다.

### RAG 서비스 — `slf rag`

```
문서 ─→ 섹션 인식 청킹 ─→ 벡터 임베딩 ─→ Qdrant 인덱스
                                              │
질문 ─→ 하이브리드 검색(벡터+BM25) ─→ 리랭킹 ─→ LLM 답변 ─→ 웹 채팅
```

### 파인튜닝 — `slf tune`

```
문서 ─→ Teacher(9B)가 컨텍스트 활용형 QA 생성 ─→ 검증/채점/증강
  ─→ Student(1B) LoRA 스타일 학습 ─→ 평가 ─→ Ollama 등록 ─→ RAG + 웹 채팅
```

Student 모델이 학습하는 것:

| 학습 대상 | 설명 |
|-----------|------|
| 컨텍스트 활용 | RAG가 전달한 문서에서 정보를 추출하는 방법 |
| 근거 인용 | "제7조에 따르면..." 같은 출처 표기 |
| 도메인 어투 | 해당 분야에 적합한 전문적 표현 |
| 거부 패턴 | 문서에 없으면 "해당 정보는 문서에 포함되어 있지 않습니다" |

> 상세 비교는 [사용 가이드 — 파인튜닝과 RAG 역할 분담](https://devdna.github.io/slm-factory/guide.html#7-파인튜닝과-rag--역할-분담)을 참고하세요.

## 주요 기능

**섹션 인식 청킹** — 문서의 장/절/항 구조를 자동 감지하여 논리적 단위로 분할합니다. 작은 청크로 정밀 검색하고, 부모 청크로 충분한 맥락을 전달합니다.

**하이브리드 검색 + 리랭킹** — 벡터 검색과 BM25 키워드 검색을 RRF로 결합하고, Cross-Encoder로 최종 순위를 재조정합니다. Lost-in-the-middle 재정렬로 LLM이 관련 문서를 놓치지 않습니다.

**HyDE + Multi-Query 검색 강화** — LLM이 가상 답변을 생성해 임베딩에 사용(HyDE)하고, 질의를 N개 패러프레이즈로 변형해 병렬 검색 후 Reciprocal Rank Fusion으로 병합합니다. 도메인 약어가 많은 corpus에서 짧은 질의-긴 문서 mismatch를 우회해 recall을 끌어올립니다.

**Contextual Retrieval** — Anthropic 패턴. 인덱싱 시 LLM이 각 청크에 부모 문서 맥락을 prefix로 부여하여 임베딩·BM25·리랭커 모두에 동일한 맥락 정보가 적용됩니다.

**자동 캘리브레이션** — 문서 길이, 밀도, 구조를 분석하여 chunk_size, 에포크 수, 학습률, 청크당 질문 수를 자동 결정합니다. 설정 없이 바로 실행해도 합리적인 결과를 냅니다.

**컨텍스트 활용 스타일 학습** — 9B Teacher 모델이 문서 기반 QA를 자동 생성하고, 검증·채점·증강을 거쳐 1B Student 모델의 응답 스타일을 학습시킵니다. 학습 데이터에 문서 컨텍스트가 포함되어, 모델은 지식이 아닌 "문서를 읽고 답변하는 패턴"을 학습합니다.

**과적합 방지 자동 조정** — 데이터 크기에 따라 learning rate, 에포크 수를 자동 조정하고, weight decay·label smoothing·NEFTune으로 과적합을 억제합니다.

**Corpus-Aware 라우팅** — 인덱싱 시 LLM이 첫 N개 청크를 표본으로 corpus의 도메인 명칭·요약·핵심 키워드를 자동 추출(`corpus_profile.json` 영속화). 라우팅·합성 단계에 컨텍스트로 주입되어 도메인 약어를 정확히 인식합니다. 어떤 도메인 문서가 들어와도 즉시 적응.

**8 카테고리 의도 분류** — IntentClassifier가 corpus profile + 8 카테고리(factual / comparative / analytical / procedural / exploratory / **chitchat** / **general** / ambiguous)로 질의를 분류:
- **chitchat** — 인사·잡담은 정규식 fast-path로 LLM 호출 없이 1초 응답
- **general** — corpus 외 일반 지식·시사 질의는 RAG 검색 우회 + 정중한 거절·도메인 안내 (학습 prior 누설·환각 방지)
- **domain** — corpus 안 질의만 Agent 경로로 다단계 검색·합성
- **ambiguous** — 진짜 모호한 질의("그거", "이거 좀")만 Clarifier가 역질문

**Agent RAG (OMO 패턴)** — `rag.agent.smart_mode: true` 원클릭으로 LLM 기반 의도 분류, 질의 명확화, Persona 라우팅, Planner/Verifier 기반 다단계 검색을 모두 활성화합니다. OpenAI 호환 엔드포인트(`/v1/chat/completions`, `/v1/models`)를 제공해 OpenWebUI 등과 바로 연동할 수 있습니다.

**도메인 외 질의 정책** — 본 시스템은 인덱싱된 도메인 corpus에 근거한 RAG 추론 전용입니다. 도메인 외 일반 시사·잡학 질의는 LLM 학습 지식이나 외부 웹으로 추측하지 않고 정중히 거절하며 도메인 안내로 유도합니다 (학습 시점 이후 사실 변동·정확도 위험 차단).

## 문서

> **[devdna.github.io/slm-factory](https://devdna.github.io/slm-factory/)**

| | |
|---|---|
| [사용 가이드](https://devdna.github.io/slm-factory/guide.html) | 설치부터 모델 배포까지 단계별 안내 |
| [빠른 참조](https://devdna.github.io/slm-factory/quick-reference.html) | 명령어·설정 치트시트 |
| [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) | 전체 명령어 옵션 |
| [설정 레퍼런스](https://devdna.github.io/slm-factory/configuration.html) | project.yaml 전체 설정 |
| [RAG 서비스 가이드](https://devdna.github.io/slm-factory/integration-guide.html) | RAG 구축·API·프로덕션 배포 |
| [아키텍처](https://devdna.github.io/slm-factory/architecture.html) | 기술 설계·데이터 흐름 |

## 시스템 요구사항

- **Python** 3.11+ · **Ollama** — [ollama.com](https://ollama.com)
- **GPU** — NVIDIA CUDA (8GB+) 또는 Apple Silicon (MPS) 권장. CPU 폴백 가능(느림).

## 라이선스

추후 결정
