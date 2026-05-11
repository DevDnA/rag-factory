# 빠른 참조 (Quick Reference)

> slm-factory의 핵심 명령어와 워크플로우를 한눈에 확인하세요. 도메인 문서를 SLM으로 변환하는 전체 파이프라인을 빠르게 참조하려는 활성 사용자를 위한 치트시트입니다.

---

## 설치

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory
./setup.sh                    # uv 설치, 의존성, Ollama 모델 준비를 한 번에 처리
```

---

## 사전 준비

`./setup.sh`가 Ollama 모델 다운로드를 자동 처리합니다. 수동으로 실행하려면 아래 명령을 사용하세요.

```bash
ollama serve           # 별도 터미널에서 실행 (백그라운드 유지)
ollama pull qwen3.5:9b   # 기본 Teacher 모델 다운로드 (최초 1회, ./setup.sh가 자동 처리)
```

---

## 프로젝트 시작

```bash
# 1. 프로젝트 초기화
slf init my-project

# 2. 학습할 문서 복사
cp /path/to/documents/*.pdf my-project/documents/

# 3. 환경 점검 (선택 권장)
slf check

# 4. 실행 (택 1)
slf rag                # RAG + Teacher(9B) 즉시 시작 — 문서 적을 때
slf tune               # 파인튜닝 + RAG + Student(1B) — 문서 20건+
```

웹 채팅 UI(`http://localhost:8000/`)에는 다크/라이트 테마 토글, 추론 표시 토글, 모델 선택기(`auto` / `rag` / `agent`)가 있습니다.

---

## CLI 명령어 요약

모든 명령어에 `--verbose` (`-v`) 또는 `--quiet` (`-q`) 전역 옵션을 사용할 수 있습니다.

### 🚀 시작하기

> `./setup.sh`가 `slf` 명령어를 자동 설치합니다. 설치 후 바로 사용할 수 있습니다.

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slf init <name>` | 새 프로젝트 초기화 | `--path <디렉토리>` |
| `slf check` | 설정 및 환경 사전 점검 | `--config <파일>` |

### ⚙️ 파이프라인

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slf tune` | 파인튜닝 + RAG (Student 모델 학습 후 RAG 서비스) | `--until <단계>`, `--from <단계>`, `--resume` / `-r`, `--chat` |
| `slf rag` | RAG 웹 채팅 (Student 자동 감지, 없으면 Teacher `qwen3.5:9b` 폴백) | `--chat/--no-chat` |
| `slf train` | LoRA 학습 실행 | `--data <jsonl>`, `--resume` / `-r` |
| `slf export` | 모델 내보내기 (LoRA 병합 + Modelfile) | `--adapter <경로>` |

### 📊 평가

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slf eval run` | BLEU/ROUGE 평가 | `--model <이름>`, `--data` |
| `slf eval compare` | Base vs Fine-tuned 비교 | `--base-model <이름>`, `--ft <이름>`, `--data` |

### 🔧 도구

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slf tool review` | QA 수동 리뷰 TUI | `--data` |
| `slf tool evolve` | 자동 진화 (증분→학습→품질게이트→배포) | `--force-update`, `--skip-gate` |
| `slf tool convert` | QA → JSONL 변환 | `--data` |
| `slf tool update` | 증분 업데이트 (변경 문서만) | |
| `slf tool compare-data` | 두 QA 데이터셋 품질 비교 | `--baseline` / `-b`, `--target` / `-t` |
| `slf tool export-corpus` | 외부 평가용 코퍼스 데이터 내보내기 | `--qa-file` |
| `slf tool rag-index` | Qdrant에 임베딩 적재 | `--corpus-dir` |
| `slf tool rag-serve` | RAG API 서버 시작 | `--host`, `--port` |
| `slf tool eval-retrieval` | RAG 검색 품질 평가 | `--top-k`, `--qa-file` |

### ℹ️ 정보

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `slf status` | 파이프라인 진행 상태 확인 | |
| `slf clean` | 중간 파일 정리 | `--all`, `--yes` / `-y` |
| `slf version` | 버전 정보 출력 | |

> 각 명령어의 전체 옵션은 [CLI 레퍼런스](cli-reference.md)를 참조하십시오.

---

## 파이프라인 단계

`slf tune` 실행 시 아래 순서로 진행됩니다. 상세한 단계 설명은 [사용 가이드](guide.md)를 참조하십시오.

1. **parse** (필수) — PDF/HWPX/HTML/TXT/MD/DOCX/HWP/DOC/PPT/PPTX/XLS/XLSX 파싱 → `output/parsed_documents.json`
2. **generate** (필수) — Teacher LLM으로 QA 쌍 생성 → `output/qa_alpaca.json` (chunking 설정 시 문서를 청크로 분할하여 생성)
3. **validate** (필수) — 규칙 + 임베딩 기반 QA 검증 및 필터링 → `qa_alpaca.json` 갱신
4. **score** (선택) — Teacher LLM 1~5점 품질 평가 → `output/qa_scored.json`
5. **augment** (선택) — 질문 패러프레이즈 데이터 증강 → `output/qa_augmented.json`
6. **analyze** (선택) — 통계 분석 보고서 생성 → `output/data_analysis.json`
7. **convert** (필수) — 채팅 템플릿 적용 JSONL 변환 → `output/training_data.jsonl`
8. **train** (필수) — LoRA 파인튜닝 → `output/checkpoints/adapter/`
9. **export** (필수) — 모델 병합 + Ollama Modelfile 생성 → `output/merged_model/`
10. **eval** (선택) — BLEU/ROUGE 평가 → `output/eval_results.json`
11. **refine** (선택) — Iterative Refinement (약점 QA 재생성 + 재학습)
12. **corpus_export** (선택) — 외부 평가용 코퍼스 데이터 내보내기 → `output/corpus/`
13. **rag_index** (선택) — Qdrant 벡터 인덱싱 → `output/qdrant_db/`

> 모든 선택 단계는 기본으로 활성화됩니다. 개별 비활성화는 `project.yaml`에서 `enabled: false`로 설정합니다.

<!-- diagram: quick-reference-diagram-pipeline-flow -->

---

## 자주 쓰는 워크플로우

### 1. 전체 자동 실행 (tune)

설정을 직접 제어하고 싶을 때 사용합니다.

```bash
slf tune
```

### 2. 단계별 실행 (tune --until)

특정 단계까지만 실행하고 결과를 확인한 후 다음 단계로 진행합니다.

```bash
slf tune --until parse
slf tune --until generate
slf tune --until validate
slf tune --until score
slf tune --until augment
slf tune --until convert
slf tune --until train
slf tune --until export
slf tune --until eval
slf tune --until rag_index
```

### 3. 기존 데이터로 학습만 (train --data)

이미 준비된 `training_data.jsonl`이 있거나 하이퍼파라미터를 반복 실험할 때 사용합니다.

```bash
slf train --data ./output/training_data.jsonl
```

### 4. 중단 후 재개 (--resume)

파이프라인이 중간에 중단된 경우, 중간 저장 파일에서 자동으로 재개합니다.

```bash
slf tune --resume
slf train --resume
```

### 5. 평가 및 비교 (eval run / eval compare)

학습 완료 후 모델 성능을 측정하고 Base 모델과 비교합니다.

```bash
# BLEU/ROUGE 평가
slf eval run --model my-project-model

# Base vs Fine-tuned 비교
slf eval compare \
  --base-model gemma:2b \
  --ft my-project-model
```

### 6. 전체 파이프라인 + RAG 채팅 (tune --chat)

SLM 학습부터 RAG 인덱싱, API 서버 시작까지 한 번에 실행합니다.

```bash
slf tune --chat
```

> 서버는 foreground로 실행됩니다. `Ctrl+C`로 종료할 수 있습니다.

---

## 출력 파일 구조

```
output/
├── parsed_documents.json       # 파싱된 문서 텍스트 및 메타데이터
├── qa_alpaca.json              # Teacher LLM이 생성한 QA 쌍 (Alpaca 형식)
├── qa_scored.json              # 품질 점수 평가를 통과한 QA 쌍
├── qa_augmented.json           # 데이터 증강이 완료된 QA 쌍
├── qa_reviewed.json            # TUI 수동 리뷰를 거친 QA 쌍
├── data_analysis.json          # 카테고리 분포, 길이 통계, 품질 경고
├── eval_results.json           # BLEU/ROUGE 평가 점수
├── compare_results.json        # Base vs Fine-tuned 비교 결과
├── training_data.jsonl         # 채팅 템플릿 적용된 최종 학습 데이터
├── checkpoints/
│   └── adapter/                # LoRA 어댑터 가중치 (PEFT 형식)
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── corpus/                     # 외부 평가용 코퍼스 데이터 (corpus.parquet, qa.parquet)
├── qdrant_db/                  # RAG 벡터 인덱스 (Qdrant)
└── merged_model/               # 병합된 최종 모델 (HuggingFace 형식)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── Modelfile               # Ollama 배포용 파일
```

---

## 설정 파일 골격

`project.yaml`의 최소 필수 설정입니다. 나머지는 기본값으로 동작합니다.

```yaml
project:
  name: "my-project"
  language: "ko"                        # "ko", "en", "ja" 등

paths:
  documents: "./documents"
  output: "./output"

teacher:
  backend: "ollama"                     # "ollama" 또는 "openai"
  model: "qwen3.5:9b"                   # 기본값. 고품질 필요 시 "qwen3.5:27b"
  api_base: "http://localhost:11434"
  temperature: 0.3

student:
  model: "google/gemma-3-1b-it"   # Pydantic 기본값. slf init 템플릿은 Qwen/Qwen2.5-1.5B-Instruct (Ollama 호환성 우수)

export:
  ollama:
    model_name: "my-project-model"
    system_prompt: "당신은 도메인 전문 도우미입니다."

chunking:
  enabled: true
  chunk_size: "auto"      # 또는 정수 (예: 10000)
  overlap_chars: 500
```

> 전체 설정 옵션(scoring, augment, validation, training, LoRA 등)은 [설정 레퍼런스](configuration.md)를 참조하십시오.

---

## 트러블슈팅 빠른 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `Failed to connect to Ollama at http://localhost:11434` | Ollama 서버 미실행 | `ollama serve` 실행 후 재시도 |
| `CUDA out of memory` | GPU VRAM 부족 | `training.batch_size: 1`, `quantization.bits: 4` 설정 |
| `Failed to parse HWPX file - section0.xml not found` | HWPX 파일 손상 | 한글에서 `.hwpx`로 다시 저장하거나 `.hwp` 형식으로 변환 |
| `Could not install packages due to an OSError` (kiwipiepy) | Python 버전 미호환 | `python --version` 확인 (3.11+) |
| `Only N QA pairs generated. Recommend at least 100` | 문서 부족 또는 질문 카테고리 부족 | 문서 추가 또는 `questions.categories` 확장 |
| 대부분의 답변이 "The document does not contain..." | Teacher 모델 타임아웃 또는 질문 부적합 | `teacher.timeout: 300`, 질문을 문서 내용에 맞게 수정 |
| `model not found` (Ollama) | Teacher 모델 미다운로드 | `ollama pull <teacher-model>` (기본값: `qwen3.5:9b`) |
| `externally-managed-environment` (pip) | 환경 미설정 | `./setup.sh` 실행 후 재시도 |

> 상세 해결 방법은 [사용 가이드](guide.md#8-트러블슈팅)를 참조하십시오.

---

## 관련 문서

| 문서 | 내용 | 대상 |
|------|------|------|
| [사용 가이드](guide.md) | 설치부터 배포까지 단계별 상세 안내 | 처음 사용자 |
| [CLI 레퍼런스](cli-reference.md) | 모든 명령어와 옵션의 완전한 설명 | 명령어 확인 |
| [설정 레퍼런스](configuration.md) | `project.yaml`의 모든 설정 옵션 상세 설명 | 설정 커스터마이징 |
| [아키텍처 가이드](architecture.md) | 내부 구조, 모듈 설계, 확장 방법 | 기여자, 고급 사용자 |
| [개발 가이드](development.md) | 개발 환경 설정, 테스트, 기여 방법 | 기여자 |
