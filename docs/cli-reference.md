# CLI 명령어 레퍼런스

> slm-factory의 모든 명령어와 옵션을 정리한 공식 레퍼런스입니다.
> 빠른 조회는 [quick-reference.md](quick-reference.md)를, 단계별 사용법은 [guide.md](guide.md)를 참조하십시오.

---

## 전역 옵션

모든 명령어 앞에 사용할 수 있는 옵션입니다.

```
slf [전역 옵션] <명령어> [옵션]
```

| 플래그 | 단축키 | 설명 |
|--------|--------|------|
| `--verbose` | `-v` | DEBUG 레벨 로그를 출력합니다. 문제 진단 시 사용합니다. |
| `--quiet` | `-q` | WARNING 이상의 로그만 출력합니다. 스크립트 자동화에 적합합니다. |
| `--version` | `-V` | 버전을 표시합니다. |
| `--help` | | 도움말을 출력합니다. |
| `--install-completion` | | 현재 셸에 자동완성을 설치합니다. |
| `--show-completion` | | 자동완성 스크립트를 출력합니다. |

`--verbose`와 `--quiet`를 동시에 지정하면 `--verbose`가 우선 적용됩니다.

---

## 명령어 목록

| 명령어 | 그룹 | 설명 |
|--------|------|------|
| `init` | 🚀 시작하기 | 새 프로젝트 디렉토리와 설정 파일을 생성합니다 |
| `check` | 🚀 시작하기 | 설정 파일, 문서 디렉토리, Ollama 연결을 사전 점검합니다 |
| `rag` | 🚀 시작하기 | RAG 웹 채팅 서비스 시작 (인덱스 자동 구축) |
| `tune` | ⚙️ 파이프라인 | 전체 파이프라인 또는 지정 단계까지 실행합니다 |
| `train` | ⚙️ 파이프라인 | LoRA 파인튜닝을 실행합니다 |
| `export` | ⚙️ 파이프라인 | 훈련된 어댑터를 병합하고 Ollama Modelfile을 생성합니다 |
| `eval run` | 📊 평가 | 학습된 모델을 BLEU/ROUGE 메트릭으로 평가합니다 |
| `eval compare` | 📊 평가 | Base 모델과 Fine-tuned 모델의 답변을 나란히 비교합니다 |
| `tool review` | 🔧 도구 | QA 쌍을 TUI에서 수동으로 승인/거부/편집합니다 |
| `tool convert` | 🔧 도구 | QA 데이터를 훈련용 JSONL 형식으로 변환합니다 |
| `tool update` | 🔧 도구 | 변경된 문서만 감지하여 증분 처리합니다 |
| `tool evolve` | 🔧 도구 | 자동 진화 (증분→학습→품질게이트→배포) |
| `tool compare-data` | 🔧 도구 | 두 QA 데이터셋의 품질 비교 |
| `tool export-corpus` | 🔧 도구 | 외부 평가용 코퍼스 데이터 내보내기 |
| `tool rag-index` | 🔧 도구 | corpus.parquet을 Qdrant에 임베딩 적재 |
| `tool rag-serve` | 🔧 도구 | RAG API 서버 시작 (Qdrant 검색 + Ollama 생성) |
| `tool eval-retrieval` | 🔧 도구 | RAG 검색 품질 평가 (Hit Rate, MRR, Recall@K) |
| `status` | ℹ️ 정보 | 각 파이프라인 단계의 진행 상태를 표시합니다 |
| `clean` | ℹ️ 정보 | 중간 생성 파일을 정리합니다 |
| `version` | ℹ️ 정보 | slm-factory 버전을 출력합니다 |

---

## 🚀 시작하기

### `init`

새로운 slm-factory 프로젝트를 초기화합니다. 프로젝트 디렉토리, `documents/`, `output/` 하위 디렉토리, 기본 `project.yaml` 설정 파일을 생성합니다.

**사용법**

```
slf init <name> [OPTIONS]
```

**인수**

| 인수 | 필수 여부 | 설명 |
|------|:---------:|------|
| `name` | 필수 | 생성할 프로젝트 이름. 디렉토리명과 설정 파일의 `project.name`에 사용됩니다. |

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--path` | | `TEXT` | `.` | 프로젝트를 생성할 상위 디렉토리입니다. |

**예시**

```bash
# 현재 디렉토리에 my-project 생성
slf init my-project

# 특정 경로에 생성
slf init policy-assistant --path /workspace/projects
```

**출력**

```
✓ 프로젝트 'my-project'가 생성되었습니다: ./my-project

프로젝트 구조:
  ./my-project/
  ./my-project/documents/
  ./my-project/output/
  ./my-project/project.yaml

다음 단계:
  1. ./my-project/documents 디렉토리에 학습할 문서(PDF, TXT 등)를 추가하세요

실행 (택 1):
  slf rag               RAG 채팅 즉시 시작 (30초)
  slf tune              파인튜닝 + RAG + 채팅 (30분)
```

**참고**

- 지정 경로에 `project.yaml`이 이미 존재하면 덮어쓰기 여부를 확인하는 프롬프트가 표시됩니다.
- 생성된 `project.yaml`의 `project.name`과 `export.ollama.model_name`에 `name` 인수가 자동으로 반영됩니다.
- 설정 파일의 모든 옵션은 [설정 레퍼런스](configuration.md)를 참조하십시오.

---

### `check`

프로젝트 설정과 실행 환경을 사전 점검합니다. 파이프라인 실행 전에 실행하여 문제를 미리 확인하십시오.

**사용법**

```
slf check [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**점검 항목**

| 항목 | 설명 |
|------|------|
| 설정 파일 | `project.yaml` 로드 및 Pydantic 스키마 검증 |
| 문서 디렉토리 | `paths.documents` 디렉토리 존재 여부 및 파일 유무 확인 |
| 출력 디렉토리 | `paths.output` 디렉토리 쓰기 권한 확인 |
| Ollama 연결 | `teacher.backend == "ollama"`일 때 서버 연결 상태 확인 |
| 모델 사용 가능 | `teacher.model`이 Ollama에 다운로드되어 있는지 확인 |
| 학생 모델 | `student.model`이 HuggingFace Hub에서 접근 가능한지 확인 (게이트 모델인 경우 로그인 필요 안내) |
| 컴퓨팅 디바이스 | GPU(CUDA/MPS) 또는 CPU 감지. GPU 미사용 시 경고 |
| 학습 정밀도 | bfloat16/float16/float32 지원 여부 확인 |
| 4bit 양자화 | CUDA 환경에서 `bitsandbytes` 설치 여부 확인 (MPS는 N/A) |

**종료 코드**

| 코드 | 의미 |
|------|------|
| `0` | 모든 항목 통과 |
| `1` | 하나 이상의 항목 실패 또는 경고 |

**예시**

```bash
slf check --config my-project/project.yaml
```

**출력 예시 (전체 통과)**

```
                slm-factory 환경 점검
┌────────────────┬────────┬────────────────────────────────┐
│ 항목           │ 상태   │ 상세                           │
├────────────────┼────────┼────────────────────────────────┤
│ 설정 파일      │ OK     │ ./my-project/project.yaml      │
│ 문서 디렉토리  │ OK     │ 3개 파일 (./documents)         │
│ 출력 디렉토리  │ OK     │ 쓰기 가능 (./output)           │
│ Ollama 연결    │ OK     │ v0.3.12 (http://localhost:11434)│
│ 모델 사용 가능 │ OK     │ qwen3.5:9b                     │
│ 학생 모델      │ OK     │ Qwen/Qwen2.5-1.5B-Instruct     │
│ 컴퓨팅 디바이스│ NVIDIA GPU (CUDA) │ NVIDIA RTX 4090       │
│ 학습 정밀도    │ OK     │ bfloat16 (bf16)                │
│ 4bit 양자화    │ OK     │ 사용 가능                      │
└────────────────┴────────┴────────────────────────────────┘

모든 점검 통과!
```

---

### `rag`

RAG 웹 채팅 서비스를 시작합니다. Qdrant 인덱스가 없으면 문서를 파싱하고 자동으로 구축합니다. 파인튜닝된 Student 모델이 Ollama에 등록되어 있으면 자동으로 해당 모델을 사용하고, 없으면 Teacher 모델(기본값: `qwen3.5:9b`)로 폴백합니다.

**사용법**

```
slf rag [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--chat/--no-chat` | | `BOOL` | `True` | RAG 인덱스 구축 후 웹 채팅 서비스를 자동으로 시작합니다. `--no-chat`으로 인덱스 구축만 수행합니다. |

**동작 방식**

- **모델 자동 선택**: ① `rag.ollama_model` 설정값 → ② Ollama에 등록된 파인튜닝 모델(`export.ollama.model_name`) → ③ Teacher 모델(`teacher.model`) 순으로 사용합니다.
- Qdrant 인덱스(`output/qdrant_db/`)가 이미 존재하면 즉시 서버를 시작합니다.
- 인덱스가 없으면 문서 파싱 → RAG 데이터 내보내기 → Qdrant 인덱싱을 자동으로 수행한 후 서버를 시작합니다.
- `--no-chat` 사용 시 인덱스 구축만 수행하고 서버를 시작하지 않습니다.

**예시**

```bash
# 기본 실행 (인덱스 자동 구축 + 서버 시작)
slf rag

# 인덱스 구축만 수행 (서버 미시작)
slf rag --no-chat

# 특정 설정 파일 사용
slf rag --config my-project/project.yaml
```

> `--host`와 `--port` 옵션이 필요한 경우 `slf tool rag-serve`를 사용하십시오.

---

## ⚙️ 파이프라인

### `tune`

문서 파싱부터 모델 배포까지 전체 파이프라인을 실행합니다. `--until` 옵션으로 특정 단계까지만 실행하고 중단할 수 있습니다.

**사용법**

```
slf tune [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--resume` | `-r` | `FLAG` | `False` | 중간 저장 파일에서 재개합니다. |
| `--chat/--no-chat` | | `BOOL` | `True` | 파이프라인 완료 후 RAG 웹 채팅 서비스를 자동으로 시작합니다. `--no-chat`으로 비활성화합니다. 서버는 foreground로 실행되며, `Ctrl+C`로 종료합니다. |
| `--from` | | `ENUM` | `None` | 지정된 단계부터 실행을 재개합니다. 이전 단계의 출력 파일이 필요합니다. |
| `--until` | | `ENUM` | `None` | 지정된 단계까지만 실행합니다. 생략하면 전체 파이프라인을 실행합니다. |

**`--until` 단계 값**

`PipelineStep` 열거형 값을 사용합니다. 각 단계는 이전 단계의 출력을 입력으로 받습니다.

| 값 | 필수/선택 | 설명 | 생성 파일 |
|----|:---------:|------|-----------|
| `parse` | 필수 | 문서를 파싱하여 텍스트와 표를 추출합니다 | `output/parsed_documents.json` |
| `generate` | 필수 | Teacher LLM으로 QA 쌍을 생성합니다 | `output/qa_alpaca.json` |
| `validate` | 필수 | 규칙 기반 및 임베딩 기반으로 QA 쌍을 검증합니다 | `output/qa_alpaca.json` (갱신) |
| `score` | 선택 | Teacher LLM이 QA 쌍을 1~5점으로 평가하고 저품질을 제거합니다 | `output/qa_scored.json` |
| `augment` | 선택 | 질문을 다양한 표현으로 변형하여 데이터를 증강합니다 | `output/qa_augmented.json` |
| `analyze` | 선택 | 카테고리 분포, 길이 통계 등 데이터 분석 보고서를 생성합니다 | `output/data_analysis.json` |
| `convert` | 필수 | 채팅 템플릿 적용 JSONL 변환 | `output/training_data.jsonl` |
| `train` | 필수 | LoRA 파인튜닝 | `output/checkpoints/adapter/` |
| `export` | 필수 | 모델 병합 + Ollama Modelfile 생성 | `output/merged_model/` |
| `eval` | 기본 활성 | BLEU/ROUGE 평가 | `output/eval_results.json` |
| `corpus_export` | 기본 활성 | 외부 평가용 코퍼스 데이터 내보내기 | `output/corpus/` |
| `rag_index` | 기본 활성 | Qdrant 벡터 인덱싱 | `output/qdrant_db/` |

`--until`을 생략하면 위 전체 12단계를 순서대로 실행합니다. 각 단계는 해당 설정의 `enabled` 값에 따라 실행 여부가 결정됩니다. Iterative Refinement(`refine`)는 `--until`로 지정할 수 없으며, `pipeline.run()` 내부에서 `eval` 단계 이후 `refinement.enabled` 설정에 따라 자동 실행됩니다.

**`--resume` 동작 방식**

출력 디렉토리에서 가장 최근 중간 저장 파일을 탐색하여 해당 단계부터 재개합니다.

| 발견된 파일 | 재개 단계 |
|-------------|-----------|
| `qa_augmented.json` | `analyze`부터 |
| `qa_scored.json` | `augment`부터 |
| `qa_alpaca.json` | `validate`부터 |
| `parsed_documents.json` | `generate`부터 |
| 없음 | 처음부터 |

**예시**

```bash
# 전체 파이프라인 실행
slf tune

# 문서 파싱만 실행
slf tune --until parse

# QA 생성까지 실행
slf tune --until generate

# 검증까지 실행
slf tune --until validate

# 품질 평가까지 실행
slf tune --until score

# 중단된 지점에서 재개
slf tune --resume

# 특정 단계까지 재개하며 실행
slf tune --until augment --resume

# 전체 파이프라인 + RAG 서버 자동 시작
slf tune --chat

# RAG 인덱싱까지 실행 (서빙은 별도)
slf tune --until rag_index

# 평가까지만 실행
slf tune --until eval
```

**출력 예시 (전체 파이프라인)**

```
slm-factory — 전체 파이프라인 시작 중...

  ✓ 5개 문서 파싱 완료
  ✓ 150개 QA 쌍 생성 완료
  ✓ 검증 완료: 132개 수락, 18개 거부
  ✓ 점수 평가: 120개 통과, 12개 제거
  ✓ 데이터 증강: 120개 → 360개
  ✓ 360개 QA 쌍 분석 완료

파이프라인 완료! 모델 저장 위치: ./my-project/output/merged_model
```

---

### `train`

LoRA 파인튜닝을 실행합니다. `--data`로 기존 학습 데이터를 직접 지정하거나, 생략하면 파싱부터 변환까지 전체 전처리를 자동으로 실행합니다.

**사용법**

```
slf train [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | 사전 생성된 `training_data.jsonl` 파일 경로입니다. 지정하면 전처리 단계를 건너뜁니다. |
| `--resume` | `-r` | `FLAG` | `False` | 중간 저장 파일에서 재개합니다. |

**동작 방식**

- `--data` 지정: 해당 JSONL 파일로 바로 학습을 시작합니다.
- `--resume` 지정 (data 없음): 중간 저장 파일을 탐색하여 해당 단계부터 전처리를 재개한 뒤 학습합니다.
- 옵션 없음: 파싱부터 변환까지 전체 전처리를 실행한 뒤 학습합니다.

**예시**

```bash
# 전체 전처리 후 학습
slf train

# 기존 학습 데이터로 학습만 실행
slf train --data ./my-project/output/training_data.jsonl

# 외부에서 준비한 데이터로 학습
slf train --data ./custom_data.jsonl

# 중단된 전처리부터 재개하여 학습
slf train --resume
```

**출력**

```
훈련 완료! 어댑터 저장 위치: ./my-project/output/checkpoints/adapter
```

**참고**

- 학습 하이퍼파라미터(`num_epochs`, `batch_size`, `learning_rate` 등)는 `project.yaml`의 `training` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.
- GPU 메모리 부족 시 해결 방법은 [사용 가이드](guide.md)를 참조하십시오.

---

### `export`

훈련된 LoRA 어댑터를 기본 모델에 병합하고 Ollama Modelfile을 생성합니다. 파이프라인 전체를 재실행하지 않고 내보내기만 단독으로 수행합니다.

**사용법**

```
slf export [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--adapter` | | `TEXT` | `None` | 어댑터 디렉토리 경로입니다. 미지정 시 `output/checkpoints/adapter/`를 사용합니다. |

**예시**

```bash
# 기본 어댑터 경로로 내보내기
slf export

# 특정 어댑터 경로 지정
slf export --adapter ./my-project/output/checkpoints/adapter
```

**출력**

```
내보내기 완료! 모델 저장 위치: ./my-project/output/merged_model
```

생성 파일:
- `output/merged_model/` — HuggingFace 형식의 병합된 모델
- `output/merged_model/Modelfile` — Ollama 배포용 Modelfile

**참고**

- Ollama 배포 설정(`model_name`, `system_prompt` 등)은 `project.yaml`의 `export.ollama` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

## 📊 평가

### `eval run`

학습된 모델을 QA 데이터로 평가합니다. BLEU와 ROUGE 메트릭을 계산하여 결과를 저장합니다.

**사용법**

```
slf eval run [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--model` | | `TEXT` | 필수 | 평가할 Ollama 모델 이름입니다. |
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로입니다. 미지정 시 출력 디렉토리에서 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# 기본 설정으로 평가
slf eval run --model my-project-model

# 특정 QA 데이터로 평가
slf eval run --model my-project-model --data ./my-project/output/qa_alpaca.json
```

**출력**

터미널에 BLEU/ROUGE 점수 요약을 출력하고, `output/eval_results.json`에 상세 결과를 저장합니다.

```
평가 완료! 결과: ./my-project/output/eval_results.json (50건)
```

**참고**

- 평가 대상 모델은 Ollama에 등록된 모델이어야 합니다. `ollama list`로 확인하십시오.
- 평가 설정(`output_file` 등)은 `project.yaml`의 `eval` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

### `eval compare`

Base 모델과 Fine-tuned 모델의 답변을 동일한 질문으로 비교합니다. 파인튜닝 전후의 성능 차이를 확인할 때 사용합니다.

**사용법**

```
slf eval compare [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--base-model` | | `TEXT` | 필수 | 비교 기준 모델 이름 (Ollama)입니다. |
| `--ft` | | `TEXT` | 필수 | 파인튜닝된 모델 이름 (Ollama)입니다. |
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로입니다. 미지정 시 출력 디렉토리에서 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# Base 모델과 파인튜닝 모델 비교
slf eval compare \
  --base-model gemma:2b \
  --ft my-project-model

# 특정 QA 데이터로 비교
slf eval compare \
  --base-model gemma:2b \
  --ft my-project-model \
  --data ./my-project/output/qa_alpaca.json
```

**출력**

터미널에 비교 요약을 출력하고, `output/compare_results.json`에 각 질문별 두 모델의 답변을 저장합니다.

```
비교 완료! 결과: ./my-project/output/compare_results.json (50건)
```

---

## 🔧 도구

### `tool review`

생성된 QA 쌍을 TUI(텍스트 기반 UI)에서 하나씩 확인하며 승인, 거부, 편집합니다. 데이터 품질을 수동으로 관리할 때 사용합니다.

**사용법**

```
slf tool review [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로입니다. 미지정 시 출력 디렉토리에서 자동 감지합니다. |

`--data` 미지정 시 자동 감지 우선순위: `qa_reviewed.json` → `qa_augmented.json` → `qa_scored.json` → `qa_alpaca.json`

**예시**

```bash
# 자동 감지된 QA 데이터로 리뷰 시작
slf tool review

# 특정 파일로 리뷰 시작
slf tool review --data ./my-project/output/qa_alpaca.json
```

**출력**

TUI 종료 후 리뷰 결과를 출력합니다.

```
리뷰 완료! 승인: 98, 거부: 12, 대기: 5
```

생성 파일: `output/qa_reviewed.json` (승인된 QA 쌍만 포함)

---

### `tool convert`

QA 데이터를 Student 모델의 채팅 템플릿이 적용된 훈련용 JSONL 형식으로 변환합니다. 파이프라인 전체를 재실행하지 않고 변환만 단독으로 수행합니다.

**사용법**

```
slf tool convert [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--data` | | `TEXT` | `None` | QA 데이터 파일 경로 (`qa_alpaca.json` 또는 `qa_augmented.json`). 미지정 시 자동 감지합니다. |

**`--data` 미지정 시 자동 감지 우선순위**

출력 디렉토리에서 다음 순서로 파일을 탐색합니다.

1. `qa_augmented.json`
2. `qa_scored.json`
3. `qa_alpaca.json`

**예시**

```bash
# 자동 감지된 QA 데이터로 변환
slf tool convert

# 특정 파일로 변환
slf tool convert --data ./my-project/output/qa_augmented.json
```

**출력**

```
변환 완료! 훈련 데이터: ./my-project/output/training_data.jsonl (360개 쌍)
```

생성 파일: `output/training_data.jsonl`

---

### `tool evolve`

문서 변경을 감지하고 증분 처리한 뒤, 자동으로 재학습 및 배포하는 통합 명령어입니다. 품질 게이트를 통과한 경우에만 새 모델을 배포하며, 진화 히스토리를 기록합니다.

**사용법**

```
slf tool evolve [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--force-update` | | `FLAG` | `False` | 변경 감지를 건너뛰고 모든 문서를 처리합니다. |
| `--skip-gate` | | `FLAG` | `False` | 품질 게이트를 건너뛰고 무조건 배포합니다. |

**5단계 진화 흐름**

| # | 단계 | 설명 |
|---|------|------|
| 1 | 증분 업데이트 | 문서 변경을 감지하고 새 QA를 생성하여 기존 QA와 병합합니다 |
| 2 | 검증/점수/증강 | 병합된 QA를 검증하고, 선택적으로 점수 평가 및 데이터 증강을 실행합니다 |
| 3 | 학습/내보내기 | 전처리된 데이터로 LoRA 파인튜닝을 실행하고 모델을 병합합니다 |
| 4 | 품질 게이트 | 새 모델을 이전 모델과 비교하여 개선 여부를 판단합니다 (설정 가능) |
| 5 | 정리/완료 | 배포 성공 시 히스토리를 기록하고 이전 버전을 관리합니다 |

**예시**

```bash
# 기본 진화 (변경 감지 + 품질 게이트 적용)
slf tool evolve

# 모든 문서 재처리 (변경 감지 건너뜀)
slf tool evolve --force-update

# 품질 게이트 건너뜀 (무조건 배포)
slf tool evolve --skip-gate

# 모든 옵션 적용
slf tool evolve --force-update --skip-gate
```

**출력 예시 (성공)**

```
진화 시작: my-project

[1/5] 증분 업데이트
  변경 문서: 2개 감지
  새 QA: 40개 생성
  전체 QA: 190개 (전략: append)

[2/5] 검증/점수/증강
  검증: 190개 → 185개 (5개 거부)
  점수 평가: 185개 → 175개 (10개 제거)
  데이터 증강: 175개 → 525개

[3/5] 학습/내보내기
  학습 완료: 20 에포크 (early stopping)
  모델 병합 완료

[4/5] 품질 게이트
  이전 모델: rougeL=0.45
  새 모델: rougeL=0.48
  개선율: +6.7% ✓ 통과

[5/5] 정리/완료
  배포 완료: v20250220
  이전 버전 보관: v20250219, v20250218, v20250217
  히스토리 저장: evolve_history.json

진화 완료! 새 모델: my-project-model (v20250220)
```

**출력 예시 (품질 게이트 실패)**

```
[4/5] 품질 게이트
  이전 모델: rougeL=0.45
  새 모델: rougeL=0.44
  개선율: -2.2% ✗ 실패

배포 취소됨. 이전 모델 유지: my-project-model (v20250219)
```

**참고**

- 진화 명령은 `tool update` + `tune --until analyze` + `train` + `export` + `eval compare` + 버전 관리를 자동으로 수행합니다.
- 품질 게이트 설정(`quality_gate`, `gate_metric`, `gate_min_improvement`)은 `project.yaml`의 `evolve` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.
- 진화 히스토리는 `evolve_history.json`에 기록되며, 각 진화 단계의 메트릭과 타임스탬프를 포함합니다.
- `--force-update` 사용 시 모든 문서를 재처리하므로 처리 시간이 증가합니다.
- `--skip-gate` 사용 시 품질 검증 없이 배포되므로 신중하게 사용하십시오.
- 품질 게이트 실패 시 새로 생성된 Ollama 모델은 자동으로 삭제(`ollama rm`)되며 이전 모델이 유지됩니다.

---

### `tool compare-data`

두 QA 데이터셋의 품질을 나란히 비교합니다. 파이프라인 설정 변경(청킹 활성화, 재생성 적용 등) 전후의 QA 품질 변화를 수치로 확인할 때 사용합니다.

**사용법**

```
slf tool compare-data [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--baseline` | `-b` | `TEXT` | 필수 | 기준 QA 데이터 파일 경로입니다. |
| `--target` | `-t` | `TEXT` | 필수 | 비교 대상 QA 데이터 파일 경로입니다. |

**비교 항목**

| 항목 | 설명 |
|------|------|
| 전체 QA 쌍 | 총 QA 수 비교 (증감 표시) |
| 원본/증강 QA 쌍 | 원본과 증강 QA 비율 변화 |
| 문서 소스 수 | QA가 생성된 문서 수 비교 |
| 답변 길이 통계 | 평균/최소/최대 답변 길이 변화 |
| 카테고리 분포 | 카테고리별 QA 수 비교 |

**예시**

```bash
# 청킹 전후 QA 품질 비교
slf tool compare-data \
  --baseline ./output/qa_alpaca_before.json \
  --target ./output/qa_alpaca.json

# 재생성 적용 전후 비교
slf tool compare-data \
  -b ./output/qa_scored_v1.json \
  -t ./output/qa_scored_v2.json
```

**출력**

Rich 테이블로 기본 통계, 답변 길이 변화, 카테고리 분포를 보여줍니다. 개선된 항목은 초록색, 악화된 항목은 빨간색으로 표시됩니다.

---

### `tool export-corpus`

slm-factory의 파싱·QA 데이터를 외부 평가용 parquet 형식으로 내보냅니다. `parsed_documents.json`에서 문서 청크를 생성하고, QA 파일에서 평가 데이터를 추출하여 `corpus.parquet`과 `qa.parquet`을 생성합니다. 결과는 RAG 인덱싱뿐 아니라 외부 평가 도구나 자체 검색 평가에서도 사용할 수 있습니다.

**사용법**

```
slf tool export-corpus [OPTIONS]
```

**옵션**

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--config` | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--qa-file` | `TEXT` | (자동 감지) | QA 데이터 파일 경로입니다. 지정하지 않으면 `qa_scored.json` → `qa_validated.json` → `qa_alpaca.json` 순서로 자동 탐색합니다. |

**예시**

```bash
# 기본 실행 (QA 파일 자동 탐색)
slf tool export-corpus

# QA 파일 직접 지정
slf tool export-corpus \
  --qa-file ./output/qa_scored.json
```

**출력**

```
output/corpus/
├── corpus.parquet    # 문서 청크 (검색 코퍼스)
└── qa.parquet        # QA 평가 데이터
```

**참고**

- `corpus.parquet`은 `parsed_documents.json`의 텍스트를 `corpus_export.chunk_size` 단위로 분할하여 생성합니다. 청크 간 중첩은 `corpus_export.overlap_chars`로 설정합니다.
- `qa.parquet`은 QA 쌍의 질문·답변·컨텍스트를 parquet 형식으로 변환합니다.
- 상세 설정은 [설정 레퍼런스](configuration.md)의 `corpus_export` 섹션을 참조하십시오.

---

### `tool rag-index`

`corpus.parquet`을 sentence-transformers 모델로 임베딩한 뒤 Qdrant에 적재합니다. `tool export-corpus`로 생성한 코퍼스 데이터를 벡터 검색이 가능한 형태로 변환합니다.

**사용법**

```
slf tool rag-index [OPTIONS]
```

**옵션**

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--config` | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. |
| `--corpus-dir` | `TEXT` | (자동 감지) | `corpus.parquet`이 있는 디렉토리 경로입니다. 기본값은 `output/corpus`입니다. |

**예시**

```bash
# 기본 실행
slf tool rag-index

# 코퍼스 디렉토리 직접 지정
slf tool rag-index --corpus-dir ./output/corpus
```

**출력**

Qdrant가 `output/qdrant_db/`에 생성됩니다. 임베딩 모델과 벡터DB 경로는 `project.yaml`의 `rag` 섹션에서 설정합니다.

**참고**

- 사전 조건: `tool export-corpus`로 `corpus.parquet`을 먼저 생성해야 합니다.
- 임베딩 모델 기본값은 `Qwen/Qwen3-Embedding-0.6B`입니다. 비대칭 인코딩(`prompt_name="query"`)으로 한국어 문서에 우수한 성능을 보입니다.
- `uv sync --extra rag --extra validation`으로 의존성을 설치하세요.
- 기존 컬렉션에 재실행하면 upsert(갱신/추가)됩니다.

---

### `tool rag-serve`

Qdrant 벡터 검색과 Ollama SLM 생성을 결합한 RAG API 서버를 실행합니다. `tool rag-index`로 구축한 벡터 DB에서 관련 문서를 검색하고, SLM이 문서 기반 답변을 생성합니다.

> 서버는 foreground로 실행됩니다. `Ctrl+C`로 종료할 수 있습니다.

**사용법**

```
slf tool rag-serve [OPTIONS]
```

**옵션**

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--config` | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. |
| `--host` | `TEXT` | (설정 파일 값) | 서버 바인드 호스트입니다. |
| `--port` | `INT` | (설정 파일 값) | 서버 포트입니다. |

**예시**

```bash
# 기본 실행
slf tool rag-serve

# 포트 지정
slf tool rag-serve --port 9000
```

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/v1/query` | 질의 → 문서 검색 → SLM 답변 생성 (`stream: true`로 SSE 스트리밍 가능) |
| `POST` | `/v1/stream` | 웹 채팅 UI 전용 SSE 스트리밍 엔드포인트 |
| `GET` | `/v1/models` | OpenAI 호환 모델 목록 (`slm-factory-auto` / `-rag` / `-agent`) |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions 호환 — OpenWebUI 등 연동. 스트리밍/비스트리밍, multimodal content 리스트 수용 |
| `GET` | `/chat` | 내장 웹 채팅 UI (HTML 페이지, 다크/라이트 테마 토글) |
| `GET` | `/health` | `/health/ready`의 별칭 — Qdrant 및 Ollama 연결 상태 확인 |
| `GET` | `/health/ready` | Qdrant 및 Ollama 연결 상태 확인 (로드밸런서 헬스체크용) |
| `GET` | `/health/live` | 라이브니스 체크 — 서버 실행 중이면 항상 200 응답 |

**OpenAI 호환 모델 라우팅**

| 모델 ID | 동작 |
|---------|------|
| `slm-factory-auto` | `AgentOrchestrator.handle_auto` — simple/agent 자동 선택 (기본값) |
| `slm-factory-rag` | 단일 패스 simple RAG |
| `slm-factory-agent` | Agent RAG (다단계 ReAct) — `rag.agent.enabled=true`일 때 목록에 추가 |

알 수 없는 모델 ID는 `slm-factory-auto`로 fallback됩니다. `body.user` 필드가 있으면 세션 키로 사용되어 Agent RAG 히스토리가 유지됩니다.

**API 호출 예시**

```bash
# 일반 응답
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "도메인 질문", "top_k": 5}'

# SSE 스트리밍 응답 (토큰 단위 실시간 전송)
curl -N -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "도메인 질문", "stream": true}'
```

**응답 형식**

```json
{
  "answer": "문서 기반 답변...",
  "sources": [
    {"content": "관련 문서 청크...", "doc_id": "...", "score": 0.85}
  ],
  "query": "도메인 질문"
}
```

**SSE 스트리밍 응답 형식** (`"stream": true` 요청 시)

`/v1/query` 및 `/v1/stream` 엔드포인트 (단일 패스 RAG):
```
data: {"token": "문서"}
data: {"token": " 기반"}
data: {"token": " 답변..."}
data: {"sources": [...], "query": "도메인 질문", "done": true}
```

`/v1/chat/completions` — 모델 `slm-factory-agent` (Agent RAG):
```
data: {"type": "route", "mode": "agent", "intent": "factual"}
data: {"type": "thought", "content": "...", "iteration": 1}
data: {"type": "action", "content": "search", "input": "...", "iteration": 1}
data: {"type": "observation", "content": "...", "iteration": 1}
data: {"type": "token", "content": "최종"}
data: {"type": "token", "content": " 답변"}
data: {"type": "sources", "sources": [...]}
data: {"type": "done", "session_id": "..."}
```

`/v1/chat/completions` — 모델 `slm-factory-auto` 또는 `slm-factory-rag` (OpenAI 호환):
```
data: {"choices":[{"delta":{"content":"문서"}}]}
data: {"choices":[{"delta":{"content":" 기반"}}]}
...
data: [DONE]
```

**참고**

- 사전 조건: `tool rag-index`로 Qdrant를 먼저 구축해야 합니다.
- Ollama가 실행 중이어야 합니다 (`ollama serve`).
- SLM 모델은 `rag.ollama_model` 또는 `export.ollama.model_name`에서 결정됩니다.
- `uv sync --extra rag --extra validation`으로 의존성을 설치하세요.

---

### `tool eval-retrieval`

RAG 검색 품질을 평가합니다. `qa.parquet`의 질문으로 Qdrant를 검색하고, 검색된 결과가 원본 문서를 얼마나 정확하게 찾아내는지 Hit@1, Hit@K, MRR(Mean Reciprocal Rank), Recall@K 메트릭으로 측정합니다.

**사용법**

```
slf tool eval-retrieval [OPTIONS]
```

**옵션**

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--config` | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--top-k` | `INT` | `5` | 검색 결과 상위 K개를 평가 기준으로 사용합니다. |
| `--qa-file` | `TEXT` | (자동 감지) | `qa.parquet` 파일 경로입니다. 미지정 시 `output/corpus/qa.parquet`을 자동으로 사용합니다. |

**예시**

```bash
# 기본 실행 (qa.parquet 자동 감지)
slf tool eval-retrieval

# 상위 10개 결과로 평가
slf tool eval-retrieval --top-k 10

# QA 파일 직접 지정
slf tool eval-retrieval --qa-file ./output/corpus/qa.parquet
```

**참고**

- 사전 조건: `tool export-corpus`로 `qa.parquet`을 생성하고, `tool rag-index`로 Qdrant를 구축해야 합니다.
- `slf rag` 실행 시 `qa.parquet`이 존재하면 자동으로 검색 평가를 수행하고 결과를 출력합니다.

---

### `tool update`

문서 디렉토리를 스캔하여 변경된 파일만 감지하고, 새 QA를 생성하여 기존 QA와 병합합니다. 문서를 추가하거나 수정했을 때 전체 파이프라인을 재실행하지 않고 증분 처리할 수 있습니다.

**사용법**

```
slf tool update [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**동작 방식**

1. `IncrementalTracker`가 각 문서 파일의 해시값을 계산합니다.
2. 이전 실행 시 저장된 해시값과 비교하여 변경된 파일을 감지합니다.
3. 변경된 파일만 파싱하고 QA를 생성합니다.
4. `project.yaml`의 `incremental.merge_strategy` 설정에 따라 기존 `qa_alpaca.json`과 병합합니다.

**예시**

```bash
# 변경된 문서만 증분 처리
slf tool update
```

**출력**

```
증분 업데이트: 2개 변경 문서 감지

증분 업데이트 완료! 변경 문서: 2개, 새 QA: 40개, 전체 QA: 190개 (전략: append)
```

변경된 문서가 없으면 "변경된 문서가 없습니다" 메시지를 출력하고 종료합니다.

**참고**

- 병합 전략(`merge_strategy`)은 `project.yaml`의 `incremental` 섹션에서 설정합니다. [설정 레퍼런스](configuration.md)를 참조하십시오.

---

## ℹ️ 정보

### `status`

파이프라인 각 단계의 진행 상태를 확인합니다. 출력 디렉토리의 파일 존재 여부와 항목 수를 테이블로 표시합니다.

**사용법**

```
slf status [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |

**예시**

```bash
slf status
```

**출력 예시**

```
              파이프라인 진행 상태
┌──────────┬──────────────────────────┬────────┬──────────────┐
│ 단계     │ 파일                     │ 상태   │ 건수         │
├──────────┼──────────────────────────┼────────┼──────────────┤
│ parse    │ parsed_documents.json    │ 존재   │ 5개 문서     │
│ generate │ qa_alpaca.json           │ 존재   │ 150개 쌍     │
│ score    │ qa_scored.json           │ 없음   │ -            │
│ augment  │ qa_augmented.json        │ 없음   │ -            │
│ analyze  │ data_analysis.json       │ 없음   │ -            │
│ convert  │ training_data.jsonl      │ 없음   │ -            │
│ train    │ checkpoints/adapter/     │ 없음   │ -            │
│ export   │ merged_model/            │ 없음   │ -            │
└──────────┴──────────────────────────┴────────┴──────────────┘

다음 --resume 실행 시 validate부터 재개됩니다
```

모든 단계가 완료된 경우 "모든 단계가 완료되었습니다" 메시지를 출력합니다.

---

### `clean`

출력 디렉토리의 중간 생성 파일을 정리합니다. 삭제 전 확인 프롬프트가 표시됩니다.

**사용법**

```
slf clean [OPTIONS]
```

**옵션**

| 플래그 | 단축키 | 타입 | 기본값 | 설명 |
|--------|--------|------|--------|------|
| `--config` | | `TEXT` | `project.yaml` | 프로젝트 설정 파일 경로입니다. 현재 디렉토리부터 상위까지 자동 탐색합니다. |
| `--all` | | `FLAG` | `False` | 출력 디렉토리의 모든 파일을 삭제합니다. |
| `--yes` | `-y` | `bool` | `false` | 확인 없이 삭제합니다 |

**기본 동작 vs `--all` 동작**

| 동작 | 삭제 대상 |
|------|-----------|
| 기본 (`--all` 없음) | `qa_scored.json`, `qa_augmented.json`, `data_analysis.json` |
| `--all` | 출력 디렉토리(`output/`)의 모든 파일과 디렉토리 |

**예시**

```bash
# 중간 파일만 정리
slf clean

# 모든 출력 파일 정리 (학습 결과 포함)
slf clean --all
```

**출력**

```
삭제 대상:
  ./my-project/output/qa_scored.json
  ./my-project/output/qa_augmented.json
  ./my-project/output/data_analysis.json

삭제하시겠습니까? [y/N]:

          삭제 결과
┌──────────────────────────────────┬────────┐
│ 파일                             │ 상태   │
├──────────────────────────────────┼────────┤
│ ./my-project/output/qa_scored... │ 삭제됨 │
│ ...                              │ 삭제됨 │
└──────────────────────────────────┴────────┘

3개 항목 삭제 완료
```

---

### `version`

slm-factory의 현재 버전을 출력합니다.

**사용법**

```
slf version
```

**출력**

```
slm-factory 0.1.0
```

---

## 출력 파일 구조

파이프라인 실행 후 `output/` 디렉토리에 생성되는 파일 목록입니다.

```
output/
├── parsed_documents.json       # parse 단계 출력
├── qa_alpaca.json              # generate + validate 단계 출력
├── qa_scored.json              # score 단계 출력 (선택)
├── qa_augmented.json           # augment 단계 출력 (선택)
├── qa_reviewed.json            # tool review 출력 (선택)
├── data_analysis.json          # analyze 단계 출력 (선택)
├── eval_results.json           # eval run 출력 (선택)
├── compare_results.json        # eval compare 출력 (선택)
├── training_data.jsonl         # convert 단계 출력
├── checkpoints/
│   └── adapter/                # train 단계 출력
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
├── corpus/                     # tool export-corpus 출력 (선택)
│   ├── corpus.parquet          # 문서 청크 (검색 코퍼스)
│   └── qa.parquet              # QA 평가 데이터
├── qdrant_db/                     # rag_index 단계 출력 (선택)
│   └── ...                        # Qdrant 벡터 인덱스
└── merged_model/               # export 단계 출력
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── Modelfile
```

**파일별 설명**

| 파일 | 생성 명령 | 설명 |
|------|-----------|------|
| `parsed_documents.json` | `tune --until parse` | 원본 문서에서 추출한 텍스트, 표, 메타데이터입니다. `--resume` 시 파싱 단계를 건너뜁니다. |
| `qa_alpaca.json` | `tune --until generate` | Teacher LLM이 생성한 QA 쌍입니다. Alpaca 형식(`instruction`, `input`, `output`)으로 저장됩니다. |
| `qa_scored.json` | `tune --until score` | 품질 점수 평가를 통과한 QA 쌍입니다. `--resume` 시 `augment`부터 재개합니다. |
| `qa_augmented.json` | `tune --until augment` | 데이터 증강이 완료된 QA 쌍입니다. `--resume` 시 `analyze`부터 재개합니다. |
| `qa_reviewed.json` | `tool review` | TUI에서 수동 리뷰를 거친 QA 쌍입니다. 승인된 항목만 포함됩니다. |
| `data_analysis.json` | `tune --until analyze` | 카테고리 분포, 길이 통계, 데이터 품질 경고를 포함한 분석 보고서입니다. |
| `eval_results.json` | `eval run` | BLEU/ROUGE 메트릭별 점수와 평균 점수를 포함한 평가 결과입니다. |
| `compare_results.json` | `eval compare` | 각 질문에 대한 Base 모델과 Fine-tuned 모델의 답변을 나란히 기록한 비교 결과입니다. |
| `training_data.jsonl` | `tool convert` | Student 모델의 채팅 템플릿이 적용된 학습 데이터입니다. 각 줄은 `{"text": "..."}` 형식입니다. |
| `checkpoints/adapter/` | `train` | PEFT 형식의 LoRA 어댑터 가중치입니다. `export` 명령으로 기본 모델과 병합합니다. |
| `corpus/` | `tool export-corpus` | 외부 평가용 parquet 데이터. `corpus.parquet`(문서 청크)과 `qa.parquet`(QA 평가 데이터)을 포함합니다. RAG 인덱싱에서도 사용됩니다. |
| `qdrant_db/` | `tool rag-index` | Qdrant 벡터 인덱스입니다. `tool rag-serve`가 이 디렉토리를 참조하여 유사도 검색을 수행합니다. |
| `merged_model/` | `export` | LoRA 어댑터가 병합된 최종 모델입니다. `Modelfile`로 Ollama에 즉시 배포할 수 있습니다. |

---

*관련 문서: [설정 레퍼런스](configuration.md) · [사용 가이드](guide.md) · [아키텍처](architecture.md)*
