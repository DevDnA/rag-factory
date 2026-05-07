# 설정 레퍼런스 (project.yaml)

> `project.yaml`의 모든 설정 옵션을 확인하세요.

## 1. 개요

`project.yaml`은 slm-factory 파이프라인의 모든 동작을 제어하는 중앙 설정 파일입니다. 코드 수정 없이 문서 파싱, QA 생성, 검증, 학습, 내보내기 등 전체 워크플로우를 구성할 수 있습니다.

**파이프라인 제어**: 문서 파싱부터 모델 내보내기까지 전 단계를 YAML 파일 하나로 제어합니다.

**타입 안전성**: Pydantic v2 기반 검증으로 잘못된 설정을 실행 전에 감지합니다.

**기본값 제공**: 모든 필드에 합리적인 기본값이 설정되어 있어 필요한 부분만 수정하면 됩니다.

**null 섹션 처리**: YAML에서 `null`로 지정된 섹션은 자동으로 기본값이 적용됩니다. 예를 들어 `eval: null`로 설정하면 `eval` 섹션 전체가 기본값으로 초기화됩니다.

### 설정 파일 생성

새 프로젝트를 시작할 때는 다음 명령으로 기본 템플릿을 생성합니다:

```bash
slf init my-project
```

이 명령은 `my-project/project.yaml` 파일을 생성하며, 모든 기본값이 주석과 함께 포함되어 있습니다.

### 파일 위치

`project.yaml` 파일은 프로젝트 루트 디렉토리에 위치해야 합니다:

```
my-project/
├── project.yaml          # 설정 파일
├── documents/            # 입력 문서 디렉토리
└── output/               # 출력 디렉토리 (자동 생성)
```

설정 로딩 프로세스 상세는 [아키텍처 가이드](architecture.md)를 참조하십시오.

<!-- diagram: configuration-config-tree -->

---

## 2. project — 프로젝트 메타데이터

> 프로젝트의 기본 정보를 정의합니다. 내보낸 모델의 이름과 버전 관리에 사용됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | `str` | `"my-project"` | 프로젝트 식별자. 내보낸 모델 이름에 반영됩니다 (빈 값 불가) |
| `version` | `str` | `"1.0.0"` | 시맨틱 버전. 모델 버전 관리에 사용됩니다 |
| `language` | `str` | `"en"` | 문서 언어 코드. `"ko"` 설정 시 QA 생성·검증·평가 프롬프트가 자동으로 한국어로 전환됩니다. `"en"` (영어), `"ko"` (한국어), `"ja"` (일본어) 등 |

```yaml
project:
  name: "company-policy-assistant"
  version: "2.1.0"
  language: "ko"
```

---

## 3. paths — 경로 설정

> 입력 문서와 출력 파일의 위치를 지정합니다. 지정된 디렉토리가 없으면 자동으로 생성됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `documents` | `Path` | `"./documents"` | 입력 문서가 저장된 디렉토리. 파싱할 PDF, HWPX 등의 파일을 여기에 배치합니다 |
| `output` | `Path` | `"./output"` | 모든 출력 파일이 저장되는 디렉토리. QA 데이터셋, 학습된 모델, 체크포인트 등이 생성됩니다 |

```yaml
paths:
  documents: "./data/source_docs"
  output: "./results"
```

상대 경로는 `project.yaml` 파일이 위치한 디렉토리를 기준으로 해석됩니다. 프로젝트 디렉토리 외부에서 `--config my-project/project.yaml`로 실행해도 올바른 경로를 참조합니다.

---

## 4. parsing — 문서 파싱 설정

> 입력 문서를 텍스트로 변환하는 파싱 단계의 설정입니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `formats` | `list[str]` | `["pdf", "txt", "html", "md", "hwpx", "hwp", "docx"]` | 파싱할 문서 형식 목록 |
| `pdf` | `PdfOptions` | (하위 참조) | PDF 파싱 옵션 |
| `hwpx` | `HwpxOptions` | (하위 참조) | HWPX 파싱 옵션 |

### 지원 형식

| 형식 | 확장자 | 필요 패키지 |
|------|--------|-------------|
| `pdf` | `.pdf` | `pymupdf` (기본 포함) |
| `hwpx` | `.hwpx` | `beautifulsoup4`, `lxml` (기본 포함), `kiwipiepy` (선택) |
| `hwp` | `.hwp` | `olefile` (`uv sync --extra hwp`) |
| `html` | `.html`, `.htm` | `beautifulsoup4` (기본 포함) |
| `txt` | `.txt` | 없음 |
| `md` | `.md` | 없음 |
| `docx` | `.docx` | `python-docx` (`uv sync --extra docx`) |

### pdf 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `extract_tables` | `bool` | `true` | PDF 내 표를 마크다운 형식으로 추출합니다. `false`로 설정하면 표를 일반 텍스트로 처리합니다 |

### hwpx 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `apply_spacing` | `bool` | `true` | 한국어 띄어쓰기 교정을 적용합니다. `kiwipiepy` 패키지가 필요합니다. `uv sync --extra korean`으로 설치하면 띄어쓰기 교정 및 형태소 분석(`kiwipiepy`) 기능이 설치됩니다 |

```yaml
parsing:
  formats: ["pdf", "hwpx", "html", "txt", "docx"]
  pdf:
    extract_tables: true
  hwpx:
    apply_spacing: true
```

---

## 5. chunking — 문서 청킹 설정

> 긴 문서를 청크(조각)로 분할하여 각 청크마다 QA를 생성합니다. 문서 전체를 빠짐없이 커버하여 QA 품질을 향상시킵니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 문서 청킹 기능 활성화 여부 |
| `chunk_size` | `int \| "auto"` | `"auto"` | 각 청크의 최대 문자 수. `"auto"`이면 문서의 단락 밀도와 길이를 분석하여 최적 크기를 자동 결정합니다. 정수를 지정하면 해당 크기를 사용합니다 (최소 1000) |
| `overlap_chars` | `int` | `500` | 연속된 청크 간 중첩 문자 수. 문맥 유지를 위해 이전 청크의 끝부분을 다음 청크에 포함합니다 |

```yaml
chunking:
  enabled: true
  chunk_size: "auto"     # 또는 정수 지정 (예: 10000)
  overlap_chars: 500
```

`teacher.max_context_chars`보다 짧은 문서는 청킹 없이 그대로 처리됩니다. 긴 문서(예: 50페이지 이상 PDF)에서 문서 뒷부분의 내용이 QA에 누락되는 문제를 해결합니다.

청크는 문단 경계(빈 줄)를 기준으로 분할되어 문장이 중간에 잘리지 않습니다.

---

## 6. teacher — Teacher LLM 설정

> QA 쌍을 생성하는 Teacher 모델의 설정입니다. Ollama 또는 OpenAI 호환 API를 사용할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `backend` | `"ollama"` \| `"openai"` | `"ollama"` | 사용할 LLM 백엔드. `"ollama"`는 로컬 Ollama 서버, `"openai"`는 OpenAI 호환 API입니다 |
| `model` | `str` | `"qwen3.5:9b"` | 모델 이름 또는 ID (빈 값 불가) |
| `api_base` | `str` | `"http://localhost:11434"` | API 엔드포인트 URL (빈 값 불가) |
| `api_key` | `str \| None` | `null` | API 키. `backend: "openai"`일 때 필수입니다. Ollama는 불필요합니다 |
| `temperature` | `float` | `0.3` | 생성 온도 (0.0~1.0). 낮을수록 일관성 있는 답변을 생성합니다 |
| `timeout` | `int` | `300` | API 요청 타임아웃 (초). 긴 문서 처리 시 늘려야 할 수 있습니다 |
| `max_context_chars` | `int` | `12000` | 문서 내용 잘라내기 한계 (문자 수). 이 길이를 초과하는 문서는 앞부분만 사용됩니다 |
| `max_concurrency` | `int` | `2` | 비동기 QA 생성 시 최대 동시 요청 수. 높이면 빠르지만 서버 부하가 증가합니다 |

```yaml
teacher:
  backend: "ollama"
  model: "qwen3.5:9b"
  api_base: "http://localhost:11434"
  temperature: 0.3
  timeout: 300
  max_context_chars: 12000
  max_concurrency: 2
```

OpenAI 호환 백엔드 (vLLM, LiteLLM, OpenRouter 등) 사용 시:

```yaml
teacher:
  backend: "openai"
  model: "gpt-4o-mini"
  api_base: "https://api.openai.com"
  api_key: "sk-proj-..."
  temperature: 0.3
```

권장 Ollama 모델: `qwen3.5:9b` (기본값, Qwen3.5 9B), `qwen3.5:27b` (고품질, 24GB+), `exaone3.5:7.8b` (한국어 최적화).

---

## 7. questions — 질문 설정

> Teacher 모델이 문서에 대해 생성할 질문을 정의합니다. 카테고리별 질문 목록 또는 외부 파일로 지정할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `categories` | `dict[str, list[str]]` | `{}` | 카테고리별 질문 목록. 키는 카테고리 이름, 값은 질문 문자열 리스트입니다 |
| `file` | `Path \| None` | `null` | 외부 질문 파일 경로. 한 줄에 하나씩 질문을 작성합니다. 이 필드가 설정되면 `categories`는 무시됩니다 |
| `system_prompt` | `str` | (하위 참조) | Teacher 모델에게 전달되는 시스템 프롬프트. 답변 스타일과 제약 조건을 정의합니다 |
| `output_format` | `str` | `"alpaca"` | QA 데이터셋 출력 형식. 현재는 `"alpaca"` 형식만 지원합니다 |
| `auto_generate` | `bool` | `false` | Teacher LLM이 각 청크를 분석하여 질문-답변 쌍을 자동으로 생성합니다. `categories` 고정질문이 있으면 자동생성 QA가 추가됩니다. `categories`가 비어있으면 자동생성만 수행합니다 |
| `questions_per_chunk` | `int \| "auto"` | `"auto"` | 청크당 생성할 질문 수. `"auto"`이면 청크의 정보 밀도(길이, 숫자, 목록 항목)를 분석하여 3~15개 범위에서 자동 결정됩니다 |

**제약조건**: `file` 경로가 설정되어 있지만 해당 파일이 존재하지 않으면 `FileNotFoundError`가 발생합니다.

기본 시스템 프롬프트:

```
You are a helpful assistant that answers questions based strictly on the provided document.
Answer only from the document content. Do not speculate or fabricate information.
Be concise and factual. Include specific numbers, dates, and names when available.
If the document does not contain relevant information, say "The document does not contain this information."
```

한국어 프로젝트 예시:

```yaml
questions:
  categories:
    개요:
      - "이 문서의 주요 목적은 무엇입니까?"
      - "이 정책이 해결하려는 문제는 무엇입니까?"
      - "대상 사용자는 누구입니까?"
      - "주요 변경 사항은 무엇입니까?"
      - "적용 범위와 제한 사항은 무엇입니까?"
    기술:
      - "주요 기술 사양은 무엇입니까?"
      - "사용된 기술이나 도구는 무엇입니까?"
      - "시스템 요구 사항은 무엇입니까?"
      - "기존 방식과 어떻게 다릅니까?"
      - "성능 지표는 무엇입니까?"
    구현:
      - "구현 일정은 어떻게 됩니까?"
      - "필요한 자원은 무엇입니까?"
      - "구현 절차는 무엇입니까?"
      - "예상되는 위험은 무엇입니까?"
      - "기대 효과는 무엇입니까?"
  system_prompt: >
    당신은 제공된 문서를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
    문서 내용에만 근거하여 답변하십시오. 추측하거나 정보를 만들어내지 마십시오.
    간결하고 사실적으로 답변하십시오. 가능한 경우 구체적인 숫자, 날짜, 이름을 포함하십시오.
    문서에 관련 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
  output_format: "alpaca"
```

외부 파일로 질문을 관리하려면 `questions.txt`에 한 줄씩 작성하고 `file: "./questions.txt"`로 참조합니다. `file`이 설정되면 `categories`는 무시됩니다.

---

## 8. validation — QA 검증 설정

> 생성된 QA 쌍의 품질을 검증하고 필터링하는 규칙을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 검증 기능 활성화 여부. `false`로 설정하면 모든 검증을 건너뜁니다 |
| `min_answer_length` | `int` | `20` | 답변 최소 길이 (문자 수). 이보다 짧은 답변은 거부됩니다 |
| `max_answer_length` | `int` | `2000` | 답변 최대 길이 (문자 수). 이보다 긴 답변은 거부됩니다 |
| `remove_empty` | `bool` | `true` | 빈 질문이나 답변을 제거합니다 |
| `deduplicate` | `bool` | `true` | 중복된 질문-답변 쌍을 제거합니다 |
| `reject_patterns` | `list[str]` | (하위 참조) | 답변에서 거부할 정규식 패턴 목록. 패턴이 매칭되면 해당 QA 쌍을 거부합니다 |
| `groundedness` | `GroundednessConfig` | (하위 참조) | 의미적 근거성 검증 설정 |

기본 거부 패턴 6개 (영어 3개 + 한국어 3개):

- `"(?i)i don't know"`
- `"(?i)not (available|provided|mentioned|found)"`
- `"(?i)the document does not contain"`
- `"(?:알|확인할|파악할|판단할) 수 없"`
- `"정보.{0,8}(?:없|부족|찾을 수 없)"`
- `"(?:문서|자료|내용)에서?.{0,12}(?:찾을 수 없|포함되어 있지 않|언급되지 않|다루고 있지 않)"`

`(?i)` 플래그는 대소문자를 구분하지 않습니다.

**제약조건**: `min_answer_length`는 반드시 `max_answer_length`보다 작아야 합니다. 그렇지 않으면 설정 로드 시 오류가 발생합니다.

### groundedness — 의미적 검증

> 답변이 원본 문서 내용에 근거하고 있는지 임베딩 기반으로 검증합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 의미적 검증 활성화 여부. `uv sync --extra validation`이 필요합니다 |
| `model` | `str` | `"all-MiniLM-L6-v2"` | 사용할 sentence-transformers 모델 |
| `threshold` | `float` | `0.3` | 코사인 유사도 임계값 (0.0~1.0). 높을수록 엄격하게 검증합니다 |

아래 예시는 한국어 문서에 권장하는 설정입니다. 기본 거부 패턴 대신 간결한 한국어 패턴을 추가하고, 다국어 임베딩 모델을 사용합니다.

```yaml
validation:
  enabled: true
  min_answer_length: 20
  max_answer_length: 2000
  remove_empty: true
  deduplicate: true
  reject_patterns:
    # 영어 패턴 (기본값)
    - "(?i)i don't know"
    - "(?i)not (available|provided|mentioned|found)"
    - "(?i)the document does not contain"
    # 한국어 패턴 (사용자 추가)
    - "(?i)알 수 없"
    - "(?i)정보가 없"
    - "(?i)언급되지 않"
    - "(?i)포함되어 있지 않"
    - "(?i)제공되지 않"
  groundedness:
    enabled: true
    model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 기본값: all-MiniLM-L6-v2
    threshold: 0.35  # 기본값: 0.3
```

한국어 문서에는 다국어 임베딩 모델 `paraphrase-multilingual-MiniLM-L12-v2`를 권장합니다.

---

## 9. scoring — 품질 점수 평가 설정

> 생성된 QA 쌍의 품질을 Teacher LLM으로 평가하고 낮은 점수의 QA 쌍을 필터링합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 품질 점수 평가 기능 활성화 여부 |
| `threshold` | `float` | `3.0` | QA 쌍 최소 합격 점수 (1.0~5.0). 이 점수 미만의 QA 쌍은 데이터셋에서 제거됩니다 |
| `max_concurrency` | `int` | `3` | 비동기 점수 평가 시 최대 동시 요청 수 |

```yaml
scoring:
   enabled: true
   threshold: 3.5
   max_concurrency: 3
```

Teacher LLM이 질문의 명확성, 답변의 정확성, 문서와의 관련성을 종합하여 1~5점으로 평가합니다. `threshold` 미만의 QA 쌍은 데이터셋에서 제거됩니다.

---

## 10. augment — 데이터 증강 설정

> 기존 QA 쌍의 질문을 패러프레이즈하여 데이터셋을 확장합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 데이터 증강 기능 활성화 여부 |
| `num_variants` | `int` | `2` | 각 QA 쌍당 생성할 패러프레이즈 변형 수. 원본 포함 총 `num_variants + 1`개의 QA 쌍이 생성됩니다 |
| `max_concurrency` | `int` | `3` | 비동기 증강 시 최대 동시 요청 수 |
| `min_similarity` | `float` | `0.3` | 패러프레이즈된 질문과 원본 질문 간 최소 토큰 겹침 비율. 이 값 미만이면 의미가 변질된 것으로 간주하여 제거합니다 (0.0~1.0) |

```yaml
augment:
  enabled: true
  num_variants: 3
  max_concurrency: 3
  min_similarity: 0.3
```

`num_variants: 3`으로 설정하면 100개의 QA 쌍이 400개로 확장됩니다. 데이터가 부족할 때 유용하지만, Teacher LLM 호출 비용이 증가합니다.

---

## 11. analyzer — 데이터 분석 설정

> 생성된 QA 데이터셋의 통계 정보를 분석하고 보고서를 생성합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 데이터 분석 기능 활성화 여부 |
| `output_file` | `str` | `"data_analysis.json"` | 분석 보고서 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

```yaml
analyzer:
  enabled: true
  output_file: "qa_analysis_report.json"
```

분석 보고서에는 전체 QA 쌍 수, 카테고리 분포, 문서별 분포, 질문/답변 길이 통계, 데이터 품질 경고가 포함됩니다. LLM 호출 없이 순수 통계 분석만 수행합니다.

---

## 12. student — Student 모델 설정

> 파인튜닝할 Student 모델을 지정합니다. HuggingFace Hub의 모든 causal language model을 사용할 수 있습니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `model` | `str` | `"google/gemma-3-1b-it"` | HuggingFace 모델 ID. causal LM이어야 합니다 (예: Qwen, Gemma, Llama) (빈 값 불가) |
| `max_seq_length` | `int` | `4096` | 학습 데이터의 최대 토큰 길이. 이보다 긴 시퀀스는 잘립니다 |

> **중요**: 위의 Pydantic 기본값(`google/gemma-3-1b-it`)은 프레임워크 기본입니다. 그러나 `slf init` 프로젝트 템플릿(`templates/project.yaml:97`)은 Ollama GGUF 변환 호환성 이슈로 `Qwen/Qwen2.5-1.5B-Instruct`를 사용합니다. 자세한 배경은 [CLAUDE.md](../CLAUDE.md)의 "Known Issues > Student Model Selection"을 참고하세요.

```yaml
student:
  model: "google/gemma-3-1b-it"  # Pydantic 기본값
  max_seq_length: 4096
```

권장 모델:

| 모델 | HuggingFace ID | VRAM (LoRA) | 특징 |
|------|----------------|-------------|------|
| Gemma 3 1B | `google/gemma-3-1b-it` | ~3GB | Gemma License (상업 허용), Pydantic 기본값이지만 Ollama 변환 이슈 있음 |
| Qwen3.5 9B | `Qwen/Qwen2.5-1.5B-Instruct` | ~2GB | Apache 2.0, **`slf init` 템플릿 기본값** — Ollama GGUF 호환성 우수 |
| Gemma 3 4B | `google/gemma-3-4b-it` | ~10GB | Gemma License (상업 허용), 128K 컨텍스트 |
| Phi-4 Mini | `microsoft/Phi-4-mini-instruct` | ~10GB (QLoRA) | MIT, 추론 강점, 한국어 공식 지원 |
| Qwen3.5 9B | `Qwen/Qwen3.5-4B` | ~10GB | Apache 2.0, 다국어 201언어 |

모델 선택 기준: 8GB VRAM은 `google/gemma-3-1b-it` 또는 `Qwen/Qwen2.5-1.5B-Instruct`, 16GB+는 `google/gemma-3-4b-it` 권장.

---

## 13. training — 학습 설정

> LoRA 파인튜닝의 하이퍼파라미터와 학습 전략을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `batch_size` | `int` | `1` | 디바이스당 배치 크기. GPU 메모리에 따라 조정합니다 |
| `gradient_accumulation_steps` | `int` | `8` | 그래디언트 누적 스텝. 실제 배치 크기는 `batch_size × gradient_accumulation_steps` (기본값: 1×8=8) |
| `learning_rate` | `float \| "auto"` | `"auto"` | 학습률. `"auto"`이면 데이터 양에 따라 자동 결정됩니다 (100개 미만: 5e-5, 100~499개: 1e-4, 500개+: 2e-4). 실수를 지정하면 해당 값을 사용합니다 |
| `lr_scheduler` | `str` | `"cosine"` | 학습률 스케줄러. `"cosine"`, `"linear"`, `"constant"` 등 |
| `warmup_ratio` | `float` | `0.1` | 워밍업 비율. 전체 학습 스텝의 10%를 워밍업에 사용합니다 (0.0~1.0) |
| `num_epochs` | `int \| "auto"` | `"auto"` | 최대 에포크 수. `"auto"`이면 학습 데이터 양에 따라 자동 결정됩니다 (100개 미만: 3, 100~499개: 2, 500개+: 1). 정수를 지정하면 해당 에포크 수로 실행됩니다. 조기 종료가 활성화되면 이보다 일찍 멈출 수 있습니다 |
| `optimizer` | `str` | `"adamw_torch_fused"` | 옵티마이저. `"adamw_torch_fused"`는 PyTorch의 fused AdamW로 가장 빠릅니다 |
| `bf16` | `bool` | `true` | bfloat16 혼합 정밀도 학습. Ampere 이상 GPU (RTX 30xx, A100 등)에서만 지원됩니다 |
| `train_split` | `float` | `0.9` | 학습 데이터 비율. 0.9는 90% 학습, 10% 검증을 의미합니다 (0.0~1.0) |
| `save_strategy` | `str` | `"epoch"` | 체크포인트 저장 전략. `"epoch"`는 에포크마다, `"steps"`는 일정 스텝마다 저장합니다 |
| `lora` | `LoraConfig` | (하위 참조) | LoRA 어댑터 설정 |
| `early_stopping` | `EarlyStoppingConfig` | (하위 참조) | 조기 종료 설정 |
| `quantization` | `QuantizationConfig` | (하위 참조) | 양자화 설정 |
| `weight_decay` | `float` | `0.05` | 가중치 감쇠(L2 정규화). 과적합을 억제합니다. 0이면 비활성 |
| `label_smoothing_factor` | `float` | `0.1` | 라벨 스무딩 계수. 모델이 특정 답변을 과도하게 확신하는 것을 방지합니다. 0이면 비활성 |
| `neftune_noise_alpha` | `float \| None` | `5.0` | NEFTune 임베딩 노이즈 강도. 학습 중 임베딩에 노이즈를 주입하여 일반화 성능을 5~15% 향상시킵니다. 권장값: `5.0` (2B 이하), `10.0` (4B 이상). `None`이면 비활성 |
| `completion_only_loss` | `bool` | `true` | assistant 응답 토큰에만 loss를 계산합니다. prompt 토큰(system/user)의 loss를 마스킹하여 모델이 응답 생성에 집중하도록 합니다. 소규모 데이터에서 과적합을 억제하는 데 효과적입니다 |

**제약조건**: `learning_rate`가 실수인 경우 반드시 0보다 커야 합니다. 0 이하로 설정하면 설정 로드 시 오류가 발생합니다.

> **참고**: 위 표는 Pydantic 기본값입니다. `slf init` 프로젝트 템플릿(`templates/project.yaml`)은 소규모 데이터(~30개 QA)·MPS(Apple Silicon) 환경에 맞춰 다음과 같이 override됩니다:
> - `lora.r: 8, alpha: 8`
> - `batch_size: 1, gradient_accumulation_steps: 4`
> - `learning_rate: 3e-5, num_epochs: 3`
> - `quantization.enabled: false` (MPS 수치 불안정성 회피)
> - `weight_decay: 0.01, label_smoothing_factor: 0.0`
> 
> 대규모 데이터(500+ QA)는 Pydantic 기본값 그대로 사용해도 무방합니다. 상세한 파라미터 튜닝은 [CLAUDE.md](../CLAUDE.md)의 "검증된 학습 파라미터"를 참고하세요.

### lora — LoRA 어댑터 설정

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `r` | `int` | `32` | LoRA rank. 높을수록 표현력이 증가하지만 메모리 사용량도 증가합니다 (일반적으로 8~64) |
| `alpha` | `int` | `32` | LoRA scaling factor. scaling은 `α/r`로 계산됩니다. `r`과 동일한 값(1:1 비율)을 권장합니다. `alpha > r`이면 LoRA 업데이트가 증폭되어 소규모 데이터에서 과적합 위험이 높아집니다 |
| `dropout` | `float` | `0.1` | LoRA 레이어의 드롭아웃 비율. 과적합 방지를 위한 정규화 기법입니다 (0.0~1.0) |
| `target_modules` | `str \| list[str]` | `"auto"` | LoRA를 적용할 모듈 이름. `"auto"`는 자동 감지, 또는 `["q_proj", "v_proj"]` 같은 리스트로 명시할 수 있습니다 |
| `use_rslora` | `bool` | `false` | Rank-Stabilized LoRA(rsLoRA) 사용 여부. 일반 LoRA의 scaling `α/r` 대신 `α/√r`을 사용합니다. `r`이 32 이상일 때만 권장합니다. 낮은 rank에서는 업데이트를 과도하게 증폭시켜 과적합을 유발할 수 있습니다 |

### early_stopping — 조기 종료

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 조기 종료 활성화 여부 |
| `patience` | `int` | `3` | 개선이 없어도 기다릴 에포크 수. 이 횟수 동안 개선이 없으면 학습을 중단합니다 |
| `threshold` | `float` | `0.01` | 개선으로 인정할 최소 변화량. 검증 손실이 이 값 이상 낮아져야 개선으로 간주합니다 |

### quantization — 양자화

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 4비트 양자화 활성화 여부. VRAM 사용량을 약 50~60% 줄입니다 |
| `bits` | `int` | `4` | 양자화 비트 수. 현재는 4비트만 지원합니다 |

```yaml
# 예시: Pydantic 기본값과 다른 설정
training:
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: "auto"
    use_rslora: false
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: "auto"  # 기본값과 동일
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  num_epochs: "auto"
  early_stopping:
    enabled: true
    patience: 3
    threshold: 0.01
  optimizer: "adamw_torch_fused"
  bf16: true
  train_split: 0.9
  save_strategy: "epoch"
  quantization:
    enabled: false
    bits: 4
```

---

## 14. export — 모델 내보내기 설정

> 학습된 모델을 내보내는 방식을 정의합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `merge_lora` | `bool` | `true` | LoRA 어댑터를 베이스 모델에 병합할지 여부. `true`면 독립 실행 가능한 모델을 생성합니다 |
| `output_format` | `str` | `"safetensors"` | 모델 저장 형식. `"safetensors"` (권장) 또는 `"pytorch"` |
| `ollama` | `OllamaExportConfig` | (하위 참조) | Ollama 내보내기 설정 |

### ollama — Ollama 내보내기

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | Ollama 내보내기 활성화 여부. Modelfile을 생성하고 `ollama create` 명령을 실행합니다 |
| `model_name` | `str` | `"my-project-model"` | Ollama에 등록할 모델 이름 (빈 값 불가) |
| `system_prompt` | `str` | `"You are a helpful domain-specific assistant."` | Ollama 모델의 시스템 프롬프트. 모델의 역할과 동작을 정의합니다 |
| `parameters` | `dict[str, Any]` | (하위 참조) | Ollama 런타임 파라미터 |

`parameters` 기본값 (Pydantic): `temperature: 0.7`, `top_p: 0.9`, `num_ctx: 4096`. 단, `slf init`으로 생성되는 템플릿에서는 `num_ctx: 32768`로 설정되어 있습니다.

```yaml
export:
  merge_lora: true
  output_format: "safetensors"
  ollama:
    enabled: true
    model_name: "company-policy-assistant"
    system_prompt: "당신은 회사 정책에 대해 답변하는 전문 어시스턴트입니다. 정확하고 간결하게 답변하십시오."
    parameters:
      temperature: 0.5
      top_p: 0.9
      num_ctx: 4096
```

---

## 15. eval — 모델 평가 설정

> 학습된 모델을 BLEU/ROUGE 메트릭으로 자동 평가합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 모델 평가 기능 활성화 여부 |
| `test_split` | `float` | `0.1` | 평가에 사용할 데이터 비율 (0.0~1.0) |
| `metrics` | `list[str]` | `["bleu", "rouge", "llm_judge"]` | 평가 메트릭 목록 |
| `max_samples` | `int` | `20` | 평가에 사용할 최대 샘플 수 |
| `max_tokens` | `int` | `512` | Ollama 생성 최대 토큰 수. 평가 시 무한 생성을 방지합니다 |
| `output_file` | `str` | `"eval_results.json"` | 평가 결과 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |
| `quality_gate` | `bool` | `true` | 품질 게이트 활성화 여부. `true`이면 평가 결과가 임계값 미달 시 경고합니다 |
| `quality_thresholds` | `dict` | `{"bleu": 0.1, "rougeL": 0.2}` | 메트릭별 최소 통과 임계값. 평균 점수가 이 값 미만이면 품질 게이트 실패입니다 |
| `llm_judge_model` | `str` | `""` | LLM-as-Judge에 사용할 모델입니다. 빈 문자열이면 `teacher.model`을 사용합니다 |

**제약조건**: `max_samples`는 1 이상이어야 합니다. 0 이하로 설정하면 설정 로드 시 오류가 발생합니다.

```yaml
eval:
  enabled: true
  test_split: 0.1
  metrics: ["bleu", "rouge"]
  max_samples: 50
  max_tokens: 512
  output_file: "eval_results.json"
  quality_gate: true
  quality_thresholds:
    bleu: 0.1
    rougeL: 0.2
```

> `project.language`가 `"ko"`인 경우, kiwipiepy 형태소 분석기를 사용하여 형태소 단위로 BLEU/ROUGE를 계산합니다. `uv sync --extra korean`으로 설치하면 자동으로 적용됩니다.

---

## 16. compare — 모델 비교 설정

> Base 모델과 Fine-tuned 모델의 답변을 나란히 비교합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 모델 비교 기능 활성화 여부 |
| `base_model` | `str` | `""` | 비교 기준 모델 이름 (Ollama) |
| `finetuned_model` | `str` | `""` | 파인튜닝된 모델 이름 (Ollama) |
| `metrics` | `list[str]` | `["bleu", "rouge"]` | 비교 메트릭 목록 |
| `max_samples` | `int` | `20` | 비교에 사용할 최대 샘플 수 |
| `max_tokens` | `int` | `512` | Ollama 생성 최대 토큰 수. 비교 시 무한 생성을 방지합니다 |
| `output_file` | `str` | `"compare_results.json"` | 비교 결과 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

**제약조건**:
- `max_samples`는 1 이상이어야 합니다. 0 이하로 설정하면 설정 로드 시 오류가 발생합니다.
- `enabled`가 `true`일 때 `base_model`과 `finetuned_model`은 반드시 비어있지 않은 값으로 설정해야 합니다. 비어있으면 설정 로드 시 `ValidationError`가 발생합니다.

```yaml
compare:
  enabled: true
  base_model: "gemma:2b"
  finetuned_model: "my-project-model"
  metrics: ["bleu", "rouge"]
  max_samples: 20
  max_tokens: 512
  output_file: "compare_results.json"
```

> `project.language`가 `"ko"`인 경우, kiwipiepy 형태소 분석기를 사용하여 형태소 단위로 BLEU/ROUGE를 계산합니다. `uv sync --extra korean`으로 설치하면 자동으로 적용됩니다.

---

## 17. evolve — 자동 진화 설정

> 증분 업데이트 → 전체 재학습 → 품질 게이트 → 조건부 배포를 단일 명령으로 실행합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `quality_gate` | `bool` | `true` | 품질 게이트 활성화 여부. `true`이면 이전 모델보다 나은 경우에만 배포합니다 |
| `gate_metric` | `str` | `"rougeL"` | 품질 비교에 사용할 메트릭. `bleu`, `rouge1`, `rouge2`, `rougeL` 중 선택 |
| `gate_min_improvement` | `float` | `0.0` | 최소 개선율(%). `0`이면 어떤 개선이든 통과합니다 |
| `version_format` | `str` | `"date"` | 버전 형식. 현재 `"date"` (`vYYYYMMDD`) 형식만 지원합니다 |
| `history_file` | `str` | `"evolve_history.json"` | 진화 히스토리 파일명. `paths.output` 디렉토리에 저장됩니다 |
| `keep_previous_versions` | `int` | `3` | 보관할 이전 버전 수. 초과분은 `ollama rm`으로 자동 제거됩니다 |

**제약조건**:
- `gate_metric`은 `bleu`, `rouge1`, `rouge2`, `rougeL` 중 하나여야 합니다. 다른 값을 설정하면 `ValidationError`가 발생합니다.
- `keep_previous_versions`는 0 이상이어야 합니다.

```yaml
evolve:
  quality_gate: true
  gate_metric: "rougeL"
  gate_min_improvement: 0.0
  version_format: "date"
  history_file: "evolve_history.json"
  keep_previous_versions: 3
```
---

## 18. corpus_export — 외부 평가용 코퍼스 데이터 내보내기 설정

> slm-factory의 파싱·QA 데이터를 외부 평가 도구나 자체 검색 평가에서 사용할 수 있는 parquet 형식으로 내보냅니다. 결과는 RAG 인덱싱 파이프라인에서도 그대로 사용됩니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 코퍼스 데이터 내보내기 기능 활성화 여부 |
| `output_dir` | `str` | `"corpus"` | 출력 디렉토리명. `paths.output` 하위에 생성됩니다 |
| `chunk_size` | `int` | `512` | 코퍼스 청크 크기 (문자 수). 100 이상이어야 합니다 |
| `overlap_chars` | `int` | `64` | 청크 간 중첩 문자 수. `chunk_size`보다 작아야 합니다 |

```yaml
corpus_export:
  enabled: true
  output_dir: "corpus"
  chunk_size: 512
  overlap_chars: 64
```

**참고**

- `tool export-corpus` 명령으로 직접 실행할 수 있습니다. CLI 사용법은 [CLI 레퍼런스](cli-reference.md)를 참조하십시오.
- `tune` 명령에서 자동 실행하려면 `enabled: true`로 설정합니다.
- 출력 파일: `output/{output_dir}/corpus.parquet` (문서 청크), `output/{output_dir}/qa.parquet` (QA 평가 데이터)
- 청크 크기는 벡터 검색 성능에 영향을 줍니다. 한국어 문서는 300~600자가 적합합니다.

---

## 19. rag — RAG 서비스 설정

> `corpus.parquet`을 Qdrant에 임베딩하여 적재하고, Ollama SLM과 연동하는 RAG API 서버를 실행합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `embedding_model` | `str` | `"Qwen/Qwen3-Embedding-0.6B"` | 임베딩 모델 이름 (sentence-transformers 호환). 비대칭 인코딩(`prompt_name="query"`)으로 한국어·다국어 문서에 우수한 성능 |
| `vector_db_path` | `str` | `"qdrant_db"` | Qdrant 저장 경로 (embedded mode). `paths.output` 하위에 생성됩니다 |
| `collection_name` | `str` | `"corpus"` | Qdrant 컬렉션 이름 |
| `top_k` | `int` | `5` | 검색 시 반환할 최대 문서 청크 수. 1 이상이어야 합니다 |
| `server_host` | `str` | `"0.0.0.0"` | RAG API 서버 바인드 호스트 |
| `server_port` | `int` | `8000` | RAG API 서버 포트 (1~65535) |
| `ollama_model` | `str` | `""` | Ollama 모델명. 빈 문자열이면 `export.ollama.model_name`을 사용합니다 |
| `workers` | `int` | `1` | uvicorn 워커 수 |
| `log_level` | `str` | `"info"` | 서버 로그 레벨 |
| `cors_origins` | `list[str]` | `["*"]` | CORS 허용 오리진 |
| `batch_size` | `int` | `64` | 인덱싱 배치 크기 |
| `request_timeout` | `float` | `120.0` | Ollama 요청 타임아웃 (초) |
| `max_tokens` | `int` | `-1` | 응답 최대 토큰 수. `-1`이면 모델 기본값을 사용합니다. 양수 지정 시 SLM의 반복 생성을 방지합니다 |
| `reranker_enabled` | `bool` | `true` | 검색 결과를 cross-encoder로 재정렬하여 정확도를 높입니다 |
| `reranker_model` | `str` | `"dragonkue/bge-reranker-v2-m3-ko"` | Reranker 모델명 (sentence-transformers CrossEncoder 호환) |
| `hybrid_search` | `bool` | `true` | 벡터 검색과 키워드 검색(BM25Okapi)을 RRF(Reciprocal Rank Fusion)로 결합하여 검색 재현율을 높입니다. `rank_bm25` 패키지 필요 (`uv sync --extra rag`) |
| `min_score` | `float` | `0.0` | 검색 결과 최소 유사도 점수 (0.0~1.0). 이 점수 미만의 문서는 제외됩니다. `0.0`이면 필터링하지 않습니다 |
| `query_rewriting` | `bool` | `false` | 짧은 질의를 LLM으로 확장하여 검색 품질을 향상합니다 |

```yaml
rag:
  embedding_model: "Qwen/Qwen3-Embedding-0.6B"
  vector_db_path: "qdrant_db"
  collection_name: "corpus"
  top_k: 5
  server_host: "0.0.0.0"
  server_port: 8000
  ollama_model: ""
  workers: 1
  log_level: "info"
  cors_origins: ["*"]
  batch_size: 64
  request_timeout: 120.0
  max_tokens: -1
  reranker_enabled: true
  reranker_model: "dragonkue/bge-reranker-v2-m3-ko"
  hybrid_search: true
  query_rewriting: false
  min_score: 0.0
```

**참고**

- `tool rag-index`로 인덱싱, `tool rag-serve`로 서버를 실행합니다. CLI 사용법은 [CLI 레퍼런스](cli-reference.md)를 참조하십시오.
- `uv sync --extra rag --extra validation`으로 의존성을 설치하세요.
- 임베딩 모델은 sentence-transformers 호환 모델이면 모두 사용 가능합니다. 한국어·다국어 문서에는 `Qwen/Qwen3-Embedding-0.6B`가 권장됩니다.
- `hybrid_search: true` (기본값) 설정 시 BM25Okapi 키워드 검색과 벡터 검색을 RRF(Reciprocal Rank Fusion)로 결합합니다. `rank_bm25` 패키지가 필요합니다 (`uv sync --extra rag`). 한국어 문서에서는 kiwipiepy 형태소 분석기를 사용하여 정확한 토큰화를 수행합니다 (`uv sync --extra korean` 필요). Lost-in-middle 문제를 완화하기 위해 검색된 문서를 관련도 순으로 재정렬합니다.

### rag.agent — Agent RAG 모드

> `AgentOrchestrator`가 구동하는 OMO 패턴 기반 12단계 지능 파이프라인 설정입니다. `/auto` 경로의 라우팅·플래닝·검증·합성·반성·재시도를 제어합니다. 아키텍처 개요는 [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)를 참고하세요.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | Agent 모드 활성화. `false`이면 기존 단일 패스 RAG를 사용합니다 |
| `max_iterations` | `int` | `5` | ReAct 루프 최대 반복 횟수. 초과 시 `force_answer` thought 이벤트 발행 후 강제 답변 생성 |
| `session_ttl` | `int` | `3600` | 세션 유지 시간(초) |
| `max_history_turns` | `int` | `20` | 대화 내역 최대 턴 수. 초과 시 오래된 턴 제거 |
| `stream_reasoning` | `bool` | `true` | 추론(Thought/Action/Observation)을 SSE로 실시간 스트리밍 |
| `persist_sessions` | `bool` | `false` | `FileBackedSessionStore`로 세션을 JSON 파일에 영속화 |
| `sessions_dir` | `str` | `"agent_sessions"` | 영속 세션 JSON 저장 경로 (`paths.output` 하위) |
| `planner_enabled` | `bool` | `false` | Planner 오케스트레이션(plan → execute → verify → synthesize) 활성화. `false`이면 기존 ReAct `AgentLoop` 사용 |
| `verifier_enabled` | `bool` | `true` | Verifier로 컨텍스트 충분성 판정 (`planner_enabled=true`에서만 유효) |
| `verifier_max_repairs` | `int` | `1` | Verifier가 허용하는 추가 검색(repair) 최대 횟수 |
| `legacy_fallback_enabled` | `bool` | `true` | Planner가 구조적으로 실패(`is_fallback=True`)하면 legacy `AgentLoop`로 자동 전환 |
| `session_source_reuse` | `bool` | `true` | 같은 세션 이전 턴의 참조 문서를 다음 synthesis 프롬프트에 주입 (`planner_enabled=true`에서만) |
| `session_source_reuse_limit` | `int` | `5` | 재사용할 이전 턴 문서 최대 개수 |
| `parallel_steps` | `bool` | `false` | Planner plan의 모든 step이 `parallel_safe`이고 2개 이상이면 `asyncio.gather`로 병렬 실행. `ToolSpec.parallel_safe` 메타데이터를 검사 |
| `reflector_enabled` | `bool` | `false` | Reflector 답변 자기 검증. 근거 부족 시 추가 검색 + 재합성 (`planner_enabled=true`에서만) |
| `reflector_max_retries` | `int` | `1` | Reflector 거부 시 추가 검색·재합성 최대 횟수. `ultra_mode=true`이면 1로 cap |
| `intent_classifier_enabled` | `bool` | `false` | LLM 기반 의도 분류(`factual`/`comparative`/`analytical`/`procedural`/`exploratory`/`ambiguous`) 활성화. 실패 시 키워드 fallback |
| `intent_classifier_cache_ttl` | `int` | `300` | 의도 분류 결과 캐싱 시간(초). `0`이면 비활성 |
| `clarifier_enabled` | `bool` | `false` | IntentClassifier가 `ambiguous`로 분류하면 Clarifier persona가 답변 대신 명확화 질문을 반환 (`intent_classifier_enabled=true` 필요) |
| `clarifier_max_questions` | `int` | `2` | Clarifier 질문 최대 개수 (1~3 권장) |
| `personas_enabled` | `bool` | `false` | Persona 시스템 활성화. intent별 전용 synthesis 프롬프트 + 도구 화이트리스트 |
| `custom_personas_dir` | `str` | `""` | 사용자 정의 persona YAML 디렉터리. 빈 문자열이면 비활성. built-in persona보다 우선 적용 |
| `review_work_enabled` | `bool` | `false` | Review-Work 병렬 검증 활성화. grounding/completeness/hallucination 3개 reviewer가 병렬 실행 (`planner_enabled=true`에서만) |
| `review_work_retry` | `bool` | `false` | Review-Work 실패 시 추가 검색 + 재합성 (`review_work_enabled=true`에서만) |
| `self_improvement_enabled` | `bool` | `false` | AnswerScorer 기반 자기 개선 루프. 점수 < `min_quality_score` 시 구체 피드백 주입 후 재합성 |
| `min_quality_score` | `float` | `7.0` | 이 점수(1~10) 미만이면 self-improvement 재시도 |
| `max_self_improvement_iterations` | `int` | `1` | self-improvement 최대 재시도 횟수. `ultra_mode=true`이면 1로 cap |
| `memory_compression_enabled` | `bool` | `false` | 긴 대화 자동 요약. `compress_after_turns` 초과 시 오래된 턴을 LLM 요약으로 압축 |
| `compress_after_turns` | `int` | `10` | 이 턴 수(user+assistant 쌍)를 초과하면 오래된 턴 압축 |
| `compress_target_chars` | `int` | `500` | 요약 결과의 목표 길이(문자) |
| `hooks_enabled` | `bool` | `false` | Hooks 시스템 활성화. `pre_query`/`post_search`/`post_synthesis` 지점에서 내장 또는 사용자 hook 실행 |
| `builtin_hooks` | `list[str]` | `[]` | 자동 등록할 내장 hook. 예: `["normalize_korean_whitespace", "dedup_sources_by_doc_id", "strip_html_from_answer"]` |
| `skills_enabled` | `bool` | `false` | Skills 시스템 활성화. `skills_dir`의 YAML에 정의된 trigger에 매칭되는 skill의 `prompt_addon`을 synthesis 프롬프트에 주입 |
| `skills_dir` | `str` | `"skills"` | Skill YAML 디렉터리. 상대 경로는 프로젝트 루트 기준 |
| `models` | `AgentModelsConfig` | (하위 참조) | 컴포넌트별 Ollama 모델 슬롯. 빈 슬롯은 `rag.ollama_model`로 fallback |
| `observation_preview_limit` | `int` | `300` | observation SSE 이벤트로 클라이언트에 보낼 때의 길이 제한(문자) |
| `ollama_keep_alive` | `str` | `"5m"` | 모든 LLM 호출에 전달하는 Ollama `keep_alive` 값. `"5m"`=5분 유지, `"-1"`=무한, `"0"`=즉시 언로드 |
| `smart_mode` | `bool` | `false` | **원클릭 프리셋 (P0)** — `true`이면 IntentClassifier + Clarifier + Personas + Review-Work + Planner + Verifier + Reflector + Legacy fallback 자동 활성화 |
| `ultra_mode` | `bool` | `false` | **최대 성능 프리셋** — `smart_mode` + Hooks + Memory Compression + Self-Improvement + Review-Work retry + Session source reuse 자동 활성화. 또한 validator가 `reflector_max_retries` / `max_self_improvement_iterations`를 1로 cap (더 큰 값을 명시하면 경고 로그 출력 후 유지) |

#### rag.agent.models — 컴포넌트별 Ollama 모델 슬롯

빈 슬롯(`""`)은 모두 `rag.ollama_model`로 자동 fallback됩니다. 워크로드별로 속도/품질 트레이드오프를 분리하려면 개별 슬롯을 지정하세요.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `router_model` | `str` | `""` | IntentClassifier용 — 빠른 소형 모델 권장 |
| `planner_model` | `str` | `""` | Planner JSON 생성용 — 중급 모델 권장 |
| `synthesis_model` | `str` | `""` | 최종 답변 합성용 — 가장 큰 모델 권장 |
| `verifier_model` | `str` | `""` | Verifier 충분성 판정용 — 작은 모델 권장 |
| `reviewer_model` | `str` | `""` | Review-Work 3개 reviewer용 — 중급 모델 권장 |
| `reflector_model` | `str` | `""` | Reflector 답변 검증용 — 중급 모델 권장 |
| `clarifier_model` | `str` | `""` | Clarifier 명확화 질문용 — 중급 모델 권장 |
| `scorer_model` | `str` | `""` | AnswerScorer 정량 평가용 — 중급 모델 권장 |

**프리셋 설정 예시**

```yaml
# 원클릭 P0 추천 (답변 품질 + 의도 정확성 + 근거 검증)
rag:
  agent:
    enabled: true
    smart_mode: true
```

```yaml
# 최대 품질 (모든 P0 + P1/P2 자동 활성화)
rag:
  agent:
    enabled: true
    ultra_mode: true
    # 선택적 확장
    skills_enabled: true
    skills_dir: "/path/to/skills"
    custom_personas_dir: "/path/to/personas"
    parallel_steps: true
    persist_sessions: true
    models:
      router_model: "qwen2.5:0.5b"
      planner_model: "qwen2.5:7b"
      synthesis_model: "qwen2.5:14b"
      verifier_model: "qwen3.5:9b"
```

**제약조건**: `max_iterations >= 1`, `session_ttl >= 0`, `max_history_turns >= 1`, `verifier_max_repairs >= 0`. 위반 시 설정 로드 단계에서 `ValidationError`가 발생합니다.

**참고**
- 모든 LLM 호출 컴포넌트(Planner / Verifier / Reflector / Scorer / Reviewer / Clarifier / IntentClassifier / Compressor)는 **never-raise 계약**을 따릅니다 — LLM 장애 시 안전한 기본값으로 fallback하여 agent 전체를 중단시키지 않습니다.
- `AnswerScorer`는 `ScoreResult.ok=False`를 반환할 수 있으며(LLM 호출/파싱 실패), orchestrator는 이 플래그를 보고 self-improvement 루프를 즉시 종료해 무한·무의미 재시도를 방지합니다.
- `ToolSpec.parallel_safe` 메타데이터: 내장 도구 중 `search`/`lookup`/`compare`는 `True`, `evaluate`/`list_documents`는 `False`입니다.

---

## 20. refinement — Iterative Refinement 설정

> 학습 후 평가에서 약점이 발견된 QA에 대해 Teacher LLM으로 추가 학습 데이터를 생성하고 재학습합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | Refinement 활성화 여부. 기본 비활성 |
| `max_rounds` | `int` | `1` | 최대 Refinement 반복 횟수. 1 이상이어야 합니다 |
| `llm_judge_threshold` | `float` | `0.6` | 이 점수 미만인 QA를 약점으로 판단합니다 (0.0~1.0) |

```yaml
refinement:
  enabled: true
  max_rounds: 2
  llm_judge_threshold: 0.6
```

Refinement가 활성화되면 `pipeline.run()`에서 평가 후 자동으로 실행됩니다. LLM-as-Judge 점수가 `llm_judge_threshold` 미만인 QA를 약점으로 판단하고, 해당 질문에 대한 타겟 QA를 Teacher LLM으로 재생성하여 재학습합니다.

---

## 21. incremental — 증분 학습 설정

> 문서 추가 시 기존 QA를 유지하면서 새 문서만 처리합니다. 해시 기반으로 변경을 감지합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | 증분 학습 기능 활성화 여부 |
| `hash_file` | `str` | `"document_hashes.json"` | 문서 해시를 저장할 파일명. `paths.output` 디렉토리에 저장됩니다 |
| `merge_strategy` | `"append"` \| `"replace"` | `"append"` | 기존 QA와 새 QA의 병합 전략. `"append"`는 추가, `"replace"`는 교체 |
| `resume_adapter` | `str` | `""` | 이전 학습의 어댑터 경로. 빈 문자열이면 새로 학습합니다 |

```yaml
incremental:
  enabled: true
  hash_file: "document_hashes.json"
  merge_strategy: "append"
  resume_adapter: "./output/checkpoints/adapter"
```

---

## 22. review — QA 리뷰 설정

> TUI에서 QA 쌍을 수동으로 검토하고 승인/거부/편집합니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `true` | QA 수동 리뷰 기능 활성화 여부 |
| `auto_open` | `bool` | `true` | 리뷰 완료 후 결과 파일을 자동으로 열지 여부 |
| `output_file` | `str` | `"qa_reviewed.json"` | 리뷰 결과 JSON 파일명. `paths.output` 디렉토리에 저장됩니다 |

```yaml
review:
  enabled: true
  auto_open: true
  output_file: "qa_reviewed.json"
```

---

## 23. 설정 레시피

자주 사용되는 설정 패턴을 레시피 형태로 제공합니다. 각 레시피는 특정 상황에 맞게 최소한의 필드만 변경합니다.

각 설정을 언제 사용할지에 대한 맥락은 [사용 가이드](guide.md)를 참조하십시오.

### 빠른 시작 — 기본값으로 바로 실행

**상황**: 처음 slm-factory를 사용하며 기본 설정으로 빠르게 테스트하고 싶을 때.

`project.yaml`을 수정할 필요가 없습니다. `slf init my-project`로 생성된 기본 템플릿을 그대로 사용하면 됩니다.

사전 준비:

```bash
# ./setup.sh가 Ollama 설치·모델 다운로드를 자동 처리합니다
# documents/ 디렉토리에 PDF 또는 TXT 파일 추가
slf tune --config my-project/project.yaml
```

기본 설정은 영어 문서, Ollama 백엔드, Gemma 3 1B 모델을 사용합니다.

---

### 소량 문서 (5개 이하) — 과적합 방지를 위한 증강

**상황**: 문서가 5개 이하로 적어서 데이터가 부족하고 과적합이 우려될 때.

```yaml
# 데이터 증강으로 데이터셋 확장 (원본의 4배)
augment:
  enabled: true
  num_variants: 3
  max_concurrency: 2

# 저품질 QA 쌍 필터링
scoring:
  enabled: true
  threshold: 3.5
  max_concurrency: 2

# 과적합 방지를 위한 학습 설정
training:
  num_epochs: 10
  lora:
    dropout: 0.1
  early_stopping:
    enabled: true
    patience: 3
```

데이터 증강은 Teacher LLM 호출 비용을 증가시키므로 로컬 Ollama를 사용하는 것이 좋습니다. 소량 데이터에서는 품질 점수 평가도 함께 활성화하는 것을 권장합니다.

---

### VRAM 제한 (8GB GPU) — 메모리 최적화

**상황**: RTX 3060, RTX 4060 등 8GB VRAM GPU에서 학습할 때.

```yaml
# 경량 Student 모델 (Pydantic 기본값. slf init 템플릿은 Ollama 호환성 우수한
# Qwen/Qwen2.5-1.5B-Instruct 사용 — guide.md "Student 모델 선택" 트러블슈팅 참조)
student:
  model: "google/gemma-3-1b-it"
  max_seq_length: 2048

# 메모리 최적화 학습 설정
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # 실제 배치 크기는 2x8=16으로 유지
  quantization:
    enabled: true
    bits: 4
  bf16: true
```

양자화를 사용하면 VRAM 사용량이 약 50% 감소하지만 품질 저하는 거의 없습니다. `max_seq_length`를 2048로 줄이면 긴 문서는 잘리지만 대부분의 QA 쌍은 이 길이 내에 들어갑니다.

---

### vLLM/OpenAI 백엔드 — 외부 API 연동

**상황**: Ollama 대신 vLLM 서버 또는 OpenAI API를 Teacher 모델로 사용할 때.

vLLM 서버:

```yaml
teacher:
  backend: "openai"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  api_base: "http://localhost:8000/v1"
  api_key: "dummy"  # vLLM은 키가 필요 없지만 필드는 채워야 합니다
  temperature: 0.3
  max_concurrency: 16  # vLLM은 높은 동시성을 지원합니다
```

OpenAI API:

```yaml
teacher:
  backend: "openai"
  model: "gpt-4o-mini"
  api_base: "https://api.openai.com"
  api_key: "sk-proj-..."
  temperature: 0.3
  max_concurrency: 8  # OpenAI는 rate limit을 고려하여 조절합니다
```

`api_key`는 환경 변수 `OPENAI_API_KEY`로 설정하는 것이 안전합니다.

---

## 24. 전체 설정 예시 (한국어 정책 문서 프로젝트)

한국어 회사 정책 문서를 처리하는 완전한 `project.yaml` 예시입니다.

```yaml
# 한국어 회사 정책 문서 처리 프로젝트
project:
  name: "company-policy-assistant"
  version: "1.0.0"
  language: "ko"

paths:
  documents: "./documents"
  output: "./output"

# PDF와 HWPX 문서 모두 파싱
parsing:
  formats: ["pdf", "hwpx"]
  pdf:
    extract_tables: true
  hwpx:
    apply_spacing: true  # 한국어 띄어쓰기 교정 (uv sync --extra korean 필요)

# 로컬 Ollama 서버 사용 (기본 Teacher 모델 qwen3.5:9b)
teacher:
  backend: "ollama"
  model: "qwen3.5:9b"
  api_base: "http://localhost:11434"
  temperature: 0.3
  timeout: 300
  max_context_chars: 12000
  max_concurrency: 2

# 한국어 질문 카테고리
questions:
  categories:
    개요:
      - "이 문서의 주요 목적은 무엇입니까?"
      - "이 정책이 해결하려는 문제는 무엇입니까?"
      - "대상 사용자는 누구입니까?"
      - "주요 변경 사항은 무엇입니까?"
      - "적용 범위와 제한 사항은 무엇입니까?"
    기술:
      - "주요 기술 사양은 무엇입니까?"
      - "사용된 기술이나 도구는 무엇입니까?"
      - "시스템 요구 사항은 무엇입니까?"
      - "기존 방식과 어떻게 다릅니까?"
      - "성능 지표는 무엇입니까?"
    구현:
      - "구현 일정은 어떻게 됩니까?"
      - "필요한 자원은 무엇입니까?"
      - "구현 절차는 무엇입니까?"
      - "예상되는 위험은 무엇입니까?"
      - "기대 효과는 무엇입니까?"
  system_prompt: >
    당신은 제공된 문서를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
    문서 내용에만 근거하여 답변하십시오. 추측하거나 정보를 만들어내지 마십시오.
    간결하고 사실적으로 답변하십시오. 가능한 경우 구체적인 숫자, 날짜, 이름을 포함하십시오.
    문서에 관련 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
  output_format: "alpaca"

# 한국어 거부 패턴 추가
validation:
  enabled: true
  min_answer_length: 20
  max_answer_length: 2000
  remove_empty: true
  deduplicate: true
  reject_patterns:
    # 영어 패턴
    - "(?i)i don't know"
    - "(?i)not (available|provided|mentioned|found)"
    - "(?i)the document does not contain"
    # 한국어 패턴
    - "(?i)알 수 없"
    - "(?i)정보가 없"
    - "(?i)언급되지 않"
    - "(?i)포함되어 있지 않"
    - "(?i)제공되지 않"
  groundedness:
    enabled: true
    model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    threshold: 0.35

# 품질 점수 평가
scoring:
  enabled: true
  threshold: 3.0
  max_concurrency: 2

# 데이터 증강
augment:
  enabled: true
  num_variants: 2
  max_concurrency: 2

# 데이터 분석
analyzer:
  enabled: true
  output_file: "data_analysis.json"

# 한국어 지원 우수한 Student 모델 (또는 slf init 템플릿 기본값
# Qwen/Qwen2.5-1.5B-Instruct — Ollama GGUF 호환성 우수)
student:
  model: "google/gemma-3-1b-it"
  max_seq_length: 4096

# 12GB VRAM 기준 설정
training:
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: "auto"
    use_rslora: false
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: "auto"
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  num_epochs: "auto"
  early_stopping:
    enabled: true
    patience: 3
    threshold: 0.01
  optimizer: "adamw_torch_fused"
  bf16: true
  train_split: 0.9
  save_strategy: "epoch"
  quantization:
    enabled: false  # 12GB VRAM이면 양자화 불필요

# Ollama로 내보내기
export:
  merge_lora: true
  output_format: "safetensors"
  ollama:
    enabled: true
    model_name: "company-policy-assistant"
    system_prompt: "당신은 회사 정책에 대해 답변하는 전문 어시스턴트입니다. 정확하고 간결하게 답변하십시오."
    parameters:
      temperature: 0.5  # 정책 문서는 일관성 있는 답변이 중요
      top_p: 0.9
      num_ctx: 4096

# compare, review는 수동 조작 또는 추가 설정이 필요하므로 명시적으로 비활성화
compare:
  enabled: false  # base_model 설정 필요

review:
  enabled: false  # TUI 대화형 실행 전용
```

주요 설정 포인트:

1. `parsing.formats: ["pdf", "hwpx"]` — PDF와 HWPX 모두 파싱합니다
2. `hwpx.apply_spacing: true` — 한국어 띄어쓰기를 자동 교정합니다
3. `teacher.model: "qwen3.5:9b"` — 한국어 지원이 우수한 모델을 선택했습니다
4. 질문과 시스템 프롬프트를 모두 한국어로 작성하여 한국어 답변을 유도합니다
5. 한국어 거부 패턴 5개를 추가했습니다
6. `groundedness.model`을 다국어 지원 모델로 변경했습니다
7. `student.model: "google/gemma-3-1b-it"` — 소량 데이터에서 파인튜닝이 용이한 경량 Instruct 모델입니다 (Ollama 배포 시 호환성 이슈가 있으면 `slf init` 템플릿 기본값인 `Qwen/Qwen2.5-1.5B-Instruct` 사용 — `guide.md` "Student 모델 선택" 트러블슈팅 참조)
8. `export.ollama.parameters.temperature: 0.5` — 정책 문서는 일관성이 중요하므로 낮췄습니다
9. `compare`, `review`는 수동 조작이 필요하므로 명시적으로 비활성화합니다. 나머지 선택적 단계는 기본값(`true`)을 그대로 사용합니다

---

## 25. 설정 검증 (slf check)

설정 파일이 올바른지 확인하려면 `check` 명령을 사용합니다:

```bash
slf check --config project.yaml
```

`--config`를 생략하면 현재 디렉토리와 상위 디렉토리(2단계)에서 `project.yaml`을 자동으로 검색합니다.

### 검증 항목

| 항목 | 설명 |
|------|------|
| 설정 파일 | YAML 로드 및 Pydantic 검증 |
| 문서 디렉토리 | 존재 여부 및 파일 유무 |
| 출력 디렉토리 | 쓰기 가능 여부 |
| Ollama 연결 | 서버 연결 상태 (`backend: "ollama"`일 때만) |
| 모델 사용 가능 | Teacher 모델이 Ollama에 다운로드되어 있는지 확인 |
| 학생 모델 | HuggingFace Hub 접근 가능 여부 (게이트 모델 시 로그인 필요 안내) |
| 컴퓨팅 디바이스 | GPU(CUDA/MPS) 또는 CPU 감지. GPU 미사용 시 경고 |
| 학습 정밀도 | bfloat16/float16/float32 지원 여부 확인 |
| 4bit 양자화 | CUDA 환경에서 `bitsandbytes` 설치 여부 (MPS는 N/A) |

### 예시 출력

```
                slm-factory 환경 점검
┌────────────────┬────────┬────────────────────────────────┐
│ 항목           │ 상태   │ 상세                           │
├────────────────┼────────┼────────────────────────────────┤
│ 설정 파일      │ OK     │ ./project.yaml                 │
│ 문서 디렉토리  │ OK     │ 5개 파일 (./documents)         │
│ 출력 디렉토리  │ OK     │ 쓰기 가능 (./output)           │
│ Ollama 연결    │ OK     │ v0.3.12 (http://localhost:11434)│
│ 모델 사용 가능 │ OK     │ qwen3.5:9b                     │
│ 학생 모델      │ OK     │ google/gemma-3-1b-it           │
│ 컴퓨팅 디바이스│ Apple Silicon GPU (MPS) │ Apple M3 Pro      │
│ 학습 정밀도    │ OK     │ float16 (fp16)                 │
│ 4bit 양자화    │ N/A    │ Apple Silicon — Unified Memory │
└────────────────┴────────┴────────────────────────────────┘

모든 점검 통과!
```

모든 항목이 통과하면 종료 코드 0, 실패하면 1을 반환합니다. 설정 파일에 오류가 있으면 (YAML 구문 오류, 잘못된 타입, 알 수 없는 필드 등) Pydantic 검증 단계에서 상세한 에러 메시지를 출력합니다.

설정 문제를 진단할 때는 `slf -v check` 명령으로 상세 로그를 확인할 수 있습니다.

---

## 관련 문서

- [사용 가이드](guide.md) — 각 설정을 언제, 어떻게 사용할지에 대한 실전 안내
- [CLI 레퍼런스](cli-reference.md) — 모든 CLI 명령어와 옵션
- [아키텍처 가이드](architecture.md) — 설정 로딩 프로세스, 내부 구조, 설계 패턴
- [빠른 참조](quick-reference.md) — 자주 쓰는 설정 한눈에 보기
