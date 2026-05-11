# 아키텍처 가이드

> slm-factory의 내부 설계와 동작 원리를 이해하세요.

---

## 1. 설계 철학

SLM Factory는 다음 5가지 핵심 원칙을 기반으로 설계되었습니다.

### 1.1 YAML 중심 설정

모든 파이프라인 동작은 `project.yaml` 파일 하나로 제어됩니다. 코드 수정 없이 문서 파싱 옵션, Teacher 모델, 질문 카테고리, 검증 규칙, Student 모델, 학습 하이퍼파라미터, 내보내기 형식을 변경할 수 있습니다.

### 1.2 단계별 독립 실행

전체 파이프라인은 독립적인 단계(parse, generate, validate, score, augment, analyze, convert, train, export, eval, refine, corpus_export, rag_index)로 구성됩니다. CLI에서 `--until`로 지정 가능한 12단계와 자동 실행되는 refine 단계를 포함하여 총 13단계입니다. 각 단계는 이전 단계의 출력 파일을 입력으로 사용합니다. 특정 단계만 재실행하거나 중간 결과를 검토할 수 있습니다. chunking은 generate 단계에 통합되어 `chunking.enabled` 설정에 따라 자동으로 수행됩니다.

### 1.3 지연 임포트 (Lazy Import)

`torch`, `transformers`, `peft`, `sentence-transformers` 등 무거운 라이브러리는 실제 사용 시점에만 로드됩니다. CLI 초기 응답 속도를 빠르게 유지하고, 특정 단계만 실행할 때 불필요한 의존성을 로드하지 않습니다. `parse` 명령어는 딥러닝 라이브러리를 전혀 로드하지 않습니다.

### 1.4 Pydantic v2 타입 안전

모든 설정은 Pydantic v2 모델로 정의되어 YAML 로드 시 자동으로 타입 검증이 수행됩니다. 기본값, 필수 필드, 값 범위 제약이 명시적으로 정의되어 있어 오타나 잘못된 값이 입력되면 즉시 에러 메시지를 출력합니다.

### 1.5 실패 격리

개별 파일 파싱 실패, 특정 문서의 QA 생성 실패, 단일 QA 쌍의 검증 실패 등은 전체 파이프라인을 중단시키지 않습니다. 실패한 항목은 로그에 기록되고 건너뛰며, 나머지 항목은 정상적으로 처리됩니다.

---

## 2. 전체 아키텍처

### 2.1 컴포넌트 구조 다이어그램

<!-- diagram: architecture-cd -->

```
                     ┌─────────────────────────────┐
                       CLI (cli.py)
                       Entry Point (Typer)
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                         Pipeline (pipeline.py)
                          Orchestrator (13 steps, --until 12)
                     └──────────────┬──────────────┘
                                    ▼

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  1. Parse             2. Generate          3. Validate
  parsers/             teacher/             validator/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  parsed_documents.json qa_alpaca.json       (filtered pairs)

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  4. Score             5. Augment           6. Analyze
  scorer.py            augmenter.py         analyzer.py
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  qa_scored.json       qa_augmented.json   data_analysis.json

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  7. Convert           8. Train             9. Export
  converter.py         trainer/             exporter/
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         ▼                    ▼                    ▼
  training_data.jsonl  adapter/            merged_model/ Modelfile

┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ ┌──────────────────┐
  10. Eval         11. Refine       12. Corpus Export    13. RAG Index
  evaluator.py     pipeline.py      exporter/            rag/
└──────┬───────┘ └──────┬───────┘ └────────┬─────────┘ └────────┬─────────┘
       ▼                ▼                    ▼                    ▼
  eval_results.json  (재학습)         corpus/*.parquet     qdrant_db/

                     ┌─────────────────────────────┐
                       Config (config.py)
                       Injected into all components
                     └─────────────────────────────┘
```

### 2.2 모듈 의존성 다이어그램

```
cli.py
  └─→ pipeline.py
        ├─→ parsers/
        │     ├─→ base.py (BaseParser, ParserRegistry)
        │     ├─→ pdf.py (PDFParser)
        │     ├─→ hwpx.py (HWPXParser)
        │     ├─→ html.py (HTMLParser)
        │     ├─→ text.py (TextParser)
        │     ├─→ docx.py (DOCXParser, 선택적)
        │     ├─→ hwp.py (HWPParser, 선택적)
        │     ├─→ doc.py (DOCParser, 선택적)
        │     ├─→ pptx.py (PPTXParser, 선택적)
        │     ├─→ ppt.py (PPTParser, 선택적)
        │     ├─→ xlsx.py (XLSXParser, 선택적)
        │     └─→ xls.py (XLSParser, 선택적)
        │
        ├─→ teacher/
        │     ├─→ base.py (BaseTeacher)
        │     ├─→ ollama.py (OllamaTeacher)
        │     ├─→ openai_compat.py (OpenAICompatTeacher)
        │     └─→ qa_generator.py (QAGenerator)
        │
        ├─→ validator/
        │     ├─→ rules.py (RuleValidator)
        │     └─→ similarity.py (GroundednessChecker)
        │
        ├─→ scorer.py (QualityScorer)
        ├─→ augmenter.py (DataAugmenter)
        ├─→ analyzer.py (DataAnalyzer)
        ├─→ converter.py (ChatFormatter)
        │
        ├─→ trainer/
        │     └─→ lora_trainer.py (DataLoader, LoRATrainer)
        │
        ├─→ exporter/
        │     ├─→ hf_export.py (HFExporter)
        │     ├─→ ollama_export.py (OllamaExporter)
        │     └─→ corpus_export.py (CorpusExporter)
        │
        ├─→ evaluator.py (ModelEvaluator)
        ├─→ comparator.py (ModelComparator)
        ├─→ incremental.py (IncrementalTracker)
        ├─→ calibration.py (auto_chunk_size, auto_num_epochs, auto_learning_rate, auto_questions_per_chunk, section_aware_chunk)
        ├─→ device.py (detect_device)
        ├─→ evolve_history.py (EvolveHistory)
        ├─→ rag/ (RAGIndexer, RAG 서버)
        └─→ tui/ (QAReviewerApp, QACard, StatusBar)

모든 모듈
  ├─→ config.py (SLMConfig + 28개 하위 모델)
  ├─→ models.py (ParsedDocument, QAPair, EvalResult, CompareResult)
  └─→ utils.py (setup_logging, get_logger)
```

---

## 3. 핵심 설계 패턴

### 3.1 Registry 패턴 (parsers/)

파서 시스템은 Registry 패턴으로 확장 가능한 파일 형식 지원을 제공합니다. `ParserRegistry`가 `BaseParser` 인스턴스 목록을 관리하며, `get_parser(path)`는 등록된 파서를 순회하여 `can_parse(path)`를 만족하는 첫 번째 파서를 반환합니다.

```python
@registry.register
class CSVParser(BaseParser):
    extensions = [".csv"]

    def parse(self, path: Path) -> ParsedDocument:
        ...
```

`parsers/__init__.py`에서 전역 `registry` 인스턴스를 생성하고 4개 기본 파서(PDF, HWPX, HTML, Text)를 무조건 등록하며, 7개 선택적 파서(DOCX, HWP, DOC, PPTX, PPT, XLSX, XLS)는 해당 패키지가 설치된 경우에만 등록합니다. 새 파서를 `@registry.register`로 등록하면 즉시 파이프라인에서 처리됩니다.

`parsers/base.py`의 `detect_encoding()` 함수는 HTML 파서와 텍스트 파서가 공유하는 인코딩 감지 유틸리티입니다. charset-normalizer를 사용하여 EUC-KR, CP949 등 한국어 인코딩을 정확하게 감지합니다. `HTMLParser`는 BeautifulSoup4의 lxml 백엔드를 사용하며, h1–h6 헤딩 태그를 마크다운 헤딩으로 변환하여 문서 구조를 보존합니다.

→ 커스텀 파서 구현 전체 예제는 [개발자 가이드](development.md) 참조

### 3.2 Factory 패턴 (teacher/)

Teacher 백엔드 생성은 Factory 패턴으로 설정 기반의 적절한 구현체를 반환합니다.

```python
def create_teacher(config: TeacherConfig) -> BaseTeacher:
    if config.backend == "ollama":
        from .ollama import OllamaTeacher
        return OllamaTeacher(config)
    elif config.backend == "openai":
        from .openai_compat import OpenAICompatTeacher
        return OpenAICompatTeacher(config)
    raise ValueError(f"Unknown backend: {config.backend}")
```

`BaseTeacher`는 `generate(prompt, **kwargs) → str`와 `health_check() → bool`을 추상 메서드로 정의합니다.

→ 새 백엔드 추가 방법은 [개발자 가이드](development.md) 참조

### 3.3 Strategy 패턴 (validator/)

검증 시스템은 Strategy 패턴으로 여러 검증 전략을 조합합니다.

- `RuleValidator`: empty → length → pattern → dedup 순서로 4개 규칙을 체인 적용합니다.
- `GroundednessChecker`: sentence-transformers 임베딩 기반 코사인 유사도 검증을 수행합니다.
- 두 검증기 모두 `(accepted, rejected)` 튜플을 반환하며, `step_validate()`에서 순차 적용됩니다.

### 3.4 Adapter 패턴 (converter.py)

`ChatFormatter`는 HuggingFace `tokenizer.apply_chat_template()`을 래핑하여 통일된 `QAPair` 형식을 모델별 채팅 형식으로 변환합니다. Gemma처럼 시스템 역할을 지원하지 않는 모델은 시스템 메시지를 제거하고 자동으로 재시도합니다. `max_seq_length` 초과 항목은 자동으로 제외됩니다.

### 3.5 비동기 동시성 패턴 (scorer, augmenter)

Scorer와 Augmenter는 `asyncio.Semaphore`로 동시 요청 수를 제한하면서 병렬 처리합니다.

```python
semaphore = asyncio.Semaphore(config.max_concurrency)

async def _bounded_task(item):
    async with semaphore:
        return await process(item)

results = await asyncio.gather(*tasks, return_exceptions=True)
```

`return_exceptions=True`로 개별 실패를 허용하며, Rich Progress 바로 실시간 진행 상황을 표시합니다.

### 3.6 통계 분석 패턴 (analyzer)

`DataAnalyzer`는 LLM 의존성 없이 순수 통계 분석을 수행합니다. `Counter`로 분포를 계산하고, `statistics` 모듈로 기초 통계를 산출합니다. 데이터 불균형, 이상치 등을 자동으로 감지하여 경고를 생성하고 Rich 콘솔로 시각적 보고서를 출력합니다.

→ 각 패턴의 실제 API는 [개발자 가이드](development.md) 참조

---

## 4. 데이터 흐름

### 4.1 파이프라인 단계별 입출력

| 단계 | 입력 타입 | 출력 타입 | 저장 파일 |
|------|----------|----------|----------|
| parse | 파일 시스템 (PDF/HWPX/HTML/TXT/MD/DOCX/HWP/DOC/PPTX/PPT/XLSX/XLS) | `list[ParsedDocument]` | `parsed_documents.json` |
| generate | `list[ParsedDocument]` | `list[QAPair]` | `qa_alpaca.json` |
| validate | `list[QAPair]` | `list[QAPair]` (필터링) | 없음 (메모리 전달) |
| score | `list[QAPair]` | `list[QAPair]` (필터링) | `qa_scored.json` |
| augment | `list[QAPair]` | `list[QAPair]` (원본+증강) | `qa_augmented.json` |
| analyze | `list[QAPair]` | 보고서 (JSON) | `data_analysis.json` |
| convert | `list[QAPair]` | JSONL 파일 경로 | `training_data.jsonl` |
| train | JSONL 파일 경로 | 어댑터 디렉토리 경로 | `checkpoints/adapter/` |
| export | 어댑터 디렉토리 경로 | 병합 모델 디렉토리 경로 | `merged_model/` + `Modelfile` |
| eval | `list[QAPair]` + 모델 이름 | `list[EvalResult]` | `eval_results.json` |
| compare | `list[QAPair]` | `list[CompareResult]` | `compare_results.json` (※ `pipeline.run()`에서 자동 실행되지 않음. `slf eval compare`로 독립 실행) |
| corpus_export | `list[ParsedDocument]` + `list[QAPair]` | parquet files | `corpus/corpus.parquet`, `corpus/qa.parquet` |
| rag_index | `corpus.parquet` | Qdrant collection | `qdrant_db/` |

### 4.2 중간 파일 체인 다이어그램

```
documents/
  (PDF, HWPX, HWP, HTML, TXT, MD, DOCX)
        │
        ▼ step_parse()
  parsed_documents.json
        │
        ├──▶ chunking (chunking.enabled=true)
        │      긴 문서를 청크로 분할 → 각 청크마다 QA 생성
        │
        ▼ step_generate()
qa_alpaca.json
        │
        ▼ step_validate()
    [메모리]
        │
        ├─▶ step_score()   ──▶  qa_scored.json
        │
        ├─▶ step_augment() ──▶  qa_augmented.json
        │
        ├─▶ step_analyze() ──▶  data_analysis.json
        │
        ▼ step_convert()
training_data.jsonl
        │
        ▼ step_train()
checkpoints/adapter/
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── tokenizer_config.json
        │
        ▼ step_export()
merged_model/
  ├── model.safetensors
  ├── config.json
  ├── tokenizer.json
  └── Modelfile
        │
        ▼ step_eval()           (eval.enabled)
  eval_results.json
        │
        ▼ step_refine()         (refinement.enabled)
  (약점 QA 재생성 + 재학습)
        │
        ▼ step_corpus_export() (corpus_export.enabled)
  corpus/corpus.parquet + qa.parquet
        │
        ▼ step_rag_index()
  qdrant_db/
```

### 4.3 재개 메커니즘 (--resume)

`--resume` 옵션을 사용하면 CLI가 중간 파일의 존재 여부를 확인하여 완료된 단계를 건너뜁니다. 체크포인트 감지 우선순위는 다음과 같습니다.

```
qa_augmented.json 존재   →  analyze 단계부터 재개
qa_scored.json 존재      →  augment 단계부터 재개
qa_alpaca.json 존재      →  validate 단계부터 재개
parsed_documents.json    →  generate 단계부터 재개
없음                     →  처음부터 실행
```

`slf status` 명령으로 각 중간 파일의 존재 여부와 항목 수를 확인할 수 있습니다.

→ 각 단계 사용법은 [사용 가이드](guide.md) 참조  
→ 각 단계 API는 [개발자 가이드](development.md) 참조

---

## 5. 설정 시스템 아키텍처

### 5.1 설정 모델 계층 구조

<!-- diagram: architecture-svg4 -->

```
SLMConfig (root)
├── project: ProjectConfig
│   ├── name: str = "my-project"
│   ├── version: str = "1.0.0"
│   └── language: str = "en"
│
├── paths: PathsConfig
│   ├── documents: Path = "./documents"
│   └── output: Path = "./output"
│
├── parsing: ParsingConfig
│   ├── formats: list[str] = ["pdf", "txt", "html", "md", "hwpx", "hwp", "docx"]
│   ├── pdf: PdfOptions
│   │   └── extract_tables: bool = True
│   └── hwpx: HwpxOptions
│       └── apply_spacing: bool = True
│
├── teacher: TeacherConfig
│   ├── backend: Literal["ollama", "openai"] = "ollama"
│   ├── model: str = "qwen3.5:9b"
│   ├── api_base: str = "http://localhost:11434"
│   ├── api_key: str | None = None
│   ├── temperature: float = 0.3
│   ├── timeout: int = 300
│   ├── max_context_chars: int = 12000
│   └── max_concurrency: int = 2
│
├── questions: QuestionsConfig
│   ├── categories: dict[str, list[str]] = {}
│   ├── file: Path | None = None
│   ├── system_prompt: str = "You are a helpful..."
│   ├── output_format: str = "alpaca"
│   ├── auto_generate: bool = False
│   └── questions_per_chunk: int | "auto" = "auto"
│
├── validation: ValidationConfig
│   ├── enabled: bool = True
│   ├── min_answer_length: int = 20
│   ├── max_answer_length: int = 2000
│   ├── remove_empty: bool = True
│   ├── deduplicate: bool = True
│   ├── reject_patterns: list[str] = [...]
│   └── groundedness: GroundednessConfig
│       ├── enabled: bool = True
│       ├── model: str = "all-MiniLM-L6-v2"
│       └── threshold: float = 0.3
│
├── chunking: ChunkingConfig
│   ├── enabled: bool = True
│   ├── chunk_size: int | "auto" = "auto"
│   └── overlap_chars: int = 500
│
├── scoring: ScoringConfig
│   ├── enabled: bool = True
│   ├── threshold: float = 3.0
│   └── max_concurrency: int = 3
│
├── augment: AugmentConfig
│   ├── enabled: bool = True
│   ├── num_variants: int = 2
│   ├── max_concurrency: int = 3
│   └── min_similarity: float = 0.3
│
├── analyzer: AnalyzerConfig
│   ├── enabled: bool = True
│   └── output_file: str = "data_analysis.json"
│
├── student: StudentConfig
│   ├── model: str = "google/gemma-3-1b-it"
│   └── max_seq_length: int = 4096
│
├── training: TrainingConfig
│   ├── batch_size: int = 1
│   ├── gradient_accumulation_steps: int = 8
│   ├── learning_rate: float | "auto" = "auto"
│   ├── lr_scheduler: str = "cosine"
│   ├── warmup_ratio: float = 0.1
│   ├── num_epochs: int | "auto" = "auto"
│   ├── optimizer: str = "adamw_torch_fused"
│   ├── bf16: bool = True
│   ├── train_split: float = 0.9
│   ├── save_strategy: str = "epoch"
│   ├── lora: LoraConfig
│   │   ├── r: int = 32
│   │   ├── alpha: int = 32
│   │   ├── dropout: float = 0.1
│   │   ├── target_modules: str | list[str] = "auto"
│   │   └── use_rslora: bool = False
│   ├── early_stopping: EarlyStoppingConfig
│   │   ├── enabled: bool = True
│   │   ├── patience: int = 3
│   │   └── threshold: float = 0.01
│   ├── quantization: QuantizationConfig
│   │   ├── enabled: bool = True
│   │   └── bits: int = 4
│   ├── weight_decay: float = 0.05
│   ├── label_smoothing_factor: float = 0.1
│   └── neftune_noise_alpha: float | None = 5.0
│
├── export: ExportConfig
│   ├── merge_lora: bool = True
│   ├── output_format: str = "safetensors"
│   └── ollama: OllamaExportConfig
│       ├── enabled: bool = True
│       ├── model_name: str = "my-project-model"
│       ├── system_prompt: str = "You are a helpful..."
│       └── parameters: dict = {temperature: 0.7, top_p: 0.9, num_ctx: 4096}
│
├── eval: EvalConfig
│   ├── enabled: bool = True
│   ├── test_split: float = 0.1
│   ├── metrics: list[str] = ["bleu", "rouge", "llm_judge"]
│   ├── max_samples: int = 20
│   ├── max_tokens: int = 512
│   ├── output_file: str = "eval_results.json"
│   ├── quality_gate: bool = True
│   ├── quality_thresholds: dict = {"bleu": 0.1, "rougeL": 0.2}
│   └── llm_judge_model: str = ""
│
├── refinement: RefinementConfig
│   ├── enabled: bool = False
│   ├── max_rounds: int = 1
│   └── llm_judge_threshold: float = 0.6
│
├── incremental: IncrementalConfig
│   ├── enabled: bool = True
│   ├── hash_file: str = "document_hashes.json"
│   ├── merge_strategy: Literal["append", "replace"] = "append"
│   └── resume_adapter: str = ""
│
├── review: ReviewConfig
│   ├── enabled: bool = True
│   ├── auto_open: bool = True
│   └── output_file: str = "qa_reviewed.json"
│
├── compare: CompareConfig
│   ├── enabled: bool = False
│   ├── base_model: str = ""
│   ├── finetuned_model: str = ""
│   ├── metrics: list[str] = ["bleu", "rouge"]
│   ├── max_samples: int = 20
│   ├── max_tokens: int = 512
│   └── output_file: str = "compare_results.json"
│
├── evolve: EvolveConfig
│   ├── quality_gate: bool = True
│   ├── gate_metric: str = "rougeL"
│   ├── gate_min_improvement: float = 0.0
│   ├── version_format: str = "date"
│   ├── history_file: str = "evolve_history.json"
│   └── keep_previous_versions: int = 3
│
├── corpus_export: CorpusExportConfig
│   ├── enabled: bool = True
│   ├── output_dir: str = "corpus"
│   ├── chunk_size: int = 512
│   └── overlap_chars: int = 64
│
└── rag: RagConfig
    ├── embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    ├── vector_db_path: str = "qdrant_db"
    ├── collection_name: str = "corpus"
    ├── top_k: int = 5
    ├── server_host: str = "0.0.0.0"
    ├── server_port: int = 8000
    ├── ollama_model: str = ""
    ├── workers: int = 1
    ├── log_level: str = "info"
    ├── cors_origins: list[str] = ["*"]
    ├── batch_size: int = 64
    ├── request_timeout: float = 120.0
    ├── max_tokens: int = -1
    ├── reranker_enabled: bool = True
    ├── reranker_model: str = "dragonkue/bge-reranker-v2-m3-ko"
    ├── hybrid_search: bool = True
    ├── min_score: float = 0.0
    ├── query_rewriting: bool = False
    └── agent: AgentRagConfig
        ├── enabled: bool = True
        ├── max_iterations: int = 5
        ├── session_ttl: int = 3600
        ├── max_history_turns: int = 20
        ├── stream_reasoning: bool = True
        ├── persist_sessions: bool = False
        ├── sessions_dir: str = "agent_sessions"
        ├── planner_enabled: bool = False
        ├── verifier_enabled: bool = True
        ├── verifier_max_repairs: int = 1
        ├── legacy_fallback_enabled: bool = True
        ├── session_source_reuse: bool = True
        ├── session_source_reuse_limit: int = 5
        ├── parallel_steps: bool = False
        ├── intent_classifier_enabled: bool = False
        ├── intent_classifier_cache_ttl: int = 300
        ├── clarifier_enabled: bool = False
        ├── clarifier_max_questions: int = 2
        ├── personas_enabled: bool = False
        ├── custom_personas_dir: str = ""
        ├── memory_compression_enabled: bool = False
        ├── compress_after_turns: int = 10
        ├── compress_target_chars: int = 500
        ├── hooks_enabled: bool = False
        ├── builtin_hooks: list[str] = []
        ├── skills_enabled: bool = False
        ├── skills_dir: str = "skills"
        ├── models: AgentModelsConfig (router/planner/synthesis/...)
        ├── observation_preview_limit: int = 300
        ├── ollama_keep_alive: str = "5m"
        └── smart_mode: bool = False
```

→ 각 필드의 상세 설명은 [설정 레퍼런스](configuration.md) 참조

### 5.2 설정 로드 프로세스

CLI에서 `--config` 옵션으로 경로를 지정하거나, 지정하지 않으면 `_find_config()`가 현재 디렉토리부터 상위 디렉토리를 순회하며 `project.yaml`을 자동 탐색합니다.

**`load_config()` 흐름:**

<!-- diagram: architecture-svg5 -->

```
CLI --config project.yaml
        │
        ▼ _find_config() (자동 탐색 시)
        │  현재 디렉토리 → 상위 디렉토리 순회
        │
        ▼ load_config(path)
        │  1. Path(path).resolve()          — 절대 경로 변환
        │  2. yaml.safe_load()              — YAML 파싱
        │  3. SLMConfig.model_validate(raw) — Pydantic 검증
        │     └─ _strip_none_sections()     — None 키 제거 (기본값 적용)
        │  4. 상대 경로 → 절대 경로 변환
        │     (paths.documents, paths.output)
        │
        ▼ SLMConfig 객체 반환
```

**`_strip_none_sections` 검증기:**

YAML에서 섹션을 생략하거나 `null`로 설정하면, 이 `model_validator`가 `None` 키를 제거하여 Pydantic 기본값이 자동으로 적용되도록 합니다.

```python
@model_validator(mode="before")
@classmethod
def _strip_none_sections(cls, values: dict) -> dict:
    if isinstance(values, dict):
        return {k: v for k, v in values.items() if v is not None}
    return values
```

예를 들어 `project.yaml`에 `parsing` 섹션이 없으면 `ParsingConfig()`의 기본값이 그대로 적용됩니다.

**에러 처리:**

| 상황 | 발생 예외 |
|------|----------|
| 파일 없음 | `FileNotFoundError` |
| 유효하지 않은 YAML | `yaml.YAMLError` |
| 스키마 불일치 | `pydantic.ValidationError` |

### 5.3 기본값 생성 로직

`create_default_config()`는 `slf init` 명령 실행 시 호출되며, 다음 순서로 기본 YAML 템플릿을 반환합니다.

```
1. 패키지 루트의 templates/project.yaml 읽기 시도
        │ 실패 시
        ▼
2. importlib.resources로 설치된 wheel에서 읽기 시도
        │ 실패 시
        ▼
3. yaml.dump(SLMConfig().model_dump(), ...) — 최소 기본값을 YAML 형식으로 반환
```

개발 환경(편집 가능 설치)과 배포된 wheel 모두에서 동일하게 동작합니다.

---

## 6. 에러 처리 전략

### 6.1 계층별 에러 처리

| 계층 | 에러 타입 | 처리 방식 | 사용자 영향 |
|------|----------|----------|------------|
| CLI | `FileNotFoundError`, `ValidationError`, 기타 | Rich 포맷 에러 출력 + `typer.Exit(1)` | 명확한 에러 메시지, 비정상 종료 |
| Pipeline | 모든 예외 | `logger.exception()` 후 예외 전파 | 단계 실패 시 파이프라인 중단 |
| Parser | 개별 파일 파싱 예외 | `logger.exception()` 후 해당 파일 건너뜀 | 나머지 파일 계속 처리 |
| Teacher | `httpx.TimeoutException`, `ConnectError`, `HTTPStatusError` | 분류된 `RuntimeError`로 변환, 지수 백오프 재시도 (3회) | 명확한 연결/타임아웃 안내 |
| Validator | 규칙 위반 | `ValidationResult.reasons` 리스트에 기록 | 거부 사유 추적 가능 |
| Trainer | `RuntimeError` (CUDA OOM 등) | OOM 감지 시 권장 사항 출력 후 재발생 | 배치 크기/양자화 조정 안내 |

### 6.2 복구 전략

| 상황 | 전략 | 구현 위치 |
|------|------|----------|
| 개별 파일 파싱 실패 | 로그 후 건너뜀, 나머지 계속 처리 | `ParserRegistry.parse_directory()` |
| QA 생성 JSON 파싱 실패 | 원본 응답 로그 출력 후 건너뜀 | `QAGenerator.parse_response()` |
| 시스템 역할 미지원 모델 | 시스템 메시지 제거 후 재시도 | `ChatFormatter.format_one()` |
| Teacher 타임아웃 | 설정 값 증가 권장 메시지 출력 후 실패 | `BaseTeacher.generate()` |
| Ollama 응답 실패 | 지수 백오프로 최대 3회 재시도 후 최종 예외 발생 | `ollama_generate()` |
| CUDA OOM | 배치 크기/양자화 권장 후 예외 재발생 | `LoRATrainer.train()` |
| QA 생성 결과 0건 | `RuntimeError` 발생하여 파이프라인 중단 | `Pipeline.step_generate()` |
| 변환 결과 빈 배치 | `RuntimeError` 발생하여 파이프라인 중단 | `ChatFormatter.format_batch()` |
| JSON 파일 손상 | 빈 기본값으로 복구 후 로그 출력 | `EvolveHistory`, `IncrementalTracker` |
| 진화 품질 게이트 실패 | 생성된 Ollama 모델 자동 삭제 후 이전 모델 유지 | `cli.py` evolve 명령 |
| 설정 파일 없음 | 기본 설정 생성 제안 (`slf init`) | `cli.py` |
| 파이프라인 중단 | `--resume`으로 중간 파일에서 재개 | `cli.py` + 중간 파일 체인 |

---

## 7. Agent RAG 파이프라인

RAG 서버 `/auto` 경로는 `AgentOrchestrator`가 구동하는 OMO 패턴 기반 12단계 지능 파이프라인을 제공합니다.

```
사용자 질의
  → [Hook] pre_query (query 정규화)
  → QueryRouter.route_async (IntentClassifier + 키워드 fallback)
  → PersonaRouter.select(intent) — Researcher/Comparator/Analyst/Procedural/Clarifier + custom
  → Planner → ExecutionPlan(strategy, steps)
  → (plan.is_fallback + legacy_fallback_enabled → legacy AgentLoop로 위임)
  → Plan step 실행 (parallel_steps + 모든 step parallel_safe면 asyncio.gather)
  → ToolRegistry.execute → ToolResult.sources → all_sources
  → Verifier → 부족하면 repair search (max verifier_max_repairs)
  → [Hook] post_search (dedup, boosting)
  → _stream_synthesis (persona template + skills addon + prior sources)
  → [Hook] post_synthesis (HTML strip 등)
  → 세션 기록 (user + assistant)
  → Memory Compression (compress_after_turns 초과 시 오래된 턴 LLM 요약)
  → 최종 답변을 단일 token 이벤트로 발행 (drafting vs publishing 분리)
  → sources 이벤트 + done 이벤트
```

### 주요 모듈

| 모듈 | 역할 |
|---|---|
| `rag/agent/orchestrator.py` | `AgentOrchestrator` — 전체 파이프라인 조율, drafting/publishing 분리 |
| `rag/agent/router.py` | `QueryRouter` — intent + 키워드 라우팅 |
| `rag/agent/intent_classifier.py` | LLM 기반 의도 분류, TTL 캐시, `_LazyIntentClassifier` asyncio.Lock |
| `rag/agent/planner.py` | 질의 → `ExecutionPlan` |
| `rag/agent/verifier.py` | 컨텍스트 충분성 판정 + repair 루프 |
| `rag/agent/personas/` | Researcher/Comparator/Analyst/Procedural/Clarifier |
| `rag/agent/persona_loader.py` | YAML 기반 custom persona 로더 |
| `rag/agent/tools.py` | `ToolRegistry`, `ToolSpec.parallel_safe` 메타데이터 |
| `rag/agent/skills/` | YAML 기반 도메인 지식 팩 로더 |
| `rag/agent/hooks.py` | `HookRegistry` — pre_query/post_search/post_synthesis |
| `rag/agent/memory.py` | `ConversationCompressor` — 긴 대화 LLM 요약 |
| `rag/agent/state.py` | `FileBackedSessionStore` — 세션 영속화, mtime 기반 cleanup |
| `rag/agent/loop.py` | ReAct `AgentLoop` (legacy), `_decompose_query` LRU 캐시 (max 50) |
| `rag/agent/parser.py` | ReAct 출력 파서. Final Answer/최종 답변 마커 + escalation |

### 아키텍처 규약

- **Drafting vs Publishing 분리**: `_collect_synthesis()`로 답변을 버퍼에 수집한 뒤 단일 `token` 이벤트로 발행합니다. OpenAI compat의 `content_parts` 누적이 항상 정확히 1개 답변만 포함합니다.
- **Session 이중 기록 방지**: user 메시지를 `planner.plan()` 호출 **전**에 세션에 기록하며, planner → legacy 경로 시 `skip_user_message=True`로 전달합니다.
- **Raw vs normalized query**: `pre_query` hook으로 정규화된 query는 router/planner/synthesis에 전달되고, 세션 `add_message`는 raw query로 기록합니다.
- **Never-raise 계약**: 모든 LLM 호출 모듈은 실패 시 안전한 기본값을 반환합니다.
- **Parallel step gate**: `parallel_steps=true` + 모든 step이 `ToolSpec.parallel_safe` + 2개 이상일 때 `asyncio.gather` 병렬 실행.

상세 아키텍처 문서와 이벤트 계약은 [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)를, 개별 config 필드 설명은 [설정 레퍼런스 — rag.agent](configuration.md#ragagent--agent-rag-모드) 섹션을 참고하세요.

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [README](../README.md) | 프로젝트 소개, 설치, 빠른 시작 |
| [사용 가이드](guide.md) | 단계별 사용법, 예제, 트러블슈팅 |
| [CLI 레퍼런스](cli-reference.md) | 모든 명령어와 옵션 상세 설명 |
| [설정 레퍼런스](configuration.md) | `project.yaml` 모든 필드 상세 설명 |
| [개발자 가이드](development.md) | 커스텀 파서/백엔드 추가, 모듈 API |
