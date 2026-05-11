# 개발자 가이드

> slm-factory의 소스 코드 구조, 모듈 API, 확장 방법을 안내합니다.
> 기여자(Contributor)와 확장 개발자를 대상으로 합니다. 사용자 튜토리얼은 [guide.md](guide.md)를, 설정 레퍼런스는 [configuration.md](configuration.md)를 참조하십시오.

---

## 1. 프로젝트 구조

### 디렉토리 구조

```
src/slm_factory/
├── __init__.py              # 패키지 초기화 및 버전 정보
├── __main__.py              # python -m slm_factory 진입점
├── cli.py                   # Typer 기반 CLI 명령어 정의
├── config.py                # Pydantic v2 설정 스키마 및 load_config()
├── models.py                # 공유 데이터 모델 (QAPair, ParsedDocument 등)
├── pipeline.py              # 파이프라인 오케스트레이터 (Pipeline 클래스)
├── scorer.py                # Teacher LLM 기반 QA 품질 점수 평가
├── augmenter.py             # 질문 패러프레이즈 데이터 증강
├── analyzer.py              # QA 데이터 통계 분석 및 보고서 생성
├── evaluator.py             # 학습된 모델 자동 평가 (BLEU/ROUGE)
├── comparator.py            # Base vs Fine-tuned 모델 비교
├── incremental.py           # 문서 해시 기반 증분 변경 추적
├── converter.py             # QA → 채팅 템플릿 JSONL 변환
├── calibration.py           # Auto 캘리브레이션 (chunk_size, num_epochs, 섹션 인식 청킹)
├── device.py                # GPU/MPS/CPU 자동 감지
├── evolve_history.py        # 자동 진화 히스토리 관리
├── utils.py                 # 로깅, 비동기 유틸리티, 파일 해시
├── parsers/
│   ├── __init__.py          # ParserRegistry 인스턴스 및 파서 등록
│   ├── base.py              # BaseParser ABC, ParserRegistry 클래스
│   ├── pdf.py               # PDF 파서 (PyMuPDF)
│   ├── hwpx.py              # HWPX 파서 (한글 문서, lxml)
│   ├── html.py              # HTML 파서 (BeautifulSoup4)
│   ├── text.py              # TXT/MD 파서
│   ├── docx.py              # DOCX 파서 (python-docx, 선택적)
│   ├── hwp.py               # HWP 파서 (olefile, 선택적)
│   ├── doc.py               # DOC 파서 (olefile, 선택적)
│   ├── pptx.py              # PPTX 파서 (python-pptx, 선택적)
│   ├── ppt.py               # PPT 파서 (olefile, 선택적)
│   ├── xlsx.py              # XLSX 파서 (openpyxl, 선택적)
│   └── xls.py               # XLS 파서 (xlrd, 선택적)
├── teacher/
│   ├── __init__.py          # create_teacher() 팩토리 함수
│   ├── base.py              # BaseTeacher ABC
│   ├── ollama.py            # OllamaTeacher (로컬 Ollama REST API)
│   ├── openai_compat.py     # OpenAICompatTeacher (OpenAI 호환 API)
│   └── qa_generator.py      # QAGenerator (문서 → QA 쌍)
├── validator/
│   ├── __init__.py
│   ├── rules.py             # RuleValidator (규칙 기반 필터링)
│   └── similarity.py        # GroundednessChecker (임베딩 유사도 검증)
├── trainer/
│   ├── __init__.py
│   └── lora_trainer.py      # LoRATrainer, DataLoader
├── exporter/
│   ├── __init__.py
│   ├── hf_export.py         # HFExporter (LoRA 병합 및 저장)
│   ├── ollama_export.py     # OllamaExporter (Modelfile 생성)
│   └── corpus_export.py     # CorpusExporter (외부 평가용 parquet 내보내기)
├── rag/
│   ├── __init__.py
│   ├── indexer.py           # RAGIndexer (Qdrant 벡터 인덱싱)
│   └── server.py            # FastAPI RAG 서버 (/v1/query, /chat)
└── tui/
    ├── __init__.py
    ├── widgets.py           # 공유 TUI 위젯 (QACard, StatusBar)
    └── reviewer.py          # QAReviewerApp (QA 수동 리뷰)
```

### 모듈 의존성

| 모듈 | 의존하는 모듈 |
|------|--------------|
| `pipeline.py` | 모든 하위 모듈 (오케스트레이터) |
| `scorer.py`, `augmenter.py` | `teacher/`, `models.py`, `utils.py` |
| `analyzer.py`, `evaluator.py`, `incremental.py` | `models.py`, `utils.py` |
| `comparator.py` | `evaluator.py`, `models.py`, `utils.py` |
| `converter.py`, `validator/*.py` | `models.py`, `config.py` |
| `teacher/qa_generator.py` | `teacher/base.py`, `models.py` |
| `trainer/lora_trainer.py`, `exporter/*.py` | `config.py` (ML 라이브러리는 지연 임포트) |
| `tui/*.py` | `models.py`, `config.py` |

> 설계 원칙과 레이어 구조는 [아키텍처 가이드](architecture.md#3-핵심-설계-패턴)를 참조하십시오.

---

## 2. 핵심 모듈

### models.py — 공유 데이터 모델

모든 모듈이 공유하는 데이터 클래스입니다. 의존성 순환을 방지하기 위해 외부 라이브러리에 의존하지 않습니다.

```python
@dataclass
class ParsedDocument:
    doc_id: str; title: str; content: str
    tables: list[str]; metadata: dict

@dataclass
class QAPair:
    question: str; answer: str; instruction: str = ""
    source_doc: str = ""; category: str = ""
    is_augmented: bool = False; content_hash: str = ""; review_status: str = ""

@dataclass
class EvalResult:
    question: str; reference_answer: str; generated_answer: str
    scores: dict  # {"bleu": 0.42, "rouge1": 0.61, ...}

@dataclass
class CompareResult:
    question: str; reference_answer: str
    base_answer: str; finetuned_answer: str
    scores: dict  # {"base_bleu": 0.1, "finetuned_bleu": 0.4, ...}
```

### config.py — 설정 시스템

`SLMConfig`는 `project.yaml` 전체 스키마를 반영하는 루트 설정 객체입니다.

```python
class SLMConfig(BaseModel):
    project: ProjectConfig      # 이름, 버전, 언어
    paths: PathsConfig          # documents, output 경로
    parsing: ParsingConfig      # 형식, PDF/HWPX 옵션
    teacher: TeacherConfig      # backend, model, api_base, temperature, timeout
    questions: QuestionsConfig  # categories, file, system_prompt
    validation: ValidationConfig; scoring: ScoringConfig
    augment: AugmentConfig;      analyzer: AnalyzerConfig
    student: StudentConfig;      training: TrainingConfig
    export: ExportConfig;        eval: EvalConfig
    refinement: RefinementConfig
    incremental: IncrementalConfig
    review: ReviewConfig
    compare: CompareConfig;      evolve: EvolveConfig
    chunking: ChunkingConfig
    corpus_export: CorpusExportConfig
    rag: RagConfig
```

각 서브 모델의 필드 상세는 [configuration.md](configuration.md)를 참조하십시오.

**공개 API:**

```python
def load_config(path: str | Path) -> SLMConfig:
    """YAML 파일을 로드하고 Pydantic으로 검증합니다.
    상대 경로는 설정 파일 위치 기준으로 절대 경로로 변환됩니다.

    Raises: FileNotFoundError, yaml.YAMLError, pydantic.ValidationError
    """

def create_default_config() -> str:
    """기본 project.yaml 템플릿을 문자열로 반환합니다."""
```

### pipeline.py — 오케스트레이터

`Pipeline`은 모든 단계를 순서대로 연결하는 파사드(Facade)입니다. 각 `step_*` 메서드는 독립적으로 호출할 수 있습니다.

```python
class Pipeline:
    def __init__(self, config: SLMConfig) -> None: ...

    # 단계별 메서드
    def step_parse(
        self, files: list[Path] | None = None
    ) -> list[ParsedDocument]:
        """문서 디렉토리를 스캔하고 파싱합니다."""

    def step_generate(
        self, docs: list[ParsedDocument],
    ) -> list[QAPair]:
        """Teacher LLM으로 QA 쌍을 생성합니다 (비동기 실행)."""

    def step_validate(
        self,
        pairs: list[QAPair],
        docs: list[ParsedDocument] | None = None,
    ) -> list[QAPair]:
        """규칙 기반 + 선택적 임베딩 검증으로 QA를 필터링합니다."""

    def step_score(
        self, pairs: list[QAPair], docs: list[ParsedDocument] | None = None,
    ) -> list[QAPair]:
        """Teacher LLM으로 1~5점 품질 평가 후 threshold 필터링합니다."""

    def step_augment(self, pairs: list[QAPair]) -> list[QAPair]:
        """질문 패러프레이즈로 데이터를 증강합니다 (원본 + 증강 반환)."""

    def step_analyze(self, pairs: list[QAPair]) -> None:
        """통계 분석 보고서를 생성하고 저장합니다."""

    def step_convert(self, pairs: list[QAPair]) -> Path:
        """QA 쌍을 채팅 템플릿 JSONL로 변환합니다."""

    def step_train(self, training_data_path: Path) -> Path:
        """LoRA 파인튜닝을 실행하고 어댑터 경로를 반환합니다."""

    def step_export(self, adapter_path: Path) -> Path:
        """LoRA 병합 + Ollama Modelfile 생성을 수행합니다."""

    def step_eval(
        self, pairs: list[QAPair], model_name: str
    ) -> list[EvalResult]:
        """BLEU/ROUGE 메트릭으로 모델을 평가합니다."""

    def step_compare(self, pairs: list[QAPair]) -> list[CompareResult]:
        """Base 모델과 Fine-tuned 모델의 답변을 비교합니다."""

    def step_corpus_export(
        self, parsed_docs: list[dict], qa_pairs: list[dict],
    ) -> tuple[Path, Path]:
        """파싱·QA 데이터를 외부 평가용 parquet으로 내보냅니다.
        반환값: (corpus.parquet 경로, qa.parquet 경로) 튜플
        """

    def step_rag_index(self, corpus_path: Path) -> Path:
        """corpus.parquet을 Qdrant에 임베딩하여 적재합니다."""

    def run(self) -> Path:
        """전체 파이프라인을 엔드-투-엔드로 실행합니다."""
```

### utils.py — 유틸리티

```python
def setup_logging(level: str = "INFO") -> logging.Logger: ...
    # Rich 핸들러로 루트 slm-factory 로거를 구성합니다.
def get_logger(name: str) -> logging.Logger: ...
    # slm_factory.<name> 네임스페이스 로거를 반환합니다.
def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str: ...
    # 파일의 해시값을 계산합니다.
async def run_bounded(semaphore, coro, progress, task_id) -> T: ...
    # 세마포어 제한 하에 코루틴을 실행하고 진행률을 갱신합니다.
async def ollama_generate(client, api_base, model_name, question, timeout, *, max_tokens=512, max_retries=3, format=None, think=None) -> str: ...
    # Ollama /api/generate 엔드포인트로 답변을 생성합니다.
    # 일시적 오류(HTTP 5xx, 타임아웃, 연결 오류)에 대해 지수 백오프로 재시도합니다.
    # max_tokens로 응답 길이를 제한하여 무한 생성을 방지합니다.
```

---

## 3. 파서 모듈 (parsers/)

### BaseParser 인터페이스

```python
class BaseParser(ABC):
    extensions: ClassVar[list[str]] = []
    # 예: ['.pdf'], ['.hwpx'], ['.html', '.htm']

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """path의 문서를 파싱하여 ParsedDocument를 반환합니다."""

    def can_parse(self, path: Path) -> bool:
        """이 파서가 주어진 파일 확장자를 지원하면 True를 반환합니다."""
```

### 파서 레지스트리

```python
class ParserRegistry:
    def register(self, parser_cls: type[BaseParser]) -> type[BaseParser]:
        """파서 클래스를 등록합니다 (데코레이터로도 사용 가능)."""

    def get_parser(self, path: Path) -> BaseParser | None:
        """path를 처리할 수 있는 첫 번째 등록된 파서를 반환합니다."""

    def parse_directory(
        self,
        dir_path: Path,
        formats: list[str] | None = None,
        files: list[Path] | None = None,
    ) -> list[ParsedDocument]:
        """디렉토리의 모든 지원 파일을 파싱합니다."""

# parsers/__init__.py에서 전역 레지스트리 인스턴스를 제공합니다
from slm_factory.parsers import registry
```

### 개별 파서

| 파서 클래스 | 파일 | 확장자 | 주요 라이브러리 |
|------------|------|--------|----------------|
| `PDFParser` | `pdf.py` | `.pdf` | PyMuPDF (fitz) |
| `HWPXParser` | `hwpx.py` | `.hwpx` | lxml, kiwipiepy (선택적) |
| `HTMLParser` | `html.py` | `.html`, `.htm` | BeautifulSoup4 (lxml 백엔드), charset-normalizer |
| `TextParser` | `text.py` | `.txt`, `.md` | charset-normalizer |
| `DOCXParser` | `docx.py` | `.docx` | python-docx (선택적) |
| `HWPParser` | `hwp.py` | `.hwp` | olefile (선택적) |
| `DOCParser` | `doc.py` | `.doc` | olefile (선택적) |
| `PPTXParser` | `pptx.py` | `.pptx` | python-pptx (선택적) |
| `PPTParser` | `ppt.py` | `.ppt` | olefile (선택적) |
| `XLSXParser` | `xlsx.py` | `.xlsx` | openpyxl (선택적) |
| `XLSParser` | `xls.py` | `.xls` | xlrd (선택적) |

선택적 파서(DOCX, HWP, DOC, PPTX, PPT, XLSX, XLS)는 해당 라이브러리가 설치되지 않은 경우 자동으로 비활성화됩니다. 레지스트리는 `parsers/__init__.py`에서 모든 파서를 자동 등록합니다.

`parsers/base.py`의 `detect_encoding()` 함수는 HTML 파서와 텍스트 파서가 공유하는 유틸리티입니다. charset-normalizer를 사용하여 EUC-KR, CP949 등 한국어 인코딩을 정확하게 감지합니다.

---

## 4. Teacher LLM 모듈 (teacher/)

### BaseTeacher 인터페이스

```python
class BaseTeacher(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """프롬프트를 Teacher LLM에 전송하고 응답 텍스트를 반환합니다.
        kwargs: 백엔드별 오버라이드 (예: temperature, format)
        """

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        """generate()의 비동기 변형. 기본 구현은 동기 호출을 래핑합니다.
        동시성이 필요한 서브클래스는 진정한 비동기 I/O로 오버라이드해야 합니다.
        """

    def health_check(self) -> bool:
        """백엔드에 도달할 수 있는지 확인합니다. 예외를 발생시키지 않습니다."""
```

### OllamaTeacher / OpenAICompatTeacher

```python
class OllamaTeacher(BaseTeacher):
    def __init__(self, config: TeacherConfig) -> None:
        # config.model, config.api_base, config.temperature, config.timeout 사용
        # generate()는 format="json" 지원, 최대 3회 재시도 + 지수 백오프 적용
        ...

class OpenAICompatTeacher(BaseTeacher):
    def __init__(self, config: TeacherConfig) -> None:
        # config.api_base, config.api_key, config.model, config.temperature 사용
        # OpenAI /v1/chat/completions 엔드포인트 호출
        ...

def create_teacher(config: TeacherConfig) -> BaseTeacher:
    """config.backend에 따라 OllamaTeacher 또는 OpenAICompatTeacher를 반환합니다.
    Raises: ValueError (알 수 없는 backend)
    """
```

### QAGenerator

```python
class QAGenerator:
    def __init__(self, config: SLMConfig) -> None: ...

    def build_prompt(
        self, doc_title: str, content: str, question: str,
        tables: list[str] | None = None, system_prompt: str | None = None,
    ) -> str: ...
    # QA 생성을 위한 전체 프롬프트를 구성합니다.

    def parse_response(self, text: str) -> dict[str, str] | None: ...
    # LLM 응답을 {"instruction": ..., "output": ...}로 파싱합니다.

    def generate_for_document(
        self, doc: ParsedDocument, questions: list[str] | None = None, category: str = "",
    ) -> list[QAPair]: ...
    # 단일 문서에 대한 QA 쌍을 동기적으로 생성합니다.

    async def generate_all_async(
        self, docs: list[ParsedDocument], questions: list[str] | None = None,
    ) -> list[QAPair]: ...
    # 세마포어 기반 동시성으로 전체 문서의 QA를 비동기 생성합니다.

    def save_alpaca(self, pairs: list[QAPair], output_path: str | Path) -> Path: ...
    # QA 쌍을 Alpaca 형식 JSON으로 저장합니다.
```

---

## 5. 검증 모듈 (validator/)

### RuleValidator

```python
class RuleValidator:
    def __init__(self, config: ValidationConfig) -> None: ...

    def validate_one(self, pair: QAPair) -> ValidationResult:
        """단일 QA 쌍을 검증합니다. ValidationResult(passed, reasons)를 반환합니다."""

    def validate_batch(
        self, pairs: list[QAPair]
    ) -> tuple[list[QAPair], list[tuple[QAPair, ValidationResult]]]:
        """QA 쌍 배치를 검증합니다.
        반환값: (수락된_쌍, 거부된_쌍_및_이유) 튜플
        """

    def reset_dedup(self) -> None:
        """중복 제거 캐시를 초기화합니다."""
```

적용되는 규칙 (순서대로):

1. **빈 값 확인**: 질문 또는 답변이 비어있거나 공백이면 거부 (`remove_empty: true`)
2. **길이 확인**: 답변이 `min_answer_length`보다 짧거나 `max_answer_length`보다 길면 거부
3. **거부 패턴**: 답변이 `reject_patterns` 정규식과 일치하면 거부 (예: "I don't know")
4. **중복 제거**: 동일한 (질문, 답변) 쌍이 이미 처리된 경우 거부 (`deduplicate: true`)

### GroundednessChecker

```python
class GroundednessChecker:
     def __init__(self, config: GroundednessConfig) -> None:
         # sentence-transformers 필요: uv sync --extra validation
         # config.model (기본값: "all-MiniLM-L6-v2"), config.threshold (기본값: 0.3)
         ...

    def score(self, answer: str, source_text: str) -> float:
        """답변과 원본 문서 청크 간의 최대 코사인 유사도를 반환합니다.
        반환값: [0, 1] 범위의 실수 (높을수록 더 근거가 있음)
        """

    def check(self, pair: QAPair, source_text: str) -> tuple[bool, float]:
        """QA 쌍의 답변이 원본 텍스트에 근거하는지 확인합니다.
        반환값: (근거_여부, 유사도_점수) 튜플
        """

    def check_batch(
        self,
        pairs: list[QAPair],
        source_texts: dict[str, str],  # doc_id → 문서 텍스트
    ) -> tuple[list[QAPair], list[tuple[QAPair, float]]]:
        """QA 쌍 배치의 근거성을 확인합니다."""
```

---

## 6. 데이터 처리 모듈

### QualityScorer (scorer.py)

```python
class QualityScorer:
    def __init__(self, teacher: BaseTeacher, config: ScoringConfig, teacher_config: TeacherConfig) -> None: ...
    async def score_one(self, pair: QAPair) -> tuple[QAPair, int, str]: ...
    # 단일 QA 쌍을 1~5점으로 평가합니다. (pair, score, reason) 반환.
    async def score_all(self, pairs: list[QAPair]) -> tuple[list[QAPair], list[tuple[QAPair, int, str]]]: ...
    # config.threshold 이상인 쌍만 반환합니다. (수락된_쌍, 필터링된_쌍_및_점수_이유) 튜플.
```

주요 설정: `scoring.enabled`, `scoring.threshold` (1.0~5.0, 기본값 3.0), `scoring.max_concurrency`

### DataAugmenter (augmenter.py)

```python
class DataAugmenter:
    def __init__(self, teacher: BaseTeacher, config: AugmentConfig, teacher_config: TeacherConfig) -> None: ...
    async def paraphrase_one(self, pair: QAPair) -> list[QAPair]: ...
    # 단일 QA 쌍의 질문을 패러프레이즈하여 증강된 QA 쌍을 생성합니다.
    async def augment_all(self, pairs: list[QAPair]) -> list[QAPair]: ...
    # 원본 + 증강 쌍을 반환합니다.
```

주요 설정: `augment.enabled`, `augment.num_variants` (기본값 2), `augment.max_concurrency`

### DataAnalyzer (analyzer.py)

```python
@dataclass
class AnalysisReport:
    total_pairs: int; original_pairs: int; augmented_pairs: int
    category_distribution: dict[str, int]; source_doc_distribution: dict[str, int]
    answer_length_stats: dict[str, float]   # min, max, mean, median, stdev
    question_length_stats: dict[str, float]; quality_score_stats: dict[str, float]
    warnings: list[str]

class DataAnalyzer:
    def analyze(self, pairs: list[QAPair]) -> AnalysisReport:
        """QA 쌍 리스트를 분석하여 AnalysisReport를 생성합니다."""

    def print_summary(self, report: AnalysisReport) -> None:
        """Rich 콘솔에 분석 요약을 출력합니다."""

    def save_report(self, report: AnalysisReport, path: Path) -> None:
        """분석 보고서를 JSON 파일로 저장합니다."""
```

### ChatFormatter (converter.py)

```python
class ChatFormatter:
    def __init__(self, config: SLMConfig) -> None: ...
    def format_one(self, pair: QAPair) -> str | None: ...
    # 단일 QA 쌍을 채팅 템플릿으로 형식화합니다. 시스템 역할 실패 시 자동 fallback.
    def format_batch(self, pairs: list[QAPair]) -> list[dict[str, str]]: ...
    # [{"text": "..."}, ...] 형식으로 반환, max_seq_length 초과 항목 제외.
    def save_training_data(self, pairs: list[QAPair], output_path: str | Path) -> Path: ...
    def format_from_alpaca_file(self, input_path: str | Path, output_path: str | Path) -> Path: ...
```

---

## 7. 학습 및 내보내기 모듈

### LoRATrainer (trainer/lora_trainer.py)

```python
class LoRATrainer:
    def __init__(self, config: SLMConfig) -> None: ...

    def train(self, dataset_dict: DatasetDict) -> Path:
        """LoRA 파인튜닝을 실행하고 어댑터 디렉토리 경로를 반환합니다.
        dataset_dict: "train"과 "eval" 분할을 포함하는 DatasetDict
        """
```

주요 학습 파라미터 (`training.*`):

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_epochs` | `"auto"` | 학습 에포크 수 (`"auto"`: 데이터 양에 따라 1~3 자동 결정) |
| `learning_rate` | `"auto"` | 학습률 (`"auto"`: 데이터 양에 따라 5e-5~2e-4 자동 결정) |
| `batch_size` | 1 | 디바이스당 배치 크기 |
| `lora.r` / `lora.alpha` | 8 / 8 | LoRA 랭크 / 스케일링 (1:1 비율 권장) |
| `early_stopping.patience` | 3 | 조기 종료 인내 에포크 수 |

```python
class DataLoader:
    def __init__(self, train_split: float = 0.9) -> None: ...

    def load_jsonl(self, path: str | Path) -> Dataset:
        """JSONL 파일을 HuggingFace Dataset으로 로드합니다."""

    def load_and_split(self, path: str | Path) -> DatasetDict:
        """JSONL 파일을 로드하고 train/eval로 분할합니다."""
```

### HFExporter (exporter/hf_export.py)

```python
class HFExporter:
    def __init__(self, config: SLMConfig) -> None: ...
    def merge_and_save(self, adapter_path: str | Path, output_dir=None) -> Path:
        """LoRA 어댑터를 기본 모델에 병합하고 safetensors 형식으로 저장합니다."""
    def save_adapter_only(self, adapter_path: str | Path, output_dir=None) -> Path:
        """병합 없이 LoRA 어댑터만 저장합니다."""
    def export(self, adapter_path: str | Path, output_dir=None) -> Path:
        """config.export.merge_lora에 따라 병합 또는 어댑터만 저장합니다."""
```

### OllamaExporter (exporter/ollama_export.py)

```python
class OllamaExporter:
    def __init__(self, config: SLMConfig) -> None: ...
    def generate_modelfile(self, model_dir: str | Path, output_path=None) -> Path:
        """Ollama Modelfile을 생성합니다 (기본 위치: model_dir/Modelfile)."""
    def create_model(
        self, modelfile_path: str | Path, model_name_override: str | None = None,
    ) -> bool:
        """ollama create 명령으로 모델을 생성합니다. 성공 여부를 반환합니다."""
    def export(
        self, model_dir: str | Path, output_dir=None, model_name_override: str | None = None,
    ) -> Path:
        """Modelfile을 생성하고 Ollama가 감지되면 모델을 자동 생성합니다."""
```

---

## 8. 평가 및 비교 모듈

### ModelEvaluator (evaluator.py)

```python
class ModelEvaluator:
    def __init__(self, config: SLMConfig) -> None: ...
    def evaluate(self, qa_pairs: list[QAPair], model_name: str) -> list[EvalResult]:
        """QA 쌍으로 Ollama 모델을 평가합니다 (config.eval.max_samples 샘플링)."""
    def save_results(self, results: list[EvalResult], path: Path) -> None: ...
    def print_summary(self, results: list[EvalResult]) -> None: ...
```

지원 메트릭: `bleu`, `rouge` (ROUGE-1, ROUGE-2, ROUGE-L), `llm_judge` (Teacher LLM 의미적 정확성 평가, 0~1 정규화)

### ModelComparator (comparator.py)

```python
class ModelComparator:
    def __init__(self, config: SLMConfig) -> None: ...
    def compare(self, qa_pairs: list[QAPair]) -> list[CompareResult]:
        """config.compare.base_model과 finetuned_model을 비교합니다."""
    def save_results(self, results: list[CompareResult], path: Path) -> None: ...
    def print_summary(self, results: list[CompareResult]) -> None: ...
```

---

## 9. TUI 모듈 (tui/)

### QAReviewerApp (reviewer.py)

Textual 기반 QA 수동 리뷰 앱입니다. `slf tool review` 명령으로 실행됩니다.

```python
class QAReviewerApp(App[None]):
    """QA 쌍을 하나씩 확인하며 승인/거부/편집하는 TUI 앱입니다.
    키 바인딩: a/Enter=승인, r=거부, e=편집, q/Escape=종료
    출력: config.review.output_file (기본값: qa_reviewed.json)
    """
```

### 위젯 (widgets.py)

| 클래스 | 역할 |
|--------|------|
| `QACard` | QA 쌍을 카드 형태로 표시하는 위젯 |
| `StatusBar` | 리뷰 진행률 및 상태를 표시하는 하단 바 |

---

## 10. 확장 가이드

### 새 파서 추가

`BaseParser`를 상속하고 `extensions`와 `parse()`를 구현한 후 레지스트리에 등록합니다.

```python
# src/slm_factory/parsers/epub.py
from pathlib import Path
from .base import BaseParser
from ..models import ParsedDocument


class EPUBParser(BaseParser):
    extensions = [".epub"]

    def parse(self, path: Path) -> ParsedDocument:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(str(path))
        parts = [
            BeautifulSoup(item.get_content(), "lxml").get_text(separator="\n")
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        ]
        title_meta = book.get_metadata("DC", "title")
        return ParsedDocument(
            doc_id=path.stem,
            title=title_meta[0][0] if title_meta else path.stem,
            content="\n\n".join(parts),
            tables=[], metadata={"format": "epub"},
        )
```

`parsers/__init__.py`에 등록합니다:

```python
try:
    from .epub import EPUBParser
    registry.register(EPUBParser)
except ImportError:
    pass
```

### 새 Teacher 백엔드 추가

`BaseTeacher`를 상속하고 `generate()`와 `agenerate()`를 구현합니다.

```python
# src/slm_factory/teacher/anthropic_teacher.py
from .base import BaseTeacher
from ..config import TeacherConfig


class AnthropicTeacher(BaseTeacher):
    def __init__(self, config: TeacherConfig) -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.model = config.model
        self.temperature = config.temperature

    def generate(self, prompt: str, **kwargs: object) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=2048, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate, prompt
        )

    def health_check(self) -> bool:
        try: self.generate("ping"); return True
        except Exception: return False
```

`teacher/__init__.py`의 `create_teacher()`에 등록합니다:

```python
def create_teacher(config: TeacherConfig) -> BaseTeacher:
    if config.backend == "ollama":
        return OllamaTeacher(config)
    elif config.backend == "openai":
        return OpenAICompatTeacher(config)
    elif config.backend == "anthropic":          # 추가
        from .anthropic_teacher import AnthropicTeacher
        return AnthropicTeacher(config)
    else:
        raise ValueError(f"Unknown teacher backend: {config.backend!r}")
```

`project.yaml`에서 사용: `teacher.backend: "anthropic"`, `teacher.model: "claude-3-haiku-20240307"`

### 커스텀 검증 규칙 추가

`RuleValidator`를 서브클래싱하거나, 파이프라인에서 `step_validate()` 이후에 커스텀 필터를 적용합니다.

```python
from slm_factory.validator.rules import RuleValidator
from slm_factory.models import QAPair


class KoreanRuleValidator(RuleValidator):
    """한국어 답변에 특화된 추가 검증 규칙을 적용합니다."""

    def validate_one(self, pair: QAPair):
        result = super().validate_one(pair)
        if not result.passed:
            return result
        korean_chars = sum(1 for c in pair.answer if "\uAC00" <= c <= "\uD7A3")
        if korean_chars < 10:
            result.passed = False
            result.reasons.append("insufficient_korean_content")
        return result


# 파이프라인에서 직접 필터링하는 방법
from slm_factory.pipeline import Pipeline
from slm_factory.config import load_config

pipeline = Pipeline(load_config("project.yaml"))
docs = pipeline.step_parse()
pairs = pipeline.step_generate(docs)
pairs = pipeline.step_validate(pairs, docs=docs)
pairs = [p for p in pairs if len(p.answer.split()) >= 20]
pairs = pipeline.step_score(pairs)
```

### 새 내보내기 형식 추가

`exporter/` 디렉토리의 패턴을 따라 새 내보내기 클래스를 작성합니다.

```python
# src/slm_factory/exporter/vllm_export.py
import json
from pathlib import Path
from ..utils import get_logger

logger = get_logger("exporter.vllm_export")


class VLLMExporter:
    """vLLM 서빙을 위한 설정 파일을 생성합니다."""

    def __init__(self, config) -> None:
        self.config = config

    def export(self, model_dir: Path) -> Path:
        config_data = {
            "model": str(model_dir),
            "served_model_name": self.config.project.name,
            "max_model_len": self.config.student.max_seq_length,
            "dtype": "bfloat16",
        }
        output_path = model_dir / "vllm_config.json"
        output_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        logger.info("vLLM 설정 파일 생성됨: %s", output_path)
        return output_path
```

> 설계 패턴 이해는 [아키텍처 가이드](architecture.md#3-핵심-설계-패턴)를 참조하십시오.

---

## 11. 테스트

### 테스트 구조

```
tests/
├── conftest.py                  # 공유 fixture 및 ML 라이브러리 mock
├── test_models.py               # 데이터 모델
├── test_config.py               # 설정 로드 및 검증
├── test_pipeline.py             # 파이프라인 통합
├── test_parsers_{base,pdf,hwpx,html,text,docx}.py
├── test_teacher.py              # OllamaTeacher, OpenAICompatTeacher
├── test_qa_generator.py         # QAGenerator
├── test_validator_{rules,similarity}.py
├── test_{scorer,augmenter,analyzer,converter}.py
├── test_exporter.py             # HFExporter, OllamaExporter
├── test_{evaluator,comparator,incremental}.py
├── test_integration.py          # 통합 테스트 (청킹, 재생성, 관계 정규화)
├── test_reviewer.py             # TUI
├── test_cli.py                  # CLI 명령어
└── test_utils.py                # 유틸리티 함수
```

### 테스트 실행

```bash
# 전체 테스트 실행
uv run pytest

# 특정 모듈 테스트
uv run pytest tests/test_parsers_pdf.py

# 특정 테스트 함수 실행
uv run pytest tests/test_validator_rules.py::test_validate_batch_rejects_empty

# 상세 출력
uv run pytest -v

# 커버리지 측정 (pytest-cov 필요: uv add --dev pytest-cov)
uv run pytest --cov=slm_factory --cov-report=html

# 빠른 실행 (병렬, pytest-xdist 필요: uv add --dev pytest-xdist)
uv run pytest -n auto
```

### 테스트 작성 가이드

**핵심 원칙:**

- `conftest.py`는 `torch`, `transformers`, `peft`, `trl`, `datasets`, `sentence_transformers`, `kiwipiepy` 등 무거운 ML 라이브러리를 `MagicMock`으로 자동 대체합니다. 실제 GPU 없이도 모든 로직을 테스트할 수 있습니다.
- `make_config` fixture로 `SLMConfig`를 쉽게 생성합니다: `config = make_config(teacher={"model": "test-model"})`
- 외부 API 호출은 `unittest.mock.patch`로 mock합니다.
- 파일 I/O 테스트에는 `tmp_path` fixture를 사용합니다.

**파서 테스트 패턴:**

```python
# tests/test_parsers_epub.py
from unittest.mock import patch, MagicMock
from slm_factory.parsers.epub import EPUBParser

def test_epub_parser_extensions():
    assert ".epub" in EPUBParser.extensions

def test_epub_parser_parse(tmp_path):
    mock_book = MagicMock()
    mock_book.get_metadata.return_value = [("테스트 문서",)]
    mock_book.get_items_of_type.return_value = []
    with patch("ebooklib.epub.read_epub", return_value=mock_book):
        doc = EPUBParser().parse(tmp_path / "test.epub")
    assert doc.doc_id == "test" and doc.title == "테스트 문서"
```

**검증 규칙 테스트 패턴:**

```python
# tests/test_validator_custom.py
from slm_factory.config import ValidationConfig
from slm_factory.models import QAPair
from slm_factory.validator.rules import RuleValidator

def test_rejects_short_answer():
    v = RuleValidator(ValidationConfig(min_answer_length=50))
    r = v.validate_one(QAPair(question="질문", answer="짧은 답변"))
    assert not r.passed and any("answer_too_short" in x for x in r.reasons)

def test_accepts_valid_pair():
    v = RuleValidator(ValidationConfig(min_answer_length=10))
    r = v.validate_one(QAPair(question="이 정책의 목적은?", answer="국민 복지 향상과 사회적 형평성 제고입니다."))
    assert r.passed
```

---

## 관련 문서

- **[아키텍처 가이드](architecture.md)**: 내부 구조, 설계 원칙, 레이어 다이어그램
- **[설정 레퍼런스](configuration.md)**: `project.yaml`의 모든 필드 상세 설명
- **[CLI 레퍼런스](cli-reference.md)**: 모든 CLI 명령어와 옵션
- **[사용자 가이드](guide.md)**: 엔드-투-엔드 사용 튜토리얼
- **[빠른 참조](quick-reference.md)**: 자주 쓰는 명령어 모음
