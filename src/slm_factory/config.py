"""Pydantic v2 + YAML를 사용한 slm-factory 설정 시스템입니다.

YAML 파일에서 프로젝트 설정을 로드하고 검증합니다.
파싱, 교사 모델, 질문, 검증, 훈련, 내보내기 등 모든 파이프라인 설정에 대한
타입 안전 접근을 제공합니다.
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 언어별 기본 프롬프트
# ---------------------------------------------------------------------------

_EN_DEFAULT_QA_SYSTEM_PROMPT = (
    "You are a domain-expert assistant that answers questions based strictly on "
    "the provided document in the [Document] section. Cite specific clauses, "
    "numbers, and dates from the document as evidence. Do not speculate or use "
    "information beyond the provided document. If the document does not contain "
    'relevant information, say "This information is not contained in the '
    'provided document."'
)

_KO_DEFAULT_QA_SYSTEM_PROMPT = (
    "당신은 제공된 문서를 참고하여 질문에 답변하는 도메인 전문 어시스턴트입니다. "
    "반드시 [문서] 섹션에 제공된 내용만을 근거로 답변하세요. "
    "답변 시 문서의 관련 조항, 수치, 날짜를 인용하세요. "
    "추측하거나 문서 외의 정보를 사용하지 마세요. "
    '문서에 관련 정보가 없으면 "해당 정보는 제공된 문서에 포함되어 있지 않습니다."라고 '
    "답변하세요."
)

_EN_DEFAULT_OLLAMA_SYSTEM_PROMPT = "You are a helpful domain-specific assistant."

_KO_DEFAULT_OLLAMA_SYSTEM_PROMPT = (
    "당신은 도메인 전문 지식을 바탕으로 사용자의 질문에 "
    "정확하고 유용하게 답변하는 어시스턴트입니다."
)


# ---------------------------------------------------------------------------
# 하위 모델
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """최상위 프로젝트 메타데이터입니다."""

    name: str = Field("my-project", min_length=1)
    version: str = "1.0.0"
    language: str = "en"


class PathsConfig(BaseModel):
    """입출력 디렉토리 경로입니다 (검증 시 자동 생성됨)."""

    documents: Path = Path("./documents")
    output: Path = Path("./output")

    def ensure_dirs(self) -> None:
        """설정된 디렉토리가 없으면 생성합니다."""
        self.documents.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)


class PdfOptions(BaseModel):
    """PDF 파싱 옵션입니다."""

    extract_tables: bool = True


class HwpxOptions(BaseModel):
    """HWPX (한글 문서) 파싱 옵션입니다."""

    apply_spacing: bool = True


class ParsingConfig(BaseModel):
    """문서 파싱 설정입니다."""

    formats: list[str] = Field(
        default_factory=lambda: ["pdf", "txt", "html", "md", "hwpx", "hwp", "docx"]
    )
    pdf: PdfOptions = Field(default_factory=PdfOptions)
    hwpx: HwpxOptions = Field(default_factory=HwpxOptions)


class TeacherConfig(BaseModel):
    """QA 생성을 위한 교사 LLM 설정입니다."""

    backend: Literal["ollama", "openai"] = "ollama"
    model: str = Field("qwen3.5:9b", min_length=1)
    api_base: str = Field("http://localhost:11434", min_length=1)
    api_key: str | None = None
    temperature: float = 0.3
    timeout: int = 300
    max_context_chars: int = 12000
    max_concurrency: int = 2


class QuestionsConfig(BaseModel):
    """질문 카테고리 및 생성 설정입니다.

    질문은 *categories* (카테고리명을 질문 문자열 리스트로 매핑하는 딕셔너리)를
    통해 인라인으로 정의하거나, *file*을 통해 외부 파일에서 로드할 수 있습니다.

    ``auto_generate=True``이면 Teacher LLM이 각 청크를 분석하여 질문-답변 쌍을
    자동으로 생성합니다. 기존 ``categories`` 고정질문이 있으면 자동생성 QA가
    **추가**됩니다. ``categories``가 비어있으면 자동생성만 수행합니다.
    """

    categories: dict[str, list[str]] = Field(default_factory=dict)
    file: Path | None = None
    auto_generate: bool = False
    questions_per_chunk: int | Literal["auto"] = Field(default="auto")
    system_prompt: str = _EN_DEFAULT_QA_SYSTEM_PROMPT
    output_format: str = "alpaca"

    def get_all_questions(self) -> list[str]:
        """모든 카테고리 질문을 단일 리스트로 평탄화합니다.

        *file*이 기존 텍스트 파일을 가리키면, 인라인 카테고리 대신
        파일의 비어있지 않은 줄들을 반환합니다.
        """
        if self.file is not None:
            path = Path(self.file)
            if path.is_file():
                return [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            raise FileNotFoundError(
                f"질문 파일을 찾을 수 없습니다: {path}. "
                f"questions.file 경로를 확인하세요."
            )
        return [q for questions in self.categories.values() for q in questions]

    def get_questions_with_categories(self) -> list[tuple[str, str]]:
        """(카테고리명, 질문) 튜플 리스트를 반환합니다.

        카테고리 정보를 유지하면서 모든 질문을 반환합니다.
        파일에서 로드한 질문은 카테고리가 빈 문자열입니다.
        """
        if self.file is not None:
            path = Path(self.file)
            if path.is_file():
                return [
                    ("", line.strip())
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            raise FileNotFoundError(
                f"질문 파일을 찾을 수 없습니다: {path}. "
                f"questions.file 경로를 확인하세요."
            )
        return [
            (cat_name, q)
            for cat_name, questions in self.categories.items()
            for q in questions
        ]


class GroundednessConfig(BaseModel):
    """생성된 답변에 대한 의미론적 근거 검증입니다."""

    enabled: bool = True
    model: str = "all-MiniLM-L6-v2"
    threshold: float = 0.3


class ValidationConfig(BaseModel):
    """QA 쌍 검증 및 필터링 규칙입니다."""

    enabled: bool = True
    min_answer_length: int = 20
    max_answer_length: int = 2000
    remove_empty: bool = True
    deduplicate: bool = True
    reject_patterns: list[str] = Field(
        default_factory=lambda: [
            "(?i)i don't know",
            "(?i)not (available|provided|mentioned|found)",
            "(?i)the document does not contain",
            "(?:알|확인할|파악할|판단할) 수 없",
            "정보.{0,8}(?:없|부족|찾을 수 없)",
            "(?:문서|자료|내용)에서?.{0,12}(?:찾을 수 없|포함되어 있지 않|언급되지 않|다루고 있지 않)",
        ]
    )
    groundedness: GroundednessConfig = Field(default_factory=GroundednessConfig)

    @model_validator(mode="after")
    def _check_answer_length_range(self) -> "ValidationConfig":
        """min_answer_length가 max_answer_length보다 작은지 검증합니다."""
        if self.min_answer_length >= self.max_answer_length:
            raise ValueError(
                f"min_answer_length({self.min_answer_length})는 "
                f"max_answer_length({self.max_answer_length})보다 작아야 합니다"
            )
        return self


class ScoringConfig(BaseModel):
    """교사 LLM을 사용한 QA 쌍 품질 점수 설정입니다."""

    enabled: bool = True
    threshold: float = 3.0
    max_concurrency: int = 3

    @model_validator(mode="after")
    def _check_scoring_threshold(self) -> "ScoringConfig":
        """점수 기준값의 유효성을 검증합니다."""
        if self.enabled and not (1.0 <= self.threshold <= 5.0):
            raise ValueError(f"threshold({self.threshold})는 1.0~5.0 범위여야 합니다")
        return self


class AugmentConfig(BaseModel):
    """QA 쌍 데이터 증강 설정입니다."""

    enabled: bool = True
    num_variants: int = 2
    max_concurrency: int = 3
    min_similarity: float = 0.3
    """패러프레이즈된 질문과 원본 질문 간 최소 토큰 겹침 비율입니다.
    이 값 미만이면 의미가 변질된 것으로 간주하여 제거합니다 (0.0~1.0)."""


class AnalyzerConfig(BaseModel):
    """학습 데이터 통계 분석 설정입니다."""

    enabled: bool = True
    output_file: str = "data_analysis.json"


class StudentConfig(BaseModel):
    """학생 (미세조정된) 모델 설정입니다."""

    model: str = Field("google/gemma-3-1b-it", min_length=1)
    max_seq_length: int = 4096


class LoraConfig(BaseModel):
    """LoRA 어댑터 하이퍼파라미터입니다."""

    r: int = 32
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Union[str, list[str]] = "auto"
    use_rslora: bool = False


class EarlyStoppingConfig(BaseModel):
    """훈련을 위한 조기 종료 기준입니다."""

    enabled: bool = True
    patience: int = 3
    threshold: float = 0.01


class QuantizationConfig(BaseModel):
    """훈련을 위한 양자화 설정입니다."""

    enabled: bool = True
    bits: int = 4


class TrainingConfig(BaseModel):
    """전체 훈련 하이퍼파라미터 설정입니다."""

    lora: LoraConfig = Field(default_factory=LoraConfig)
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float | Literal["auto"] = "auto"
    """학습률입니다. ``"auto"``이면 학습 데이터 양에 따라 자동 결정합니다.
    소규모(<100): 5e-5, 중규모(<500): 1e-4, 대규모: 2e-4."""
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    num_epochs: int | Literal["auto"] = "auto"
    """에포크 수입니다. ``"auto"``이면 학습 데이터 양에 따라 자동 결정합니다."""
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    optimizer: str = "adamw_torch_fused"  # 디바이스 자동 감지로 런타임에 조정됨
    bf16: bool = True  # 디바이스 자동 감지로 런타임에 조정됨
    train_split: float = 0.9
    save_strategy: str = "epoch"
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    weight_decay: float = 0.05
    """가중치 감쇠(L2 정규화)입니다. 과적합을 억제합니다. 0이면 비활성."""
    label_smoothing_factor: float = 0.1
    """라벨 스무딩 계수입니다. 모델이 특정 답변을 과도하게 확신하는 것을 방지합니다.
    0이면 비활성, 0.1이면 정답 확률을 90%로 낮추어 일반화 성능을 높입니다."""
    neftune_noise_alpha: float | None = 5.0
    """NEFTune 임베딩 노이즈 강도입니다. 학습 중 임베딩에 노이즈를
    주입하여 일반화 성능을 5~15% 향상시킵니다. 권장값: 5.0 (2B 이하), 10.0 (4B 이상).
    None이면 비활성."""
    completion_only_loss: bool = True
    """assistant 응답 토큰에만 loss를 계산합니다. prompt 토큰(system/user)의
    loss를 마스킹하여 모델이 응답 생성에 집중하도록 합니다.
    소규모 데이터에서 과적합을 억제하는 데 효과적입니다."""

    @model_validator(mode="after")
    def _check_training_params(self) -> "TrainingConfig":
        """학습 파라미터의 유효성을 검증합니다."""
        if not (0.0 < self.train_split < 1.0):
            raise ValueError(f"train_split({self.train_split})은 0과 1 사이여야 합니다")
        if self.learning_rate != "auto" and self.learning_rate <= 0:
            raise ValueError(f"learning_rate({self.learning_rate})는 양수여야 합니다")
        return self


class OllamaExportConfig(BaseModel):
    """Ollama 특화 내보내기 설정입니다."""

    enabled: bool = True
    model_name: str = Field("my-project-model", min_length=1)
    system_prompt: str = _EN_DEFAULT_OLLAMA_SYSTEM_PROMPT
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 4096,
        }
    )

    @model_validator(mode="after")
    def _sanitize_model_name(self) -> "OllamaExportConfig":
        from pathlib import PurePosixPath

        if "/" in self.model_name:
            sanitized = PurePosixPath(self.model_name).name
            if sanitized:
                self.model_name = sanitized
        return self


class ExportConfig(BaseModel):
    """모델 내보내기 및 패키징 설정입니다."""

    merge_lora: bool = True
    output_format: str = "safetensors"
    ollama: OllamaExportConfig = Field(default_factory=OllamaExportConfig)


# ---------------------------------------------------------------------------
# 신규 기능 설정
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """학습된 모델의 자동 평가 설정입니다."""

    enabled: bool = True
    test_split: float = 0.1
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge", "llm_judge"])
    max_samples: int = 20
    max_tokens: int = 512
    """Ollama 생성 최대 토큰 수. 평가 시 응답 길이를 제한하여 무한 생성을 방지합니다."""
    output_file: str = "eval_results.json"
    quality_gate: bool = True
    """품질 게이트 활성화 여부입니다. True이면 평가 결과가 임계값 미달 시 경고합니다."""
    quality_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"bleu": 0.1, "rougeL": 0.2}
    )
    """메트릭별 최소 통과 임계값입니다. 평균 점수가 이 값 미만이면 품질 게이트 실패입니다."""
    llm_judge_model: str = ""
    """LLM-as-Judge에 사용할 모델입니다. 빈 문자열이면 teacher.model을 사용합니다."""

    @model_validator(mode="after")
    def _check_eval_params(self) -> "EvalConfig":
        """평가 설정의 유효성을 검증합니다."""
        if not (0.0 < self.test_split < 1.0):
            raise ValueError(f"test_split({self.test_split})은 0과 1 사이여야 합니다")
        if self.max_samples < 1:
            raise ValueError(f"max_samples({self.max_samples})는 1 이상이어야 합니다")
        return self


class RefinementConfig(BaseModel):
    """Iterative Refinement 설정입니다.

    학습 후 평가에서 약점이 발견된 QA에 대해
    Teacher LLM으로 추가 학습 데이터를 생성하고 재학습합니다.
    """

    enabled: bool = False
    """Refinement 활성화 여부입니다. 기본 비활성."""

    max_rounds: int = 1
    """최대 Refinement 반복 횟수입니다."""

    llm_judge_threshold: float = 0.6
    """이 점수 미만인 QA를 약점으로 판단합니다 (0.0~1.0)."""

    @model_validator(mode="after")
    def _check_refinement_params(self) -> "RefinementConfig":
        """Refinement 설정의 유효성을 검증합니다."""
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds({self.max_rounds})는 1 이상이어야 합니다")
        if not (0.0 <= self.llm_judge_threshold <= 1.0):
            raise ValueError(
                f"llm_judge_threshold({self.llm_judge_threshold})는 "
                "0.0~1.0 범위여야 합니다"
            )
        return self


class AgentModelsConfig(BaseModel):
    """Phase 9 — Agent 각 컴포넌트별 모델 라우팅.

    각 슬롯은 선택적이며, 빈 문자열이면 ``rag.ollama_model`` (기본 synthesis 모델)로
    fallback합니다. 사용자가 여러 크기의 Ollama 모델을 보유했을 때 라이트한 작업은
    작은 모델, 최종 답변은 큰 모델로 분리하여 성능·비용을 최적화합니다.
    """

    router_model: str = ""
    """IntentClassifier용 — 빠른 분류. 작은 모델 권장."""
    planner_model: str = ""
    """Planner용 — JSON 구조 생성. 중급 모델."""
    synthesis_model: str = ""
    """최종 답변 합성용. 가장 큰 모델 권장."""
    verifier_model: str = ""
    """Verifier 충분성 판정용. 작은 모델."""
    reviewer_model: str = ""
    """Review-Work 3개 checker용. 중급 모델."""
    reflector_model: str = ""
    """Reflector 답변 자기 검증용. 중급 모델."""
    clarifier_model: str = ""
    """Clarifier 명확화 질문 생성용. 중급 모델."""
    scorer_model: str = ""
    """AnswerScorer 정량 평가용. 중급 모델."""


class AgentRagConfig(BaseModel):
    """Agent RAG 모드 설정입니다.

    활성화하면 LLM이 ReAct 루프를 통해 도구를 반복 사용하여
    복합 질문에 다단계 검색·추론으로 답변합니다.
    """

    enabled: bool = True
    """Agent 모드 활성화 여부. ``false``이면 기존 단일 패스 RAG를 사용합니다."""

    max_iterations: int = 5
    """ReAct 루프 최대 반복 횟수. 무한 루프를 방지합니다."""

    session_ttl: int = 3600
    """세션 유지 시간 (초). 이 시간 동안 대화 내역을 보존합니다."""

    max_history_turns: int = 20
    """대화 내역 최대 턴 수. 초과 시 오래된 턴을 제거합니다."""

    stream_reasoning: bool = True
    """추론 과정(Thought/Action/Observation)을 클라이언트에 실시간 스트리밍합니다."""

    web_search_for_general: bool = True
    """``general`` 라우팅 경로에서 합성 전 DuckDuckGo로 외부 사실 조회.
    시기 의존·시사·통계 질의의 정확도 향상을 위함. ``false``로 두면 LLM
    학습 지식만으로 답변 — 최신 정보·고유명사에 부정확할 수 있음.
    인터넷이 차단된 환경에서는 ``false`` 권장."""

    native_thinking: bool = True
    """Ollama native thinking(Qwen3/DeepSeek-R1 등)을 **판정 경로**에만 활성화합니다.

    적용 대상 — Planner, Verifier, Reflector 3곳.
    최종 답변 합성(``_stream_synthesis``)은 thinking 누적으로 스트리밍 레이턴시가
    폭증하는 문제가 실측되어 의도적으로 제외됩니다. ReAct AgentLoop / QA 생성 /
    Scorer / Augmenter / 단순 RAG 경로는 JSON 파싱·포맷 라벨 안정성과 배치
    처리량을 위해 항상 ``think=False``로 고정됩니다(이 플래그의 영향을 받지 않습니다).

    Qwen3·DeepSeek-R1 등 thinking 지원 모델이 아니면 효과가 없습니다.
    thinking이 켜져도 기존 ``<think>`` 태그 제거·fallback 로직은 유지되므로
    비지원 모델에서도 안전합니다."""

    persist_sessions: bool = False
    """세션을 파일 시스템에 영속화합니다. ``true``이면 서버 재시작 후에도 대화 내역이 유지됩니다."""

    sessions_dir: str = "agent_sessions"
    """세션 JSON 파일 저장 경로. ``paths.output`` 하위에 생성됩니다."""

    planner_enabled: bool = False
    """Planner 기반 오케스트레이션 활성화. ``true``이면 plan → execute → verify → synthesize 경로를 사용하고, ``false``이면 기존 ReAct AgentLoop를 사용합니다."""

    verifier_enabled: bool = True
    """Planner 경로에서 Verifier로 컨텍스트 충분성을 판정합니다. ``planner_enabled=false``이면 무시됩니다."""

    verifier_max_repairs: int = 1
    """Verifier가 허용하는 추가 검색(repair)의 최대 횟수."""

    legacy_fallback_enabled: bool = True
    """Planner가 구조적으로 실패(``is_fallback=True``)할 때 기존 ReAct ``AgentLoop``로 자동 전환합니다.
    ``false``로 설정하면 planner의 fallback 계획(단일 search)을 그대로 실행합니다."""

    session_source_reuse: bool = True
    """같은 세션의 이전 턴에서 참조한 문서를 다음 질의의 synthesis 프롬프트에 주입합니다.
    follow-up 질의의 대화 연속성을 높입니다. ``planner_enabled=true``에서만 작동합니다."""

    session_source_reuse_limit: int = 5
    """이전 턴에서 재사용할 참조 문서의 최대 개수."""

    parallel_steps: bool = False
    """Planner plan이 2개 이상의 ``search`` step으로 구성된 경우 이를 ``asyncio.gather``로 동시 실행합니다.
    이벤트는 plan 순서대로 emit되어 SSE 계약이 유지됩니다. ``search`` 이외의 도구 또는 단일 step에서는 무시됩니다."""

    reflector_enabled: bool = False
    """Reflector (답변 자기 검증) 활성화. 답변 생성 후 LLM이 품질·근거를 평가하고
    부족하면 추가 검색으로 재시도합니다. ``planner_enabled=true``에서만 작동합니다."""

    reflector_max_retries: int = 1
    """Reflector가 답변을 거부할 때 허용할 추가 검색·재합성 횟수."""

    intent_classifier_enabled: bool = False
    """IntentClassifier (LLM 기반 의도 분류) 활성화. ``/auto`` 라우팅에 키워드 휴리스틱
    대신 LLM 분류를 사용합니다. 실패 시 키워드 fallback."""

    intent_classifier_cache_ttl: int = 300
    """같은 질의에 대한 의도 분류 결과 캐싱 시간(초). 0이면 비활성."""

    clarifier_enabled: bool = False
    """Clarifier persona 활성화. IntentClassifier가 ``ambiguous``로 분류한 질의에
    답변 대신 명확화 질문을 반환합니다. ``intent_classifier_enabled=true``에서만 작동."""

    clarifier_max_questions: int = 2
    """Clarifier가 반환할 질문의 최대 개수 (1~3 권장)."""

    personas_enabled: bool = False
    """Persona 시스템 활성화. intent별 전용 synthesis prompt + 도구 화이트리스트 사용.
    ``intent_classifier_enabled=true``와 ``planner_enabled=true``에서 효과적."""

    custom_personas_dir: str = ""
    """Phase 14 — 사용자 정의 persona YAML 디렉터리. 빈 문자열이면 비활성. 이 디렉터리의
    persona가 built-in persona보다 우선 적용됩니다."""

    review_work_enabled: bool = False
    """Review-Work 병렬 검증 활성화. 답변 생성 후 3개 reviewer(grounding, completeness,
    hallucination)가 병렬로 검증하여 품질 이벤트를 발행합니다. ``planner_enabled=true``에서만 작동."""

    review_work_retry: bool = False
    """Review-Work 실패 시 추가 검색 + 재합성 자동 재시도. ``review_work_enabled=true``에서만 효과."""

    self_improvement_enabled: bool = False
    """Phase 13 — AnswerScorer 기반 자기 개선 루프 활성화. 답변 점수가 ``min_quality_score``
    미만이면 구체 피드백을 주입해 재합성을 시도합니다."""

    min_quality_score: float = 7.0
    """이 점수 미만이면 self-improvement 재시도 (1~10 범위)."""

    max_self_improvement_iterations: int = 1
    """self-improvement의 최대 재시도 횟수."""

    memory_compression_enabled: bool = False
    """Phase 12 — 긴 대화 자동 요약 활성화. ``compress_after_turns`` 초과 시 가장 오래된
    대화를 LLM 요약 한 줄로 압축하여 토큰 예산을 관리합니다."""

    compress_after_turns: int = 10
    """이 턴 수를 초과하면 오래된 턴 압축을 시도합니다 (user+assistant 쌍 기준)."""

    compress_target_chars: int = 500
    """요약 결과의 목표 길이 (문자 수)."""

    hooks_enabled: bool = False
    """Hooks 시스템 활성화. 파이프라인 주요 지점(pre_query, post_search, post_synthesis)
    에서 등록된 hook을 실행합니다."""

    builtin_hooks: list[str] = Field(default_factory=list)
    """자동 등록할 내장 hook 이름 목록. 예: ``["normalize_korean_whitespace",
    "dedup_sources_by_doc_id", "strip_html_from_answer"]``."""

    skills_enabled: bool = False
    """Skills 시스템 활성화. ``skills_dir``의 YAML 파일을 로드하여 질의의 trigger에
    매칭되는 skill의 ``prompt_addon``을 synthesis prompt에 자동 주입합니다."""

    skills_dir: str = "skills"
    """Skill YAML 파일이 위치한 디렉터리. 상대 경로는 프로젝트 루트 기준."""

    models: AgentModelsConfig = Field(default_factory=AgentModelsConfig)
    """컴포넌트별 모델 라우팅. 빈 슬롯은 ``rag.ollama_model``로 fallback."""

    observation_preview_limit: int = 300
    """observation 이벤트로 클라이언트에 보낼 때의 길이 제한(문자)."""

    ollama_keep_alive: str = "5m"
    """모든 LLM 호출에 사용할 Ollama keep_alive 값. 모델 워밍업 유지 시간을
    의미합니다. 기본 ``"5m"``은 5분 유지, ``"-1"`` 또는 ``-1`` 은 무한 유지,
    ``"0"`` 은 즉시 언로드."""

    # ------------------------------------------------------------------
    # Ralph Loop — oh-my-openagent의 self-referential loop을 RAG에 적응
    # ------------------------------------------------------------------

    ralph_loop_enabled: bool = False
    """Ralph 스타일 통합 품질 루프 활성화. ``true``이면 기존 reflector→review-work→
    self-improvement 직렬 체인 대신 단일 통합 루프(synthesize → gates → 완료
    약속(promise) 발행 → 종료, 실패 시 보완 검색 + 재합성)를 사용합니다.
    ``planner_enabled=true``에서만 작동하며 기본 비활성으로 기존 동작을 보존합니다."""

    ralph_loop_max_iterations: int = 5
    """Ralph 루프의 최대 반복 횟수. 각 반복은 합성 + 게이트 평가를 포함하므로
    LLM 호출이 다수 발생합니다. 권장 3~5."""

    ralph_loop_quality_threshold: float = 7.0
    """Ralph 루프에서 통과로 간주할 scorer 점수 임계값(1~10)."""

    ralph_loop_strategy: Literal["reset", "continue"] = "continue"
    """Ralph 루프의 피드백 누적 전략. ``continue``는 이전 반복의 피드백을 누적해
    프롬프트에 모두 주입(점진적 개선), ``reset``은 매 반복마다 가장 최근 피드백만
    사용(트래킹 단순)."""

    ralph_loop_state_dir: str = ""
    """Ralph 루프 상태 JSON 파일 디렉터리(빈 문자열이면 비영속). 세션별
    ``<sid>.json``으로 저장하여 다음 턴에서 재개 가능. ``paths.output`` 기준 상대 경로."""

    ralph_loop_completion_promise: str = "DONE"
    """모든 게이트를 통과했을 때 최종 답변 직전에 발행하는 promise 토큰.
    oh-my-openagent의 ``<promise>DONE</promise>`` 컨벤션을 따릅니다."""

    intent_verbalization_enabled: bool = False
    """라우팅 결정 직후 의도(IntentClassifier 결과)를 짧은 thought 이벤트로
    명시 발화합니다. oh-my-openagent의 'Verbalize Intent' 패턴을 따라 라우팅
    결정의 투명성을 높이고 follow-up 처리 일관성을 확보합니다."""

    quality_mode: bool = False
    """**고품질 답변 프리셋 (Ralph 통합)** — ``true``이면 Ralph 통합 quality
    loop을 활성화하고 의도 발화·세션 재사용·legacy fallback을 함께 켭니다.
    ``ultra_mode``와 달리 reflector/review-work/self-improvement 직렬 chain을
    실행하지 않고, 매 반복마다 reflector + reviewers + scorer를 **병렬 평가**
    하는 단일 통합 루프를 사용합니다.

    LLM 호출 수가 많으므로(반복당 약 6~7회) 답변 합성 모델과 판정 모델 분리,
    반복 횟수 조정이 중요합니다. 권장 설정:
    - ``models.synthesis_model``: 가장 큰 모델 (예: ``qwen3.5:35b-a3b``)
    - ``models.reviewer_model`` / ``models.scorer_model``: 중급 판정 모델
    - ``ralph_loop_max_iterations``: 2~3 권장 (기본 5는 latency 부담)
    """

    smart_mode: bool = False
    """**원클릭 프리셋 (P0)** — ``true``이면 Phase 5+6+8+11 + 기반 (Planner/Verifier/
    Reflector/Legacy fallback)이 자동 활성화."""

    ultra_mode: bool = False
    """**최대 성능 프리셋 (Phase 15b)** — ``smart_mode`` + P1/P2 전체 (Hooks, Memory
    Compression, Self-Improvement, Review-Work retry). 가장 똑똑하지만 LLM 호출 수·지연
    증가. 파일 기반 기능(skills, custom_personas)은 디렉터리 지정 시에만 활성화됩니다."""

    @model_validator(mode="after")
    def _apply_smart_mode(self) -> "AgentRagConfig":
        """smart_mode/ultra_mode/quality_mode 프리셋을 개별 플래그에 cascade."""
        if self.quality_mode:
            # 고품질 답변 프리셋 — Ralph 통합 quality loop을 진입점으로 사용.
            # ralph_loop_enabled가 켜지면 orchestrator가 reflector/review-work/
            # self-improvement 직렬 chain을 자동 우회하므로, 그 플래그들은
            # 강제로 켤 필요가 없습니다(켜져 있어도 ralph_active 게이트로 무시).
            self.planner_enabled = True
            self.verifier_enabled = True
            self.legacy_fallback_enabled = True
            self.intent_classifier_enabled = True
            self.intent_verbalization_enabled = True
            self.clarifier_enabled = True
            self.personas_enabled = True
            self.session_source_reuse = True
            # ralph_loop_enabled는 사용자가 명시적으로 끌 수 있도록 cascade가
            # 강제하지 않습니다 (예: latency 우선으로 quality_mode + ralph OFF 구성).
            # 다른 cascade 필드(planner/verifier 등)는 quality_mode의 정의상 강제 ON.
            if "ralph_loop_enabled" not in self.model_fields_set:
                self.ralph_loop_enabled = True
            # 사용자가 명시한 max_iterations는 1 이상이면 그대로 존중 (권고는 2~3,
            # 단 latency 우선 시 1도 유효 — retry 없이 단일 평가만 수행).
            if self.ralph_loop_max_iterations < 1:
                self.ralph_loop_max_iterations = 3
        if self.smart_mode or self.ultra_mode:
            # P0 = Phase 5 + 6 + 8 + 11 + 기반 Phase들(1c planner/verifier, 2 fallback, 4 reflector)
            self.intent_classifier_enabled = True
            self.clarifier_enabled = True
            self.personas_enabled = True
            self.review_work_enabled = True
            self.planner_enabled = True
            self.verifier_enabled = True
            self.reflector_enabled = True
            self.legacy_fallback_enabled = True
        if self.ultra_mode:
            # Ultra = P0 + P1/P2 전체
            self.hooks_enabled = True
            self.memory_compression_enabled = True
            self.self_improvement_enabled = True
            self.review_work_retry = True
            self.session_source_reuse = True
            # ultra_mode는 retry 폭발을 막기 위해 retry/iter 한도를 1로 권고합니다.
            # 사용자가 명시적으로 더 큰 값을 설정하면 경고만 출력하고 그대로 둡니다.
            if self.reflector_max_retries > 1:
                logging.getLogger("slm_factory.config").warning(
                    "ultra_mode와 reflector_max_retries=%d가 동시에 설정되었습니다 — "
                    "답변 재합성이 누적되어 지연·중복이 커질 수 있습니다.",
                    self.reflector_max_retries,
                )
            else:
                self.reflector_max_retries = 1
            if self.max_self_improvement_iterations > 1:
                logging.getLogger("slm_factory.config").warning(
                    "ultra_mode와 max_self_improvement_iterations=%d가 동시에 설정되었습니다 — "
                    "self-improvement 루프가 답변 지연을 키울 수 있습니다.",
                    self.max_self_improvement_iterations,
                )
            else:
                self.max_self_improvement_iterations = 1
            # skills_enabled, custom_personas_dir는 디렉터리 지정 시에만 의미가 있으므로 건드리지 않음.
        return self

    @model_validator(mode="after")
    def _check_agent_params(self) -> "AgentRagConfig":
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations({self.max_iterations})는 1 이상이어야 합니다"
            )
        if self.session_ttl < 0:
            raise ValueError(
                f"session_ttl({self.session_ttl})은 0 이상이어야 합니다"
            )
        if self.max_history_turns < 1:
            raise ValueError(
                f"max_history_turns({self.max_history_turns})는 1 이상이어야 합니다"
            )
        if self.verifier_max_repairs < 0:
            raise ValueError(
                f"verifier_max_repairs({self.verifier_max_repairs})는 0 이상이어야 합니다"
            )
        return self


class RagConfig(BaseModel):
    """RAG 서비스 구축 설정입니다.

    ``corpus.parquet``을 Qdrant에 임베딩하여 적재하고,
    Ollama SLM과 연동하는 RAG API 서버를 실행합니다.
    """

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    """임베딩 모델 이름 (sentence-transformers 호환)."""

    vector_db_path: str = "qdrant_db"
    """Qdrant 저장 경로. ``paths.output`` 하위에 생성됩니다."""

    collection_name: str = "corpus"
    """Qdrant 컬렉션 이름."""

    top_k: int = 5
    """검색 시 반환할 최대 문서 청크 수."""

    batch_size: int = 64
    """Qdrant 인덱싱 시 배치 크기."""

    server_host: str = "0.0.0.0"
    """RAG API 서버 바인드 호스트."""

    server_port: int = 8000
    """RAG API 서버 포트."""

    workers: int = 1
    """uvicorn 워커 프로세스 수."""

    log_level: str = "info"
    """uvicorn 로그 레벨."""

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    """CORS 허용 오리진 목록."""

    request_timeout: float = 120.0
    """RAG 질의 타임아웃 (초)."""

    max_tokens: int = -1
    """Ollama 생성 최대 토큰 수. -1이면 모델이 자연스럽게 종료할 때까지 생성합니다."""

    ollama_model: str = ""
    """Ollama 모델명. 빈 문자열이면 ``export.ollama.model_name``을 사용합니다."""

    reranker_enabled: bool = True
    """검색 결과를 cross-encoder로 재정렬하여 정확도를 높입니다."""

    reranker_model: str = "dragonkue/bge-reranker-v2-m3-ko"
    """Reranker 모델명 (sentence-transformers CrossEncoder 호환).
    한국어 파인튜닝된 bge-reranker-v2-m3-ko가 기본값입니다.
    한국어 금융 벤치마크에서 F1 0.9123 (원본 bge-v2-m3: 0.8772)."""

    hybrid_search: bool = True
    """벡터 검색과 키워드 검색(BM25)을 결합하여 검색 재현율을 높입니다."""

    query_rewriting: bool = False
    """짧은 질의를 LLM으로 확장하여 검색 품질을 향상합니다."""

    min_score: float = 0.0
    """검색 결과 최소 유사도 점수 (0.0~1.0). 이 값 미만의 문서는 컨텍스트에서 제외됩니다.
    0.0으로 설정하면 필터링을 비활성화합니다."""

    agent: AgentRagConfig = Field(default_factory=AgentRagConfig)
    """Agent RAG 모드 설정. ``agent.enabled: true``로 활성화합니다."""

    corpus_profile: "CorpusProfileConfig" = Field(
        default_factory=lambda: CorpusProfileConfig()
    )
    """이 corpus가 어떤 도메인을 다루는지의 자기 기술 — 라우팅·합성 컨텍스트로 사용."""

    @model_validator(mode="after")
    def _check_rag_params(self) -> "RagConfig":
        """RAG 설정의 유효성을 검증합니다."""
        if self.top_k < 1:
            raise ValueError(f"top_k({self.top_k})는 1 이상이어야 합니다")
        if self.server_port < 1 or self.server_port > 65535:
            raise ValueError(
                f"server_port({self.server_port})는 1~65535 범위여야 합니다"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size({self.batch_size})는 1 이상이어야 합니다")
        if self.workers < 1:
            raise ValueError(f"workers({self.workers})는 1 이상이어야 합니다")
        if self.request_timeout <= 0:
            raise ValueError(
                f"request_timeout({self.request_timeout})은 0보다 커야 합니다"
            )
        return self


class ContextualRetrievalConfig(BaseModel):
    """Anthropic Contextual Retrieval 설정입니다.

    각 청크에 대해 Teacher LLM이 "이 청크가 전체 문서에서 어떤 위치에
    어떤 맥락으로 위치하는지"를 50~100토큰으로 생성하여 청크 본문 앞에
    덧붙입니다. 임베딩·BM25·리랭커 모두에 동일한 prefix가 적용되어
    검색 실패율이 평균 ~35% (단독), ~49% (BM25+리랭커 결합) 감소합니다.

    LLM 호출이 청크 수만큼 발생하므로 인덱싱 시간·비용이 증가합니다.
    Teacher LLM은 ``teacher`` 설정을 그대로 재사용합니다 (Ollama 사용 시 추가 비용 0).
    """

    enabled: bool = False
    """Contextual Retrieval 활성화 여부. 기본 비활성."""

    max_concurrency: int = 4
    """청크 컨텍스트 생성의 동시 호출 수."""

    max_chars: int = 300
    """생성된 컨텍스트의 최대 길이(문자). 초과 시 잘립니다."""

    doc_truncate_chars: int = 8000
    """LLM에 전달할 부모 문서의 최대 길이. 너무 길면 컨텍스트가 부정확해지고
    Teacher 컨텍스트 윈도우를 초과할 수 있어 절단합니다."""

    skip_short_docs: int = 800
    """이 길이 미만의 문서는 청크가 1개뿐이므로 컨텍스트 생성을 생략합니다."""

    @model_validator(mode="after")
    def _check_contextual_retrieval_params(self) -> "ContextualRetrievalConfig":
        if self.max_concurrency < 1:
            raise ValueError(
                f"max_concurrency({self.max_concurrency})는 1 이상이어야 합니다"
            )
        if self.max_chars < 50:
            raise ValueError(f"max_chars({self.max_chars})는 50 이상이어야 합니다")
        if self.doc_truncate_chars < 500:
            raise ValueError(
                f"doc_truncate_chars({self.doc_truncate_chars})는 500 이상이어야 합니다"
            )
        return self


class CorpusProfileConfig(BaseModel):
    """RAG corpus의 도메인 자기 기술 설정입니다.

    인덱싱 후 첫 N개 청크를 표본으로 LLM이 corpus profile을 자동 생성하여
    ``<paths.output>/<profile_path>``에 저장합니다. 서버 시작 시 로드되어
    IntentClassifier·합성 프롬프트의 도메인 컨텍스트로 주입됩니다.

    사용자 override(``name``/``summary``/``keywords``)가 비어 있지 않으면
    자동 생성 결과보다 우선합니다.
    """

    enabled: bool = True
    """corpus profile 사용 여부. false이면 IntentClassifier에 도메인 컨텍스트 주입 없음."""

    auto_generate: bool = True
    """서버 시작 시 profile 파일이 없으면 자동 생성. false이면 사용자 override만 사용."""

    profile_path: str = "corpus_profile.json"
    """profile JSON 파일 경로 (``paths.output`` 하위)."""

    sample_size: int = 8
    """자동 생성 시 LLM에 전달할 표본 청크 수."""

    sample_strategy: str = "stratified"
    """표본 추출 전략 — ``head``(처음 N개) | ``stratified``(균등 분포 N개) | ``random``."""

    name: str = ""
    """사용자 override — 도메인 명칭. 빈 문자열이면 자동 생성 결과 사용."""

    summary: str = ""
    """사용자 override — 도메인 요약. 빈 문자열이면 자동 생성 결과 사용."""

    keywords: list[str] = Field(default_factory=list)
    """사용자 override — 도메인 핵심 키워드. 비어 있지 않으면 자동 생성 결과를 완전 대체."""

    @model_validator(mode="after")
    def _check_corpus_profile_params(self) -> "CorpusProfileConfig":
        if self.sample_size < 1:
            raise ValueError(f"sample_size({self.sample_size})는 1 이상이어야 합니다")
        if self.sample_strategy not in {"head", "stratified", "random"}:
            raise ValueError(
                f"sample_strategy({self.sample_strategy})는 head/stratified/random 중 하나여야 합니다"
            )
        return self


class AutoRAGExportConfig(BaseModel):
    """AutoRAG 연동을 위한 데이터 내보내기 설정입니다.

    slm-factory의 파싱된 문서와 QA 쌍을 AutoRAG 평가용
    ``corpus.parquet`` + ``qa.parquet`` 형식으로 변환합니다.
    """

    enabled: bool = True
    output_dir: str = "autorag"
    chunk_size: int = 512
    overlap_chars: int = 64
    contextual_retrieval: ContextualRetrievalConfig = Field(
        default_factory=ContextualRetrievalConfig
    )
    """Anthropic Contextual Retrieval 설정. ``enabled: true``로 활성화하면
    각 청크에 LLM 생성 컨텍스트가 prefix로 추가됩니다."""

    @model_validator(mode="after")
    def _check_autorag_export_params(self) -> "AutoRAGExportConfig":
        """AutoRAG 내보내기 설정의 유효성을 검증합니다."""
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size({self.chunk_size})는 100 이상이어야 합니다")
        if self.overlap_chars >= self.chunk_size:
            raise ValueError(
                f"overlap_chars({self.overlap_chars})는 "
                f"chunk_size({self.chunk_size})보다 작아야 합니다"
            )
        return self


class IncrementalConfig(BaseModel):
    """증분 학습 설정입니다."""

    enabled: bool = True
    hash_file: str = "document_hashes.json"
    merge_strategy: Literal["append", "replace"] = "append"
    resume_adapter: str = ""


class ReviewConfig(BaseModel):
    """QA 수동 리뷰 설정입니다."""

    enabled: bool = True
    auto_open: bool = True
    output_file: str = "qa_reviewed.json"


class CompareConfig(BaseModel):
    """모델 비교 설정입니다."""

    enabled: bool = False
    base_model: str = ""
    finetuned_model: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge"])
    max_samples: int = 20
    max_tokens: int = 512
    """Ollama 생성 최대 토큰 수. 비교 시 응답 길이를 제한하여 무한 생성을 방지합니다."""
    output_file: str = "compare_results.json"

    @model_validator(mode="after")
    def _check_compare_params(self) -> "CompareConfig":
        """비교 설정의 유효성을 검증합니다."""
        if self.max_samples < 1:
            raise ValueError(f"max_samples({self.max_samples})는 1 이상이어야 합니다")
        if self.enabled and not self.base_model:
            raise ValueError("compare.enabled가 true일 때 base_model은 필수입니다")
        if self.enabled and not self.finetuned_model:
            raise ValueError("compare.enabled가 true일 때 finetuned_model은 필수입니다")
        return self


class EvolveConfig(BaseModel):
    """자동 진화 파이프라인 설정입니다.

    ``tool evolve`` 명령의 동작을 제어합니다.
    증분 업데이트 → 전체 재학습 → 품질 게이트 → 조건부 배포를
    단일 명령으로 실행합니다.
    """

    quality_gate: bool = True
    """품질 게이트 활성화 여부입니다. True이면 이전 모델보다 나은 경우에만 배포합니다."""

    gate_metric: str = "rougeL"
    """품질 비교에 사용할 메트릭입니다 (bleu, rouge1, rouge2, rougeL)."""

    gate_min_improvement: float = 0.0
    """최소 개선율(%)입니다. 0이면 어떤 개선이든 통과합니다."""

    version_format: str = "date"
    """버전 형식입니다. 현재 ``date`` (vYYYYMMDD) 형식만 지원합니다."""

    history_file: str = "evolve_history.json"
    """진화 히스토리 파일명입니다 (출력 디렉토리에 저장)."""

    keep_previous_versions: int = 3
    """보관할 이전 버전 수입니다. 초과분은 ``ollama rm``으로 제거됩니다."""

    @model_validator(mode="after")
    def _check_evolve_params(self) -> "EvolveConfig":
        """진화 설정의 유효성을 검증합니다."""
        valid_metrics = {"bleu", "rouge1", "rouge2", "rougeL"}
        if self.gate_metric not in valid_metrics:
            raise ValueError(
                f"gate_metric({self.gate_metric})이 올바르지 않습니다. "
                f"지원: {', '.join(sorted(valid_metrics))}"
            )
        if self.keep_previous_versions < 0:
            raise ValueError(
                f"keep_previous_versions({self.keep_previous_versions})는 "
                "0 이상이어야 합니다"
            )
        return self


class ChunkingConfig(BaseModel):
    """문서 청킹 설정입니다.

    긴 문서를 중첩(overlap)이 있는 청크로 분할하여 QA 생성 품질을 향상합니다.
    ``max_context_chars``보다 짧은 문서는 청킹하지 않습니다.
    """

    enabled: bool = True
    chunk_size: int | Literal["auto"] = "auto"
    """각 청크의 최대 문자 수입니다. ``"auto"``이면 문서 분석으로 자동 결정합니다."""

    overlap_chars: int = 500
    """연속된 청크 간 중첩 문자 수입니다. 문맥 연속성을 유지합니다."""

    @model_validator(mode="after")
    def _check_chunking_params(self) -> "ChunkingConfig":
        """청킹 파라미터의 유효성을 검증합니다."""
        if self.overlap_chars < 0:
            raise ValueError(
                f"overlap_chars({self.overlap_chars})는 0 이상이어야 합니다"
            )
        if self.chunk_size == "auto":
            return self
        if self.chunk_size < 1000:
            raise ValueError(f"chunk_size({self.chunk_size})는 1000 이상이어야 합니다")
        if self.overlap_chars >= self.chunk_size:
            raise ValueError(
                f"overlap_chars({self.overlap_chars})는 "
                f"chunk_size({self.chunk_size})보다 작아야 합니다"
            )
        return self


# ---------------------------------------------------------------------------
# 루트 설정
# ---------------------------------------------------------------------------


class SLMConfig(BaseModel):
    """slm-factory 프로젝트의 루트 설정 객체입니다.

    전체 ``project.yaml`` 스키마를 반영합니다. :func:`load_config`를 통해
    또는 딕셔너리/YAML에서 직접 인스턴스화됩니다.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    analyzer: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    autorag_export: AutoRAGExportConfig = Field(default_factory=AutoRAGExportConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)
    evolve: EvolveConfig = Field(default_factory=EvolveConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @model_validator(mode="before")
    @classmethod
    def _strip_none_sections(cls, values: dict[str, Any]) -> dict[str, Any]:
        """값이 ``None``인 최상위 키를 제거하여 기본값이 적용되도록 합니다."""
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if v is not None}
        return values

    @model_validator(mode="after")
    def _apply_language_defaults(self) -> "SLMConfig":
        if self.project.language != "ko":
            return self
        if self.questions.system_prompt == _EN_DEFAULT_QA_SYSTEM_PROMPT:
            self.questions.system_prompt = _KO_DEFAULT_QA_SYSTEM_PROMPT
        if self.export.ollama.system_prompt == _EN_DEFAULT_OLLAMA_SYSTEM_PROMPT:
            self.export.ollama.system_prompt = _KO_DEFAULT_OLLAMA_SYSTEM_PROMPT
        return self


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = "templates/project.yaml"


def load_config(path: str | Path) -> SLMConfig:
    """프로젝트 YAML 설정 파일을 로드하고 검증합니다.

    ``paths.documents``와 ``paths.output``이 상대 경로인 경우,
    설정 파일이 위치한 디렉토리를 기준으로 절대 경로로 변환합니다.

    매개변수
    ----------
    path:
        ``project.yaml`` 파일의 파일시스템 경로입니다.

    반환값
    -------
    SLMConfig
        완전히 검증된 설정 객체입니다.

    예외
    ------
    FileNotFoundError
        *path*가 존재하지 않으면 발생합니다.
    yaml.YAMLError
        파일이 유효한 YAML이 아니면 발생합니다.
    pydantic.ValidationError
        YAML 내용이 예상 스키마와 일치하지 않으면 발생합니다.
    """
    filepath = Path(path).resolve()
    if not filepath.is_file():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    raw = yaml.safe_load(filepath.read_text(encoding="utf-8")) or {}
    config = SLMConfig.model_validate(raw)

    config_dir = filepath.parent
    if not config.paths.documents.is_absolute():
        config.paths.documents = (config_dir / config.paths.documents).resolve()
    if not config.paths.output.is_absolute():
        config.paths.output = (config_dir / config.paths.output).resolve()

    return config


def create_default_config() -> str:
    """기본 YAML 프로젝트 템플릿을 문자열로 반환합니다.

    패키지와 함께 제공되는 ``templates/project.yaml``을 읽습니다.
    개발 환경 (편집 가능 설치)과 빌드된 wheel 모두에서 작동하도록
    패키지 리소스 조회로 폴백합니다.
    """
    # 먼저 형제 경로 시도 (소스 / 편집 가능 설치에서 작동)
    pkg_root = Path(__file__).resolve().parent.parent.parent
    template = pkg_root / _TEMPLATE_PATH
    if template.is_file():
        return template.read_text(encoding="utf-8")

    # 폴백: importlib.resources (설치된 wheel)
    try:
        ref = importlib.resources.files("slm_factory").joinpath(
            "../../" + _TEMPLATE_PATH
        )
        return ref.read_text(encoding="utf-8")
    except Exception as e:
        logging.getLogger("slm_factory.config").debug(
            "importlib.resources에서 템플릿 로드 실패: %s", e
        )

    # 최후의 수단: 최소한의 내장 기본값 반환 (YAML 형식)
    return yaml.dump(
        SLMConfig().model_dump(),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
