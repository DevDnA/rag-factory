"""여러 모듈에서 공유하는 핵심 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """모든 문서 파서의 구조화된 출력입니다."""

    doc_id: str
    """파일명 기반 고유 식별자입니다."""

    title: str
    """문서 제목 (메타데이터 또는 파일명에서 추출)입니다."""

    content: str
    """마크다운 형식의 전체 추출 텍스트입니다."""

    tables: list[str] = field(default_factory=list)
    """마크다운 문자열로 추출된 표입니다."""

    metadata: dict = field(default_factory=dict)
    """임의의 메타데이터: 날짜, 저자, 페이지 수 등입니다."""


@dataclass
class QAPair:
    """메타데이터를 포함한 단일 질문-답변 쌍."""

    question: str
    answer: str
    instruction: str = ""
    source_doc: str = ""
    category: str = ""
    context: str = ""
    """QA 생성에 사용된 문서 청크입니다. 학습 시 user 메시지에 포함되어
    모델이 컨텍스트를 활용하는 스타일을 학습합니다."""
    is_augmented: bool = False
    content_hash: str = ""
    review_status: str = ""


@dataclass
class EvalResult:
    """단일 평가 결과."""

    question: str
    reference_answer: str
    generated_answer: str
    scores: dict = field(default_factory=dict)


@dataclass
class CompareResult:
    """모델 비교 결과."""

    question: str
    reference_answer: str
    base_answer: str
    finetuned_answer: str
    scores: dict = field(default_factory=dict)
