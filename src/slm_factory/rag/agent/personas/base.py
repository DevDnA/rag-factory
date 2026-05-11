"""Persona 공통 인터페이스 — 각 persona의 최소 규약 정의."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersonaResult:
    """Persona 실행 결과의 공통 envelope.

    각 persona는 자신에 맞는 필드를 채웁니다.

    Attributes
    ----------
    kind:
        결과 종류 — ``"answer"`` (일반 답변) / ``"clarification"`` (역질문)
        / ``"refusal"`` (답변 불가) 등.
    answer:
        일반 답변 텍스트 (kind == "answer").
    sources:
        참조 문서 목록 (kind == "answer").
    questions:
        역질문 목록 (kind == "clarification").
    metadata:
        persona별 추가 정보 (iterations, model 등).
    """

    kind: str
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Persona:
    """Persona 기본 클래스 — 필요한 최소 속성만 정의.

    Researcher·Comparator·Analyst·Procedural·Clarifier가 이 기반을 확장합니다.

    Attributes
    ----------
    name:
        persona 고유 이름.
    description:
        사람이 읽을 수 있는 persona 설명.
    allowed_tools:
        이 persona가 호출 가능한 도구 이름 집합. 빈 집합은 "도구 없음".
        ``None``이면 제한 없음 (모든 도구 허용).
    synthesis_prompt_template:
        이 persona의 답변 합성 프롬프트 템플릿. ``None``이면 기본
        ``ANSWER_SYNTHESIS_PROMPT`` 사용. ``{history}``, ``{context}``, ``{query}``
        placeholder를 지원해야 함.
    plan_strategy_hint:
        Planner가 선호할 전략 힌트 — ``"fact" | "compare" | "decompose"`` 또는
        ``None``. Planner가 참고는 하지만 강제하지 않음.
    """

    name: str = "base"
    description: str = "generic persona"
    allowed_tools: frozenset[str] | None = None
    synthesis_prompt_template: str | None = None
    plan_strategy_hint: str | None = None


__all__ = ["Persona", "PersonaResult"]
