"""Comparator persona — 비교·대조 질의 전용."""

from __future__ import annotations

from ..prompts import COMPARATOR_SYNTHESIS_PROMPT
from .base import Persona


class Comparator(Persona):
    name = "comparator"
    description = "두 개 이상 개념을 표 형식으로 비교·대조하는 persona"
    allowed_tools = frozenset({"search", "compare", "lookup"})
    synthesis_prompt_template = COMPARATOR_SYNTHESIS_PROMPT
    plan_strategy_hint = "compare"


__all__ = ["Comparator"]
