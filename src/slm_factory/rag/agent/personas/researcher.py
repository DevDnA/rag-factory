"""Researcher persona — 단일 사실·조항 정확 인용."""

from __future__ import annotations

from ..prompts import RESEARCHER_SYNTHESIS_PROMPT
from .base import Persona


class Researcher(Persona):
    name = "researcher"
    description = "단일 사실·조항·수치의 정확 인용 전문 persona"
    allowed_tools = frozenset({"search", "lookup"})
    synthesis_prompt_template = RESEARCHER_SYNTHESIS_PROMPT
    plan_strategy_hint = "fact"


__all__ = ["Researcher"]
