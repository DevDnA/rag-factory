"""Analyst persona — 다각도 분석·종합·시사점."""

from __future__ import annotations

from ..prompts import ANALYST_SYNTHESIS_PROMPT
from .base import Persona


class Analyst(Persona):
    name = "analyst"
    description = "다각도 분석·원인·영향·시사점 도출 persona"
    allowed_tools = frozenset({"search", "lookup", "compare"})
    synthesis_prompt_template = ANALYST_SYNTHESIS_PROMPT
    plan_strategy_hint = "decompose"


__all__ = ["Analyst"]
