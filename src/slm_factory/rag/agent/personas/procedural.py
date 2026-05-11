"""Procedural persona — 절차·방법·단계별 지침."""

from __future__ import annotations

from ..prompts import PROCEDURAL_SYNTHESIS_PROMPT
from .base import Persona


class Procedural(Persona):
    name = "procedural"
    description = "절차·방법을 번호 매긴 단계별 지침으로 제시하는 persona"
    allowed_tools = frozenset({"search", "lookup"})
    synthesis_prompt_template = PROCEDURAL_SYNTHESIS_PROMPT
    plan_strategy_hint = "fact"


__all__ = ["Procedural"]
