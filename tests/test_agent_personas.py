"""Persona 시스템 테스트 — 각 persona 속성 + PersonaRouter 매핑."""

from __future__ import annotations

import pytest

from slm_factory.rag.agent.persona_router import PersonaRouter
from slm_factory.rag.agent.personas import (
    Analyst,
    Clarifier,
    Comparator,
    Persona,
    PersonaResult,
    Procedural,
    Researcher,
)


class TestPersonaBase:
    def test_기본_속성_default(self):
        assert Persona.name == "base"
        assert Persona.allowed_tools is None
        assert Persona.synthesis_prompt_template is None

    def test_PersonaResult_envelope(self):
        r = PersonaResult(kind="answer", answer="x", sources=[{"doc_id": "d1"}])
        assert r.kind == "answer"
        assert r.answer == "x"
        assert r.sources == [{"doc_id": "d1"}]
        assert r.questions == []


class TestSpecializedPersonas:
    @pytest.mark.parametrize("persona_cls, expected_name, expected_hint", [
        (Researcher, "researcher", "fact"),
        (Comparator, "comparator", "compare"),
        (Analyst, "analyst", "decompose"),
        (Procedural, "procedural", "fact"),
    ])
    def test_각_persona_속성(self, persona_cls, expected_name, expected_hint):
        p = persona_cls()
        assert p.name == expected_name
        assert p.plan_strategy_hint == expected_hint
        assert p.synthesis_prompt_template is not None
        assert "{query}" in p.synthesis_prompt_template
        assert "{context}" in p.synthesis_prompt_template
        assert "{history}" in p.synthesis_prompt_template

    def test_Researcher_도구_화이트리스트(self):
        p = Researcher()
        assert "search" in p.allowed_tools
        assert "lookup" in p.allowed_tools
        assert "compare" not in p.allowed_tools

    def test_Comparator_compare_도구_포함(self):
        p = Comparator()
        assert "compare" in p.allowed_tools

    def test_Analyst_모든_도구(self):
        p = Analyst()
        assert "search" in p.allowed_tools
        assert "compare" in p.allowed_tools
        assert "lookup" in p.allowed_tools

    def test_Clarifier_도구_없음(self):
        assert Clarifier.allowed_tools == frozenset()


class TestPersonaRouter:
    def test_비활성화시_None_반환(self):
        r = PersonaRouter(enabled=False)
        assert r.select("factual") is None
        assert r.select("comparative") is None

    @pytest.mark.parametrize("intent, expected_cls", [
        ("factual", Researcher),
        ("comparative", Comparator),
        ("analytical", Analyst),
        ("procedural", Procedural),
        ("exploratory", Analyst),
    ])
    def test_intent_매핑(self, intent, expected_cls):
        r = PersonaRouter(enabled=True)
        p = r.select(intent)
        assert isinstance(p, expected_cls)

    def test_None_intent(self):
        r = PersonaRouter(enabled=True)
        assert r.select(None) is None

    def test_알_수_없는_intent(self):
        r = PersonaRouter(enabled=True)
        assert r.select("alien_intent") is None

    def test_ambiguous는_매핑없음_orchestrator가_처리(self):
        # ambiguous는 Clarifier 경로로 가므로 일반 persona 매핑에는 포함되지 않음
        r = PersonaRouter(enabled=True)
        assert r.select("ambiguous") is None

    def test_available_personas_목록(self):
        names = PersonaRouter.available_personas()
        assert "researcher" in names
        assert "comparator" in names
        assert "analyst" in names
        assert "procedural" in names
        # Clarifier는 intent 매핑에 없음
        assert "clarifier" not in names


class TestPromptTemplates:
    """각 persona의 synthesis prompt가 핵심 키워드를 포함하는지."""

    def test_Researcher_prompt는_인용_강조(self):
        t = Researcher.synthesis_prompt_template
        assert "Researcher" in t
        assert "문서" in t

    def test_Comparator_prompt는_표_언급(self):
        t = Comparator.synthesis_prompt_template
        assert "Comparator" in t
        assert "표" in t or "Markdown" in t

    def test_Analyst_prompt는_관점_분석(self):
        t = Analyst.synthesis_prompt_template
        assert "Analyst" in t
        assert "관점" in t or "분석" in t

    def test_Procedural_prompt는_단계_번호(self):
        t = Procedural.synthesis_prompt_template
        assert "Procedural" in t
        assert "단계" in t or "번호" in t
