"""CompositePersona 테스트 — 합성 결과의 속성·placeholder 안전성·prompt 구조."""

from __future__ import annotations

import pytest

from rag_factory.rag.agent.personas.analyst import Analyst
from rag_factory.rag.agent.personas.base import Persona
from rag_factory.rag.agent.personas.comparator import Comparator
from rag_factory.rag.agent.personas.composite import CompositePersona
from rag_factory.rag.agent.personas.procedural import Procedural
from rag_factory.rag.agent.personas.researcher import Researcher

_PERSONA_CLASSES = [Researcher, Comparator, Analyst, Procedural]


# ---------------------------------------------------------------------------
# 기본 속성 합성
# ---------------------------------------------------------------------------


class TestBasicComposition:
    def test_name은_primary_plus_secondary(self):
        c = CompositePersona(Comparator(), Analyst())
        assert c.name == "comparator+analyst"

    def test_description_표시(self):
        c = CompositePersona(Comparator(), Analyst())
        assert "comparator" in c.description
        assert "analyst" in c.description
        assert "primary" in c.description and "secondary" in c.description

    def test_allowed_tools는_union(self):
        # Comparator: {search, compare, lookup}, Analyst: {search, lookup, compare}
        c = CompositePersona(Comparator(), Analyst())
        assert c.allowed_tools is not None
        assert "search" in c.allowed_tools
        assert "compare" in c.allowed_tools
        assert "lookup" in c.allowed_tools

    def test_allowed_tools가_둘_다_None이면_None(self):
        class NoToolsA(Persona):
            name = "a"
            allowed_tools = None
            synthesis_prompt_template = "A {history}\n{context}\n{query}\n답변:"

        class NoToolsB(Persona):
            name = "b"
            allowed_tools = None
            synthesis_prompt_template = "B {history}\n{context}\n{query}\n답변:"

        c = CompositePersona(NoToolsA(), NoToolsB())
        assert c.allowed_tools is None

    def test_한쪽_None이면_None(self):
        """둘 중 하나라도 None(무제한)이면 결과도 None."""

        class NoTools(Persona):
            name = "n"
            allowed_tools = None
            synthesis_prompt_template = "X {history}\n{context}\n{query}\n답변:"

        c = CompositePersona(Comparator(), NoTools())
        assert c.allowed_tools is None

    def test_plan_strategy_hint는_primary_우선(self):
        # Comparator.plan_strategy_hint == "compare"
        # Analyst.plan_strategy_hint == "decompose"
        c = CompositePersona(Comparator(), Analyst())
        assert c.plan_strategy_hint == "compare"

        c2 = CompositePersona(Analyst(), Comparator())
        assert c2.plan_strategy_hint == "decompose"


# ---------------------------------------------------------------------------
# Synthesis prompt template 합성
# ---------------------------------------------------------------------------


class TestPromptComposition:
    """secondary의 핵심 규칙이 primary template에 명시 마커로 부착되는지."""

    def test_marker가_포함됨(self):
        c = CompositePersona(Comparator(), Analyst())
        assert c.synthesis_prompt_template is not None
        assert "## 추가 분석 지침" in c.synthesis_prompt_template

    def test_placeholder_3개_정확히_1번씩(self):
        """{.format()} 안전성 — primary placeholder는 단 한 번만 등장해야."""
        c = CompositePersona(Comparator(), Analyst())
        assert c.synthesis_prompt_template is not None
        assert c.synthesis_prompt_template.count("{history}") == 1
        assert c.synthesis_prompt_template.count("{context}") == 1
        assert c.synthesis_prompt_template.count("{query}") == 1

    def test_format_호출이_깨지지_않음(self):
        """실제로 .format()을 호출했을 때 placeholder 충돌 없음."""
        c = CompositePersona(Comparator(), Analyst())
        assert c.synthesis_prompt_template is not None
        # 실제 .format 호출
        rendered = c.synthesis_prompt_template.format(
            history="HIST",
            context="CTX",
            query="Q",
        )
        assert "HIST" in rendered
        assert "CTX" in rendered
        assert "Q" in rendered

    def test_secondary_핵심_규칙이_포함됨(self):
        """Comparator + Analyst — Analyst의 '다각도 분석' 키워드가 들어가야 함."""
        c = CompositePersona(Comparator(), Analyst())
        assert c.synthesis_prompt_template is not None
        # Analyst template은 '문서 간 상충하면', '단정적 주장 금지' 등을 포함
        # 첫 4줄 추출이므로 적어도 한국어 키워드가 있어야 함
        addendum_start = c.synthesis_prompt_template.find("## 추가 분석 지침")
        addendum = c.synthesis_prompt_template[addendum_start:]
        # secondary의 규칙 라인이 부착됐는지 — 한글 문자가 일정 분량 있어야
        addendum_ko = sum(1 for ch in addendum if "가" <= ch <= "힣")
        assert addendum_ko > 20

    def test_primary가_None이면_None(self):
        class NullPrimary(Persona):
            name = "null"
            synthesis_prompt_template = None

        c = CompositePersona(NullPrimary(), Analyst())
        assert c.synthesis_prompt_template is None

    def test_secondary가_None이면_primary_그대로(self):
        class NullSecondary(Persona):
            name = "null"
            synthesis_prompt_template = None

        c = CompositePersona(Comparator(), NullSecondary())
        assert c.synthesis_prompt_template == Comparator().synthesis_prompt_template


# ---------------------------------------------------------------------------
# 추출 라인의 placeholder 충돌 방지
# ---------------------------------------------------------------------------


class TestPlaceholderEscape:
    """secondary template에서 추출한 라인이 우연히 단일 중괄호 placeholder를
    포함할 경우 escape되어야 .format()이 깨지지 않음."""

    def test_secondary에_placeholder가_있으면_escape(self):
        # 인위적으로 규칙 안에 {foo} 같은 placeholder를 포함한 secondary
        class WeirdSecondary(Persona):
            name = "weird"
            synthesis_prompt_template = (
                "규칙:\n- {foo}를 출력하지 말 것\n- 정확한 인용\n"
                "{history}\n[참고 문서]\n{context}\n질문: {query}\n답변:"
            )

        c = CompositePersona(Comparator(), WeirdSecondary())
        assert c.synthesis_prompt_template is not None
        # 합성된 template을 .format() 호출 — {foo}가 escape되었으면 깨지지 않음
        rendered = c.synthesis_prompt_template.format(
            history="H",
            context="C",
            query="Q",
        )
        # {foo}는 그대로 문자열로 남아있어야 함 (이스케이프 결과)
        assert "{foo}" in rendered


# ---------------------------------------------------------------------------
# 모든 built-in 페르소나 쌍에 대한 합성 안정성
# ---------------------------------------------------------------------------


class TestAllBuiltinPairs:
    """4개 built-in persona의 모든 순서 쌍에 대해 합성이 깨지지 않음."""

    @pytest.mark.parametrize(
        "primary_cls,secondary_cls",
        [
            (p, s)
            for p in _PERSONA_CLASSES
            for s in _PERSONA_CLASSES
            if p is not s
        ],
    )
    def test_쌍_합성_format_안정(self, primary_cls, secondary_cls):
        c = CompositePersona(primary_cls(), secondary_cls())
        assert c.synthesis_prompt_template is not None
        # 실제 .format() 호출이 깨지지 않아야 함
        rendered = c.synthesis_prompt_template.format(
            history="",
            context="dummy context",
            query="dummy query",
        )
        assert "dummy context" in rendered
        assert "dummy query" in rendered
