"""Phase 14 — Custom Personas (YAML) 테스트."""

from __future__ import annotations

import pytest

from slm_factory.rag.agent.persona_loader import (
    CustomPersona,
    CustomPersonaRegistry,
    load_custom_personas)


class TestLoadCustomPersonas:
    def test_없는_디렉터리(self, tmp_path):
        personas = load_custom_personas(tmp_path / "nope")
        assert personas == []

    def test_빈_디렉터리(self, tmp_path):
        personas = load_custom_personas(tmp_path)
        assert personas == []

    def test_단일_persona_로드(self, tmp_path):
        (tmp_path / "legal.yaml").write_text(
            """
name: legal-expert
description: 법률 전문가
intent: factual
allowed_tools: [search, lookup]
plan_strategy_hint: fact
synthesis_prompt_template: |
  법률 전문: {query}
  {history}[참고 문서]
  {context}
  답변:
""",
            encoding="utf-8")
        personas = load_custom_personas(tmp_path)
        assert len(personas) == 1
        p = personas[0]
        assert p.name == "legal-expert"
        assert p.intent == "factual"
        assert "search" in p.allowed_tools
        assert "법률 전문" in p.synthesis_prompt_template

    def test_name_없으면_건너뜀(self, tmp_path):
        (tmp_path / "noname.yaml").write_text("intent: factual\n", encoding="utf-8")
        personas = load_custom_personas(tmp_path)
        assert personas == []

    def test_잘못된_YAML은_건너뜀(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("{not valid", encoding="utf-8")
        (tmp_path / "ok.yaml").write_text(
            "name: ok\nintent: factual\nsynthesis_prompt_template: 'Q {query} C {context} H {history}'\n",
            encoding="utf-8")
        personas = load_custom_personas(tmp_path)
        assert len(personas) == 1

    def test_hint_과_tools_정규화(self, tmp_path):
        (tmp_path / "p.yaml").write_text(
            """
name: p
intent: analytical
plan_strategy_hint: decompose
allowed_tools:
  - search
  - compare
""",
            encoding="utf-8")
        personas = load_custom_personas(tmp_path)
        p = personas[0]
        assert p.plan_strategy_hint == "decompose"
        assert p.allowed_tools == frozenset({"search", "compare"})


class TestCustomPersonaRegistry:
    def test_intent별_조회(self):
        p1 = CustomPersona(name="legal", intent="factual")
        p2 = CustomPersona(name="finance", intent="comparative")
        reg = CustomPersonaRegistry([p1, p2])
        assert reg.select_for_intent("factual") is p1
        assert reg.select_for_intent("comparative") is p2
        assert reg.select_for_intent("analytical") is None

    def test_intent_없는_persona(self):
        p = CustomPersona(name="generic")
        reg = CustomPersonaRegistry([p])
        assert reg.select_for_intent("factual") is None
        assert len(reg) == 1

    def test_all_반환(self):
        p1 = CustomPersona(name="a", intent="factual")
        p2 = CustomPersona(name="b", intent="comparative")
        reg = CustomPersonaRegistry([p1, p2])
        assert len(reg.all()) == 2


class TestPersonaRouterWithCustom:
    def test_custom_persona_우선_적용(self):
        from slm_factory.rag.agent.persona_router import PersonaRouter

        custom = CustomPersona(
            name="legal-custom",
            intent="factual",
            synthesis_prompt_template="CUSTOM {query} {context} {history}")
        reg = CustomPersonaRegistry([custom])
        router = PersonaRouter(enabled=True, custom_registry=reg)

        p = router.select("factual")
        assert p is not None
        assert p.name == "legal-custom"

    def test_custom_없으면_builtin_fallback(self):
        from slm_factory.rag.agent.persona_router import PersonaRouter
        from slm_factory.rag.agent.personas import Researcher

        reg = CustomPersonaRegistry([])
        router = PersonaRouter(enabled=True, custom_registry=reg)
        p = router.select("factual")
        assert isinstance(p, Researcher)

    def test_custom_intent_매핑_없으면_builtin(self):
        """custom이 analytical에만 매핑됐는데 factual 질의면 builtin 사용."""
        from slm_factory.rag.agent.persona_router import PersonaRouter
        from slm_factory.rag.agent.personas import Researcher

        custom = CustomPersona(name="analytical-custom", intent="analytical")
        reg = CustomPersonaRegistry([custom])
        router = PersonaRouter(enabled=True, custom_registry=reg)

        # factual → built-in Researcher
        assert isinstance(router.select("factual"), Researcher)
        # analytical → custom
        p = router.select("analytical")
        assert p.name == "analytical-custom"

    def test_enabled_False면_None(self):
        from slm_factory.rag.agent.persona_router import PersonaRouter

        custom = CustomPersona(name="x", intent="factual")
        reg = CustomPersonaRegistry([custom])
        router = PersonaRouter(enabled=False, custom_registry=reg)
        assert router.select("factual") is None


class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_custom_persona_synthesis_prompt_적용(self, tmp_path, monkeypatch):
        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_plan,
            _FakeToolResult,
            _collect)
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.intent_classifier import IntentDecision
        from slm_factory.rag.agent.router import QueryRouter
        from types import SimpleNamespace

        # custom persona 파일 작성
        (tmp_path / "legal.yaml").write_text(
            """
name: legal-custom
intent: factual
synthesis_prompt_template: |
  LEGAL_CUSTOM_TAG
  {history}{context}
  질문: {query}
  답변:
""",
            encoding="utf-8")

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])

        class _FakeClassifier:
            async def classify(self, query):
                return IntentDecision(intent="factual", confidence=0.95)

        config_ns = SimpleNamespace(
            rag=SimpleNamespace(
                agent=SimpleNamespace(
                    enabled=True,
                    max_iterations=3,
                    stream_reasoning=False,
                    planner_enabled=True,
                    verifier_enabled=False,
                    verifier_max_repairs=0,
                    legacy_fallback_enabled=True,
                    session_source_reuse=False,
                    session_source_reuse_limit=5,
                    parallel_steps=False,
                    clarifier_enabled=False,
                    clarifier_max_questions=2,
                    personas_enabled=True,
                    skills_enabled=False,
                    skills_dir="",
                    hooks_enabled=False,
                    builtin_hooks=[],
                    memory_compression_enabled=False,
                    compress_after_turns=10,
                    compress_target_chars=500,
                    custom_personas_dir=str(tmp_path),
                    models=SimpleNamespace(
                        router_model="", planner_model="", synthesis_model="",
                        verifier_model="",                         clarifier_model="")),
                request_timeout=60.0))

        async def _simple(q):
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=QueryRouter(agent_enabled=True, intent_classifier=_FakeClassifier()),
            app_state=fixtures.app_state,
            config=config_ns,
            ollama_model="base",
            api_base="http://localhost:11434",
            rag_max_tokens=-1,
            simple_stream_fn=_simple)
        # "분석" 키워드로 agent 경로 강제 — "비교" 키워드는 tabular_intent
        # override가 작동해 built-in Comparator로 강제 전환되므로 custom persona
        # 합성 prompt 검증 불가. _COMPLEX_KEYWORDS \ _TABULAR_KEYWORDS인 "분석"
        # 사용으로 override 회피.
        await _collect(orch.handle_auto("분석 관련 질의"))

        prompt = fixtures.app_state.http_client.stream.call_args.kwargs["json"]["prompt"]
        assert "LEGAL_CUSTOM_TAG" in prompt
