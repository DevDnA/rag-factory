"""Phase 9 — Multi-Model Routing 테스트.

각 컴포넌트가 구성된 모델 슬롯을 사용하는지, 슬롯이 비면 기본 모델로 fallback하는지 검증.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from slm_factory.config import AgentModelsConfig, AgentRagConfig


class TestAgentModelsConfig:
    def test_기본값은_모두_빈_문자열(self):
        cfg = AgentModelsConfig()
        assert cfg.router_model == ""
        assert cfg.planner_model == ""
        assert cfg.synthesis_model == ""
        assert cfg.verifier_model == ""
        assert cfg.clarifier_model == ""

    def test_AgentRagConfig에_models_nested(self):
        cfg = AgentRagConfig()
        assert isinstance(cfg.models, AgentModelsConfig)

    def test_부분_지정(self):
        cfg = AgentRagConfig(
            models={"planner_model": "small:7b", "synthesis_model": "big:14b"}
        )
        assert cfg.models.planner_model == "small:7b"
        assert cfg.models.synthesis_model == "big:14b"
        assert cfg.models.verifier_model == ""  # fallback 유지


class TestOrchestratorModelRouting:
    """orchestrator._model_for가 슬롯 지정/fallback을 올바르게 처리."""

    def _make_orch(self, models: dict | None = None):
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.router import QueryRouter

        models_ns = SimpleNamespace(
            router_model="",
            planner_model="",
            synthesis_model="",
            verifier_model="",
            clarifier_model="")
        if models:
            for k, v in models.items():
                setattr(models_ns, k, v)

        config_ns = SimpleNamespace(
            rag=SimpleNamespace(
                agent=SimpleNamespace(
                    enabled=True,
                    max_iterations=3,
                    stream_reasoning=False,
                    planner_enabled=False,
                    verifier_enabled=False,
                    verifier_max_repairs=0,
                    legacy_fallback_enabled=True,
                    session_source_reuse=False,
                    session_source_reuse_limit=5,
                    parallel_steps=False,
                    clarifier_enabled=False,
                    clarifier_max_questions=2,
                    personas_enabled=False,
                    skills_enabled=False,
                    skills_dir="skills",
                    hooks_enabled=False,
                    builtin_hooks=[],
                    models=models_ns),
                request_timeout=60.0))

        async def _simple(q):
            yield {"type": "done"}

        app_state = SimpleNamespace(
            agent_session_manager=MagicMock(),
            agent_tool_registry=MagicMock(),
            http_client=MagicMock())
        return AgentOrchestrator(
            router=QueryRouter(agent_enabled=True),
            app_state=app_state,
            config=config_ns,
            ollama_model="base:3b",
            api_base="x",
            rag_max_tokens=-1,
            simple_stream_fn=_simple)

    def test_모든_슬롯_빈_경우_base로_fallback(self):
        orch = self._make_orch()
        assert orch._model_for("planner") == "base:3b"
        assert orch._model_for("synthesis") == "base:3b"
        assert orch._model_for("verifier") == "base:3b"
        assert orch._model_for("clarifier") == "base:3b"
        assert orch._model_for("router") == "base:3b"

    def test_슬롯별_지정_사용(self):
        orch = self._make_orch(
            models={
                "planner_model": "plan:7b",
                "synthesis_model": "syn:14b",
                "verifier_model": "ver:1b",
            }
        )
        assert orch._model_for("planner") == "plan:7b"
        assert orch._model_for("synthesis") == "syn:14b"
        assert orch._model_for("verifier") == "ver:1b"
        # 미지정 슬롯은 base
        assert orch._model_for("clarifier") == "base:3b"

    def test_공백만_있는_슬롯은_base로(self):
        orch = self._make_orch(models={"planner_model": "   "})
        assert orch._model_for("planner") == "base:3b"

    def test_알_수_없는_슬롯도_base(self):
        orch = self._make_orch()
        assert orch._model_for("nonexistent") == "base:3b"


class TestOrchestratorUsesSlots:
    """실제 orchestrator 실행 중 각 컴포넌트가 올바른 모델 슬롯을 사용하는지."""

    @pytest.mark.asyncio
    async def test_synthesis_slot이_prompt_모델로_전달(self, monkeypatch):
        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_plan,
            _FakeToolResult,
            _collect)
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.router import QueryRouter

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])

        models_ns = SimpleNamespace(
            router_model="",
            planner_model="",
            synthesis_model="BIG_SYNTHESIS",
            verifier_model="",
            clarifier_model="")

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
                    personas_enabled=False,
                    skills_enabled=False,
                    skills_dir="skills",
                    hooks_enabled=False,
                    builtin_hooks=[],
                    models=models_ns),
                request_timeout=60.0))

        async def _simple(q):
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=QueryRouter(agent_enabled=True),
            app_state=fixtures.app_state,
            config=config_ns,
            ollama_model="base",
            api_base="http://localhost:11434",
            rag_max_tokens=-1,
            simple_stream_fn=_simple)
        await _collect(orch.handle_agent("질의"))

        call_kwargs = fixtures.app_state.http_client.stream.call_args.kwargs
        assert call_kwargs["json"]["model"] == "BIG_SYNTHESIS"
