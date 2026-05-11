"""Hooks 시스템 테스트 — HookRegistry + 내장 hooks + orchestrator 통합."""

from __future__ import annotations

import asyncio

import pytest

from rag_factory.rag.agent.hooks import (
    BUILT_IN_HOOKS,
    HookRegistry,
    build_default_registry,
    dedup_sources_by_doc_id,
    normalize_korean_whitespace,
    strip_html_from_answer)


class TestHookRegistry:
    @pytest.mark.asyncio
    async def test_비활성이면_payload_그대로(self):
        reg = HookRegistry(enabled=False)
        reg.register("pre_query", lambda q: q.upper())
        result = await reg.run("pre_query", "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_sync_hook_실행(self):
        reg = HookRegistry()
        reg.register("pre_query", lambda q: q.strip())
        result = await reg.run("pre_query", "  hi  ")
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_async_hook_실행(self):
        async def expand(q):
            await asyncio.sleep(0)
            return f"{q} 확장됨"

        reg = HookRegistry()
        reg.register("pre_query", expand)
        result = await reg.run("pre_query", "질의")
        assert result == "질의 확장됨"

    @pytest.mark.asyncio
    async def test_여러_hook_순서대로_chained(self):
        reg = HookRegistry()
        reg.register("pre_query", lambda q: q + "1")
        reg.register("pre_query", lambda q: q + "2")
        reg.register("pre_query", lambda q: q + "3")
        result = await reg.run("pre_query", "x")
        assert result == "x123"

    @pytest.mark.asyncio
    async def test_hook_예외는_삼키고_이전값_유지(self):
        reg = HookRegistry()
        reg.register("pre_query", lambda q: q + "!")

        def boom(q):
            raise RuntimeError("oops")

        reg.register("pre_query", boom)
        reg.register("pre_query", lambda q: q + "?")
        result = await reg.run("pre_query", "x")
        # boom이 실패해서 이전값 'x!'가 다음 hook으로, 최종 'x!?'
        assert result == "x!?"

    def test_clear_특정_지점(self):
        reg = HookRegistry()
        reg.register("pre_query", lambda q: q)
        reg.register("post_search", lambda s: s)
        reg.clear("pre_query")
        assert reg.count("pre_query") == 0
        assert reg.count("post_search") == 1

    def test_clear_전체(self):
        reg = HookRegistry()
        reg.register("pre_query", lambda q: q)
        reg.register("post_search", lambda s: s)
        reg.clear()
        assert reg.count("pre_query") == 0
        assert reg.count("post_search") == 0

    @pytest.mark.asyncio
    async def test_등록되지_않은_지점은_payload_그대로(self):
        reg = HookRegistry()
        result = await reg.run("unknown_point", "x")
        assert result == "x"


class TestBuiltinHooks:
    def test_normalize_korean_whitespace(self):
        assert normalize_korean_whitespace("  hello   world  ") == "hello world"
        assert normalize_korean_whitespace("  한국어\n\n테스트 ") == "한국어 테스트"
        assert normalize_korean_whitespace("") == ""

    def test_normalize_non_str는_그대로(self):
        assert normalize_korean_whitespace(None) is None
        assert normalize_korean_whitespace(123) == 123

    def test_dedup_sources_by_doc_id(self):
        sources = [
            {"doc_id": "a", "content": "첫번째"},
            {"doc_id": "b", "content": "두번째"},
            {"doc_id": "a", "content": "중복"},
        ]
        out = dedup_sources_by_doc_id(sources)
        assert len(out) == 2
        assert out[0]["content"] == "첫번째"
        assert out[1]["doc_id"] == "b"

    def test_dedup_빈_목록(self):
        assert dedup_sources_by_doc_id([]) == []

    def test_dedup_non_list는_그대로(self):
        assert dedup_sources_by_doc_id("not a list") == "not a list"

    def test_dedup_non_dict_entry_건너뜀(self):
        sources = [{"doc_id": "a"}, "not a dict", {"doc_id": "b"}]
        out = dedup_sources_by_doc_id(sources)
        assert len(out) == 2

    def test_strip_html(self):
        assert strip_html_from_answer("<b>bold</b> text") == "bold text"
        assert strip_html_from_answer("<div class='x'>x</div>") == "x"
        assert strip_html_from_answer("plain text") == "plain text"


class TestBuildDefaultRegistry:
    @pytest.mark.asyncio
    async def test_내장_hook_자동_등록(self):
        reg = build_default_registry(
            enabled=True,
            builtin_names=["normalize_korean_whitespace", "dedup_sources_by_doc_id"])
        assert reg.count("pre_query") == 1
        assert reg.count("post_search") == 1

        r1 = await reg.run("pre_query", "  hi  ")
        assert r1 == "hi"

        r2 = await reg.run("post_search", [{"doc_id": "a"}, {"doc_id": "a"}])
        assert len(r2) == 1

    def test_알_수_없는_hook_무시(self):
        reg = build_default_registry(
            enabled=True, builtin_names=["unknown_hook", "normalize_korean_whitespace"]
        )
        assert reg.count("pre_query") == 1

    @pytest.mark.asyncio
    async def test_enabled_False면_빈_registry(self):
        reg = build_default_registry(
            enabled=False, builtin_names=["normalize_korean_whitespace"]
        )
        # enabled=False → 비활성화. 등록도 건너뜀 (엄격).
        assert reg.count("pre_query") == 0


class TestBuiltinHooksDict:
    def test_모든_내장_hook_등록(self):
        assert "normalize_korean_whitespace" in BUILT_IN_HOOKS
        assert "dedup_sources_by_doc_id" in BUILT_IN_HOOKS
        assert "strip_html_from_answer" in BUILT_IN_HOOKS


class TestOrchestratorIntegration:
    """orchestrator가 hook 지점에서 적절히 호출하는지."""

    @pytest.mark.asyncio
    async def test_pre_query_hook이_query_수정(self, monkeypatch):
        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_orchestrator,
            _make_plan,
            _FakeToolResult,
            _collect)

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])
        orch = _make_orchestrator(
            planner_enabled=True,
            app_state=fixtures.app_state)
        # 외부에서 hook 등록
        received_queries: list[str] = []

        def tracker(q):
            received_queries.append(q)
            return q + "_MUTATED"

        orch.register_hook("pre_query", tracker)
        await _collect(orch.handle_agent("  원본  "))

        # hook은 기본 enabled=True일 때만 실행됨. _make_orchestrator는 hooks_enabled=False로
        # config를 만들어서 비활성 상태 → hook 실행 안 됨.
        assert received_queries == []

    @pytest.mark.asyncio
    async def test_hooks_enabled_True일때만_실행(self, monkeypatch):
        from types import SimpleNamespace
        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_plan,
            _FakeToolResult,
            _collect)
        from rag_factory.rag.agent.orchestrator import AgentOrchestrator
        from rag_factory.rag.agent.router import QueryRouter

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])

        config_ns = SimpleNamespace(
            rag=SimpleNamespace(
                agent=SimpleNamespace(
                    enabled=True,
                    max_iterations=3,
                    stream_reasoning=False,
                    planner_enabled=True,
                    verifier_enabled=True,
                    verifier_max_repairs=1,
                    legacy_fallback_enabled=True,
                    refusal_min_score=0.0,
                    session_source_reuse=False,
                    session_source_reuse_limit=5,
                    parallel_steps=False,
                    clarifier_enabled=False,
                    clarifier_max_questions=2,
                    personas_enabled=False,
                    native_thinking=False,
                    skills_enabled=False,
                    skills_dir="skills",
                    hooks_enabled=True,
                    builtin_hooks=[]),
                request_timeout=60.0))

        async def _simple(q):
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=QueryRouter(agent_enabled=True),
            app_state=fixtures.app_state,
            config=config_ns,
            ollama_model="test",
            api_base="http://localhost:11434",
            rag_max_tokens=-1,
            simple_stream_fn=_simple)

        received_queries: list[str] = []
        orch.register_hook("pre_query", lambda q: received_queries.append(q) or q)
        await _collect(orch.handle_agent("질의"))
        assert received_queries == ["질의"]
