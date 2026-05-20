"""AgentOrchestrator 테스트 — 라우팅 + 스트리밍 통합을 검증합니다.

``AgentLoop``와 ``http_client``는 모킹하여 HTTP 호출 없이 이벤트 시퀀스만
검증합니다. 실제 end-to-end는 ``test_integration.py`` 및 수동 /auto 테스트에
맡깁니다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from rag_factory.rag.agent.orchestrator import AgentOrchestrator
from rag_factory.rag.agent.router import QueryRouter
from rag_factory.rag.agent.session import Message, SessionManager


# ---------------------------------------------------------------------------
# Fake AgentLoop — loop.AgentLoop를 대체합니다.
# ---------------------------------------------------------------------------


class _FakeAgentEvent:
    def __init__(self, type: str, content: str = "", iteration: int = 1, metadata=None):
        self.type = type
        self.content = content
        self.iteration = iteration
        self.metadata = metadata or {}


class _FakeAgentLoop:
    """run_stream에서 미리 정의한 이벤트 시퀀스를 yield하는 테스트용 AgentLoop."""

    script: list[_FakeAgentEvent] = []

    def __init__(self, **_kwargs):
        pass

    async def run_stream(self, query: str, history: str = "") -> AsyncGenerator:
        for event in _FakeAgentLoop.script:
            yield event


@pytest.fixture(autouse=True)
def _patch_agent_loop(monkeypatch):
    """orchestrator가 import하는 AgentLoop를 _FakeAgentLoop로 바꿉니다."""
    from rag_factory.rag.agent import orchestrator as orch_mod
    from rag_factory.rag.agent import loop as loop_mod

    monkeypatch.setattr(loop_mod, "AgentLoop", _FakeAgentLoop)
    # orchestrator가 지연 import하므로 loop 모듈만 패치하면 충분합니다.
    return orch_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    agent_enabled: bool = True,
    stream_reasoning: bool = True,
    planner_enabled: bool = False,
    verifier_enabled: bool = True,
    verifier_max_repairs: int = 1,
    legacy_fallback_enabled: bool = True,
    session_source_reuse: bool = True,
    session_source_reuse_limit: int = 5,
    parallel_steps: bool = False,
    clarifier_enabled: bool = False,
    clarifier_max_questions: int = 2,
    personas_enabled: bool = False,
    native_thinking: bool = False,
    refusal_min_score: float = 0.0,
    refusal_relative_margin: float = 0.0,
    in_domain_score_threshold: float = 0.0,
    planner_preserve_first_query: bool = False,
):
    return SimpleNamespace(
        rag=SimpleNamespace(
            agent=SimpleNamespace(
                enabled=agent_enabled,
                max_iterations=3,
                stream_reasoning=stream_reasoning,
                planner_enabled=planner_enabled,
                verifier_enabled=verifier_enabled,
                verifier_max_repairs=verifier_max_repairs,
                legacy_fallback_enabled=legacy_fallback_enabled,
                session_source_reuse=session_source_reuse,
                session_source_reuse_limit=session_source_reuse_limit,
                parallel_steps=parallel_steps,
                clarifier_enabled=clarifier_enabled,
                clarifier_max_questions=clarifier_max_questions,
                personas_enabled=personas_enabled,
                native_thinking=native_thinking,
                refusal_min_score=refusal_min_score,
                refusal_relative_margin=refusal_relative_margin,
                in_domain_score_threshold=in_domain_score_threshold,
                planner_preserve_first_query=planner_preserve_first_query,
            ),
            request_timeout=60.0,
        ),
    )


def _make_app_state():
    return SimpleNamespace(
        agent_session_manager=SessionManager(ttl=3600, max_turns=10),
        agent_tool_registry=MagicMock(),
        http_client=MagicMock(),
    )


async def _collect(agen) -> list[dict]:
    return [ev async for ev in agen]


async def _simple_stream_fixture(query: str):
    """고정된 단순 스트림 — 토큰 두 개 + sources + done."""
    yield {"type": "token", "content": f"answer for {query}"}
    yield {"type": "token", "content": "."}
    yield {"type": "sources", "sources": []}
    yield {"type": "done"}


def _make_orchestrator(
    *,
    agent_enabled: bool = True,
    stream_reasoning: bool = True,
    planner_enabled: bool = False,
    verifier_enabled: bool = True,
    verifier_max_repairs: int = 1,
    legacy_fallback_enabled: bool = True,
    session_source_reuse: bool = True,
    session_source_reuse_limit: int = 5,
    parallel_steps: bool = False,
    clarifier_enabled: bool = False,
    clarifier_max_questions: int = 2,
    personas_enabled: bool = False,
    refusal_min_score: float = 0.0,
    refusal_relative_margin: float = 0.0,
    in_domain_score_threshold: float = 0.0,
    planner_preserve_first_query: bool = False,
    router=None,
    simple_fn=_simple_stream_fixture,
    app_state=None,
) -> AgentOrchestrator:
    config = _make_config(
        agent_enabled=agent_enabled,
        stream_reasoning=stream_reasoning,
        planner_enabled=planner_enabled,
        verifier_enabled=verifier_enabled,
        verifier_max_repairs=verifier_max_repairs,
        legacy_fallback_enabled=legacy_fallback_enabled,
        session_source_reuse=session_source_reuse,
        session_source_reuse_limit=session_source_reuse_limit,
        parallel_steps=parallel_steps,
        clarifier_enabled=clarifier_enabled,
        clarifier_max_questions=clarifier_max_questions,
        personas_enabled=personas_enabled,
        refusal_min_score=refusal_min_score,
        refusal_relative_margin=refusal_relative_margin,
        in_domain_score_threshold=in_domain_score_threshold,
        planner_preserve_first_query=planner_preserve_first_query,
    )
    return AgentOrchestrator(
        router=router or QueryRouter(agent_enabled=agent_enabled),
        app_state=app_state or _make_app_state(),
        config=config,
        ollama_model="test",
        api_base="http://localhost:11434",
        rag_max_tokens=-1,
        simple_stream_fn=simple_fn,
    )


# ---------------------------------------------------------------------------
# 단순 경로 라우팅
# ---------------------------------------------------------------------------


class TestSimpleRoute:
    """단순 질의는 simple_stream_fn을 통과시킵니다."""

    @pytest.mark.asyncio
    async def test_단순_질의는_simple_경로로(self):
        orch = _make_orchestrator()
        events = await _collect(orch.handle_auto("오늘 날씨"))

        assert events[0] == {"type": "route", "mode": "simple"}
        assert events[-1] == {"type": "done"}
        token_events = [e for e in events if e["type"] == "token"]
        assert any("오늘 날씨" in e["content"] for e in token_events)

    @pytest.mark.asyncio
    async def test_agent_비활성화시_복합질의도_simple로(self):
        orch = _make_orchestrator(agent_enabled=False)
        events = await _collect(orch.handle_auto("A와 B의 차이 비교"))
        assert events[0] == {"type": "route", "mode": "simple"}

    @pytest.mark.asyncio
    async def test_simple_경로는_agent_loop_호출안함(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="AGENT-SHOULD-NOT-RUN")
        ]
        orch = _make_orchestrator()
        events = await _collect(orch.handle_auto("간단한 질의"))
        assert not any("AGENT-SHOULD-NOT-RUN" in str(e) for e in events)


# ---------------------------------------------------------------------------
# Agent 경로 라우팅
# ---------------------------------------------------------------------------


class TestAgentRoute:
    """복합 질의는 _stream_agent를 통해 AgentLoop.run_stream을 호출합니다."""

    @pytest.mark.asyncio
    async def test_복합_질의는_agent_경로로(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="thought", content="검색합니다"),
            _FakeAgentEvent(type="action", content="search", metadata={"input": {"query": "x"}}),
            _FakeAgentEvent(type="observation", content="결과"),
            _FakeAgentEvent(type="token", content="최종 "),
            _FakeAgentEvent(type="token", content="답변"),
            _FakeAgentEvent(
                type="done",
                metadata={"sources": [{"doc_id": "d1", "score": 0.9, "content": "c"}]},
            ),
        ]

        orch = _make_orchestrator()
        events = await _collect(orch.handle_auto("A와 B의 차이 비교"))

        assert events[0] == {"type": "route", "mode": "agent"}

        types = [e["type"] for e in events]
        assert "thought" in types
        assert "action" in types
        assert "observation" in types
        assert types.count("token") == 2
        assert types[-2] == "sources"
        assert types[-1] == "done"

        done = events[-1]
        assert "session_id" in done
        assert done["session_id"]

        sources = events[-2]
        assert sources["sources"][0]["doc_id"] == "d1"

    @pytest.mark.asyncio
    async def test_stream_reasoning_false면_추론이벤트_생략(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="thought", content="t"),
            _FakeAgentEvent(type="action", content="a"),
            _FakeAgentEvent(type="observation", content="o"),
            _FakeAgentEvent(type="token", content="ans"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator(stream_reasoning=False)
        events = await _collect(orch.handle_auto("비교해줘"))

        types = [e["type"] for e in events]
        assert "thought" not in types
        assert "action" not in types
        assert "observation" not in types
        assert "token" in types

    @pytest.mark.asyncio
    async def test_observation_300자_초과시_truncate(self):
        long_obs = "x" * 500
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="observation", content=long_obs),
            _FakeAgentEvent(type="token", content="ans"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator()
        events = await _collect(orch.handle_auto("비교해줘"))

        obs_events = [e for e in events if e["type"] == "observation"]
        assert len(obs_events) == 1
        truncated = obs_events[0]["content"]
        assert truncated.endswith("...")
        assert len(truncated) < len(long_obs)  # truncation 발생 확인 — 정확 임계값에는 의존하지 않음

    @pytest.mark.asyncio
    async def test_agent_error_이벤트는_오류_토큰으로_변환(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="error", content="boom"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator()
        events = await _collect(orch.handle_auto("비교"))

        token_events = [e for e in events if e["type"] == "token"]
        assert any("오류" in e["content"] for e in token_events)

    @pytest.mark.asyncio
    async def test_planner_enabled_false_면_legacy_사용(self, monkeypatch):
        """planner_enabled=False일 때 Planner가 instantiate조차 되지 않아야 함."""
        from rag_factory.rag.agent import planner as planner_mod

        class _ShouldNotBeCalled:
            def __init__(self, **_kwargs):
                raise AssertionError("Planner가 호출되면 안됨")

            async def plan(self, query):
                raise AssertionError("plan()이 호출되면 안됨")

        monkeypatch.setattr(planner_mod, "Planner", _ShouldNotBeCalled)

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="legacy"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator(planner_enabled=False)
        events = await _collect(orch.handle_auto("비교"))
        assert any(e.get("content") == "legacy" for e in events)


# ---------------------------------------------------------------------------
# 세션 부수효과
# ---------------------------------------------------------------------------


class TestSessionSideEffects:
    """agent/planner 경로 × handle_auto/handle_agent dispatcher가 모두 user/assistant 메시지를 세션에 기록합니다."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dispatcher,path,query,assistant_tokens",
        [
            ("handle_auto", "legacy", "A와 B 비교", ["안녕", "하세요"]),
            ("handle_auto", "planner", "비교 질의", ["최종 ", "답변"]),
            ("handle_agent", "legacy", "질문", ["안녕"]),
            ("handle_agent", "planner", "질의", ["agent_via_planner"]),
        ],
    )
    async def test_user_assistant_메시지_기록(
        self, monkeypatch, dispatcher, path, query, assistant_tokens
    ):
        expected_assistant = "".join(assistant_tokens)

        if path == "legacy":
            _FakeAgentLoop.script = [
                *(_FakeAgentEvent(type="token", content=t) for t in assistant_tokens),
                _FakeAgentEvent(type="done", metadata={"sources": []}),
            ]
            app_state = _make_app_state()
            orch = _make_orchestrator(app_state=app_state, planner_enabled=False)
        else:  # planner
            plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
            fixtures = _PlannerPathFixtures(
                monkeypatch,
                plan=plan,
                tool_script=[_FakeToolResult(text="r", sources=[])],
                synthesis_tokens=assistant_tokens,
            )
            app_state = fixtures.app_state
            orch = _make_orchestrator(planner_enabled=True, app_state=app_state)

        events = await _collect(getattr(orch, dispatcher)(query))

        sid = events[-1]["session_id"]
        _, msgs = app_state.agent_session_manager.get_or_create(sid)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == query
        assert msgs[1].role == "assistant"
        assert msgs[1].content == expected_assistant

    @pytest.mark.asyncio
    async def test_답변없으면_assistant_메시지_미기록(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        app_state = _make_app_state()
        orch = _make_orchestrator(app_state=app_state)
        events = await _collect(orch.handle_auto("비교"))

        sid = events[-1]["session_id"]
        _, msgs = app_state.agent_session_manager.get_or_create(sid)
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    @pytest.mark.asyncio
    async def test_기존_세션_재사용(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="x"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        app_state = _make_app_state()
        sm = app_state.agent_session_manager
        existing_sid = sm.create_session()
        sm.add_message(existing_sid, Message(role="user", content="이전 질문"))

        orch = _make_orchestrator(app_state=app_state)
        events = await _collect(orch.handle_auto("비교", existing_sid))

        sid = events[-1]["session_id"]
        assert sid == existing_sid
        _, msgs = sm.get_or_create(sid)
        assert len(msgs) == 3  # 이전 + 새 user + assistant


# ---------------------------------------------------------------------------
# 이벤트 계약: route event 가장 먼저
# ---------------------------------------------------------------------------


class TestEventContract:
    """SSE 클라이언트가 의존하는 이벤트 순서 계약."""

    @pytest.mark.asyncio
    async def test_route_이벤트가_항상_첫_이벤트(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="x"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        for query in ("단순", "A와 B 비교"):
            orch = _make_orchestrator()
            events = await _collect(orch.handle_auto(query))
            assert events[0]["type"] == "route"

    @pytest.mark.asyncio
    async def test_done_이벤트가_항상_마지막(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="x"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        for query in ("단순", "A와 B 비교"):
            orch = _make_orchestrator()
            events = await _collect(orch.handle_auto(query))
            assert events[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# Planner 경로 (planner_enabled=True)
# ---------------------------------------------------------------------------


class _FakeToolResult:
    def __init__(self, text: str, sources: list[dict] | None = None):
        self.text = text
        self.sources = sources or []


class _FakeToolRegistry:
    """script를 순서대로 반환하는 테스트용 ToolRegistry."""

    def __init__(self, script: list[_FakeToolResult]):
        self._script = list(script)
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, args: dict):
        self.calls.append((name, dict(args)))
        if not self._script:
            return _FakeToolResult(text="(빈 결과)", sources=[])
        return self._script.pop(0)


class _FakeStreamResponse:
    """httpx.AsyncClient.stream()이 반환하는 async context manager 모방."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _make_stream_lines(tokens: list[str]) -> list[str]:
    """답변 합성용 Ollama 스트림 라인을 생성합니다."""
    import json as _json

    lines = [
        _json.dumps({"response": t, "done": False}) for t in tokens
    ]
    lines.append(_json.dumps({"response": "", "done": True}))
    return lines


class _PlannerPathFixtures:
    """planner/verifier/http_client를 한번에 세팅하는 헬퍼."""

    def __init__(
        self,
        monkeypatch,
        *,
        plan,
        tool_script: list[_FakeToolResult],
        verifier_decisions=None,
        synthesis_tokens=None,
        synthesis_scripts: list[list[str]] | None = None,
    ):
        from rag_factory.rag.agent import planner as planner_mod
        from rag_factory.rag.agent import verifier as verifier_mod

        # Planner.plan()을 상수 반환으로 패치
        class _FakePlanner:
            def __init__(self, **_kwargs):
                pass

            async def plan(self, query):
                return plan

        monkeypatch.setattr(planner_mod, "Planner", _FakePlanner)

        # Verifier.evaluate()를 순차 반환으로 패치
        decisions = list(verifier_decisions or [])

        class _FakeVerifier:
            def __init__(self, **_kwargs):
                pass

            async def evaluate(self, query, context):
                if not decisions:
                    return verifier_mod.VerifierDecision(
                        sufficient=True, reason="default"
                    )
                return decisions.pop(0)

        monkeypatch.setattr(verifier_mod, "Verifier", _FakeVerifier)

        # ToolRegistry + http_client.stream 모킹
        self.tool_registry = _FakeToolRegistry(tool_script)
        self.app_state = _make_app_state()
        self.app_state.agent_tool_registry = self.tool_registry

        # synthesis_scripts 우선: 여러 합성 호출에 대해 각각 다른 토큰 시퀀스 반환
        mock_http = MagicMock()
        if synthesis_scripts:
            scripts = [_make_stream_lines(s) for s in synthesis_scripts]
            responses = [_FakeStreamResponse(lines) for lines in scripts]
            mock_http.stream = MagicMock(side_effect=responses)
        else:
            lines = _make_stream_lines(synthesis_tokens or ["답변"])
            mock_http.stream = MagicMock(return_value=_FakeStreamResponse(lines))
        self.app_state.http_client = mock_http


def _make_plan(steps, strategy="fact", rationale="test plan"):
    from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

    return ExecutionPlan(
        strategy=strategy,
        steps=[PlanStep(**s) if isinstance(s, dict) else s for s in steps],
        rationale=rationale,
    )


class TestPlannerRoute:
    """planner_enabled=True 경로 — plan → execute → verify → synthesize."""

    @pytest.mark.asyncio
    async def test_fact_plan_실행_후_합성(self, monkeypatch):
        plan = _make_plan([
            {"tool": "search", "args": {"query": "x"}, "reason": "찾기"},
        ])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="검색 결과 텍스트",
                    sources=[{"doc_id": "d1", "score": 0.9, "content": "c1"}],
                ),
            ],
            synthesis_tokens=["안녕", "하세요"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("복합 질의 비교"))

        # route가 agent로 결정됨
        assert events[0] == {"type": "route", "mode": "agent"}

        types = [e["type"] for e in events]
        assert "thought" in types  # plan summary + step.reason
        assert "action" in types
        assert "observation" in types

        # token 이벤트로 합성 결과가 스트리밍됨
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "안녕하세요"

        # sources는 plan step 결과에서 수집됨
        sources_event = [e for e in events if e["type"] == "sources"][0]
        assert sources_event["sources"][0]["doc_id"] == "d1"

        # done에 session_id 포함
        assert events[-1]["type"] == "done"
        assert events[-1]["session_id"]

        # tool_registry가 plan의 도구로 호출됨
        assert fixtures.tool_registry.calls == [("search", {"query": "x"})]

    @pytest.mark.asyncio
    async def test_verifier_필요시_추가_검색(self, monkeypatch):
        from rag_factory.rag.agent.verifier import VerifierDecision

        plan = _make_plan([
            {"tool": "search", "args": {"query": "초기 검색"}, "reason": "첫 검색"},
        ])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(text="초기 결과 (부족)", sources=[]),
                _FakeToolResult(
                    text="보충 결과",
                    sources=[{"doc_id": "d2", "score": 0.8, "content": "c"}],
                ),
            ],
            verifier_decisions=[
                VerifierDecision(
                    sufficient=False,
                    reason="수치 부족",
                    suggested_query="구체 수치",
                ),
                VerifierDecision(sufficient=True, reason="충분"),
            ],
            synthesis_tokens=["답"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            verifier_enabled=True,
            verifier_max_repairs=1,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교 질의"))

        # repair search 호출 확인
        assert fixtures.tool_registry.calls == [
            ("search", {"query": "초기 검색"}),
            ("search", {"query": "구체 수치"}),
        ]

        # repair 이벤트가 iteration=2로 emit됨
        repair_thoughts = [
            e for e in events
            if e["type"] == "thought" and "추가 검색" in e.get("content", "")
        ]
        assert len(repair_thoughts) == 1

        # d2 source 포함
        sources = [e for e in events if e["type"] == "sources"][0]["sources"]
        assert any(s["doc_id"] == "d2" for s in sources)

    @pytest.mark.asyncio
    async def test_verifier_비활성시_repair_안함(self, monkeypatch):
        from rag_factory.rag.agent.verifier import VerifierDecision

        plan = _make_plan([
            {"tool": "search", "args": {"query": "q"}},
        ])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="결과", sources=[])],
            verifier_decisions=[
                VerifierDecision(
                    sufficient=False,
                    reason="부족",
                    suggested_query="재검색",
                ),
            ],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            verifier_enabled=False,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("비교"))

        # verifier_enabled=False이므로 repair search가 일어나지 않음
        assert fixtures.tool_registry.calls == [("search", {"query": "q"})]

    @pytest.mark.asyncio
    async def test_verifier_max_repairs_존중(self, monkeypatch):
        from rag_factory.rag.agent.verifier import VerifierDecision

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        # 계속 repair 필요하다고 해도 max_repairs=1로 1회만 실행
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(text="r1", sources=[]),
                _FakeToolResult(text="r2", sources=[]),
                _FakeToolResult(text="r3", sources=[]),
            ],
            verifier_decisions=[
                VerifierDecision(
                    sufficient=False,
                    reason="부족",
                    suggested_query="재검색1",
                ),
                VerifierDecision(
                    sufficient=False,
                    reason="부족",
                    suggested_query="재검색2",
                ),
            ],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            verifier_max_repairs=1,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("비교"))

        # 초기 1 + repair 1 = 2번 호출
        assert len(fixtures.tool_registry.calls) == 2

    @pytest.mark.asyncio
    async def test_stream_reasoning_false면_추론_이벤트_생략(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}, "reason": "r"}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            stream_reasoning=False,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        types = [e["type"] for e in events]
        assert "thought" not in types
        assert "action" not in types
        assert "observation" not in types
        assert "token" in types

    @pytest.mark.asyncio
    async def test_fallback_gate_비활성시_planner_가_fallback_실행(self, monkeypatch):
        # legacy_fallback_enabled=False이면 planner fallback 계획도 그대로 실행
        from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

        fallback_plan = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search", args={"query": "비교"})],
            rationale="fallback: parse-error",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=fallback_plan,
            tool_script=[_FakeToolResult(text="결과", sources=[])],
            synthesis_tokens=["답"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            legacy_fallback_enabled=False,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        assert events[-1]["type"] == "done"
        assert fixtures.tool_registry.calls == [("search", {"query": "비교"})]

    @pytest.mark.asyncio
    async def test_tool_실행_예외시_계속_진행(self, monkeypatch):
        plan = _make_plan([
            {"tool": "search", "args": {"query": "q1"}},
            {"tool": "search", "args": {"query": "q2"}},
        ])

        # 첫번째 실행에서 예외를 던지는 registry. 두번째 step의 sources에 식별
        # 가능한 doc_id를 부여해 meta-요약(검색 완료 N건, doc IDs: …)에서 본 step의
        # observation임을 확인할 수 있게 함.
        class _ErrorRegistry(_FakeToolRegistry):
            async def execute(self, name, args):
                self.calls.append((name, dict(args)))
                if len(self.calls) == 1:
                    raise RuntimeError("tool error")
                return _FakeToolResult(
                    text="정상",
                    sources=[{"doc_id": "ok2", "score": 0.85, "content": "x"}],
                )

        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[],
            synthesis_tokens=["ans"],
        )
        fixtures.tool_registry = _ErrorRegistry([])
        fixtures.app_state.agent_tool_registry = fixtures.tool_registry

        orch = _make_orchestrator(
            planner_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        # 두 step 모두 시도됨
        assert len(fixtures.tool_registry.calls) == 2
        # 두번째 step의 observation만 emit됨 (meta-요약 포맷)
        obs = [e for e in events if e["type"] == "observation"]
        assert len(obs) == 1
        assert "ok2" in obs[0]["content"]

    @pytest.mark.asyncio
    async def test_답변은_Ollama_토큰_단위로_즉시_발행(self, monkeypatch):
        """진짜 스트리밍: Ollama가 yield하는 토큰마다 token 이벤트가 1:1로 발행됨."""
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        # 합성 모델이 5개 토큰을 yield하는 시나리오
        synthesis_tokens = ["안녕", "하세", "요. ", "반갑", "습니다"]
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="결과", sources=[])],
            synthesis_tokens=synthesis_tokens,
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        # Ollama 토큰 1개 = SSE 이벤트 1개 (인공 chunking 없음)
        assert tokens == synthesis_tokens
        # 재조립 시 원본과 일치
        assert "".join(tokens) == "".join(synthesis_tokens)


# ---------------------------------------------------------------------------
# handle_agent() — /agent endpoint path (no route event)
# ---------------------------------------------------------------------------


class TestHandleAgent:
    """handle_agent는 라우팅 없이 항상 agent 경로로 진입하며 route 이벤트를 발행하지 않습니다."""

    @pytest.mark.asyncio
    async def test_route_이벤트_없음(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="ans"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator()
        events = await _collect(orch.handle_agent("간단한 질의"))

        # /agent 엔드포인트는 route 이벤트를 발행하지 않아야 함
        assert not any(e.get("type") == "route" for e in events)
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_legacy_경로_기본사용(self):
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="thought", content="검색합니다"),
            _FakeAgentEvent(type="token", content="답변"),
            _FakeAgentEvent(type="done", metadata={"sources": [{"doc_id": "d1", "score": 0.9, "content": "c"}]}),
        ]

        orch = _make_orchestrator(planner_enabled=False)
        events = await _collect(orch.handle_agent("질의"))

        types = [e["type"] for e in events]
        assert "thought" in types
        assert "token" in types
        assert "sources" in types
        assert events[-1]["type"] == "done"
        assert events[-1]["session_id"]

    @pytest.mark.asyncio
    async def test_planner_enabled이면_planner_경로_사용(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["agent_via_planner"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "agent_via_planner" in "".join(tokens)
        assert not any(e.get("type") == "route" for e in events)

# ---------------------------------------------------------------------------
# Phase 2 — Legacy fallback gate
# ---------------------------------------------------------------------------


class TestPersonaIntegration:
    """Phase 6 — Persona가 intent별 synthesis prompt와 도구 권한에 반영됨."""

    def _make_router_with_intent(self, intent_label: str, confidence: float = 0.9):
        from rag_factory.rag.agent.intent_classifier import IntentDecision
        from rag_factory.rag.agent.router import QueryRouter

        class _FakeClassifier:
            async def classify(self, query):
                return IntentDecision(intent=intent_label, confidence=confidence)

        return QueryRouter(agent_enabled=True, intent_classifier=_FakeClassifier())

    @pytest.mark.asyncio
    async def test_comparator_persona_synthesis_prompt_사용(self, monkeypatch):
        """comparative intent → Comparator persona → COMPARATOR_SYNTHESIS_PROMPT 주입."""

        plan = _make_plan(
            [
                {"tool": "compare", "args": {"query_a": "x", "query_b": "y"}},
            ],
            strategy="compare",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="비교 결과", sources=[])],
            synthesis_tokens=["표 답변"],
        )

        router = self._make_router_with_intent("comparative")
        orch = _make_orchestrator(
            planner_enabled=True,
            personas_enabled=True,
            router=router,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("A와 B 비교"))

        call_kwargs = fixtures.app_state.http_client.stream.call_args.kwargs
        prompt = call_kwargs["json"]["prompt"]
        assert "Comparator" in prompt

    @pytest.mark.asyncio
    async def test_researcher_persona_도구_화이트리스트_적용(self, monkeypatch):
        """Researcher는 compare 도구 금지 — plan의 compare step이 필터링됨."""

        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "x"}},
                {"tool": "compare", "args": {"query_a": "a", "query_b": "b"}},
            ]
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="검색만", sources=[])],
            synthesis_tokens=["답"],
        )

        router = self._make_router_with_intent("factual")
        orch = _make_orchestrator(
            planner_enabled=True,
            personas_enabled=True,
            router=router,
            app_state=fixtures.app_state,
        )
        # "분석" 키워드로 agent 경로 강제 (LLM은 factual 반환 → Researcher persona 선택,
        # keyword가 agent 경로로 확정). "비교" 키워드는 orchestrator의 tabular_intent
        # override가 작동해 Comparator로 강제 전환되므로 Researcher 화이트리스트 검증 불가.
        await _collect(orch.handle_auto("분석 관련 사실 질의"))

        # Researcher 도구 화이트리스트로 compare step은 filter됨 — search만 실행
        assert fixtures.tool_registry.calls == [("search", {"query": "x"})]

    @pytest.mark.asyncio
    async def test_personas_비활성이면_기본_synthesis(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"],
        )

        router = self._make_router_with_intent("comparative")
        orch = _make_orchestrator(
            planner_enabled=True,
            personas_enabled=False,  # 비활성
            router=router,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("비교"))

        prompt = fixtures.app_state.http_client.stream.call_args.kwargs["json"]["prompt"]
        # Comparator persona prompt가 아닌 기본 prompt
        assert "Comparator" not in prompt

    @pytest.mark.asyncio
    async def test_analyst_persona_analytical_intent(self, monkeypatch):
        plan = _make_plan(
            [{"tool": "search", "args": {"query": "q"}}], strategy="decompose"
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"],
        )

        router = self._make_router_with_intent("analytical")
        orch = _make_orchestrator(
            planner_enabled=True,
            personas_enabled=True,
            router=router,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("왜 이렇게 됐나"))

        prompt = fixtures.app_state.http_client.stream.call_args.kwargs["json"]["prompt"]
        assert "Analyst" in prompt

    @pytest.mark.asyncio
    async def test_procedural_persona_intent(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"],
        )

        router = self._make_router_with_intent("procedural")
        orch = _make_orchestrator(
            planner_enabled=True,
            personas_enabled=True,
            router=router,
            app_state=fixtures.app_state,
        )
        # "변경" 키워드로 agent 경로 확보
        await _collect(orch.handle_auto("변경된 절차는 어떻게 하나요"))

        prompt = fixtures.app_state.http_client.stream.call_args.kwargs["json"]["prompt"]
        assert "Procedural" in prompt


class TestClarifierIntegration:
    """Phase 11 — Clarifier: ambiguous 의도 시 역질문 반환."""

    def _make_router_with_intent(self, intent_label: str, confidence: float = 0.8):
        from rag_factory.rag.agent.intent_classifier import IntentDecision
        from rag_factory.rag.agent.router import QueryRouter

        class _FakeClassifier:
            async def classify(self, query):
                return IntentDecision(intent=intent_label, confidence=confidence)

        return QueryRouter(agent_enabled=True, intent_classifier=_FakeClassifier())

    def _patch_clarifier(self, monkeypatch, *, questions, is_fallback=False):
        from rag_factory.rag.agent.personas import clarifier as clarifier_mod
        from rag_factory.rag.agent.personas.base import PersonaResult

        class _FakeClarifier(clarifier_mod.Clarifier):
            def __init__(self, **_kwargs):
                pass

            async def generate_questions(self, query, history=""):
                return PersonaResult(
                    kind="clarification",
                    questions=list(questions),
                    metadata={"is_fallback": is_fallback, "persona": "clarifier"},
                )

        monkeypatch.setattr(clarifier_mod, "Clarifier", _FakeClarifier)

    @pytest.mark.asyncio
    async def test_ambiguous_의도는_clarification_이벤트_발행(self, monkeypatch):
        self._patch_clarifier(
            monkeypatch,
            questions=["주제는?", "시점은?"],
        )

        router = self._make_router_with_intent("ambiguous")
        orch = _make_orchestrator(
            clarifier_enabled=True,
            router=router,
        )
        events = await _collect(orch.handle_auto("그거"))

        types = [e["type"] for e in events]
        assert types[0] == "route"
        assert "clarification" in types
        assert types[-1] == "done"

        clar = [e for e in events if e["type"] == "clarification"][0]
        assert clar["questions"] == ["주제는?", "시점은?"]
        assert clar["is_fallback"] is False

    @pytest.mark.asyncio
    async def test_ambiguous_clarifier_비활성이면_일반_경로(self, monkeypatch):
        # Clarifier가 호출되면 실패하도록
        from rag_factory.rag.agent.personas import clarifier as clarifier_mod

        class _ShouldNotBeCalled:
            def __init__(self, **_kwargs):
                raise AssertionError("Clarifier가 호출되면 안됨")

            async def generate_questions(self, query, history=""):
                raise AssertionError("호출되면 안됨")

        monkeypatch.setattr(clarifier_mod, "Clarifier", _ShouldNotBeCalled)

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="일반답변"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        router = self._make_router_with_intent("ambiguous")
        orch = _make_orchestrator(
            clarifier_enabled=False,
            router=router,
        )
        events = await _collect(orch.handle_auto("그거"))

        types = [e["type"] for e in events]
        assert "clarification" not in types
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_factual_의도는_clarifier_트리거_안함(self, monkeypatch):
        from rag_factory.rag.agent.personas import clarifier as clarifier_mod

        class _ShouldNotBeCalled:
            def __init__(self, **_kwargs):
                raise AssertionError("호출되면 안됨")

            async def generate_questions(self, query, history=""):
                raise AssertionError()

        monkeypatch.setattr(clarifier_mod, "Clarifier", _ShouldNotBeCalled)

        router = self._make_router_with_intent("factual", confidence=0.95)
        orch = _make_orchestrator(
            clarifier_enabled=True,
            router=router,
        )
        events = await _collect(orch.handle_auto("사실"))

        # factual 고신뢰 → simple 경로
        assert events[0]["mode"] == "simple"
        assert not any(e["type"] == "clarification" for e in events)

    @pytest.mark.asyncio
    async def test_clarifier_세션_기록(self, monkeypatch):
        self._patch_clarifier(monkeypatch, questions=["Q1"])

        router = self._make_router_with_intent("ambiguous")
        app_state = _make_app_state()
        orch = _make_orchestrator(
            clarifier_enabled=True,
            router=router,
            app_state=app_state,
        )
        events = await _collect(orch.handle_auto("모호한 질의"))

        sid = events[-1]["session_id"]
        _, msgs = app_state.agent_session_manager.get_or_create(sid)
        # user + assistant(요약) 2개
        assert len(msgs) == 2
        assert msgs[0].content == "모호한 질의"
        assert msgs[1].role == "assistant"
        assert "Q1" in msgs[1].content


class TestParallelSteps:
    """Phase 3-b — multi-search plan의 병렬 실행."""

    @pytest.mark.asyncio
    async def test_2개_search_step은_병렬_실행_이벤트는_순서대로(self, monkeypatch):
        """병렬 실행이 켜진 경우, action/observation이 plan 순서대로 emit."""

        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "A"}, "reason": "A검색"},
                {"tool": "search", "args": {"query": "B"}, "reason": "B검색"},
            ],
            strategy="decompose",
        )

        # 도구 결과는 특정 순서로 반환되지만, 이벤트는 plan 순서대로 emit되어야 함
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="A 결과",
                    sources=[{"doc_id": "a1", "score": 0.9, "content": "A"}],
                ),
                _FakeToolResult(
                    text="B 결과",
                    sources=[{"doc_id": "b1", "score": 0.85, "content": "B"}],
                ),
            ],
            synthesis_tokens=["종합 답변"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            parallel_steps=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("A와 B 비교"))

        # action 이벤트 순서가 plan 순서(A → B)
        actions = [e for e in events if e["type"] == "action"]
        assert len(actions) == 2
        assert actions[0]["input"] == {"query": "A"}
        assert actions[1]["input"] == {"query": "B"}

        # observation 순서도 plan 순서 — observation은 메타 요약이므로 doc_id로 확인
        observations = [e for e in events if e["type"] == "observation"]
        assert "a1" in observations[0]["content"]
        assert "b1" in observations[1]["content"]

        # iteration 번호가 plan 순서대로 1, 2
        assert actions[0]["iteration"] == 1
        assert actions[1]["iteration"] == 2

    @pytest.mark.asyncio
    async def test_2개_search_step은_실제로_concurrent하게_실행(self, monkeypatch):
        """asyncio.gather를 사용하므로 두 도구 호출의 실행 시간이 겹쳐야 함."""

        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "A"}},
                {"tool": "search", "args": {"query": "B"}},
            ],
            strategy="decompose",
        )

        concurrency_log: list[tuple[str, str]] = []

        class _TrackingRegistry:
            def __init__(self):
                self.calls: list[tuple[str, dict]] = []

            async def execute(self, name: str, args: dict):
                self.calls.append((name, dict(args)))
                concurrency_log.append(("start", args["query"]))
                # 짧게 await하여 다른 태스크가 먼저 start할 기회를 줌
                await asyncio.sleep(0.01)
                concurrency_log.append(("end", args["query"]))
                return _FakeToolResult(text=f"결과:{args['query']}", sources=[])

        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[],
            synthesis_tokens=["ans"],
        )
        fixtures.tool_registry = _TrackingRegistry()
        fixtures.app_state.agent_tool_registry = fixtures.tool_registry

        orch = _make_orchestrator(
            planner_enabled=True,
            parallel_steps=True,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("A와 B 비교"))

        # 두 쿼리 모두 start가 end보다 먼저 모두 찍혀야 함 (concurrent).
        # 순차 실행이면 [start A, end A, start B, end B] 패턴.
        # 병렬 실행이면 [start A, start B, end A, end B] (또는 유사 교차).
        start_a_idx = concurrency_log.index(("start", "A"))
        end_a_idx = concurrency_log.index(("end", "A"))
        start_b_idx = concurrency_log.index(("start", "B"))

        # B가 A end 전에 start되어야 함 → 진정한 병렬 실행의 증거
        assert start_b_idx < end_a_idx

    @pytest.mark.asyncio
    async def test_parallel_steps_false이면_직렬_실행(self, monkeypatch):
        """기본값은 직렬 — 기존 동작 유지."""

        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "A"}},
                {"tool": "search", "args": {"query": "B"}},
            ]
        )

        concurrency_log: list[str] = []

        class _SerialRegistry:
            calls: list = []

            async def execute(self, name, args):
                self.calls.append((name, dict(args)))
                concurrency_log.append(f"start:{args['query']}")
                concurrency_log.append(f"end:{args['query']}")
                return _FakeToolResult(text="r", sources=[])

        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[],
            synthesis_tokens=["ans"],
        )
        fixtures.tool_registry = _SerialRegistry()
        fixtures.app_state.agent_tool_registry = fixtures.tool_registry

        orch = _make_orchestrator(
            planner_enabled=True,
            parallel_steps=False,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("비교"))

        # 직렬 실행 → [start A, end A, start B, end B]
        assert concurrency_log == [
            "start:A",
            "end:A",
            "start:B",
            "end:B",
        ]

    @pytest.mark.asyncio
    async def test_비_search_도구_섞이면_직렬(self, monkeypatch):
        """compare 등 search가 아닌 도구가 있으면 안전하게 직렬 실행."""

        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "A"}},
                {"tool": "compare", "args": {"query_a": "x", "query_b": "y"}},
            ]
        )

        concurrency_log: list[str] = []

        class _MixedRegistry:
            calls: list = []

            async def execute(self, name, args):
                self.calls.append((name, dict(args)))
                concurrency_log.append(f"start:{name}")
                concurrency_log.append(f"end:{name}")
                return _FakeToolResult(text="r", sources=[])

        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[],
            synthesis_tokens=["ans"],
        )
        fixtures.tool_registry = _MixedRegistry()
        fixtures.app_state.agent_tool_registry = fixtures.tool_registry

        orch = _make_orchestrator(
            planner_enabled=True,
            parallel_steps=True,  # 켜져있어도 search만이 아니므로 직렬로 fallback
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("비교"))

        # 직렬 패턴 확인
        assert concurrency_log == [
            "start:search",
            "end:search",
            "start:compare",
            "end:compare",
        ]

    @pytest.mark.asyncio
    async def test_단일_step은_병렬화_무의미_직렬(self, monkeypatch):
        """step이 1개뿐이면 parallel_steps=True여도 직렬로 실행."""

        plan = _make_plan([{"tool": "search", "args": {"query": "x"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            parallel_steps=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        # 정상적으로 done 이벤트까지 emit
        assert events[-1]["type"] == "done"
        assert fixtures.tool_registry.calls == [("search", {"query": "x"})]

    @pytest.mark.asyncio
    async def test_병렬_실행_중_예외는_해당_step만_건너뜀(self, monkeypatch):
        """하나의 병렬 step이 실패해도 나머지는 정상 처리."""

        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "fail"}},
                {"tool": "search", "args": {"query": "ok"}},
            ]
        )

        class _PartialFailureRegistry:
            calls: list = []

            async def execute(self, name, args):
                self.calls.append((name, dict(args)))
                if args["query"] == "fail":
                    raise RuntimeError("tool boom")
                return _FakeToolResult(
                    text="성공",
                    sources=[{"doc_id": "ok", "score": 0.9, "content": "c"}],
                )

        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[],
            synthesis_tokens=["ans"],
        )
        fixtures.tool_registry = _PartialFailureRegistry()
        fixtures.app_state.agent_tool_registry = fixtures.tool_registry

        orch = _make_orchestrator(
            planner_enabled=True,
            parallel_steps=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        # 성공한 step의 observation만 emit됨 — meta-요약에서 doc_id로 확인
        observations = [e for e in events if e["type"] == "observation"]
        assert len(observations) == 1
        assert "ok" in observations[0]["content"]

        # sources는 성공한 step만
        sources = [e for e in events if e["type"] == "sources"][0]["sources"]
        assert len(sources) == 1
        assert sources[0]["doc_id"] == "ok"


class TestSessionSourceReuse:
    """Phase 3-a — 이전 턴 sources를 synthesis에 주입."""

    @pytest.mark.asyncio
    async def test_planner_경로는_sources를_세션에_저장(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="r",
                    sources=[{"doc_id": "d1", "score": 0.9, "content": "c1"}],
                )
            ],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            session_source_reuse=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        sid = events[-1]["session_id"]
        stored = fixtures.app_state.agent_session_manager.get_last_sources(sid)
        assert len(stored) == 1
        assert stored[0]["doc_id"] == "d1"

    @pytest.mark.asyncio
    async def test_follow_up_턴은_이전_sources를_synthesis에_주입(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="두번째 결과", sources=[])],
            synthesis_tokens=["follow_up_answer"],
        )

        # 세션에 이전 sources 미리 주입
        sm = fixtures.app_state.agent_session_manager
        sid_existing = sm.create_session()
        sm.set_last_sources(
            sid_existing,
            [{"doc_id": "prior_d1", "score": 0.88, "content": "이전 턴에서 참조한 내용"}],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            session_source_reuse=True,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_agent("후속 질문", sid_existing))

        # synthesis prompt에 이전 문서가 포함되어야 함
        call_kwargs = fixtures.app_state.http_client.stream.call_args.kwargs
        prompt = call_kwargs["json"]["prompt"]
        assert "[이전 대화 참조 문서]" in prompt
        assert "prior_d1" in prompt
        assert "이전 턴에서 참조한 내용" in prompt

    @pytest.mark.asyncio
    async def test_session_source_reuse_false이면_주입하지_않음(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="결과", sources=[])],
            synthesis_tokens=["ans"],
        )

        sm = fixtures.app_state.agent_session_manager
        sid_existing = sm.create_session()
        sm.set_last_sources(sid_existing, [{"doc_id": "prior", "content": "이전"}])

        orch = _make_orchestrator(
            planner_enabled=True,
            session_source_reuse=False,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_agent("후속 질문", sid_existing))

        call_kwargs = fixtures.app_state.http_client.stream.call_args.kwargs
        prompt = call_kwargs["json"]["prompt"]
        assert "[이전 대화 참조 문서]" not in prompt

    @pytest.mark.asyncio
    async def test_session_source_reuse_limit_존중(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="r",
                    sources=[{"doc_id": f"d{i}", "score": 0.9, "content": f"c{i}"} for i in range(10)],
                )
            ],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            session_source_reuse=True,
            session_source_reuse_limit=3,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        sid = events[-1]["session_id"]
        stored = fixtures.app_state.agent_session_manager.get_last_sources(sid)
        assert len(stored) == 3


class TestLegacyFallbackGate:
    """Planner가 is_fallback을 반환할 때 legacy AgentLoop로 자동 전환."""

    @pytest.mark.asyncio
    async def test_fallback_plan이면_legacy로_전환(self, monkeypatch):
        from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

        fallback_plan = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search", args={"query": "비교"})],
            rationale="fallback: llm-error",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=fallback_plan,
            tool_script=[],  # planner 경로에서 호출되면 안됨
            synthesis_tokens=["SHOULD_NOT_APPEAR"],
        )

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="LEGACY_TOKEN"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator(
            planner_enabled=True,
            legacy_fallback_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        # legacy 경로 실행됨
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "LEGACY_TOKEN" in tokens
        assert "SHOULD_NOT_APPEAR" not in "".join(tokens)

        # planner 도구 호출이 일어나지 않음
        assert fixtures.tool_registry.calls == []

    @pytest.mark.asyncio
    async def test_fallback_plan이어도_gate_off면_planner_실행(self, monkeypatch):
        from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

        fallback_plan = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search", args={"query": "x"})],
            rationale="fallback: parse-error",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=fallback_plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["planner_path"],
        )

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="LEGACY_SHOULD_NOT_RUN"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator(
            planner_enabled=True,
            legacy_fallback_enabled=False,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        joined = "".join(tokens)
        assert "planner_path" in joined
        assert "LEGACY_SHOULD_NOT_RUN" not in joined
        assert fixtures.tool_registry.calls == [("search", {"query": "x"})]

    @pytest.mark.asyncio
    async def test_정상_plan은_fallback_gate_영향없음(self, monkeypatch):
        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}], rationale="정상 계획")
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["ok"],
        )

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="LEGACY_SHOULD_NOT_RUN"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator(
            planner_enabled=True,
            legacy_fallback_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        joined = "".join(tokens)
        assert "ok" in joined
        assert "LEGACY_SHOULD_NOT_RUN" not in joined

    @pytest.mark.asyncio
    async def test_fallback_전환시_세션_이중기록_방지(self, monkeypatch):
        from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

        fallback_plan = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search", args={"query": "x"})],
            rationale="fallback: llm-error",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=fallback_plan,
            tool_script=[],
            synthesis_tokens=[],
        )

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="ans"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        orch = _make_orchestrator(
            planner_enabled=True,
            legacy_fallback_enabled=True,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_auto("비교 질의"))

        sid = events[-1]["session_id"]
        _, msgs = fixtures.app_state.agent_session_manager.get_or_create(sid)
        # user 1 + assistant 1 = 2 (이중 기록되면 3이상)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "비교 질의"
        assert msgs[1].role == "assistant"


# ---------------------------------------------------------------------------
# Chitchat 경로 — RAG 검색 우회 LLM 직답
# ---------------------------------------------------------------------------


class TestChitchatRoute:
    """잡담 라우팅: planner/verifier/Qdrant 호출 없이 LLM 직답 + 세션 기록."""

    @pytest.mark.asyncio
    async def test_인사_질의는_chitchat_route_이벤트(self, monkeypatch):
        # http_client.stream을 chitchat 합성용으로 모킹.
        app_state = _make_app_state()
        mock_http = MagicMock()
        lines = _make_stream_lines(["안녕", "하세요!"])
        mock_http.stream = MagicMock(return_value=_FakeStreamResponse(lines))
        app_state.http_client = mock_http

        orch = _make_orchestrator(
            planner_enabled=True,  # planner 켜져 있어도 chitchat이면 우회되어야 함
            app_state=app_state,
        )
        events = await _collect(orch.handle_auto("안녕하세요"))

        # 첫 이벤트는 route(mode=chitchat)
        assert events[0] == {"type": "route", "mode": "chitchat"}
        # 마지막 이벤트는 done
        assert events[-1]["type"] == "done"
        # 토큰이 chitchat 합성기에서 흘러나옴
        tokens = [e for e in events if e["type"] == "token"]
        assert tokens, "chitchat이 token 이벤트를 발행해야 함"
        assert "".join(t["content"] for t in tokens) == "안녕하세요!"

        # ToolRegistry는 chitchat에서 호출되지 않음 (Qdrant 우회 검증).
        app_state.agent_tool_registry.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_chitchat_세션_기록(self, monkeypatch):
        app_state = _make_app_state()
        mock_http = MagicMock()
        lines = _make_stream_lines(["반갑습니다."])
        mock_http.stream = MagicMock(return_value=_FakeStreamResponse(lines))
        app_state.http_client = mock_http

        orch = _make_orchestrator(app_state=app_state)
        events = await _collect(orch.handle_auto("안녕"))

        sid = events[-1]["session_id"]
        _, msgs = app_state.agent_session_manager.get_or_create(sid)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "안녕"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "반갑습니다."

    @pytest.mark.asyncio
    async def test_도메인_질의는_chitchat_아님(self, monkeypatch):
        """도메인 키워드를 포함한 질의는 잡담 패턴에 매치되지 않아 simple/agent로 라우팅."""
        orch = _make_orchestrator()
        events = await _collect(orch.handle_auto("RFP에서 NMS 개발자가 무엇을 만들어야 하는지"))

        # chitchat이 아니라 simple로 라우팅되어 _simple_stream_fixture를 통과.
        assert events[0]["mode"] != "chitchat"


# ---------------------------------------------------------------------------
# 거부 게이트 (Refusal gate) — 무관한 chunk 위 합성 방지
# ---------------------------------------------------------------------------


class TestRefusalGate:
    """rag.agent.refusal_min_score > 0이고 best score가 미만이면 거부 응답."""

    @pytest.mark.asyncio
    async def test_빈_sources면_거부_응답(self, monkeypatch):
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="(빈 결과)", sources=[])],
            synthesis_tokens=["이 토큰은 yield되지 않아야 함"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.15,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == [_REFUSAL_MESSAGE]
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_낮은_score_sources면_거부_응답(self, monkeypatch):
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="무관 텍스트",
                    sources=[
                        {"doc_id": "x", "score": 0.05, "content": "noise"},
                    ],
                )
            ],
            synthesis_tokens=["yield되지 않아야 함"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.15,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == [_REFUSAL_MESSAGE]
        # 검토한 문서는 sources 이벤트로 노출
        sources_evt = [e for e in events if e["type"] == "sources"]
        assert len(sources_evt) == 1
        assert sources_evt[0]["sources"][0]["doc_id"] == "x"

    @pytest.mark.asyncio
    async def test_충분한_score면_거부_없이_정상_합성(self, monkeypatch):
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="관련 텍스트",
                    sources=[
                        {"doc_id": "ok", "score": 0.8, "content": "good"},
                    ],
                )
            ],
            synthesis_tokens=["정상", " 답변"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.15,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert _REFUSAL_MESSAGE not in tokens
        assert "".join(tokens) == "정상 답변"

    @pytest.mark.asyncio
    async def test_threshold_0이면_게이트_비활성(self, monkeypatch):
        """기본값 0 — 빈 sources여도 합성을 시도해야 함."""
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="(빈)", sources=[])],
            synthesis_tokens=["답변"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.0,  # 명시적으로 OFF
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert _REFUSAL_MESSAGE not in tokens
        assert "답변" in "".join(tokens)

    @pytest.mark.asyncio
    async def test_low_floor_0_01에서_corpus_relative_score_통과(self, monkeypatch):
        """2026-05 calibration — 0.01 floor에서 RFP corpus score 0.03대도 통과.

        Cross-encoder reranker 출력이 corpus·쿼리에 따라 0.03 ~ 0.9 범위로 변동.
        새 default 0.01은 "0에 가까운 garbage만 차단"하는 안전 하한 — 0.0315 같은
        valid corpus-relative score는 통과해 합성으로 진행해야 한다.
        """
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="RFP 컨텍스트",
                    sources=[
                        # 실측 corpus best_score (0.0315 ~ 0.0328 range)
                        {"doc_id": "rfp", "score": 0.0328, "content": "rfp chunk"},
                    ],
                )
            ],
            synthesis_tokens=["정상", " 답변"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.01,  # 새 default
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert _REFUSAL_MESSAGE not in tokens
        assert "정상" in "".join(tokens)

    @pytest.mark.asyncio
    async def test_garbage_score_0_005는_0_01_floor에서_거부(self, monkeypatch):
        """0.01 floor — 0에 가까운(0.005) score는 garbage로 간주, 거부."""
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="무관 잡음",
                    sources=[
                        {"doc_id": "noise", "score": 0.005, "content": "noise"},
                    ],
                )
            ],
            synthesis_tokens=["yield되지 않아야 함"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.01,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == [_REFUSAL_MESSAGE]

    @pytest.mark.asyncio
    async def test_relative_margin_signal_noise_분리_안되면_거부(self, monkeypatch):
        """동적 게이트 — best가 나머지 평균 대비 충분히 크지 않으면 거부.

        scores = [0.10, 0.09, 0.10] → best=0.10, mean(rest)=(0.09+0.10)/2=0.095.
        margin=0.20 → threshold = 0.095 * 1.20 = 0.114. best=0.10 < 0.114 → 거부.
        절대 score는 낮지 않은데(0.10) "1등이 나머지와 구별되지 않는" 케이스.
        """
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="구분 없음",
                    sources=[
                        {"doc_id": "a", "score": 0.10, "content": "top"},
                        {"doc_id": "b", "score": 0.09, "content": "near"},
                        {"doc_id": "c", "score": 0.10, "content": "tied"},
                    ],
                )
            ],
            synthesis_tokens=["yield되지 않아야 함"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.0,
            refusal_relative_margin=0.20,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == [_REFUSAL_MESSAGE]

    @pytest.mark.asyncio
    async def test_relative_margin_signal_분리되면_통과(self, monkeypatch):
        """best가 나머지 대비 명확히 크면 동적 게이트 통과."""
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        # scores=[0.50, 0.10, 0.05] → best=0.50, mean(rest)=0.075.
        # margin=0.20 → threshold=0.075*1.20=0.09. best=0.50 > 0.09 → 통과.
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="명확한 1등",
                    sources=[
                        {"doc_id": "a", "score": 0.50, "content": "top"},
                        {"doc_id": "b", "score": 0.10, "content": "weak"},
                        {"doc_id": "c", "score": 0.05, "content": "weak"},
                    ],
                )
            ],
            synthesis_tokens=["정상", " 답변"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.0,
            refusal_relative_margin=0.20,
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert _REFUSAL_MESSAGE not in tokens
        assert "정상" in "".join(tokens)

    @pytest.mark.asyncio
    async def test_relative_margin_single_source면_무시(self, monkeypatch):
        """source가 1개뿐이면 비교 대상 없음 → 상대 게이트 발동 안 함."""
        from rag_factory.rag.agent.orchestrator import _REFUSAL_MESSAGE

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(
                    text="단일",
                    sources=[{"doc_id": "only", "score": 0.20, "content": "x"}],
                )
            ],
            synthesis_tokens=["답변"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            refusal_min_score=0.0,
            refusal_relative_margin=0.50,  # 활성이지만 rest 비어 있어 무시
            app_state=fixtures.app_state,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert _REFUSAL_MESSAGE not in tokens
        assert "답변" in "".join(tokens)


# ---------------------------------------------------------------------------
# Corpus override 안전망 — agent 경로
# ---------------------------------------------------------------------------


class _StubEmbedding:
    """encode → numpy ndarray-like(.tolist()) 모방."""

    def encode(self, query, prompt_name=None, show_progress_bar=False):
        class _Vec:
            def tolist(self):
                return [0.0]

        return _Vec()


class _StubQdrantClient:
    """query_points → score 고정값."""

    def __init__(self, score: float):
        self._score = score

    def query_points(self, collection_name, query, limit, with_payload=False):
        class _Point:
            def __init__(self, score):
                self.score = score

        class _Result:
            def __init__(self, points):
                self.points = points

        return _Result([_Point(self._score)])


class TestAgentCorpusOverride:
    """agent 경로의 corpus override 안전망 — 키워드 없이 LLM-only로 agent 분류된
    경우 corpus 유사도가 임계 이상이면 simple로 정정합니다."""

    def _make_router(self, intent: str, conf: float = 0.90):
        from rag_factory.rag.agent.intent_classifier import IntentDecision
        from rag_factory.rag.agent.router import QueryRouter

        class _FakeClassifier:
            async def classify(self, query):
                return IntentDecision(intent=intent, confidence=conf)

        return QueryRouter(agent_enabled=True, intent_classifier=_FakeClassifier())

    def _make_app_state_with_corpus(self, score: float, collection: str = "c1"):
        app_state = _make_app_state()
        app_state.embedding_model = _StubEmbedding()
        app_state.qdrant_client = _StubQdrantClient(score)
        return app_state

    @pytest.mark.asyncio
    async def test_exploratory_intent도_corpus_매칭_되면_simple로(self):
        """exploratory(키워드 없음) → corpus 0.59 ≥ 0.55 → simple 정정."""

        router = self._make_router("exploratory", conf=0.90)
        app_state = self._make_app_state_with_corpus(score=0.59)
        # collection_name 참조용 — config의 rag.collection_name이 필요.
        config = _make_config(in_domain_score_threshold=0.55)
        config.rag.collection_name = "c1"

        captured: list[str] = []

        async def _simple(q: str):
            captured.append(q)
            yield {"type": "token", "content": "SIMPLE_OK"}
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=router,
            app_state=app_state,
            config=config,
            ollama_model="t",
            api_base="http://x",
            rag_max_tokens=-1,
            simple_stream_fn=_simple,
        )
        events = await _collect(orch.handle_auto("그래프형식으로 정리해줘"))

        # 첫 route는 agent, override 후 두번째 route는 simple
        route_events = [e for e in events if e["type"] == "route"]
        assert route_events[0]["mode"] == "agent"
        assert route_events[1]["mode"] == "simple"

        # 보정 thought + 사용자 원문 그대로 simple_fn에 전달됨
        thoughts = [e["content"] for e in events if e["type"] == "thought"]
        assert any("의도 보정" in t and "agent" in t for t in thoughts)
        assert captured == ["그래프형식으로 정리해줘"]

    @pytest.mark.asyncio
    async def test_keyword_매칭된_agent는_override_안함(self):
        """matched_keyword가 있으면 corpus 점수 무관하게 agent 유지."""

        # router는 키워드 휴리스틱 ON → "비교"가 keyword_decision.matched_keyword로 잡힘
        router = self._make_router("comparative", conf=0.90)
        app_state = self._make_app_state_with_corpus(score=0.99)
        config = _make_config(in_domain_score_threshold=0.55, planner_enabled=False)
        config.rag.collection_name = "c1"

        # legacy agent(planner OFF) → AgentLoop 사용. _FakeAgentLoop에 토큰 주입.
        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="AGENT_RAN"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        async def _simple(q: str):  # 호출되면 안됨
            yield {"type": "token", "content": "SHOULD_NOT_RUN"}

        orch = AgentOrchestrator(
            router=router,
            app_state=app_state,
            config=config,
            ollama_model="t",
            api_base="http://x",
            rag_max_tokens=-1,
            simple_stream_fn=_simple,
        )
        events = await _collect(orch.handle_auto("A와 B의 차이 비교"))

        tokens = "".join(e["content"] for e in events if e["type"] == "token")
        assert "AGENT_RAN" in tokens
        assert "SHOULD_NOT_RUN" not in tokens

    @pytest.mark.asyncio
    async def test_ambiguous는_clarifier로_위임_override_안함(self):
        """intent=ambiguous + clarifier_enabled → corpus 매칭과 무관하게 clarifier."""

        router = self._make_router("ambiguous", conf=0.50)
        app_state = self._make_app_state_with_corpus(score=0.99)
        # clarifier_enabled=True + ambiguous → clarifier 호출.
        config = _make_config(
            in_domain_score_threshold=0.55,
            planner_enabled=False,
            clarifier_enabled=True,
        )
        config.rag.collection_name = "c1"

        # Clarifier는 http_client.post를 사용 — JSON 응답을 모킹.
        import json as _json

        post_resp = MagicMock()
        post_resp.raise_for_status = MagicMock(return_value=None)
        post_resp.json = MagicMock(
            return_value={"response": _json.dumps({"questions": ["무엇을 원하시나요?"]})}
        )

        async def _post(*a, **kw):
            return post_resp

        app_state.http_client.post = _post

        async def _simple(q):
            yield {"type": "token", "content": "SHOULD_NOT_RUN"}

        orch = AgentOrchestrator(
            router=router,
            app_state=app_state,
            config=config,
            ollama_model="t",
            api_base="http://x",
            rag_max_tokens=-1,
            simple_stream_fn=_simple,
        )
        events = await _collect(orch.handle_auto("그거 어떻게 돼요?"))

        # clarification 이벤트가 발행되어야 함 (override 안 일어남)
        types = [e["type"] for e in events]
        assert "clarification" in types
        tokens = "".join(e.get("content", "") for e in events if e["type"] == "token")
        assert "SHOULD_NOT_RUN" not in tokens

    @pytest.mark.asyncio
    async def test_score_낮으면_agent_유지(self):
        """score < threshold이면 override 안하고 agent 경로 진행."""

        router = self._make_router("exploratory", conf=0.90)
        app_state = self._make_app_state_with_corpus(score=0.20)  # 낮음
        config = _make_config(in_domain_score_threshold=0.55, planner_enabled=False)
        config.rag.collection_name = "c1"

        _FakeAgentLoop.script = [
            _FakeAgentEvent(type="token", content="AGENT_RAN"),
            _FakeAgentEvent(type="done", metadata={"sources": []}),
        ]

        async def _simple(q):
            yield {"type": "token", "content": "SHOULD_NOT_RUN"}

        orch = AgentOrchestrator(
            router=router,
            app_state=app_state,
            config=config,
            ollama_model="t",
            api_base="http://x",
            rag_max_tokens=-1,
            simple_stream_fn=_simple,
        )
        events = await _collect(orch.handle_auto("탐색적 질의"))

        tokens = "".join(e["content"] for e in events if e["type"] == "token")
        assert "AGENT_RAN" in tokens
        assert "SHOULD_NOT_RUN" not in tokens


# ---------------------------------------------------------------------------
# Planner 첫 search step query 보존
# ---------------------------------------------------------------------------


class TestPlannerPreserveFirstQuery:
    """planner_preserve_first_query=True이면 plan의 첫 search/lookup step ``query``
    인자를 사용자 원문으로 강제 — planner의 추상화로 임베딩 매칭이 폭락하는
    케이스를 방지합니다."""

    @pytest.mark.asyncio
    async def test_단일_search_step의_query는_사용자_원문으로_치환(self, monkeypatch):
        plan = _make_plan(
            [{"tool": "search", "args": {"query": "추상화된 검색어"}, "reason": ""}]
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            planner_preserve_first_query=True,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("사용자 원문 비교"))

        # planner가 만든 "추상화된 검색어" 대신 사용자 원문이 그대로 전달됨
        assert fixtures.tool_registry.calls == [
            ("search", {"query": "사용자 원문 비교"})
        ]

    @pytest.mark.asyncio
    async def test_다단계_decompose는_첫_step만_치환(self, monkeypatch):
        """decomposition은 보존 — 첫 search만 사용자 원문, 이후는 planner 분해 유지."""
        plan = _make_plan(
            [
                {"tool": "search", "args": {"query": "abstract1"}},
                {"tool": "search", "args": {"query": "분해 검색2"}},
                {"tool": "search", "args": {"query": "분해 검색3"}},
            ],
            strategy="decompose",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(text="r1", sources=[]),
                _FakeToolResult(text="r2", sources=[]),
                _FakeToolResult(text="r3", sources=[]),
            ],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            planner_preserve_first_query=True,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("RFP 핵심 비교"))

        assert fixtures.tool_registry.calls == [
            ("search", {"query": "RFP 핵심 비교"}),
            ("search", {"query": "분해 검색2"}),
            ("search", {"query": "분해 검색3"}),
        ]

    @pytest.mark.asyncio
    async def test_플래그_off이면_planner_query_그대로(self, monkeypatch):
        plan = _make_plan(
            [{"tool": "search", "args": {"query": "추상화"}}]
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            planner_preserve_first_query=False,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("사용자 비교 질의"))

        assert fixtures.tool_registry.calls == [("search", {"query": "추상화"})]

    @pytest.mark.asyncio
    async def test_fallback_plan은_치환_안함(self, monkeypatch):
        """is_fallback=True인 plan은 이미 사용자 query를 사용 — 추가 치환 불필요."""
        from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

        plan = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search", args={"query": "원래 fallback query"})],
            rationale="fallback: parse-error",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            planner_preserve_first_query=True,
            legacy_fallback_enabled=False,  # planner 경로에 머무름
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("사용자 비교 질의"))

        # fallback plan의 query가 그대로 보존됨
        assert fixtures.tool_registry.calls == [
            ("search", {"query": "원래 fallback query"})
        ]

    @pytest.mark.asyncio
    async def test_compare_도구는_치환_대상_아님(self, monkeypatch):
        """compare 도구는 query 인자 의미가 다름 — 치환하지 않음."""
        plan = _make_plan(
            [
                {"tool": "compare", "args": {"query_a": "a", "query_b": "b"}},
                {"tool": "search", "args": {"query": "abstract"}},
            ],
            strategy="compare",
        )
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[
                _FakeToolResult(text="r1", sources=[]),
                _FakeToolResult(text="r2", sources=[]),
            ],
            synthesis_tokens=["ans"],
        )

        orch = _make_orchestrator(
            planner_enabled=True,
            planner_preserve_first_query=True,
            app_state=fixtures.app_state,
        )
        await _collect(orch.handle_auto("진짜 사용자 비교 질의"))

        # compare는 그대로, 첫 search step query만 사용자 원문으로 치환
        assert fixtures.tool_registry.calls == [
            ("compare", {"query_a": "a", "query_b": "b"}),
            ("search", {"query": "진짜 사용자 비교 질의"}),
        ]
