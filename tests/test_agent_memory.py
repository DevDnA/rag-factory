"""Phase 12 — ConversationCompressor + session 압축 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.memory import ConversationCompressor
from slm_factory.rag.agent.session import Message, SessionManager
from slm_factory.rag.agent.state import FileBackedSessionStore


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_compressor(http_client, **kwargs) -> ConversationCompressor:
    return ConversationCompressor(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
        target_chars=kwargs.get("target_chars", 500))


class TestCompressor:
    @pytest.mark.asyncio
    async def test_정상_요약(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("요약된 내용"))

        msgs = [
            Message(role="user", content="질문1"),
            Message(role="assistant", content="답변1"),
            Message(role="user", content="질문2"),
        ]
        summary = await _make_compressor(http).summarize(msgs)
        assert summary == "요약된 내용"

    @pytest.mark.asyncio
    async def test_think_태그_제거(self):
        http = MagicMock()
        http.post = AsyncMock(
            return_value=_ollama_response("<think>생각</think>최종 요약")
        )

        summary = await _make_compressor(http).summarize(
            [Message(role="user", content="x")]
        )
        assert summary == "최종 요약"

    @pytest.mark.asyncio
    async def test_빈_messages는_None(self):
        http = MagicMock()
        summary = await _make_compressor(http).summarize([])
        assert summary is None

    @pytest.mark.asyncio
    async def test_빈_content만_있으면_None(self):
        http = MagicMock()
        msgs = [Message(role="user", content=""), Message(role="assistant", content="")]
        summary = await _make_compressor(http).summarize(msgs)
        assert summary is None

    @pytest.mark.asyncio
    async def test_HTTP_오류는_None(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("t"))

        summary = await _make_compressor(http).summarize(
            [Message(role="user", content="x")]
        )
        assert summary is None

    @pytest.mark.asyncio
    async def test_빈_응답은_None(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(""))

        summary = await _make_compressor(http).summarize(
            [Message(role="user", content="x")]
        )
        assert summary is None

    @pytest.mark.asyncio
    async def test_transcript가_prompt에_포함(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("요약"))

        msgs = [
            Message(role="user", content="첫번째 질문"),
            Message(role="assistant", content="첫번째 답변"),
        ]
        await _make_compressor(http).summarize(msgs)

        prompt = http.post.call_args.kwargs["json"]["prompt"]
        assert "첫번째 질문" in prompt
        assert "첫번째 답변" in prompt
        assert "사용자:" in prompt
        assert "어시스턴트:" in prompt


class TestSessionManagerCompression:
    def test_compress_old_turns_기본(self):
        sm = SessionManager()
        sid = sm.create_session()
        for i in range(5):
            sm.add_message(sid, Message(role="user", content=f"q{i}"))
            sm.add_message(sid, Message(role="assistant", content=f"a{i}"))
        # 10개 메시지, keep_recent=4 → 6개 요약됨
        removed = sm.compress_old_turns(sid, keep_recent=4, summary_text="요약")
        assert removed == 6
        _, msgs = sm.get_or_create(sid)
        # summary 1 + 최근 4 = 5
        assert len(msgs) == 5
        assert msgs[0].role == "assistant"
        assert "요약" in msgs[0].content
        assert msgs[-1].content == "a4"

    def test_임계값_미만이면_무작업(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, Message(role="user", content="q"))
        removed = sm.compress_old_turns(sid, keep_recent=4, summary_text="요약")
        assert removed == 0
        _, msgs = sm.get_or_create(sid)
        assert len(msgs) == 1

    def test_없는_세션(self):
        sm = SessionManager()
        removed = sm.compress_old_turns("nonexistent", keep_recent=4, summary_text="x")
        assert removed == 0


class TestFileBackedCompression:
    def test_compress_재시작_후_유지(self, tmp_path):
        base = tmp_path / "s"
        store1 = FileBackedSessionStore(base_dir=base, max_turns=100)
        sid = store1.create_session()
        for i in range(5):
            store1.add_message(sid, Message(role="user", content=f"q{i}"))
            store1.add_message(sid, Message(role="assistant", content=f"a{i}"))

        removed = store1.compress_old_turns(sid, keep_recent=4, summary_text="압축")
        assert removed == 6

        # 재시작
        store2 = FileBackedSessionStore(base_dir=base, max_turns=100)
        _, msgs = store2.get_or_create(sid)
        assert len(msgs) == 5
        assert "압축" in msgs[0].content


class TestOrchestratorCompression:
    """orchestrator가 assistant 메시지 추가 후 자동 압축을 시도하는지."""

    @pytest.mark.asyncio
    async def test_compression_enabled면_긴_세션_압축(self, monkeypatch):
        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_plan,
            _FakeToolResult,
            _collect)
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.router import QueryRouter
        from types import SimpleNamespace

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_tokens=["답"])
        # Memory compressor도 동일 http_client 사용 — 동일 stream 응답.
        # ConversationCompressor는 post를 쓰므로 AsyncMock 필요.
        fixtures.app_state.http_client.post = AsyncMock(
            return_value=_ollama_response("압축된 요약")
        )

        # 세션에 미리 긴 이력 주입
        sm = fixtures.app_state.agent_session_manager
        sid = sm.create_session()
        for i in range(15):
            sm.add_message(sid, Message(role="user", content=f"q{i}"))
            sm.add_message(sid, Message(role="assistant", content=f"a{i}"))

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
                    memory_compression_enabled=True,
                    compress_after_turns=5,  # 5 turns = 10 messages threshold
                    compress_target_chars=200,
                    models=SimpleNamespace(
                        router_model="", planner_model="", synthesis_model="",
                        verifier_model="",                         clarifier_model="")),
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
        await _collect(orch.handle_agent("새 질의", sid))

        _, msgs = sm.get_or_create(sid)
        # 압축 후: summary 1 + keep_recent 5 + (user 새 + assistant 새) = 8
        # compress_after_turns=5, keep_recent=5
        assert len(msgs) < 32  # 원래 30+2 보다 적어야 함
        # 첫 메시지는 요약 메시지
        assert "요약" in msgs[0].content or "압축" in msgs[0].content
