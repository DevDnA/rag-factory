"""Phase 14 wire-in 통합 테스트 — answer_verifier + citation audit + persona 합성.

기존 ``test_agent_orchestrator.py``의 Planner 경로 fixture를 부분 재사용해
새 SSE 이벤트(`verification`, `warning`)와 재합성 흐름을 검증합니다.
"""

from __future__ import annotations

import json as _json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_factory.rag.agent.orchestrator import AgentOrchestrator
from rag_factory.rag.agent.router import QueryRouter
from rag_factory.rag.agent.session import SessionManager


# ---------------------------------------------------------------------------
# 기본 fixture (test_agent_orchestrator.py 패턴 미러)
# ---------------------------------------------------------------------------


def _make_smart_config(
    *,
    planner_enabled: bool = True,
    answer_verifier_enabled: bool = False,
    answer_verifier_max_repairs: int = 1,
    synthesis_require_citations: bool = False,
    persona_composition_enabled: bool = False,
    persona_composition_confidence_threshold: float = 0.7,
    personas_enabled: bool = False,
):
    return SimpleNamespace(
        rag=SimpleNamespace(
            agent=SimpleNamespace(
                enabled=True,
                max_iterations=3,
                stream_reasoning=True,
                planner_enabled=planner_enabled,
                verifier_enabled=False,
                verifier_max_repairs=1,
                legacy_fallback_enabled=False,
                session_source_reuse=False,
                session_source_reuse_limit=5,
                parallel_steps=False,
                clarifier_enabled=False,
                clarifier_max_questions=2,
                personas_enabled=personas_enabled,
                native_thinking=False,
                refusal_min_score=0.0,
                in_domain_score_threshold=0.0,
                planner_preserve_first_query=False,
                answer_verifier_enabled=answer_verifier_enabled,
                answer_verifier_max_repairs=answer_verifier_max_repairs,
                synthesis_require_citations=synthesis_require_citations,
                persona_composition_enabled=persona_composition_enabled,
                persona_composition_confidence_threshold=persona_composition_confidence_threshold,
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


class _FakeToolResult:
    def __init__(self, text: str, sources):
        self.text = text
        self.sources = sources


class _FakeToolRegistry:
    def __init__(self, script):
        self._script = list(script)

    async def execute(self, name: str, args: dict):
        if not self._script:
            return _FakeToolResult(text="(빈 결과)", sources=[])
        return self._script.pop(0)


class _FakeStreamResponse:
    def __init__(self, lines):
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


def _stream_lines(tokens):
    out = [_json.dumps({"response": t, "done": False}) for t in tokens]
    out.append(_json.dumps({"response": "", "done": True}))
    return out


async def _collect(agen):
    return [ev async for ev in agen]


async def _simple_stream(query: str):
    yield {"type": "token", "content": "simple"}
    yield {"type": "done"}


def _make_planner_plan():
    from rag_factory.rag.agent.planner import ExecutionPlan, PlanStep

    return ExecutionPlan(
        strategy="fact",
        steps=[PlanStep(tool="search", args={"query": "x"}, reason="r")],
        rationale="test plan",
    )


def _patch_planner_verifier_loop(monkeypatch):
    """Planner / Verifier / AgentLoop 를 deterministic 모킹으로 패치."""
    from rag_factory.rag.agent import (
        loop as loop_mod,
        planner as planner_mod,
        verifier as verifier_mod,
    )

    fixed_plan = _make_planner_plan()

    class _FakePlanner:
        def __init__(self, **_kwargs):
            pass

        async def plan(self, query):
            return fixed_plan

    class _FakeVerifier:
        def __init__(self, **_kwargs):
            pass

        async def evaluate(self, query, context):
            return verifier_mod.VerifierDecision(sufficient=True, reason="default")

    class _FakeLoop:
        def __init__(self, **_kwargs):
            pass

        async def run_stream(self, query, history=""):
            if False:
                yield None  # pragma: no cover

    monkeypatch.setattr(planner_mod, "Planner", _FakePlanner)
    monkeypatch.setattr(verifier_mod, "Verifier", _FakeVerifier)
    monkeypatch.setattr(loop_mod, "AgentLoop", _FakeLoop)


def _make_orchestrator(config, app_state, *, simple_fn=_simple_stream):
    router = QueryRouter(agent_enabled=True)
    return AgentOrchestrator(
        router=router,
        app_state=app_state,
        config=config,
        ollama_model="test-model",
        api_base="http://localhost:11434",
        rag_max_tokens=512,
        simple_stream_fn=simple_fn,
    )


# ---------------------------------------------------------------------------
# V.1 — answer_verifier 발동, FAIL+repair_hint이면 1회 재합성
# ---------------------------------------------------------------------------


class TestV1_AnswerVerifier:
    """answer_verifier_enabled=True 경로의 verification 이벤트 발행."""

    @pytest.mark.asyncio
    async def test_FAIL_시_verification_이벤트_발행_후_재합성(self, monkeypatch):
        _patch_planner_verifier_loop(monkeypatch)

        # AnswerVerifier mock: 첫 호출에 FAIL+repair_hint, 두 번째는 호출되지 않음
        from rag_factory.rag.agent import answer_verifier as av_mod

        class _FakeAV:
            def __init__(self, **_kwargs):
                pass

            async def evaluate(self, query, answer, context):
                return av_mod.AnswerVerdict(
                    verdict="FAIL",
                    issues=["주장 X 미지지"],
                    repair_hint="X는 문서에 없음을 명시",
                )

        monkeypatch.setattr(av_mod, "AnswerVerifier", _FakeAV)

        # 합성 LLM 응답: 첫 합성(FAIL 유도) + 재합성(retry)
        first = _FakeStreamResponse(_stream_lines(["원본 답변"]))
        retry = _FakeStreamResponse(_stream_lines(["수정된 답변"]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="검색 결과",
                    sources=[{"doc_id": "d1", "content": "c", "score": 0.9}],
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(side_effect=[first, retry])
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True,
            answer_verifier_enabled=True,
            answer_verifier_max_repairs=1,
        )
        orch = _make_orchestrator(config, app_state)

        events = await _collect(orch.handle_agent("질문"))

        # verification 이벤트가 정확히 1번 발행
        ver_events = [e for e in events if e.get("type") == "verification"]
        assert len(ver_events) == 1
        assert ver_events[0]["verdict"] == "FAIL"
        assert "주장 X 미지지" in ver_events[0]["issues"]
        assert "1회 재합성 수행" in ver_events[0]["issues"]

        # 재합성 답변이 token으로 발행 (원본 아닌 retried)
        token_events = [e for e in events if e.get("type") == "token"]
        joined = "".join(e.get("content", "") for e in token_events)
        assert "수정된 답변" in joined
        assert "원본 답변" not in joined

        # 합성 LLM은 정확히 2번 호출
        assert mock_http.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_max_repairs_0이면_재합성_안함(self, monkeypatch):
        _patch_planner_verifier_loop(monkeypatch)

        from rag_factory.rag.agent import answer_verifier as av_mod

        class _FakeAV:
            async def evaluate(self, query, answer, context):
                return av_mod.AnswerVerdict(
                    verdict="FAIL",
                    issues=["x"],
                    repair_hint="hint",
                )

            def __init__(self, **_):
                pass

        monkeypatch.setattr(av_mod, "AnswerVerifier", _FakeAV)

        first = _FakeStreamResponse(_stream_lines(["원본 답변"]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="검색", sources=[{"doc_id": "d", "content": "c", "score": 0.9}]
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(side_effect=[first])
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True,
            answer_verifier_enabled=True,
            answer_verifier_max_repairs=0,
        )
        orch = _make_orchestrator(config, app_state)

        events = await _collect(orch.handle_agent("질문"))
        # 재합성 일어나지 않음 — token에 원본 답변만 등장
        joined = "".join(e.get("content", "") for e in events if e.get("type") == "token")
        assert "원본 답변" in joined
        # verification 이벤트는 여전히 발행됨 (PASS가 아니므로)
        ver_events = [e for e in events if e.get("type") == "verification"]
        assert len(ver_events) == 1
        # 합성은 1번만
        assert mock_http.stream.call_count == 1


# ---------------------------------------------------------------------------
# V.2 — answer_verifier timeout은 답변을 막지 않음 (never-raise)
# ---------------------------------------------------------------------------


class TestV2_NeverBlock:
    @pytest.mark.asyncio
    async def test_AV_timeout시_token_그대로_발행(self, monkeypatch):
        """AnswerVerifier.evaluate가 발생시킨 예외가 답변을 막아서는 안 됨.

        AnswerVerifier 자체는 never-raise이지만, evaluate()가 PASS를 반환하면
        verification 이벤트가 발행되지 않아야 한다는 게 핵심.
        """
        _patch_planner_verifier_loop(monkeypatch)

        from rag_factory.rag.agent import answer_verifier as av_mod

        class _FakeAV:
            def __init__(self, **_):
                pass

            async def evaluate(self, query, answer, context):
                # AnswerVerifier.evaluate의 never-raise 정책을 흉내 — PASS 반환
                return av_mod.AnswerVerdict(verdict="PASS")

        monkeypatch.setattr(av_mod, "AnswerVerifier", _FakeAV)

        first = _FakeStreamResponse(_stream_lines(["답변 본문"]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="x", sources=[{"doc_id": "d", "content": "c", "score": 0.9}]
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(side_effect=[first])
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True, answer_verifier_enabled=True
        )
        orch = _make_orchestrator(config, app_state)

        events = await _collect(orch.handle_agent("질문"))

        # verification 이벤트 미발행 (PASS는 silent)
        assert not any(e.get("type") == "verification" for e in events)
        # 답변은 token으로 전달
        joined = "".join(e.get("content", "") for e in events if e.get("type") == "token")
        assert "답변 본문" in joined


# ---------------------------------------------------------------------------
# V.3 — smart_mode=False (모든 신규 플래그 False) 회귀 — 기존 스트리밍 보존
# ---------------------------------------------------------------------------


class TestV3_Regression:
    @pytest.mark.asyncio
    async def test_신규_플래그_모두_False면_기존_token_스트리밍(self, monkeypatch):
        _patch_planner_verifier_loop(monkeypatch)

        # 합성 LLM이 토큰 4개를 순차 yield
        resp = _FakeStreamResponse(_stream_lines(["조각", "1", "조각", "2"]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="x", sources=[{"doc_id": "d", "content": "c", "score": 0.9}]
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=resp)
        app_state.http_client = mock_http

        config = _make_smart_config(planner_enabled=True)  # 신규 플래그 모두 False
        orch = _make_orchestrator(config, app_state)

        events = await _collect(orch.handle_agent("질문"))

        # verification, warning 이벤트는 미발행
        assert not any(e.get("type") == "verification" for e in events)
        assert not any(e.get("type") == "warning" for e in events)

        # token 이벤트는 4개 (collect-then-emit이 아닌 streaming)
        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) == 4
        assert "".join(e["content"] for e in token_events) == "조각1조각2"


# ---------------------------------------------------------------------------
# V.4 — citation audit warning 이벤트 발행
# ---------------------------------------------------------------------------


class TestV4_CitationAudit:
    @pytest.mark.asyncio
    async def test_미매칭_인용이면_warning_이벤트(self, monkeypatch):
        _patch_planner_verifier_loop(monkeypatch)

        # 합성 LLM이 nonexistent.pdf를 인용
        answer_with_bad_cite = (
            "내용은 [doc:law.pdf]에 있고 추가는 [doc:nonexistent.pdf]에 있다."
        )
        resp = _FakeStreamResponse(_stream_lines([answer_with_bad_cite]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="x",
                    sources=[{"doc_id": "law.pdf::p1", "content": "c", "score": 0.9}],
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=resp)
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True, synthesis_require_citations=True
        )
        orch = _make_orchestrator(config, app_state)

        events = await _collect(orch.handle_agent("질문"))

        warning_events = [e for e in events if e.get("type") == "warning"]
        assert len(warning_events) == 1
        assert warning_events[0]["items"] == ["nonexistent.pdf"]
        assert "근거 없는 인용" in warning_events[0]["content"]

        # 답변 본문은 그대로 발행 (막지 않음)
        joined = "".join(e.get("content", "") for e in events if e.get("type") == "token")
        assert "nonexistent.pdf" in joined

    @pytest.mark.asyncio
    async def test_모든_인용_매칭이면_warning_미발행(self, monkeypatch):
        _patch_planner_verifier_loop(monkeypatch)

        good_answer = "내용은 [doc:law.pdf]에 있다."
        resp = _FakeStreamResponse(_stream_lines([good_answer]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="x",
                    sources=[{"doc_id": "law.pdf::p1", "content": "c", "score": 0.9}],
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=resp)
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True, synthesis_require_citations=True
        )
        orch = _make_orchestrator(config, app_state)

        events = await _collect(orch.handle_agent("질문"))
        assert not any(e.get("type") == "warning" for e in events)


# ---------------------------------------------------------------------------
# V.5 — composite persona 활성화 시 _maybe_compose_persona 동작
# ---------------------------------------------------------------------------


class TestV5_PersonaComposition:
    def test_low_confidence_comparator_composes_with_analyst(self):
        """confidence < threshold이면 comparator + analyst 합성."""
        from rag_factory.rag.agent.personas.comparator import Comparator
        from rag_factory.rag.agent.personas.composite import CompositePersona

        config = _make_smart_config(
            planner_enabled=True,
            personas_enabled=True,
            persona_composition_enabled=True,
            persona_composition_confidence_threshold=0.8,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(config, app_state)

        primary = Comparator()
        decision = SimpleNamespace(
            confidence=0.5, reason="비교 요청", intent="comparative"
        )
        result = orch._maybe_compose_persona(decision, primary)
        assert isinstance(result, CompositePersona)
        assert result.name == "comparator+analyst"

    def test_high_confidence_no_composition(self):
        """confidence >= threshold이고 hybrid 신호 없으면 primary 그대로."""
        from rag_factory.rag.agent.personas.comparator import Comparator

        config = _make_smart_config(
            planner_enabled=True,
            personas_enabled=True,
            persona_composition_enabled=True,
            persona_composition_confidence_threshold=0.7,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(config, app_state)

        primary = Comparator()
        decision = SimpleNamespace(
            confidence=0.95, reason="단순 비교", intent="comparative"
        )
        result = orch._maybe_compose_persona(decision, primary)
        assert result is primary  # 합성 없음

    def test_disabled_flag_no_composition(self):
        from rag_factory.rag.agent.personas.comparator import Comparator

        config = _make_smart_config(
            planner_enabled=True,
            personas_enabled=True,
            persona_composition_enabled=False,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(config, app_state)

        primary = Comparator()
        decision = SimpleNamespace(confidence=0.1, reason="x", intent="comparative")
        result = orch._maybe_compose_persona(decision, primary)
        assert result is primary

    def test_non_comparator_analyst_primary는_합성_안함(self):
        """researcher / procedural은 합성 대상 아님 (단일 의도 명확)."""
        from rag_factory.rag.agent.personas.researcher import Researcher

        config = _make_smart_config(
            planner_enabled=True,
            personas_enabled=True,
            persona_composition_enabled=True,
            persona_composition_confidence_threshold=0.9,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(config, app_state)

        primary = Researcher()
        decision = SimpleNamespace(confidence=0.1, reason="x", intent="factual")
        result = orch._maybe_compose_persona(decision, primary)
        assert result is primary

    def test_None_primary는_None_반환(self):
        config = _make_smart_config(
            planner_enabled=True,
            persona_composition_enabled=True,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(config, app_state)

        decision = SimpleNamespace(confidence=0.1, reason="x", intent="comparative")
        result = orch._maybe_compose_persona(decision, None)
        assert result is None


# ---------------------------------------------------------------------------
# V.6 — citation preamble은 synthesis prompt에 prepend됨
# ---------------------------------------------------------------------------


class TestV7_SimplePathWireIn:
    """Phase 14 simple-path 확장 — _wrap_simple_with_verification.

    simple_stream_fn이 이벤트를 yield하면 orchestrator가 wrap해서:
    - 토글 OFF → 그대로 통과 (회귀 차단)
    - 토글 ON → token collect, answer_verifier·citation_audit, verification/warning 발행
    """

    @staticmethod
    async def _stub_simple_with_citations(query: str):
        """답변에 [doc:law.pdf]가 포함된 simple_stream_fn 모의 — citation 일치."""
        yield {"type": "token", "content": "내용은 [doc:law.pdf]에 있습니다."}
        yield {
            "type": "sources",
            "sources": [{"doc_id": "law.pdf::p1", "content": "법령 내용", "score": 0.9}],
        }
        yield {"type": "done"}

    @staticmethod
    async def _stub_simple_with_bad_citation(query: str):
        """답변에 nonexistent doc 인용 — citation_audit이 잡아야 함."""
        yield {"type": "token", "content": "내용은 [doc:nonexistent.pdf]에 있습니다."}
        yield {
            "type": "sources",
            "sources": [{"doc_id": "law.pdf::p1", "content": "x", "score": 0.9}],
        }
        yield {"type": "done"}

    @staticmethod
    async def _stub_simple_plain(query: str):
        """citation 없는 평범한 simple stream."""
        yield {"type": "token", "content": "답변 본문"}
        yield {"type": "sources", "sources": [{"doc_id": "d1", "content": "c", "score": 0.9}]}
        yield {"type": "done"}

    @pytest.mark.asyncio
    async def test_토글_OFF면_원본_그대로_통과(self):
        """answer_verifier_enabled=False + synthesis_require_citations=False이면
        simple_stream_fn의 이벤트가 어떤 후처리도 없이 통과."""
        config = _make_smart_config(
            planner_enabled=True,
            answer_verifier_enabled=False,
            synthesis_require_citations=False,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(
            config, app_state, simple_fn=self._stub_simple_plain
        )

        events = await _collect(orch._wrap_simple_with_verification("q"))
        # 토큰이 한 번만 발행되어야 함 (collect-then-emit이 아닌 원본 streaming)
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["content"] == "답변 본문"
        # verification, warning 절대 없음
        assert not any(e["type"] in ("verification", "warning") for e in events)

    @pytest.mark.asyncio
    async def test_citation_audit이_simple_경로에서_미매칭_warning_발행(self):
        config = _make_smart_config(
            planner_enabled=True,
            synthesis_require_citations=True,
            answer_verifier_enabled=False,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(
            config, app_state, simple_fn=self._stub_simple_with_bad_citation
        )

        events = await _collect(orch._wrap_simple_with_verification("q"))

        warnings = [e for e in events if e["type"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["items"] == ["nonexistent.pdf"]

        # 답변 본문은 collect-then-emit으로 단일 token에 담겨야 함
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 1
        assert "nonexistent.pdf" in token_events[0]["content"]

    @pytest.mark.asyncio
    async def test_citation_매칭이면_warning_미발행(self):
        config = _make_smart_config(
            planner_enabled=True,
            synthesis_require_citations=True,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(
            config, app_state, simple_fn=self._stub_simple_with_citations
        )

        events = await _collect(orch._wrap_simple_with_verification("q"))
        assert not any(e["type"] == "warning" for e in events)

    @pytest.mark.asyncio
    async def test_answer_verifier가_simple에서_FAIL이면_verification_발행(
        self, monkeypatch
    ):
        from rag_factory.rag.agent import answer_verifier as av_mod

        class _FakeAV:
            def __init__(self, **_):
                pass

            async def evaluate(self, query, answer, context):
                return av_mod.AnswerVerdict(
                    verdict="FAIL",
                    issues=["주장 X 미지지"],
                    repair_hint="hint",
                )

        monkeypatch.setattr(av_mod, "AnswerVerifier", _FakeAV)

        config = _make_smart_config(
            planner_enabled=True,
            answer_verifier_enabled=True,
            synthesis_require_citations=False,
        )
        app_state = _make_app_state()
        orch = _make_orchestrator(
            config, app_state, simple_fn=self._stub_simple_plain
        )

        events = await _collect(orch._wrap_simple_with_verification("q"))
        verifications = [e for e in events if e["type"] == "verification"]
        assert len(verifications) == 1
        assert verifications[0]["verdict"] == "FAIL"
        assert "주장 X 미지지" in verifications[0]["issues"]
        # simple 경로는 재합성 없음 — 답변 본문은 그대로 emit
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["content"] == "답변 본문"


class TestV6_CitationPreamble:
    @pytest.mark.asyncio
    async def test_synthesis_require_citations이면_prompt에_preamble_prepend(
        self, monkeypatch
    ):
        _patch_planner_verifier_loop(monkeypatch)

        resp = _FakeStreamResponse(_stream_lines(["답변"]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="x", sources=[{"doc_id": "d", "content": "c", "score": 0.9}]
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=resp)
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True, synthesis_require_citations=True
        )
        orch = _make_orchestrator(config, app_state)

        await _collect(orch.handle_agent("질문"))

        # http_client.stream의 prompt가 인용 강제 preamble을 포함하는지 확인
        prompt = mock_http.stream.call_args.kwargs["json"]["prompt"]
        assert "인용 규칙" in prompt
        assert "[doc:" in prompt

    @pytest.mark.asyncio
    async def test_플래그_off이면_preamble_미적용(self, monkeypatch):
        _patch_planner_verifier_loop(monkeypatch)

        resp = _FakeStreamResponse(_stream_lines(["답변"]))
        app_state = _make_app_state()
        app_state.agent_tool_registry = _FakeToolRegistry(
            [
                _FakeToolResult(
                    text="x", sources=[{"doc_id": "d", "content": "c", "score": 0.9}]
                )
            ]
        )
        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=resp)
        app_state.http_client = mock_http

        config = _make_smart_config(
            planner_enabled=True, synthesis_require_citations=False
        )
        orch = _make_orchestrator(config, app_state)

        await _collect(orch.handle_agent("질문"))

        prompt = mock_http.stream.call_args.kwargs["json"]["prompt"]
        assert "인용 규칙" not in prompt
