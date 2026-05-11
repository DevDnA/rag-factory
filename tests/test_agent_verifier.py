"""Verifier 테스트 — 충분성 판정, JSON 파싱, fallback 동작."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.verifier import Verifier, VerifierDecision


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_verifier(http_client) -> Verifier:
    return Verifier(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidDecision:
    """구조화된 JSON 응답."""

    @pytest.mark.asyncio
    async def test_sufficient_true(self):
        raw = '{"sufficient": true, "reason": "모든 정보 확인됨"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("질의", "컨텍스트")
        assert decision.sufficient is True
        assert "확인" in decision.reason
        assert not decision.needs_repair

    @pytest.mark.asyncio
    async def test_sufficient_false_with_suggestion(self):
        raw = (
            '{"sufficient": false, "reason": "핵심 수치 누락", '
            '"suggestion": "2024년 매출 수치"}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("질의", "컨텍스트")
        assert decision.sufficient is False
        assert decision.suggested_query == "2024년 매출 수치"
        assert decision.needs_repair

    @pytest.mark.asyncio
    async def test_suggested_query_대체키(self):
        raw = (
            '{"sufficient": false, "reason": "부족", '
            '"suggested_query": "추가 키워드"}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.suggested_query == "추가 키워드"

    @pytest.mark.asyncio
    async def test_제안_없으면_repair_불필요(self):
        raw = '{"sufficient": false, "reason": "애매"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is False
        assert decision.suggested_query is None
        assert not decision.needs_repair

    @pytest.mark.asyncio
    async def test_빈_문자열_suggestion은_None으로_정규화(self):
        raw = '{"sufficient": false, "reason": "x", "suggestion": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.suggested_query is None


# ---------------------------------------------------------------------------
# 문자열 "true"/"false" 방어 처리
# ---------------------------------------------------------------------------


class TestBooleanCoercion:
    """LLM이 sufficient를 문자열로 반환할 때."""

    @pytest.mark.parametrize("raw_value, expected", [
        ('"true"', True),
        ('"yes"', True),
        ('"예"', True),
        ('"충분"', True),
        ('"false"', False),
        ('"no"', False),
        ('"unknown"', False),
    ])
    @pytest.mark.asyncio
    async def test_문자열_sufficient_처리(self, raw_value, expected):
        raw = f'{{"sufficient": {raw_value}, "reason": "x"}}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is expected

    @pytest.mark.asyncio
    async def test_sufficient_필드_없으면_True(self):
        raw = '{"reason": "키가 없음"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is True


# ---------------------------------------------------------------------------
# Fallback — LLM 실패 시 sufficient=True (안전한 default)
# ---------------------------------------------------------------------------


class TestFallback:
    """evaluate()는 어떤 실패에서도 raise하지 않고 sufficient=True로 반환."""

    @pytest.mark.asyncio
    async def test_HTTP_타임아웃시_sufficient_True(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is True
        assert "unavailable" in decision.reason
        assert not decision.needs_repair

    @pytest.mark.asyncio
    async def test_연결오류시_sufficient_True(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is True

    @pytest.mark.asyncio
    async def test_빈_응답시_sufficient_True(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(""))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is True
        assert "parse failure" in decision.reason

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패시_sufficient_True(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("garbage {not json"))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is True

    @pytest.mark.asyncio
    async def test_JSON_배열은_dict_아님_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("[true]"))

        decision = await _make_verifier(http).evaluate("q", "ctx")
        assert decision.sufficient is True


# ---------------------------------------------------------------------------
# 부수적 동작
# ---------------------------------------------------------------------------


class TestContextTruncation:
    """긴 컨텍스트는 2000자로 잘려 LLM에 전달됩니다."""

    @pytest.mark.asyncio
    async def test_긴_컨텍스트_전달시_잘림(self):
        raw = '{"sufficient": true, "reason": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        long_ctx = "a" * 5000
        await _make_verifier(http).evaluate("질문", long_ctx)

        # post가 호출한 prompt를 검사
        call_kwargs = http.post.call_args.kwargs
        prompt = call_kwargs["json"]["prompt"]
        # 컨텍스트는 2000자로 잘렸으므로 prompt 전체에 "a"*5000이 들어갈 수 없음
        assert "a" * 2000 in prompt
        assert "a" * 2001 not in prompt


class TestVerifierDecision:
    """VerifierDecision 편의 속성."""

    def test_needs_repair_true(self):
        d = VerifierDecision(sufficient=False, reason="x", suggested_query="새 키워드")
        assert d.needs_repair

    def test_needs_repair_sufficient면_false(self):
        d = VerifierDecision(sufficient=True, reason="x", suggested_query="무시됨")
        assert not d.needs_repair

    def test_needs_repair_제안없으면_false(self):
        d = VerifierDecision(sufficient=False, reason="x", suggested_query=None)
        assert not d.needs_repair
