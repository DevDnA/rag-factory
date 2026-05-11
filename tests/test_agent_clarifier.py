"""Clarifier persona 테스트 — 명확화 질문 생성, fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.personas.clarifier import Clarifier


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_clarifier(http_client, **kwargs) -> Clarifier:
    return Clarifier(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
        max_questions=kwargs.get("max_questions", 2),
    )


class TestValidGeneration:
    @pytest.mark.asyncio
    async def test_질문_2개_반환(self):
        raw = '{"questions": ["주제는 무엇인가요?", "특정 시점을 전제하시나요?"]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_clarifier(http).generate_questions("그거")
        assert r.kind == "clarification"
        assert len(r.questions) == 2
        assert "주제" in r.questions[0]
        assert r.metadata["is_fallback"] is False

    @pytest.mark.asyncio
    async def test_max_questions_제한(self):
        raw = (
            '{"questions": ["Q1", "Q2", "Q3", "Q4"]}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_clarifier(http, max_questions=2).generate_questions("x")
        assert len(r.questions) == 2

    @pytest.mark.asyncio
    async def test_긴_질문은_300자로_잘림(self):
        raw = '{"questions": ["' + "가" * 500 + '"]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_clarifier(http).generate_questions("x")
        assert len(r.questions[0]) == 300

    @pytest.mark.asyncio
    async def test_공백만_있는_질문은_제외(self):
        raw = '{"questions": ["  ", "실제 질문", ""]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_clarifier(http).generate_questions("x")
        assert r.questions == ["실제 질문"]

    @pytest.mark.asyncio
    async def test_history가_prompt에_포함(self):
        raw = '{"questions": ["Q"]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        await _make_clarifier(http).generate_questions(
            "후속", history="[이전 대화]\n사용자: 처음 질문"
        )
        prompt = http.post.call_args.kwargs["json"]["prompt"]
        assert "[이전 대화]" in prompt
        assert "처음 질문" in prompt


class TestFallback:
    """모든 실패는 일반 fallback 질문으로 graceful하게."""

    @pytest.mark.asyncio
    async def test_HTTP_타임아웃(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("t"))

        r = await _make_clarifier(http).generate_questions("x")
        assert r.kind == "clarification"
        assert r.metadata["is_fallback"] is True
        assert r.metadata["fallback_reason"] == "llm-error"
        assert len(r.questions) == 2  # default

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("not json"))

        r = await _make_clarifier(http).generate_questions("x")
        assert r.metadata["fallback_reason"] == "parse-error"

    @pytest.mark.asyncio
    async def test_빈_질문_목록은_fallback(self):
        raw = '{"questions": []}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_clarifier(http).generate_questions("x")
        assert r.metadata["is_fallback"] is True
        assert r.metadata["fallback_reason"] == "empty-questions"
        assert len(r.questions) > 0

    @pytest.mark.asyncio
    async def test_questions_필드_없음(self):
        raw = '{"other": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_clarifier(http).generate_questions("x")
        assert r.metadata["is_fallback"] is True


class TestPersonaAttributes:
    def test_allowed_tools는_빈_집합(self):
        assert Clarifier.allowed_tools == frozenset()

    def test_name_is_clarifier(self):
        assert Clarifier.name == "clarifier"
