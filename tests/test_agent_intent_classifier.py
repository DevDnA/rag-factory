"""IntentClassifier 테스트 — JSON 파싱, fallback, 캐싱."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.intent_classifier import IntentClassifier, IntentDecision


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_classifier(http_client, **kwargs) -> IntentClassifier:
    return IntentClassifier(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
        cache_ttl=kwargs.get("cache_ttl", 300),
        cache_max_size=kwargs.get("cache_max_size", 512),
    )


class TestValidClassification:
    """올바른 JSON 응답 파싱."""

    @pytest.mark.parametrize("intent_label", [
        "factual", "comparative", "analytical",
        "procedural", "exploratory", "ambiguous",
    ])
    @pytest.mark.asyncio
    async def test_6가지_카테고리_모두_지원(self, intent_label):
        raw = f'{{"intent": "{intent_label}", "confidence": 0.9, "reason": "r"}}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_classifier(http).classify("테스트")
        assert d.intent == intent_label
        assert d.confidence == 0.9

    @pytest.mark.asyncio
    async def test_is_agent_intent_편의속성(self):
        raw = '{"intent": "comparative", "confidence": 0.9, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_classifier(http).classify("비교")
        assert d.is_agent_intent is True

        raw2 = '{"intent": "factual", "confidence": 0.95, "reason": "r"}'
        http2 = MagicMock()
        http2.post = AsyncMock(return_value=_ollama_response(raw2))

        d2 = await _make_classifier(http2).classify("사실")
        assert d2.is_agent_intent is False


class TestValidation:
    """잘못된 값 정규화."""

    @pytest.mark.asyncio
    async def test_알_수_없는_intent는_ambiguous로(self):
        raw = '{"intent": "weird_thing", "confidence": 0.8, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_classifier(http).classify("x")
        assert d.intent == "ambiguous"

    @pytest.mark.asyncio
    async def test_confidence_범위_벗어나면_clamp(self):
        raw = '{"intent": "factual", "confidence": 5.0, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_classifier(http).classify("x")
        assert d.confidence == 1.0

        raw2 = '{"intent": "factual", "confidence": -0.5, "reason": "r"}'
        http2 = MagicMock()
        http2.post = AsyncMock(return_value=_ollama_response(raw2))

        d2 = await _make_classifier(http2).classify("y")
        assert d2.confidence == 0.0

    @pytest.mark.asyncio
    async def test_confidence_문자열이면_기본값(self):
        raw = '{"intent": "factual", "confidence": "high", "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_classifier(http).classify("x")
        assert d.confidence == 0.5


class TestFallback:
    """LLM·파싱 실패는 ambiguous(confidence=0)로 안전하게 반환."""

    @pytest.mark.asyncio
    async def test_HTTP_타임아웃시_ambiguous(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("t"))

        d = await _make_classifier(http).classify("x")
        assert d.intent == "ambiguous"
        assert d.confidence == 0.0
        assert "llm-error" in d.reason

    @pytest.mark.asyncio
    async def test_연결_오류(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        d = await _make_classifier(http).classify("x")
        assert d.intent == "ambiguous"

    @pytest.mark.asyncio
    async def test_빈_응답(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(""))

        d = await _make_classifier(http).classify("x")
        assert d.intent == "ambiguous"
        assert "parse-error" in d.reason

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("{not valid"))

        d = await _make_classifier(http).classify("x")
        assert d.intent == "ambiguous"

    @pytest.mark.asyncio
    async def test_JSON_배열은_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("[1,2,3]"))

        d = await _make_classifier(http).classify("x")
        assert d.intent == "ambiguous"


class TestCaching:
    """TTL 캐싱 동작."""

    @pytest.mark.asyncio
    async def test_같은_질의는_한번만_LLM_호출(self):
        raw = '{"intent": "factual", "confidence": 0.9, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        cls = _make_classifier(http)
        d1 = await cls.classify("동일 질의")
        d2 = await cls.classify("동일 질의")

        assert d1 == d2
        assert http.post.call_count == 1

    @pytest.mark.asyncio
    async def test_공백_정규화(self):
        raw = '{"intent": "factual", "confidence": 0.9, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        cls = _make_classifier(http)
        await cls.classify("hello   world")
        await cls.classify("HELLO world")  # 정규화 후 같은 키
        assert http.post.call_count == 1

    @pytest.mark.asyncio
    async def test_다른_질의는_분리_호출(self):
        raw = '{"intent": "factual", "confidence": 0.9, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        cls = _make_classifier(http)
        await cls.classify("query A")
        await cls.classify("query B")
        assert http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_ttl_0이면_캐싱_비활성(self):
        raw = '{"intent": "factual", "confidence": 0.9, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        cls = _make_classifier(http, cache_ttl=0)
        await cls.classify("q")
        await cls.classify("q")
        assert http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_max_size_초과시_최고령_퇴거(self):
        raw = '{"intent": "factual", "confidence": 0.9, "reason": "r"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        cls = _make_classifier(http, cache_max_size=2)
        await cls.classify("q1")
        time.sleep(0.001)
        await cls.classify("q2")
        time.sleep(0.001)
        await cls.classify("q3")  # q1 퇴거

        # q1 재조회 시 새 LLM 호출
        await cls.classify("q1")
        assert http.post.call_count == 4  # q1, q2, q3, q1(재)


class TestIntentDecision:
    def test_frozen_dataclass(self):
        d = IntentDecision(intent="factual", confidence=0.9)
        with pytest.raises(Exception):
            d.intent = "comparative"  # type: ignore[misc]
