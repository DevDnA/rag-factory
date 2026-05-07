"""IntentClassifier 테스트 — corpus_profile 주입·프롬프트 헤더 동작."""

from __future__ import annotations

import json

import pytest

from slm_factory.rag.agent.intent_classifier import IntentClassifier
from slm_factory.rag.corpus_profile import CorpusProfile


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    """post(...)가 호출될 때마다 마지막 prompt를 self.last_prompt에 저장."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.last_prompt: str = ""
        self.calls = 0

    async def post(self, url, json=None, timeout=None):
        self.calls += 1
        self.last_prompt = (json or {}).get("prompt", "")
        return _FakeResponse(self._payload)


class TestCorpusProfileInjection:
    @pytest.mark.asyncio
    async def test_profile_없으면_헤더_없음(self):
        payload = {"response": json.dumps({"intent": "factual", "confidence": 0.9})}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=None,
        )
        await classifier.classify("질의")
        assert "[본 corpus 도메인 정보]" not in http.last_prompt

    @pytest.mark.asyncio
    async def test_빈_profile은_헤더_없음(self):
        payload = {"response": json.dumps({"intent": "factual", "confidence": 0.9})}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=CorpusProfile(),
        )
        await classifier.classify("질의")
        assert "[본 corpus 도메인 정보]" not in http.last_prompt

    @pytest.mark.asyncio
    async def test_profile_채워지면_헤더_주입(self):
        payload = {"response": json.dumps({"intent": "factual", "confidence": 0.9})}
        http = _FakeHttp(payload)
        profile = CorpusProfile(
            name="한국 통신사 RFP",
            summary="통신 인프라 사양 요구사항.",
            keywords=["NMS", "BIS", "MIMO"],
        )
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=profile,
        )
        await classifier.classify("NMS는 무엇입니까")

        prompt = http.last_prompt
        assert "[본 corpus 도메인 정보]" in prompt
        assert "한국 통신사 RFP" in prompt
        assert "NMS" in prompt and "BIS" in prompt
        # 헤더가 사용자 질의보다 먼저 등장해야 LLM 컨텍스트로 작동.
        assert prompt.index("[본 corpus 도메인 정보]") < prompt.index("질의:")

    @pytest.mark.asyncio
    async def test_헤더_있어도_분류_결과는_그대로_파싱(self):
        # 헤더가 추가돼도 LLM 응답 파싱 로직은 영향 없음 — 분류 정확도만 변함.
        payload = {"response": json.dumps({"intent": "comparative", "confidence": 0.88, "reason": "비교"})}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=CorpusProfile(name="N", keywords=["k"]),
        )
        decision = await classifier.classify("질의")
        assert decision.intent == "comparative"
        assert decision.confidence == pytest.approx(0.88)
