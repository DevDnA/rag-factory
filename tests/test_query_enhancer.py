"""query_enhancer 모듈 테스트 — HyDE / multi-query / RRF."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from slm_factory.rag.query_enhancer import (
    generate_hyde_doc,
    generate_multi_queries,
    rrf_merge,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, payload: dict | None = None, *, exc: Exception | None = None):
        self._payload = payload or {}
        self._exc = exc
        self.calls = 0
        self.last_prompt: str = ""

    async def post(self, url, json=None, timeout=None):
        self.calls += 1
        if json:
            self.last_prompt = json.get("prompt", "")
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._payload)


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------


class TestHyde:
    @pytest.mark.asyncio
    async def test_정상_생성(self):
        http = _FakeHttp({"response": "이 질문에 대한 가상 문서 본문입니다."})
        doc = await generate_hyde_doc(
            "SLA란 무엇인가",
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert "가상 문서" in doc
        assert "SLA란 무엇인가" in http.last_prompt

    @pytest.mark.asyncio
    async def test_빈_쿼리는_LLM_호출_없이_빈_문자열(self):
        http = _FakeHttp()
        doc = await generate_hyde_doc("", http_client=http, ollama_model="m", api_base="http://x")
        assert doc == ""
        assert http.calls == 0

    @pytest.mark.asyncio
    async def test_LLM_예외는_빈_문자열(self):
        http = _FakeHttp(exc=RuntimeError("boom"))
        doc = await generate_hyde_doc("질문", http_client=http, ollama_model="m", api_base="http://x")
        assert doc == ""

    @pytest.mark.asyncio
    async def test_think_태그_제거(self):
        http = _FakeHttp({"response": "<think>분석</think>가상 본문 텍스트"})
        doc = await generate_hyde_doc("질문", http_client=http, ollama_model="m", api_base="http://x")
        assert "<think>" not in doc
        assert "가상 본문 텍스트" in doc


# ---------------------------------------------------------------------------
# Multi-Query
# ---------------------------------------------------------------------------


class TestMultiQuery:
    @pytest.mark.asyncio
    async def test_정상_생성_3개(self):
        payload = {"response": json.dumps({"queries": ["변형1", "변형2", "변형3"]})}
        http = _FakeHttp(payload)
        out = await generate_multi_queries(
            "원본",
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            n=3,
        )
        assert out == ["변형1", "변형2", "변형3"]

    @pytest.mark.asyncio
    async def test_n_제한(self):
        payload = {"response": json.dumps({"queries": ["a", "b", "c", "d", "e"]})}
        http = _FakeHttp(payload)
        out = await generate_multi_queries(
            "원본",
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            n=2,
        )
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_원본과_동일한_변형은_제거(self):
        payload = {"response": json.dumps({"queries": ["원본", "변형A"]})}
        http = _FakeHttp(payload)
        out = await generate_multi_queries(
            "원본",
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            n=3,
        )
        assert "원본" not in out
        assert out == ["변형A"]

    @pytest.mark.asyncio
    async def test_중복_변형_dedup(self):
        payload = {"response": json.dumps({"queries": ["A", "A", "B"]})}
        http = _FakeHttp(payload)
        out = await generate_multi_queries(
            "원본",
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            n=5,
        )
        assert out == ["A", "B"]

    @pytest.mark.asyncio
    async def test_LLM_예외는_빈_리스트(self):
        http = _FakeHttp(exc=RuntimeError("boom"))
        out = await generate_multi_queries(
            "원본", http_client=http, ollama_model="m", api_base="http://x"
        )
        assert out == []

    @pytest.mark.asyncio
    async def test_파싱_실패는_빈_리스트(self):
        http = _FakeHttp({"response": "not json"})
        out = await generate_multi_queries(
            "원본", http_client=http, ollama_model="m", api_base="http://x"
        )
        assert out == []


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


@dataclass
class _Doc:
    doc_id: str
    score: float = 0.0


class TestRRF:
    def test_빈_입력은_빈_결과(self):
        assert rrf_merge([], top_k=5) == []

    def test_단일_리스트는_그대로_top_k_제한(self):
        items = [_Doc("a"), _Doc("b"), _Doc("c"), _Doc("d")]
        out = rrf_merge([items], top_k=2)
        assert [d.doc_id for d in out] == ["a", "b"]

    def test_두_리스트_병합(self):
        list_a = [_Doc("x"), _Doc("y"), _Doc("z")]  # ranks: x=1, y=2, z=3
        list_b = [_Doc("y"), _Doc("x"), _Doc("w")]  # ranks: y=1, x=2, w=3
        out = rrf_merge([list_a, list_b], top_k=4, k=60)
        ids = [d.doc_id for d in out]
        # x: 1/(60+1) + 1/(60+2) = 0.01639+0.01613 = 0.03252
        # y: 1/(60+2) + 1/(60+1) = 0.01613+0.01639 = 0.03252  (tied with x)
        # z: 1/(60+3) = 0.01587
        # w: 1/(60+3) = 0.01587  (tied with z)
        # x and y are tied — order between them is implementation-dependent (sort stability)
        assert set(ids[:2]) == {"x", "y"}
        assert set(ids[2:]) == {"z", "w"}

    def test_top_k_초과_안함(self):
        list_a = [_Doc(str(i)) for i in range(10)]
        list_b = [_Doc(str(i)) for i in range(5, 15)]
        out = rrf_merge([list_a, list_b], top_k=3)
        assert len(out) == 3

    def test_빈_doc_id는_무시(self):
        list_a = [_Doc(""), _Doc("x")]
        out = rrf_merge([list_a], top_k=5)
        assert [d.doc_id for d in out] == ["x"]

    def test_k_값_변경(self):
        # k=1이면 rank 차이가 더 크게 반영됨.
        list_a = [_Doc("a"), _Doc("b")]
        list_b = [_Doc("b"), _Doc("a")]
        # k=1: a=1/2+1/3=0.833, b=1/3+1/2=0.833 → tie
        # k=60: 둘 다 같음 → tie
        out = rrf_merge([list_a, list_b], top_k=2, k=1)
        ids = {d.doc_id for d in out}
        assert ids == {"a", "b"}
