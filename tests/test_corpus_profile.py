"""CorpusProfile 모듈 테스트 — dataclass·persistence·LLM 자동 생성."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_factory.rag.corpus_profile import (
    CorpusProfile,
    generate_corpus_profile,
    load_corpus_profile,
    merge_with_override,
    save_corpus_profile,
)


class TestCorpusProfile:
    def test_빈_profile은_is_empty_True(self):
        assert CorpusProfile().is_empty() is True

    def test_name만_있어도_not_empty(self):
        assert CorpusProfile(name="X").is_empty() is False

    def test_keywords만_있어도_not_empty(self):
        assert CorpusProfile(keywords=["a"]).is_empty() is False

    def test_to_prompt_header_빈_profile은_빈_문자열(self):
        assert CorpusProfile().to_prompt_header() == ""

    def test_to_prompt_header_모든_필드_포함(self):
        p = CorpusProfile(name="N", summary="S", keywords=["k1", "k2"])
        h = p.to_prompt_header()
        assert "[본 corpus 도메인 정보]" in h
        assert "명칭: N" in h
        assert "요약: S" in h
        assert "k1" in h and "k2" in h

    def test_to_prompt_header_keywords_상한_20개(self):
        p = CorpusProfile(name="N", keywords=[f"k{i}" for i in range(30)])
        h = p.to_prompt_header()
        # k20 이상은 잘림
        assert "k0" in h
        assert "k19" in h
        assert "k20" not in h


class TestPersistence:
    def test_없는_파일은_빈_profile(self, tmp_path: Path):
        p = load_corpus_profile(tmp_path / "nope.json")
        assert p.is_empty()

    def test_save_load_왕복(self, tmp_path: Path):
        original = CorpusProfile(
            name="N", summary="S", keywords=["a", "b"], generated_at="2026-05-07T00:00:00Z", model="m"
        )
        path = tmp_path / "profile.json"
        save_corpus_profile(original, path)

        loaded = load_corpus_profile(path)
        assert loaded == original

    def test_파싱_실패_파일은_빈_profile(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not-json", encoding="utf-8")
        p = load_corpus_profile(path)
        assert p.is_empty()

    def test_save_실패는_raise_안함(self):
        # 존재할 수 없는 경로 — 호출만으로는 예외 안 남.
        p = CorpusProfile(name="X")
        save_corpus_profile(p, Path("/proc/1/nope.json"))


class TestMergeOverride:
    def test_override_없으면_auto_그대로(self):
        auto = CorpusProfile(name="auto-N", summary="auto-S", keywords=["k"])
        merged = merge_with_override(auto)
        assert merged.name == "auto-N"
        assert merged.summary == "auto-S"
        assert merged.keywords == ["k"]

    def test_name_override_우선(self):
        auto = CorpusProfile(name="auto-N", summary="auto-S")
        merged = merge_with_override(auto, name_override="user-N")
        assert merged.name == "user-N"
        assert merged.summary == "auto-S"

    def test_keywords_override_완전_대체(self):
        auto = CorpusProfile(keywords=["a", "b", "c"])
        merged = merge_with_override(auto, keywords_override=["x", "y"])
        assert merged.keywords == ["x", "y"]

    def test_빈_keywords_override는_무시(self):
        auto = CorpusProfile(keywords=["a", "b"])
        merged = merge_with_override(auto, keywords_override=[])
        assert merged.keywords == ["a", "b"]


# ---------------------------------------------------------------------------
# generate_corpus_profile — LLM 호출 모킹
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, payload: dict, *, raise_exc: Exception | None = None):
        self._payload = payload
        self._raise = raise_exc
        self.calls = 0

    async def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return _FakeResponse(self._payload)


class TestGenerate:
    @pytest.mark.asyncio
    async def test_정상_생성(self):
        payload = {
            "response": json.dumps({
                "name": "한국 통신사 RFP",
                "summary": "통신 인프라 요구사항.",
                "keywords": ["NMS", "BIS", "MIMO"],
            })
        }
        http = _FakeHttp(payload)
        p = await generate_corpus_profile(
            chunks=["청크 1 본문", "청크 2 본문"],
            http_client=http,
            ollama_model="qwen3.5:9b",
            api_base="http://x",
        )
        assert p.name == "한국 통신사 RFP"
        assert p.summary == "통신 인프라 요구사항."
        assert p.keywords == ["NMS", "BIS", "MIMO"]
        assert p.model == "qwen3.5:9b"
        assert p.chunks_sampled == 2
        assert p.generated_at  # ISO 시각 채워짐

    @pytest.mark.asyncio
    async def test_빈_청크는_빈_profile(self):
        http = _FakeHttp({"response": ""})
        p = await generate_corpus_profile(
            chunks=[],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.is_empty()
        assert http.calls == 0  # LLM 호출도 안 함

    @pytest.mark.asyncio
    async def test_LLM_예외는_빈_profile(self):
        http = _FakeHttp({}, raise_exc=RuntimeError("boom"))
        p = await generate_corpus_profile(
            chunks=["x"],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.is_empty()

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패는_빈_profile(self):
        payload = {"response": "그냥 텍스트, JSON 아님"}
        http = _FakeHttp(payload)
        p = await generate_corpus_profile(
            chunks=["x"],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.is_empty()

    @pytest.mark.asyncio
    async def test_think_태그_제거(self):
        # qwen 등 thinking 모드 모델의 응답에 포함될 수 있는 태그는 사전 strip.
        payload = {
            "response": (
                "<think>도메인 분석</think>"
                "{\"name\": \"X\", \"summary\": \"Y\", \"keywords\": [\"a\"]}"
            )
        }
        http = _FakeHttp(payload)
        p = await generate_corpus_profile(
            chunks=["x"],
            http_client=http,
            ollama_model="m",
            api_base="http://x",
        )
        assert p.name == "X"
        assert p.keywords == ["a"]
