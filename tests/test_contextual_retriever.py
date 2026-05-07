"""Contextual Retrieval (Anthropic) 모듈의 단위 테스트입니다."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from slm_factory.config import ContextualRetrievalConfig
from slm_factory.contextual_retriever import (
    _build_prompt,
    _truncate_document,
    generate_chunk_contexts_async,
    prepend_context,
)


# ---------------------------------------------------------------------------
# 프롬프트 빌더
# ---------------------------------------------------------------------------


class Test프롬프트빌더:
    def test_한국어_프롬프트는_문서와_청크를_포함(self):
        prompt = _build_prompt("DOC", "CHUNK", max_chars=200, language="ko")
        assert "DOC" in prompt
        assert "CHUNK" in prompt
        assert "200" in prompt
        assert "<document>" in prompt
        assert "<chunk>" in prompt

    def test_영어_프롬프트는_영어_지시문(self):
        prompt = _build_prompt("DOC", "CHUNK", max_chars=200, language="en")
        assert "DOC" in prompt
        assert "Please write a short" in prompt


# ---------------------------------------------------------------------------
# 문서 절단
# ---------------------------------------------------------------------------


class Test문서절단:
    def test_짧은_문서는_그대로(self):
        doc = "짧은 문서"
        assert _truncate_document(doc, max_chars=1000) == doc

    def test_긴_문서는_앞뒤_절반만(self):
        doc = "A" * 100 + "B" * 100
        truncated = _truncate_document(doc, max_chars=20)
        assert "[중략]" in truncated
        assert truncated.startswith("A")
        assert truncated.endswith("B")


# ---------------------------------------------------------------------------
# prepend_context
# ---------------------------------------------------------------------------


class TestPrependContext:
    def test_빈_컨텍스트는_본문만_반환(self):
        assert prepend_context("본문", "") == "본문"

    def test_컨텍스트_있으면_prefix_추가(self):
        result = prepend_context("본문", "이 청크는 서론입니다.")
        assert result.startswith("[Context] 이 청크는 서론입니다.")
        assert result.endswith("본문")


# ---------------------------------------------------------------------------
# generate_chunk_contexts_async
# ---------------------------------------------------------------------------


class TestGenerate청크컨텍스트:
    @pytest.mark.asyncio
    async def test_빈_청크리스트는_빈리스트(self):
        teacher = MagicMock()
        teacher.agenerate = AsyncMock()
        result = await generate_chunk_contexts_async(
            teacher, "doc", [], ContextualRetrievalConfig(enabled=True)
        )
        assert result == []
        teacher.agenerate.assert_not_called()

    @pytest.mark.asyncio
    async def test_짧은_문서는_빈컨텍스트_반환(self):
        """skip_short_docs 미만 문서는 LLM 호출하지 않고 빈 prefix를 반환합니다."""
        teacher = MagicMock()
        teacher.agenerate = AsyncMock()
        cfg = ContextualRetrievalConfig(enabled=True, skip_short_docs=1000)
        result = await generate_chunk_contexts_async(
            teacher, "짧은 문서", ["청크1", "청크2"], cfg
        )
        assert result == ["", ""]
        teacher.agenerate.assert_not_called()

    @pytest.mark.asyncio
    async def test_각_청크마다_LLM_호출(self):
        teacher = MagicMock()
        teacher.agenerate = AsyncMock(return_value="이 청크는 서론입니다.")
        cfg = ContextualRetrievalConfig(
            enabled=True, skip_short_docs=10, max_chars=200
        )
        chunks = ["A" * 100, "B" * 100, "C" * 100]
        result = await generate_chunk_contexts_async(
            teacher, "X" * 5000, chunks, cfg
        )
        assert len(result) == 3
        assert all(c == "이 청크는 서론입니다." for c in result)
        assert teacher.agenerate.call_count == 3

    @pytest.mark.asyncio
    async def test_LLM_실패시_빈_컨텍스트(self):
        teacher = MagicMock()
        teacher.agenerate = AsyncMock(side_effect=RuntimeError("ollama down"))
        cfg = ContextualRetrievalConfig(enabled=True, skip_short_docs=10)
        result = await generate_chunk_contexts_async(
            teacher, "X" * 5000, ["청크1"], cfg
        )
        assert result == [""]

    @pytest.mark.asyncio
    async def test_max_chars_초과시_절단(self):
        teacher = MagicMock()
        teacher.agenerate = AsyncMock(return_value="가" * 500)
        cfg = ContextualRetrievalConfig(
            enabled=True, skip_short_docs=10, max_chars=100
        )
        result = await generate_chunk_contexts_async(
            teacher, "X" * 5000, ["청크"], cfg
        )
        assert len(result[0]) <= 101  # …문자 1개 추가 허용
        assert result[0].endswith("…")

    @pytest.mark.asyncio
    async def test_temperature_및_think_고정(self):
        """판정 안정성을 위해 temperature=0.0, think=False로 호출되어야 합니다."""
        teacher = MagicMock()
        teacher.agenerate = AsyncMock(return_value="ctx")
        cfg = ContextualRetrievalConfig(enabled=True, skip_short_docs=10)
        await generate_chunk_contexts_async(teacher, "X" * 5000, ["청크"], cfg)
        _args, kwargs = teacher.agenerate.call_args
        assert kwargs["temperature"] == 0.0
        assert kwargs["think"] is False


# ---------------------------------------------------------------------------
# pytest-asyncio 의존성 안내
# ---------------------------------------------------------------------------

# 위 테스트는 pytest-asyncio가 필요합니다. 없으면 다음과 같이 동기 실행:
#   pip install pytest-asyncio
# 또는 pyproject.toml의 dev extra에 추가하세요.
