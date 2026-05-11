"""도구 레지스트리 및 도구 실행 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from slm_factory.rag.agent.tools import ToolRegistry, ToolSpec, ToolResult
from slm_factory.rag.search import SearchResult, SearchOutput


def _make_app_state():
    """테스트용 app_state mock."""
    state = MagicMock()
    state.qdrant_client = MagicMock()
    state.collection_name = "test_corpus"
    state.embedding_model = MagicMock()
    state.reranker = None
    state.bm25_index = None
    state.bm25_docs = None
    state.bm25_ids = None
    state.bm25_metadatas = None
    return state


def _make_config():
    """테스트용 config mock — query enhancement 비활성으로 단순 검색 동작 검증."""
    config = MagicMock()
    config.rag.top_k = 3
    config.rag.min_score = 0.0
    config.rag.hybrid_search = False
    config.rag.hyde_enabled = False
    config.rag.multi_query_enabled = False
    return config


@pytest.fixture
def registry():
    return ToolRegistry(
        app_state=_make_app_state(),
        config=_make_config(),
    )


class TestToolRegistry:
    """ToolRegistry 기본 기능 테스트."""

    def test_내장_도구_등록(self, registry):
        assert "search" in registry.tool_names
        assert "lookup" in registry.tool_names
        assert "compare" in registry.tool_names
        assert "list_documents" in registry.tool_names

    def test_도구_설명_생성(self, registry):
        desc = registry.get_tool_descriptions()
        assert "search" in desc
        assert "lookup" in desc
        assert "compare" in desc

    def test_존재하지_않는_도구(self, registry):
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_알수없는_도구_실행(self, registry):
        result = await registry.execute("unknown_tool", {})
        assert "[오류]" in result.text
        assert "unknown_tool" in result.text

    def test_커스텀_도구_등록(self, registry):
        async def my_tool(args):
            return ToolResult(text="custom result")

        registry.register(ToolSpec(
            name="custom",
            description="커스텀 도구",
            parameters="{}",
            fn=my_tool,
        ))
        assert "custom" in registry.tool_names

    @pytest.mark.asyncio
    async def test_커스텀_도구_실행(self, registry):
        async def my_tool(args):
            return ToolResult(text=f"result: {args.get('x', 0)}")

        registry.register(ToolSpec(
            name="custom",
            description="test",
            parameters="{}",
            fn=my_tool,
        ))
        result = await registry.execute("custom", {"x": 42})
        assert result.text == "result: 42"


class TestToolSearch:
    """search 도구 테스트."""

    @pytest.mark.asyncio
    async def test_query_없으면_에러(self, registry):
        result = await registry.execute("search", {})
        assert "[오류]" in result.text

    @pytest.mark.asyncio
    async def test_검색_실행(self, registry):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "slm_factory.rag.agent.tools.search_documents",
                lambda *a, **kw: SearchOutput(
                    sources=[SearchResult(content="테스트 문서", doc_id="doc1", score=0.9)],
                    context_parts=["[문서 1]\n테스트 문서"],
                ),
            )
            result = await registry.execute("search", {"query": "테스트"})
            assert "doc1" in result.text
            assert "0.90" in result.text
            assert len(result.sources) == 1
            assert result.sources[0]["doc_id"] == "doc1"


class TestToolLookup:
    """lookup 도구 테스트."""

    @pytest.mark.asyncio
    async def test_doc_id_없으면_에러(self, registry):
        result = await registry.execute("lookup", {})
        assert "[오류]" in result.text


class TestToolCompare:
    """compare 도구 테스트."""

    @pytest.mark.asyncio
    async def test_query_하나만_있으면_에러(self, registry):
        result = await registry.execute("compare", {"query_a": "test"})
        assert "[오류]" in result.text

    @pytest.mark.asyncio
    async def test_두_query_모두_없으면_에러(self, registry):
        result = await registry.execute("compare", {})
        assert "[오류]" in result.text
