"""Agent RAG 도구 정의 및 레지스트리입니다.

에이전트가 ReAct 루프에서 사용할 수 있는 도구(search, lookup,
compare, list_documents)를 정의합니다.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ...utils import get_logger
from ..search import search_documents, SearchOutput

logger = get_logger("rag.agent.tools")


@dataclass
class ToolResult:
    """도구 실행 결과 — 텍스트와 구조화된 소스를 함께 반환."""

    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolSpec:
    """도구 명세.

    ``parallel_safe``는 동일 plan 내 다른 step과 ``asyncio.gather``로 병렬
    실행해도 안전한지를 표시합니다. 외부 상태를 변경하지 않고 read-only로
    동작하며 결과가 호출 순서에 의존하지 않는 경우에만 ``True``로 두세요.
    Orchestrator의 ``parallel_steps`` 게이트가 이 값을 검사합니다.
    """

    name: str
    description: str
    parameters: str
    fn: Callable[..., Awaitable[ToolResult]]
    parallel_safe: bool = False


class ToolRegistry:
    """에이전트가 사용할 수 있는 도구 레지스트리.

    Parameters
    ----------
    app_state:
        FastAPI ``app.state`` — Qdrant, 임베딩 모델 등을 보유.
    config:
        RAG 설정.
    tokenize_fn:
        한국어 토큰화 함수 (BM25 검색용).
    """

    def __init__(
        self,
        app_state: Any,
        config: Any,
        tokenize_fn: Any = None,
        *,
        keep_alive: str = "5m",
    ) -> None:
        self._app_state = app_state
        self._config = config
        self._tokenize_fn = tokenize_fn
        self._keep_alive = keep_alive
        self._tools: dict[str, ToolSpec] = {}

        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        # search/lookup/compare는 read-only & 결과가 호출 순서에 의존하지 않으므로
        # parallel_safe=True. evaluate/list_documents는 의존성·일관성 보장을 위해
        # 보수적으로 직렬 처리.
        self.register(ToolSpec(
            name="search",
            description="벡터 DB에서 관련 문서를 검색합니다. 핵심 키워드로 검색하세요.",
            parameters='{"query": "검색할 핵심 키워드"}',
            fn=self._tool_search,
            parallel_safe=True,
        ))
        self.register(ToolSpec(
            name="lookup",
            description="특정 문서 ID로 문서 전체 내용을 조회합니다.",
            parameters='{"doc_id": "조회할 문서 ID"}',
            fn=self._tool_lookup,
            parallel_safe=True,
        ))
        self.register(ToolSpec(
            name="compare",
            description="두 검색어의 결과를 비교합니다. 차이점 분석에 유용합니다.",
            parameters='{"query_a": "첫 번째 검색어", "query_b": "두 번째 검색어"}',
            fn=self._tool_compare,
            parallel_safe=True,
        ))
        self.register(ToolSpec(
            name="evaluate",
            description="수집된 정보가 질문에 답하기 충분한지 평가합니다. 부족하면 추가 검색 키워드를 제안합니다.",
            parameters='{"query": "원래 질문", "context": "지금까지 수집된 정보 요약"}',
            fn=self._tool_evaluate,
            parallel_safe=False,
        ))
        self.register(ToolSpec(
            name="list_documents",
            description="인덱싱된 문서 목록과 ID를 조회합니다.",
            parameters="{}",
            fn=self._tool_list_documents,
            parallel_safe=False,
        ))

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """프롬프트에 삽입할 도구 설명 텍스트를 생성합니다."""
        lines: list[str] = []
        for spec in self._tools.values():
            lines.append(f"- **{spec.name}**: {spec.description}")
            lines.append(f"  입력: {spec.parameters}")
        return "\n".join(lines)

    async def execute(self, name: str, args: dict) -> ToolResult:
        """도구를 실행하고 결과를 반환합니다."""
        spec = self._tools.get(name)
        if spec is None:
            available = ", ".join(self._tools.keys())
            return ToolResult(text=f"[오류] 알 수 없는 도구 '{name}'. 사용 가능: {available}")

        try:
            return await spec.fn(args)
        except Exception as e:
            logger.warning("도구 '%s' 실행 실패: %s", name, e, exc_info=True)
            return ToolResult(text=f"[오류] 도구 '{name}' 실행 중 문제가 발생했습니다.")

    # ------------------------------------------------------------------
    # 내장 도구 구현
    # ------------------------------------------------------------------

    def _search_common(self, query: str, top_k: int | None = None) -> SearchOutput:
        """공통 검색 로직 — search, compare에서 재사용."""
        return search_documents(
            query,
            top_k=top_k or self._config.rag.top_k,
            qdrant_client=self._app_state.qdrant_client,
            collection_name=self._app_state.collection_name,
            embedding_model=self._app_state.embedding_model,
            min_score=self._config.rag.min_score,
            reranker=self._app_state.reranker,
            hybrid_search=self._config.rag.hybrid_search,
            bm25_index=self._app_state.bm25_index,
            bm25_docs=getattr(self._app_state, "bm25_docs", None),
            bm25_ids=getattr(self._app_state, "bm25_ids", None),
            bm25_metadatas=getattr(self._app_state, "bm25_metadatas", None),
            tokenize_fn=self._tokenize_fn,
        )

    @staticmethod
    def _format_search_output(output: SearchOutput, label: str = "") -> str:
        """SearchOutput을 에이전트가 읽을 수 있는 텍스트로 변환."""
        if not output.sources:
            prefix = f"({label}) " if label else ""
            return f"{prefix}관련 문서를 찾을 수 없습니다."

        parts: list[str] = []
        for i, src in enumerate(output.sources, 1):
            prefix = f"[{label} " if label else "["
            parts.append(
                f"{prefix}문서 {i}] (ID: {src.doc_id}, 유사도: {src.score:.2f})\n"
                f"{src.content[:500]}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _extract_sources(output: SearchOutput) -> list[dict[str, Any]]:
        """SearchOutput에서 구조화된 소스 목록을 추출합니다."""
        return [
            {"doc_id": s.doc_id, "score": s.score, "content": s.content[:300]}
            for s in output.sources
        ]

    async def _tool_search(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(text="[오류] 'query' 파라미터가 필요합니다.")

        top_k = args.get("top_k")
        output = await self._enhanced_search(query, top_k)
        return ToolResult(
            text=self._format_search_output(output),
            sources=self._extract_sources(output),
        )

    async def _enhanced_search(self, query: str, top_k: int | None) -> SearchOutput:
        """HyDE + Multi-Query + RRF로 검색 품질을 향상시킨 검색.

        config 플래그가 모두 false면 단일 ``_search_common`` 호출과 동일하게
        동작 — 기존 동작 보존.
        """
        from ..query_enhancer import (
            generate_hyde_doc,
            generate_multi_queries,
            rrf_merge,
        )

        rag_cfg = self._config.rag
        use_hyde = bool(getattr(rag_cfg, "hyde_enabled", False))
        use_mq = bool(getattr(rag_cfg, "multi_query_enabled", False))

        if not use_hyde and not use_mq:
            return await asyncio.to_thread(self._search_common, query, top_k)

        # 보조 LLM 호출용 모델·엔드포인트 — 라우터 모델 슬롯 재사용 (없으면 기본).
        agent_cfg = getattr(rag_cfg, "agent", None)
        models_cfg = getattr(agent_cfg, "models", None)
        aux_model = (
            getattr(models_cfg, "router_model", "")
            or getattr(models_cfg, "synthesis_model", "")
            or rag_cfg.ollama_model
            or self._config.teacher.model
        ).strip()
        api_base = self._config.teacher.api_base
        keep_alive = self._keep_alive
        aux_timeout = min(rag_cfg.request_timeout, 30.0)
        n_variants = max(1, int(getattr(rag_cfg, "multi_query_count", 3) or 3))
        http_client = self._app_state.http_client

        # 두 enhancer를 병렬 호출 — LLM 라운드트립 1회로 묶음.
        hyde_task = (
            asyncio.create_task(
                generate_hyde_doc(
                    query,
                    http_client=http_client,
                    ollama_model=aux_model,
                    api_base=api_base,
                    request_timeout=aux_timeout,
                    keep_alive=keep_alive,
                )
            )
            if use_hyde
            else None
        )
        mq_task = (
            asyncio.create_task(
                generate_multi_queries(
                    query,
                    http_client=http_client,
                    ollama_model=aux_model,
                    api_base=api_base,
                    n=n_variants,
                    request_timeout=aux_timeout,
                    keep_alive=keep_alive,
                )
            )
            if use_mq
            else None
        )

        hyde_doc = await hyde_task if hyde_task is not None else ""
        variants = await mq_task if mq_task is not None else []

        # 검색할 쿼리 목록 구성.
        search_queries: list[str] = []
        if hyde_doc:
            # HyDE: 가상 문서 + 원본 질의 결합 — 둘 다 임베딩에 기여.
            search_queries.append(f"{hyde_doc}\n\n{query}")
        else:
            search_queries.append(query)
        # 변형들 추가 (원본은 이미 포함됨)
        search_queries.extend(variants)

        if len(search_queries) == 1:
            return await asyncio.to_thread(self._search_common, search_queries[0], top_k)

        # 병렬 검색 — top_k는 각 검색마다 동일 적용 (RRF가 융합 후 자르도록).
        per_query_top_k = top_k or rag_cfg.top_k
        results = await asyncio.gather(
            *[
                asyncio.to_thread(self._search_common, q, per_query_top_k)
                for q in search_queries
            ],
            return_exceptions=True,
        )

        valid_outputs: list[SearchOutput] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Multi-query 검색 부분 실패: %s", r)
                continue
            valid_outputs.append(r)

        if not valid_outputs:
            return await asyncio.to_thread(self._search_common, query, top_k)

        if len(valid_outputs) == 1:
            return valid_outputs[0]

        # RRF 병합 — 각 SearchOutput.sources를 정렬된 리스트로 사용.
        source_lists = [out.sources for out in valid_outputs]
        merged_sources = rrf_merge(
            source_lists,
            top_k=per_query_top_k,
            id_attr="doc_id",
        )

        # 병합된 sources에 해당하는 context_parts 재구성 (search.py의 contract 유지).
        merged_ids = {s.doc_id for s in merged_sources}
        merged_context: list[str] = []
        seen_ctx: set[str] = set()
        for out in valid_outputs:
            for src, ctx in zip(out.sources, out.context_parts):
                if src.doc_id in merged_ids and src.doc_id not in seen_ctx:
                    merged_context.append(ctx)
                    seen_ctx.add(src.doc_id)

        from ..search import SearchOutput as SO

        return SO(sources=merged_sources, context_parts=merged_context)

    async def _tool_lookup(self, args: dict) -> ToolResult:
        doc_id = args.get("doc_id", "")
        if not doc_id:
            return ToolResult(text="[오류] 'doc_id' 파라미터가 필요합니다.")

        qdrant_client = self._app_state.qdrant_client
        collection_name = self._app_state.collection_name

        def _do_lookup():
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                limit=1,
                with_payload=True,
            )
            return results[0] if results else []

        try:
            points = await asyncio.to_thread(_do_lookup)
            if not points:
                return ToolResult(text=f"문서 ID '{doc_id}'를 찾을 수 없습니다.")

            point = points[0]
            content = point.payload.get("document", "")
            parent = point.payload.get("parent_content", content)
            # lookup 결과도 길이 제한
            text = f"[문서 ID: {doc_id}]\n{parent[:2000]}"
            return ToolResult(text=text)
        except Exception as e:
            logger.warning("문서 조회 실패: %s", e, exc_info=True)
            return ToolResult(text="[오류] 문서 조회 중 문제가 발생했습니다.")

    async def _tool_compare(self, args: dict) -> ToolResult:
        query_a = args.get("query_a", "")
        query_b = args.get("query_b", "")
        if not query_a or not query_b:
            return ToolResult(text="[오류] 'query_a'와 'query_b' 파라미터가 모두 필요합니다.")

        top_k = args.get("top_k", max(3, self._config.rag.top_k // 2))
        output_a, output_b = await asyncio.gather(
            asyncio.to_thread(self._search_common, query_a, top_k),
            asyncio.to_thread(self._search_common, query_b, top_k),
        )

        text_a = self._format_search_output(output_a, label=f"A: {query_a}")
        text_b = self._format_search_output(output_b, label=f"B: {query_b}")

        all_sources = self._extract_sources(output_a) + self._extract_sources(output_b)
        return ToolResult(
            text=f"--- 검색 A: {query_a} ---\n{text_a}\n\n--- 검색 B: {query_b} ---\n{text_b}",
            sources=all_sources,
        )

    async def _tool_evaluate(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        context = args.get("context", "")
        if not query:
            return ToolResult(text="[오류] 'query' 파라미터가 필요합니다.")

        from .prompts import SUFFICIENCY_PROMPT

        prompt = SUFFICIENCY_PROMPT.format(query=query, context=context[:2000])

        try:
            http_client = self._app_state.http_client
            ollama_model = self._config.rag.ollama_model or self._config.teacher.model
            api_base = self._config.teacher.api_base

            response = await http_client.post(
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "keep_alive": self._keep_alive,
                    "options": {"num_predict": 200},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "")

            try:
                result = json.loads(text)
                sufficient = result.get("sufficient", False)
                reason = result.get("reason", "")
                suggestion = result.get("suggestion", "")
                summary = f"충분성: {'예' if sufficient else '아니오'}\n이유: {reason}"
                if not sufficient and suggestion:
                    summary += f"\n추가 검색 제안: {suggestion}"
                return ToolResult(text=summary)
            except json.JSONDecodeError:
                return ToolResult(text=text[:500])
        except Exception as e:
            logger.warning("자기평가 실패: %s", e, exc_info=True)
            return ToolResult(text="[오류] 평가 중 문제가 발생했습니다.")

    async def _tool_list_documents(self, args: dict) -> ToolResult:
        qdrant_client = self._app_state.qdrant_client
        collection_name = self._app_state.collection_name

        def _do_scroll():
            results = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True,
            )
            return results[0] if results else []

        try:
            points = await asyncio.to_thread(_do_scroll)
            if not points:
                return ToolResult(text="인덱싱된 문서가 없습니다.")

            seen: dict[str, str] = {}
            for point in points:
                doc_id = point.payload.get("doc_id", str(point.id))
                content = point.payload.get("document", "")
                if doc_id not in seen:
                    seen[doc_id] = content[:80]

            lines = [f"총 {len(seen)}개 문서:"]
            for doc_id, preview in list(seen.items())[:50]:
                lines.append(f"- {doc_id}: {preview}...")
            if len(seen) > 50:
                lines.append(f"  ... 외 {len(seen) - 50}개")
            return ToolResult(text="\n".join(lines))
        except Exception as e:
            logger.warning("문서 목록 조회 실패: %s", e, exc_info=True)
            return ToolResult(text="[오류] 문서 목록 조회 중 문제가 발생했습니다.")
