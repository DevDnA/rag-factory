"""독립 검색 모듈 — 벡터 검색 + 리랭킹 + BM25 하이브리드 검색 로직입니다.

``server.py``의 ``_search_documents`` 클로저에서 추출하여
Agent RAG 모듈에서도 재사용할 수 있도록 모듈 레벨 함수로 제공합니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..utils import get_logger

logger = get_logger("rag.search")


# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """검색된 문서 하나의 정보."""

    content: str
    doc_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchOutput:
    """검색 결과 집합."""

    sources: list[SearchResult]
    context_parts: list[str]


# ---------------------------------------------------------------------------
# BM25 검색
# ---------------------------------------------------------------------------


def bm25_search(
    query: str,
    top_k: int,
    bm25_index: Any,
    bm25_docs: list[str],
    bm25_ids: list[str],
    bm25_metadatas: list[dict] | None,
    tokenize_fn: Any = None,
) -> list[tuple[str, str, float, dict]]:
    """BM25Okapi를 사용한 키워드 기반 문서 검색입니다.

    Parameters
    ----------
    query:
        검색 질의 문자열.
    top_k:
        반환할 최대 문서 수.
    bm25_index:
        사전 구축된 BM25Okapi 인스턴스.
    bm25_docs:
        BM25 인덱스에 대응하는 문서 텍스트 목록.
    bm25_ids:
        BM25 인덱스에 대응하는 문서 ID 목록.
    bm25_metadatas:
        BM25 인덱스에 대응하는 메타데이터 목록.
    tokenize_fn:
        토큰화 함수. ``None``이면 소문자 공백 분할.
    """
    if bm25_index is None or not bm25_docs:
        return []

    if tokenize_fn is not None:
        query_tokens = tokenize_fn(query)
    else:
        query_tokens = query.lower().split()

    if not query_tokens:
        return []

    scores = bm25_index.get_scores(query_tokens)
    top_indices = scores.argsort()[::-1][:top_k]

    results: list[tuple[str, str, float, dict]] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break
        meta = (
            bm25_metadatas[idx]
            if bm25_metadatas and idx < len(bm25_metadatas)
            else {}
        )
        results.append((bm25_docs[idx], bm25_ids[idx], score, meta))
    return results


# ---------------------------------------------------------------------------
# 메인 검색 함수
# ---------------------------------------------------------------------------


def search_documents(
    query: str,
    *,
    top_k: int,
    qdrant_client: Any,
    collection_name: str,
    embedding_model: Any,
    min_score: float = 0.0,
    reranker: Any = None,
    hybrid_search: bool = False,
    bm25_index: Any = None,
    bm25_docs: list[str] | None = None,
    bm25_ids: list[str] | None = None,
    bm25_metadatas: list[dict] | None = None,
    tokenize_fn: Any = None,
) -> SearchOutput:
    """벡터 검색 + 리랭킹 + BM25 하이브리드 검색을 수행합니다.

    ``server.py``의 ``_search_documents``와 동일한 로직이며,
    ``app.state`` 대신 명시적 파라미터를 받습니다.

    Returns
    -------
    SearchOutput
        검색 결과(sources)와 컨텍스트 파트(context_parts).
    """
    use_reranker = reranker is not None
    initial_k = top_k * 3 if use_reranker else top_k

    t0 = time.monotonic()
    query_embedding = embedding_model.encode(
        query, prompt_name="query", show_progress_bar=False
    ).tolist()
    t_embed = time.monotonic()

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=initial_k,
        with_payload=True,
    )
    t_search = time.monotonic()

    documents = [p.payload.get("document", "") for p in results.points]
    ids = [p.payload.get("doc_id", str(p.id)) for p in results.points]
    distances = [1.0 - p.score for p in results.points]
    metadatas = [
        {k: v for k, v in p.payload.items() if k not in ("document", "doc_id")}
        for p in results.points
    ]

    # -- 리랭킹 ---
    t_rerank = t_search
    if use_reranker and documents:
        pairs = [(query, doc) for doc in documents]
        rerank_scores = reranker.predict(pairs)
        t_rerank = time.monotonic()
        ranked = sorted(
            zip(rerank_scores, documents, ids, distances, metadatas),
            reverse=True,
        )
        ranked = ranked[:top_k]
        if ranked:
            _, documents, ids, distances, metadatas = zip(*ranked)
            documents = list(documents)
            ids = list(ids)
            distances = list(distances)
            metadatas = list(metadatas)
        else:
            documents, ids, distances, metadatas = [], [], [], []
    else:
        documents = documents[:top_k]
        ids = ids[:top_k]
        distances = distances[:top_k]
        metadatas = metadatas[:top_k]

    # -- BM25 하이브리드 (RRF) ---
    t_bm25 = t_rerank
    if (
        hybrid_search
        and bm25_index is not None
        and bm25_docs is not None
        and bm25_ids is not None
    ):
        rrf_k = 60

        rrf_scores: dict[str, float] = {}
        rrf_data: dict[str, tuple[str, dict]] = {}
        for rank, (doc, doc_id, _dist, meta) in enumerate(
            zip(documents, ids, distances, metadatas)
        ):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            if doc_id not in rrf_data:
                rrf_data[doc_id] = (doc, meta)

        bm25_results = bm25_search(
            query, top_k * 2, bm25_index, bm25_docs, bm25_ids, bm25_metadatas,
            tokenize_fn,
        )
        for rank, (doc, doc_id, _score, meta) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            if doc_id not in rrf_data:
                rrf_data[doc_id] = (doc, meta)

        sorted_ids = sorted(
            rrf_scores, key=lambda did: rrf_scores[did], reverse=True
        )[:top_k]
        documents = [rrf_data[did][0] for did in sorted_ids]
        ids = list(sorted_ids)
        distances = [1.0 - rrf_scores[did] for did in sorted_ids]
        metadatas = [rrf_data[did][1] for did in sorted_ids]
        t_bm25 = time.monotonic()

    # -- 유사도 필터링 ---
    if min_score > 0:
        filtered = [
            (doc, did, dist, meta)
            for doc, did, dist, meta in zip(documents, ids, distances, metadatas)
            if max(0.0, min(1.0, 1.0 - dist)) >= min_score
        ]
        if filtered:
            documents = [x[0] for x in filtered]
            ids = [x[1] for x in filtered]
            distances = [x[2] for x in filtered]
            metadatas = [x[3] for x in filtered]

    # -- Lost-in-the-middle 재정렬 ---
    if len(documents) >= 3:
        items = list(zip(documents, ids, distances, metadatas))
        front = [items[i] for i in range(0, len(items), 2)]
        back = [items[i] for i in range(1, len(items), 2)]
        reordered = front + list(reversed(back))
        documents = [x[0] for x in reordered]
        ids = [x[1] for x in reordered]
        distances = [x[2] for x in reordered]
        metadatas = [x[3] for x in reordered]

    # -- 소스 & 컨텍스트 조합 ---
    sources: list[SearchResult] = []
    context_parts: list[str] = []
    seen_parents: set[str] = set()

    doc_num = 0
    for doc, doc_id, distance, metadata in zip(
        documents, ids, distances, metadatas
    ):
        score = max(0.0, min(1.0, 1.0 - distance))
        parent = metadata.get("parent_content", doc) if metadata else doc
        sources.append(SearchResult(content=doc, doc_id=doc_id, score=score, metadata=metadata))
        parent_key = parent[:100]
        if parent_key not in seen_parents:
            seen_parents.add(parent_key)
            doc_num += 1
            context_parts.append(f"[문서 {doc_num}]\n{parent}")

    t_end = time.monotonic()
    logger.info(
        "검색 완료 %.3fs (임베딩 %.3fs, 벡터검색 %.3fs, 리랭커 %.3fs, BM25/RRF %.3fs, 후처리 %.3fs) — %d건",
        t_end - t0,
        t_embed - t0,
        t_search - t_embed,
        t_rerank - t_search,
        t_bm25 - t_rerank,
        t_end - t_bm25,
        len(sources),
    )
    return SearchOutput(sources=sources, context_parts=context_parts)
