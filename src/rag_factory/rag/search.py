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

    # -- BM25 하이브리드 융합 (RRF) — 리랭킹보다 먼저 수행한다 ---
    # 과거에는 벡터 결과만 먼저 리랭킹→top_k로 자른 뒤 BM25와 융합했다. 그 경우
    # BM25만 찾아낸(=벡터가 놓친) 청크는 cross-encoder 판정을 한 번도 받지 못했고,
    # 리랭크 점수의 크기도 RRF 순위로 뭉개졌다. 이제 벡터(top_k*3) + BM25(top_k*2)를
    # 먼저 RRF로 합쳐 union을 만든 뒤, 그 union 전체를 리랭커가 재정렬한다.
    t_bm25 = t_search
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

        # 리랭커가 있으면 union pool을 넓게(initial_k=top_k*3) 유지해 cross-encoder가
        # BM25-only 후보까지 보게 하고, 없으면 곧장 top_k로 자른다. pool을 initial_k로
        # 제한해 리랭크 부하는 기존(벡터 top_k*3)과 동일하게 유지한다.
        pool_k = initial_k if use_reranker else top_k
        sorted_ids = sorted(
            rrf_scores, key=lambda did: rrf_scores[did], reverse=True
        )[:pool_k]
        documents = [rrf_data[did][0] for did in sorted_ids]
        ids = list(sorted_ids)
        distances = [1.0 - rrf_scores[did] for did in sorted_ids]
        metadatas = [rrf_data[did][1] for did in sorted_ids]
        t_bm25 = time.monotonic()

    # -- 리랭킹 — 융합된 union(또는 hybrid 미사용 시 벡터 결과)을 cross-encoder로 재정렬 ---
    t_rerank = t_bm25
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
        # Phase 14 — LLM에 전달할 본문은 reranker가 선택한 *chunk* 그 자체(doc).
        # 과거에는 ``parent_content`` (major section 단위, 최대 12K자) 를 사용했으나
        # 이는 chunk가 속한 章節 전체를 합쳐 만든 거대 텍스트라 chunk의 핵심 문구를
        # 평균적인 노이즈에 묻어버리는 부작용이 있다. 실측: q4(48개월) / q5(32,857대) /
        # q10(6개월 이내) 모두 chunk에 정답이 명시되어 있지만 parent에는 TOC·역사적
        # 통계만 들어 있어 LLM이 "정보 없음"으로 abstain. doc로 바꾸면 정상 답변.
        # 빈 chunk fallback으로만 parent_content를 유지한다.
        body = doc or (metadata.get("parent_content", "") if metadata else "")
        sources.append(SearchResult(content=doc, doc_id=doc_id, score=score, metadata=metadata))
        body_key = body[:100]
        if body_key not in seen_parents:
            seen_parents.add(body_key)
            doc_num += 1
            # Phase 14 citation discipline — context 라벨에 doc 파일명을 노출해
            # LLM이 ``[doc:파일명]`` 인용 토큰을 생성할 수 있게 함.
            source_name = ""
            if metadata:
                source_name = str(metadata.get("source_doc_id") or "").strip()
            label = (
                f"[문서 {doc_num} | doc:{source_name}]"
                if source_name
                else f"[문서 {doc_num}]"
            )
            context_parts.append(f"{label}\n{body}")

    t_end = time.monotonic()
    logger.info(
        "검색 완료 %.3fs (임베딩 %.3fs, 벡터검색 %.3fs, 리랭커 %.3fs, BM25/RRF %.3fs, 후처리 %.3fs) — %d건",
        t_end - t0,
        t_embed - t0,
        t_search - t_embed,
        t_rerank - t_bm25,
        t_bm25 - t_search,
        t_end - t_rerank,
        len(sources),
    )
    return SearchOutput(sources=sources, context_parts=context_parts)
