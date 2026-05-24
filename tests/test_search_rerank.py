"""search_documents의 union-rerank 순서 검증.

핵심 속성: hybrid 검색에서 BM25만 찾아낸(=벡터가 놓친) 문서도 cross-encoder
리랭커를 거쳐 최종 top_k에 오를 수 있어야 한다. 과거 구현은 벡터 결과만 먼저
리랭킹→top_k로 자른 뒤 BM25와 RRF 융합했으므로, BM25-only 문서는 리랭커 판정을
영영 받지 못했다. 이 테스트는 그 회귀를 막는다.
"""

from __future__ import annotations

import numpy as np

from rag_factory.rag.search import search_documents


class _FakePoint:
    def __init__(self, pid, doc, doc_id, source):
        self.id = pid
        self.score = 0.9  # cosine sim (distance = 1 - score)
        self.payload = {"document": doc, "doc_id": doc_id, "source_doc_id": source}


class _FakeQdrant:
    """벡터 검색 결과만 반환 — BM25-only 문서(b1)는 포함하지 않는다."""

    def __init__(self, points):
        self._points = points

    def query_points(self, *, collection_name, query, limit, with_payload):
        class _R:
            pass

        r = _R()
        r.points = self._points[:limit]
        return r


class _FakeReranker:
    """prefer_text를 가장 높게 매기는 cross-encoder. 호출 인자를 기록한다."""

    def __init__(self, prefer_text):
        self.prefer_text = prefer_text
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return np.array(
            [10.0 if doc == self.prefer_text else 0.1 * i for i, (_q, doc) in enumerate(pairs)]
        )


class _FakeBM25:
    """corpus 순서에 정렬된 점수 배열을 반환한다."""

    def __init__(self, scores):
        self._scores = np.array(scores, dtype=float)

    def get_scores(self, tokens):
        return self._scores


def test_bm25_only_doc_reaches_reranker_and_can_win():
    # 벡터는 v1,v2,v3만 반환 (b1을 놓침). top_k=1 → initial_k=3.
    vec_points = [
        _FakePoint(1, "vector doc one", "v1", "v1.pdf"),
        _FakePoint(2, "vector doc two", "v2", "v2.pdf"),
        _FakePoint(3, "vector doc three", "v3", "v3.pdf"),
    ]
    qdrant = _FakeQdrant(vec_points)

    # BM25 corpus: 벡터 문서 + BM25-only 문서 b1. b1만 양(+) 점수.
    bm25_docs = ["vector doc one", "vector doc two", "vector doc three", "bm25 only doc"]
    bm25_ids = ["v1", "v2", "v3", "b1"]
    bm25_metadatas = [
        {"source_doc_id": "v1.pdf"},
        {"source_doc_id": "v2.pdf"},
        {"source_doc_id": "v3.pdf"},
        {"source_doc_id": "b1.pdf"},
    ]
    bm25_index = _FakeBM25([0.0, 0.0, 0.0, 5.0])

    reranker = _FakeReranker(prefer_text="bm25 only doc")

    class _FakeEmbed:
        def encode(self, *a, **k):
            return np.array([0.0, 0.0, 0.0])

    out = search_documents(
        "어떤 질의",
        top_k=1,
        qdrant_client=qdrant,
        collection_name="corpus",
        embedding_model=_FakeEmbed(),
        min_score=0.0,
        reranker=reranker,
        hybrid_search=True,
        bm25_index=bm25_index,
        bm25_docs=bm25_docs,
        bm25_ids=bm25_ids,
        bm25_metadatas=bm25_metadatas,
        tokenize_fn=lambda t: t.split(),
    )

    # 1) BM25-only 문서가 리랭커 입력(pairs)에 포함되어야 한다 (union이 리랭킹됨).
    assert reranker.calls, "리랭커가 호출되지 않았다"
    reranked_docs = {doc for (_q, doc) in reranker.calls[0]}
    assert "bm25 only doc" in reranked_docs, (
        "BM25-only 문서가 리랭커를 거치지 못했다 — union이 아니라 벡터 가지만 리랭킹됨"
    )

    # 2) 리랭커가 최고점을 준 BM25-only 문서가 최종 top_k=1을 차지해야 한다.
    assert len(out.sources) == 1
    assert out.sources[0].doc_id == "b1", (
        f"리랭커가 선호한 BM25-only 문서가 최종 결과에 없음: {out.sources[0].doc_id}"
    )
