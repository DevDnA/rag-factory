#!/usr/bin/env python
"""파이프라인-aware retrieval probe.

실제 ``search_documents``(vector + RRF hybrid + cross-encoder rerank)를
allganize 50-query 벤치마크에 돌려 **문서 단위** Hit@1 / Hit@K / MRR을 측정한다.

왜 별도 probe가 필요한가:
    ``rf tool eval-retrieval`` (evaluator.RetrievalEvaluator) 는 (a) ``search_documents``
    를 거치지 않고 Qdrant에 raw vector query만 날리며, (b) 현재 qa.parquet이 0행이다.
    따라서 search.py의 rerank/RRF 순서 변경을 측정할 수 없다. 이 probe는 서버
    lifespan(server.py:252-338)과 동일하게 embedding/reranker/bm25를 조립한 뒤
    production 검색 함수 그 자체를 호출하므로 변경 효과를 그대로 잰다.

Ground truth:
    각 query의 ``target_file_name`` (정답 문서 1개). 검색된 chunk의 ``source_doc_id``
    를 stem 정규화하여 매칭한다.

사용:
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \\
    uv run python scripts/eval_retrieval_pipeline.py \\
        --config allganize-eval-project/project.yaml \\
        --queries scripts/eval_allganize_queries_full.json \\
        --top-k 5 --out /tmp/retrieval_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag_factory.config import load_config
from rag_factory.rag.search import search_documents


def normalize_stem(name: str) -> str:
    """파일명을 확장자·공백·구분자 제거 후 소문자 stem으로 정규화한다."""
    s = Path(str(name)).stem
    s = re.sub(r"[\s_\-\[\]()+]+", "", s)
    return s.lower()


def make_korean_tokenize():
    """server.py ``_korean_tokenize`` 와 동일한 kiwi 형태소 토크나이저."""
    kiwi = None

    def tok(text: str) -> list[str]:
        nonlocal kiwi
        if kiwi is None:
            try:
                from kiwipiepy import Kiwi

                kiwi = Kiwi()
            except ImportError:
                kiwi = False
        if kiwi is False:
            return text.lower().split()
        tokens = kiwi.tokenize(text)
        return [t.form for t in tokens if len(t.form) > 1 or not t.tag.startswith("J")]

    return tok


def is_control(target_answer: object) -> bool:
    """OOD / ambiguous 컨트롤 항목(retrieval 채점 제외)인지 판정한다."""
    if not target_answer:
        return True
    ta = str(target_answer).strip()
    return ta.startswith("[OUT OF CORPUS") or ta.startswith("[AMBIGUOUS")


def build_search_state(config) -> dict:
    """server.py lifespan(252-338)과 동일한 검색 상태를 조립한다."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    db_path = Path(config.paths.output) / config.rag.vector_db_path
    embedding_model = SentenceTransformer(config.rag.embedding_model)
    client = QdrantClient(path=str(db_path))
    collection = config.rag.collection_name

    reranker = None
    if config.rag.reranker_enabled:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(config.rag.reranker_model, max_length=512)

    tok = make_korean_tokenize()
    bm25_index = bm25_docs = bm25_ids = bm25_metadatas = None
    if config.rag.hybrid_search:
        from rank_bm25 import BM25Okapi

        all_points = []
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset
        bm25_docs = [p.payload.get("document", "") for p in all_points]
        bm25_ids = [p.payload.get("doc_id", str(p.id)) for p in all_points]
        bm25_metadatas = [
            {k: v for k, v in p.payload.items() if k not in ("document", "doc_id")}
            for p in all_points
        ]
        bm25_index = BM25Okapi([tok(d) for d in bm25_docs])

    return {
        "qdrant_client": client,
        "collection_name": collection,
        "embedding_model": embedding_model,
        "reranker": reranker,
        "hybrid_search": config.rag.hybrid_search,
        "bm25_index": bm25_index,
        "bm25_docs": bm25_docs,
        "bm25_ids": bm25_ids,
        "bm25_metadatas": bm25_metadatas,
        "tokenize_fn": tok,
        "min_score": config.rag.min_score,
    }


def retrieved_docs_in_order(output) -> list[tuple[str, str]]:
    """source 목록을 문서 단위로 dedupe하되 최초 등장 순서를 유지한다.

    Returns (normalized_stem, raw_name) 튜플 리스트 (rank 1-indexed).
    """
    seen: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for s in output.sources:
        name = (str((s.metadata or {}).get("source_doc_id") or "").strip()) or s.doc_id
        key = normalize_stem(name)
        if key and key not in seen_keys:
            seen_keys.add(key)
            seen.append((key, name))
    return seen


def rank_of_target(docs: list[tuple[str, str]], target_norm: str) -> int:
    """정답 문서의 1-indexed rank (없으면 0). 정확 매칭 후 substring fallback."""
    norms = [k for k, _ in docs]
    if target_norm in norms:
        return norms.index(target_norm) + 1
    for i, (k, _) in enumerate(docs, 1):
        if target_norm and (target_norm in k or k in target_norm):
            return i
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="allganize-eval-project/project.yaml")
    ap.add_argument("--queries", default="scripts/eval_allganize_queries_full.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="/tmp/retrieval_probe.json")
    args = ap.parse_args()

    config = load_config(args.config)
    state = build_search_state(config)
    queries = json.load(open(args.queries, encoding="utf-8"))

    search_kwargs = {
        k: state[k]
        for k in (
            "qdrant_client",
            "collection_name",
            "embedding_model",
            "reranker",
            "hybrid_search",
            "bm25_index",
            "bm25_docs",
            "bm25_ids",
            "bm25_metadatas",
            "tokenize_fn",
            "min_score",
        )
    }

    rows = []
    n = hit1 = hitk = 0
    rr_sum = 0.0
    for q in queries:
        if is_control(q.get("target_answer")) or not q.get("target_file_name"):
            continue
        target = normalize_stem(q["target_file_name"])
        out = search_documents(q["query"], top_k=args.top_k, **search_kwargs)
        docs = retrieved_docs_in_order(out)
        rank = rank_of_target(docs, target)

        n += 1
        if rank == 1:
            hit1 += 1
        if rank >= 1:
            hitk += 1
            rr_sum += 1.0 / rank
        rows.append(
            {
                "id": q.get("id"),
                "intent": q.get("intent"),
                "query": q["query"],
                "target": q["target_file_name"],
                "rank": rank,
                "retrieved": [raw for _, raw in docs],
            }
        )
        print(f"[{q.get('id')}] rank={rank} target={q['target_file_name'][:48]}")

    metrics = {
        "n": n,
        "hit@1": round(hit1 / n, 4) if n else 0.0,
        f"hit@{args.top_k}": round(hitk / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
    }
    print("\n=== METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
