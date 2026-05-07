"""AutoRAG 연동 데이터 내보내기 — 문서와 QA 쌍을 AutoRAG 평가용 parquet으로 변환합니다.

slm-factory의 ``parsed_documents.json``과 QA 쌍 파일(``qa_alpaca.json`` 등)을
AutoRAG가 요구하는 ``corpus.parquet`` + ``qa.parquet`` 형식으로 변환합니다.

AutoRAG 스키마:
- ``corpus.parquet``: doc_id(str), contents(str), metadata(dict)
- ``qa.parquet``: qid(str), query(str), retrieval_gt(list[list[str]]),
  generation_gt(list[str])
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger, run_async

logger = get_logger("exporter.autorag_export")

# 결정적 UUID 생성을 위한 네임스페이스
_NAMESPACE = uuid.UUID("b7e23f8a-4c19-4d7b-9a31-0f8e6d2c5a47")


def _chunk_for_retrieval(content: str, chunk_size: int, overlap: int) -> list[str]:
    """검색용 문서 청킹을 수행합니다.

    문단 경계(``\\n\\n``)를 우선 존중하여 자연스러운 분할을 시도합니다.
    QA 생성용 청킹(``teacher.qa_generator.chunk_document``)과 동일한
    알고리즘을 사용하되, 검색에 최적화된 크기(기본 512자)로 분할합니다.
    """
    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = start + chunk_size

        if end >= len(content):
            chunks.append(content[start:])
            break

        boundary = content.rfind("\n\n", start + chunk_size // 2, end)
        if boundary > start:
            end = boundary

        chunks.append(content[start:end])
        new_start = end - overlap
        start = new_start if new_start > start else end

    return chunks


def _tfidf_char_ngram_cosine(query: str, texts: list[str], n: int = 3) -> list[float]:
    """TF-IDF char n-gram 코사인 유사도. sklearn 없이 직접 구현."""
    import math
    from collections import Counter

    import numpy as np

    def _ngrams(text: str) -> Counter:
        t = text.lower()
        return Counter(t[i : i + n] for i in range(max(0, len(t) - n + 1)))

    all_docs = texts + [query]
    doc_ngrams = [_ngrams(d) for d in all_docs]
    num_docs = len(all_docs)

    # IDF: log(N / df) — 공통 n-gram을 억제하여 희귀 n-gram에 가중치 부여
    df: Counter = Counter()
    for ng in doc_ngrams:
        df.update(ng.keys())
    idf = {term: math.log(num_docs / freq) for term, freq in df.items()}

    def _tfidf_vector(ng: Counter) -> dict[str, float]:
        total = sum(ng.values()) or 1
        return {term: (count / total) * idf.get(term, 0) for term, count in ng.items()}

    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = np.sqrt(sum(v * v for v in a.values()))
        norm_b = np.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    query_vec = _tfidf_vector(doc_ngrams[-1])
    return [_cosine(query_vec, _tfidf_vector(ng)) for ng in doc_ngrams[:-1]]


def _find_best_chunks(
    answer: str,
    chunk_texts: list[str],
    chunk_ids: list[str],
    question: str = "",
) -> list[str]:
    """BM25 + char n-gram hybrid로 답변에 가장 관련 높은 청크를 찾습니다.

    question + answer를 결합하여 매칭합니다.
    질문이 주제를 제한하고, 답변이 구체적 위치를 식별합니다.
    """
    if not chunk_texts:
        return []

    import numpy as np
    from rank_bm25 import BM25Okapi

    query = f"{question} {answer}".strip()
    tokenize = _get_tokenizer()

    tokenized_corpus = [tokenize(t) for t in chunk_texts]
    tokenized_query = tokenize(query)

    # BM25 점수 (TF-IDF + 문서 길이 정규화)
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    # TF-IDF char n-gram 코사인 유사도 (패러프레이즈 대응, IDF로 공통 n-gram 억제)
    ngram_scores = np.array(_tfidf_char_ngram_cosine(query, chunk_texts))

    # 가중 합산: n-gram 60%, BM25 40%
    hybrid_scores = 0.4 * bm25_scores + 0.6 * ngram_scores

    top_indices = hybrid_scores.argsort()[::-1][:3]
    threshold = 0.05
    return [chunk_ids[i] for i in top_indices if hybrid_scores[i] > threshold] or [
        chunk_ids[top_indices[0]]
    ]


def _get_tokenizer():
    """kiwi 형태소 토크나이저를 반환합니다. 미설치 시 whitespace 폴백."""
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()

        def _tokenize(text: str) -> list[str]:
            tokens = kiwi.tokenize(text)
            return [
                t.form for t in tokens if len(t.form) > 1 or not t.tag.startswith("J")
            ]

        return _tokenize
    except ImportError:
        return lambda text: text.lower().split()


class AutoRAGExporter:
    """slm-factory 데이터를 AutoRAG 평가용 parquet 형식으로 변환합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.output_dir = Path(config.paths.output) / config.autorag_export.output_dir
        self.chunk_size = config.autorag_export.chunk_size
        self.overlap = config.autorag_export.overlap_chars

    def export(
        self,
        parsed_docs: list[dict[str, Any]],
        qa_pairs: list[dict[str, Any]],
    ) -> tuple[Path, Path]:
        """AutoRAG 호환 corpus.parquet + qa.parquet 파일을 생성합니다.

        매개변수
        ----------
        parsed_docs:
            ``ParsedDocument`` 딕셔너리 리스트
            (``parsed_documents.json``\\ 에서 로드).
        qa_pairs:
            QA 쌍 딕셔너리 리스트
            (``qa_alpaca.json`` 또는 ``qa_scored.json``\\ 에서 로드).

        반환값
        -------
        tuple[Path, Path]
            ``(corpus.parquet 경로, qa.parquet 경로)``
        """
        import pandas as pd  # datasets 의존성으로 이미 설치됨

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "AutoRAG 데이터 내보내기 시작 — 문서 %d건, QA %d건",
            len(parsed_docs),
            len(qa_pairs),
        )

        corpus_rows, doc_chunk_map, doc_full_text = self._build_corpus(parsed_docs)
        logger.info("문서 청킹 완료 — %d개 청크 생성", len(corpus_rows))

        if self.config.autorag_export.contextual_retrieval.enabled:
            self._apply_contextual_retrieval(corpus_rows, doc_chunk_map, doc_full_text)

        qa_rows = self._build_qa(qa_pairs, doc_chunk_map, corpus_rows)
        logger.info("QA 매핑 완료 — %d개 QA 항목 생성", len(qa_rows))

        corpus_path = self.output_dir / "corpus.parquet"
        qa_path = self.output_dir / "qa.parquet"

        pd.DataFrame(corpus_rows).to_parquet(corpus_path, index=False)
        pd.DataFrame(qa_rows).to_parquet(qa_path, index=False)

        logger.info(
            "AutoRAG 데이터 내보내기 완료 — %s, %s",
            corpus_path,
            qa_path,
        )
        return corpus_path, qa_path

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _build_corpus(
        self,
        parsed_docs: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, list[tuple[str, str]]],
        dict[str, str],
    ]:
        """문서를 청킹하여 corpus 데이터와 문서→청크 매핑을 생성합니다.

        반환값
        -------
        tuple
            ``(corpus_rows, doc_chunk_map, doc_full_text)``

            - ``corpus_rows``: AutoRAG corpus 행 리스트
            - ``doc_chunk_map``: ``{source_doc_id: [(chunk_doc_id, chunk_text), ...]}``
            - ``doc_full_text``: ``{source_doc_id: full_content}`` —
              Contextual Retrieval에서 LLM에 전달할 부모 문서 원문.
        """
        corpus_rows: list[dict[str, Any]] = []
        doc_chunk_map: dict[str, list[tuple[str, str]]] = {}
        doc_full_text: dict[str, str] = {}

        for doc in parsed_docs:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            title = doc.get("title", "")
            metadata = doc.get("metadata", {})

            if not content or not content.strip():
                logger.debug("빈 문서 건너뜀: %s", doc_id)
                continue

            # 테이블 내용 병합
            tables = doc.get("tables", [])
            if tables:
                content = content + "\n\n" + "\n\n".join(tables)

            doc_full_text[doc_id] = content

            context_chunks = self._chunk_with_context(content)
            if context_chunks is not None:
                chunk_entries = self._append_context_chunks(
                    context_chunks,
                    corpus_rows,
                    doc_id,
                    title,
                    metadata,
                )
                doc_chunk_map[doc_id] = chunk_entries
                continue

            chunks = _chunk_for_retrieval(content, self.chunk_size, self.overlap)
            chunk_entries_plain: list[tuple[str, str]] = []
            chunk_start = len(corpus_rows)

            offset = 0
            for i, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid5(_NAMESPACE, f"{doc_id}::{i}"))
                start_idx = offset
                end_idx = offset + len(chunk_text)
                offset = end_idx - self.overlap if i < len(chunks) - 1 else end_idx

                corpus_rows.append(
                    {
                        "doc_id": chunk_id,
                        "contents": chunk_text,
                        "path": metadata.get("path", title),
                        "start_end_idx": [start_idx, end_idx],
                        "metadata": {
                            "last_modified_datetime": datetime.now(),
                            "prev_id": None,
                            "next_id": None,
                            "source_doc_id": doc_id,
                            "source_title": title,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    }
                )
                chunk_entries_plain.append((chunk_id, chunk_text))

            for j in range(len(chunks)):
                idx = chunk_start + j
                if j > 0:
                    corpus_rows[idx]["metadata"]["prev_id"] = corpus_rows[idx - 1][
                        "doc_id"
                    ]
                if j < len(chunks) - 1:
                    corpus_rows[idx]["metadata"]["next_id"] = corpus_rows[idx + 1][
                        "doc_id"
                    ]

            doc_chunk_map[doc_id] = chunk_entries_plain

        return corpus_rows, doc_chunk_map, doc_full_text

    # ------------------------------------------------------------------
    # Contextual Retrieval (Anthropic)
    # ------------------------------------------------------------------

    def _apply_contextual_retrieval(
        self,
        corpus_rows: list[dict[str, Any]],
        doc_chunk_map: dict[str, list[tuple[str, str]]],
        doc_full_text: dict[str, str],
    ) -> None:
        """각 청크의 ``contents`` 앞에 LLM 생성 컨텍스트를 prefix로 부여합니다.

        ``corpus_rows``를 in-place로 수정하며, ``doc_chunk_map``의 chunk_text도
        새 prefix가 반영된 값으로 갱신합니다 (QA 매칭이 prefix를 보지 않도록 하려면
        후속 단계에서 별도 처리하지만, 여기서는 검색 정확도 우선이므로 동기화합니다).
        """
        from ..contextual_retriever import (
            generate_chunk_contexts_async,
            prepend_context,
        )
        from ..teacher import create_teacher

        cfg = self.config.autorag_export.contextual_retrieval
        language = self.config.project.language

        try:
            teacher = create_teacher(self.config.teacher)
        except Exception as exc:
            logger.warning(
                "Teacher 초기화 실패 (%s) — Contextual Retrieval 건너뜀",
                exc,
            )
            return

        # corpus_rows를 source_doc_id별로 묶어 부모 문서 단위로 LLM 호출.
        rows_by_doc: dict[str, list[int]] = {}
        for idx, row in enumerate(corpus_rows):
            src = row.get("metadata", {}).get("source_doc_id", "")
            if src:
                rows_by_doc.setdefault(src, []).append(idx)

        total_chunks = sum(len(v) for v in rows_by_doc.values())
        logger.info(
            "Contextual Retrieval 시작 — 문서 %d개 / 청크 %d개 (concurrency=%d)",
            len(rows_by_doc),
            total_chunks,
            cfg.max_concurrency,
        )

        async def _run() -> None:
            for src_doc_id, indices in rows_by_doc.items():
                document = doc_full_text.get(src_doc_id, "")
                if not document:
                    continue
                chunks = [corpus_rows[i]["contents"] for i in indices]
                contexts = await generate_chunk_contexts_async(
                    teacher, document, chunks, cfg, language=language
                )
                for i, ctx in zip(indices, contexts):
                    new_contents = prepend_context(corpus_rows[i]["contents"], ctx)
                    corpus_rows[i]["contents"] = new_contents
                    corpus_rows[i].setdefault("metadata", {})[
                        "context_prefix"
                    ] = ctx
                # doc_chunk_map의 텍스트도 동기화 (QA 매칭은 prefix까지 보게 됨).
                if src_doc_id in doc_chunk_map:
                    refreshed: list[tuple[str, str]] = []
                    for (cid, _old), i in zip(doc_chunk_map[src_doc_id], indices):
                        refreshed.append((cid, corpus_rows[i]["contents"]))
                    doc_chunk_map[src_doc_id] = refreshed

        run_async(_run())
        logger.info("Contextual Retrieval 완료")

    def _chunk_with_context(
        self,
        content: str,
    ) -> list | None:
        """``section_aware_chunk_with_context``로 컨텍스트 청킹을 시도합니다.

        사용 불가능하면 ``None``을 반환하여 기존 방식으로 폴백합니다.
        """
        try:
            from ..calibration import section_aware_chunk_with_context
        except ImportError:
            return None
        chunks = section_aware_chunk_with_context(content, self.chunk_size)
        if not chunks:
            return None
        return chunks

    def _append_context_chunks(
        self,
        context_chunks: list,
        corpus_rows: list[dict[str, Any]],
        doc_id: str,
        title: str,
        metadata: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """``ChunkWithContext`` 리스트를 corpus_rows에 추가합니다."""
        chunk_entries: list[tuple[str, str]] = []
        chunk_start = len(corpus_rows)
        total = len(context_chunks)

        for i, ctx_chunk in enumerate(context_chunks):
            chunk_id = str(uuid.uuid5(_NAMESPACE, f"{doc_id}::{i}"))
            prefixed = (
                ctx_chunk.context_prefix + "\n" + ctx_chunk.content
                if ctx_chunk.context_prefix
                else ctx_chunk.content
            )
            corpus_rows.append(
                {
                    "doc_id": chunk_id,
                    "contents": prefixed,
                    "path": metadata.get("path", title),
                    "start_end_idx": [0, len(ctx_chunk.content)],
                    "metadata": {
                        "last_modified_datetime": datetime.now(),
                        "prev_id": None,
                        "next_id": None,
                        "source_doc_id": doc_id,
                        "source_title": title,
                        "chunk_index": i,
                        "total_chunks": total,
                        "parent_content": ctx_chunk.parent_content,
                    },
                }
            )
            chunk_entries.append((chunk_id, prefixed))

        for j in range(total):
            idx = chunk_start + j
            if j > 0:
                corpus_rows[idx]["metadata"]["prev_id"] = corpus_rows[idx - 1]["doc_id"]
            if j < total - 1:
                corpus_rows[idx]["metadata"]["next_id"] = corpus_rows[idx + 1]["doc_id"]

        return chunk_entries

    def _build_qa(
        self,
        qa_pairs: list[dict[str, Any]],
        doc_chunk_map: dict[str, list[tuple[str, str]]],
        corpus_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """QA 쌍을 AutoRAG 평가 데이터 형식으로 변환합니다.

        각 QA의 ``source_doc``을 기반으로 관련 코퍼스 청크를 찾아
        ``retrieval_gt`` (list[list[str]])를 생성합니다.
        """
        qa_rows: list[dict[str, Any]] = []
        all_chunk_ids = [row["doc_id"] for row in corpus_rows]

        for qa in qa_pairs:
            question = qa.get("question", qa.get("instruction", ""))
            answer = qa.get("answer", qa.get("output", ""))
            source_doc = qa.get("source_doc", "")

            if not question or not answer:
                continue

            if source_doc and source_doc in doc_chunk_map:
                entries = doc_chunk_map[source_doc]
                chunk_ids = [cid for cid, _ in entries]
                chunk_texts = [ct for _, ct in entries]
                relevant_ids = _find_best_chunks(
                    answer, chunk_texts, chunk_ids, question=question
                )
            else:
                logger.debug(
                    "source_doc '%s' 매핑 없음, QA 건너뜀: %s",
                    source_doc,
                    question[:50],
                )
                continue

            if not relevant_ids:
                logger.warning(
                    "QA에 대한 관련 청크를 찾을 수 없음: %s",
                    question[:50],
                )
                continue

            qa_rows.append(
                {
                    "qid": str(uuid.uuid4()),
                    "query": question,
                    "retrieval_gt": [relevant_ids],
                    "generation_gt": [answer],
                }
            )

        return qa_rows
