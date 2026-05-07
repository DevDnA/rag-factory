"""AutoRAG 내보내기(autorag_export) 모듈의 단위 테스트입니다.

AutoRAGExporter의 corpus/qa 변환, 청킹, 매핑 기능을 검증합니다.
pandas/pyarrow는 conftest.py에서 mock 처리되지 않으므로 실제 parquet 생성을 테스트합니다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_factory.exporter.autorag_export import (
    AutoRAGExporter,
    _chunk_for_retrieval,
    _find_best_chunks,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_autorag_exporter(make_config, **overrides):
    """테스트용 AutoRAGExporter를 생성합니다."""
    export_cfg = {
        "enabled": True,
        "output_dir": "autorag",
        "chunk_size": 512,
        "overlap_chars": 64,
        **overrides,
    }
    config = make_config(autorag_export=export_cfg)
    return AutoRAGExporter(config)


def _sample_parsed_docs():
    """테스트용 파싱된 문서 리스트를 반환합니다."""
    return [
        {
            "doc_id": "doc_001",
            "title": "테스트 문서 1",
            "content": "인공지능은 컴퓨터 과학의 한 분야입니다. "
            "기계 학습과 딥러닝은 인공지능의 하위 분야로, "
            "데이터를 기반으로 패턴을 학습합니다. "
            "자연어 처리는 텍스트 데이터를 분석하는 기술입니다.",
            "tables": [],
            "metadata": {"path": "docs/ai.pdf"},
        },
        {
            "doc_id": "doc_002",
            "title": "테스트 문서 2",
            "content": "한국어 형태소 분석은 자연어 처리의 기초입니다. "
            "Kiwi와 Mecab은 대표적인 한국어 형태소 분석기입니다.",
            "tables": ["| 분석기 | 속도 |\n| Kiwi | 빠름 |"],
            "metadata": {"path": "docs/nlp.pdf"},
        },
    ]


def _sample_qa_pairs():
    """테스트용 QA 쌍 리스트를 반환합니다."""
    return [
        {
            "question": "인공지능이란 무엇인가요?",
            "answer": "인공지능은 컴퓨터 과학의 한 분야입니다.",
            "source_doc": "doc_001",
        },
        {
            "question": "한국어 형태소 분석기에는 무엇이 있나요?",
            "answer": "Kiwi와 Mecab은 대표적인 한국어 형태소 분석기입니다.",
            "source_doc": "doc_002",
        },
        {
            "instruction": "딥러닝에 대해 설명하세요.",
            "output": "기계 학습과 딥러닝은 인공지능의 하위 분야입니다.",
            "source_doc": "doc_001",
        },
    ]


# ---------------------------------------------------------------------------
# _chunk_for_retrieval
# ---------------------------------------------------------------------------


class Test청킹함수:
    """_chunk_for_retrieval 함수의 테스트입니다."""

    def test_짧은_문서_단일_청크(self):
        """chunk_size보다 짧은 문서는 그대로 하나의 청크가 됩니다."""
        result = _chunk_for_retrieval("짧은 텍스트", chunk_size=512, overlap=64)
        assert len(result) == 1
        assert result[0] == "짧은 텍스트"

    def test_긴_문서_다중_청크(self):
        """chunk_size를 초과하는 문서는 여러 청크로 분할됩니다."""
        content = "A" * 1200
        result = _chunk_for_retrieval(content, chunk_size=512, overlap=64)
        assert len(result) > 1

    def test_중첩_영역_존재(self):
        """연속된 청크는 overlap만큼 겹치는 영역이 있어야 합니다."""
        content = "가" * 1200
        overlap = 64
        result = _chunk_for_retrieval(content, chunk_size=512, overlap=overlap)
        assert len(result) >= 2
        # 첫 번째 청크 끝부분과 두 번째 청크 시작부분이 겹침
        tail = result[0][-overlap:]
        head = result[1][:overlap:]
        assert tail == head

    def test_빈_문서(self):
        """빈 문서는 빈 문자열 하나가 반환됩니다."""
        result = _chunk_for_retrieval("", chunk_size=512, overlap=64)
        assert len(result) == 1
        assert result[0] == ""

    def test_문단_경계_우선(self):
        r"""문단 경계(\\n\\n)가 있으면 거기서 분할합니다."""
        # 256자 텍스트 + 문단 경계 + 256자 텍스트 = ~514자
        content = "가" * 256 + "\n\n" + "나" * 256
        result = _chunk_for_retrieval(content, chunk_size=512, overlap=32)
        # 문단 경계에서 분할되어 첫 청크가 "가"*256 이어야 함
        assert result[0].endswith("가" * 10)


# ---------------------------------------------------------------------------
# _find_best_chunks
# ---------------------------------------------------------------------------


class Test청크매칭함수:
    """_find_best_chunks 함수의 테스트입니다."""

    def test_정확히_일치하는_청크_반환(self):
        """답변 텍스트를 포함하는 청크가 최우선 반환됩니다."""
        texts = ["전혀 관련없는 텍스트", "인공지능은 컴퓨터 과학의 한 분야입니다."]
        ids = ["chunk_a", "chunk_b"]
        result = _find_best_chunks(
            "인공지능은 컴퓨터 과학의 한 분야입니다.", texts, ids
        )
        assert "chunk_b" in result

    def test_빈_청크_리스트(self):
        """청크 리스트가 비어 있으면 빈 리스트를 반환합니다."""
        result = _find_best_chunks("무언가", [], [])
        assert result == []

    def test_최대_3개_반환(self):
        """관련 청크가 많아도 최대 3개까지만 반환합니다."""
        # 모든 청크가 동일 텍스트 — 모두 동점
        texts = ["공통 단어 포함 텍스트"] * 5
        ids = [f"chunk_{i}" for i in range(5)]
        result = _find_best_chunks("공통 단어 포함", texts, ids)
        assert len(result) <= 3

    def test_겹침_없으면_첫번째_반환(self):
        """단어 겹침이 전혀 없어도 최소 하나는 반환합니다."""
        texts = ["완전히 다른 내용"]
        ids = ["chunk_only"]
        result = _find_best_chunks("전혀 관련 없는 질의", texts, ids)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# AutoRAGExporter._build_corpus
# ---------------------------------------------------------------------------


class TestBuildCorpus:
    """AutoRAGExporter._build_corpus 메서드의 테스트입니다."""

    def test_정상_문서_청킹(self, make_config):
        """정상 문서가 올바른 corpus 행으로 변환되는지 확인합니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=200, overlap_chars=32)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)

        assert len(corpus_rows) > 0
        assert "doc_001" in doc_chunk_map
        assert "doc_002" in doc_chunk_map

        # AutoRAG 필수 필드 확인
        row = corpus_rows[0]
        assert "doc_id" in row
        assert "contents" in row
        assert "metadata" in row
        assert "path" in row
        assert "start_end_idx" in row

    def test_빈_문서_건너뜀(self, make_config):
        """빈 content인 문서는 건너뛰어야 합니다."""
        exporter = _make_autorag_exporter(make_config)
        docs = [{"doc_id": "empty", "content": "", "title": "빈 문서", "metadata": {}}]
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)

        assert len(corpus_rows) == 0
        assert "empty" not in doc_chunk_map

    def test_테이블_내용_병합(self, make_config):
        """tables 필드가 있으면 content에 병합됩니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = [
            {
                "doc_id": "with_table",
                "content": "본문 내용",
                "title": "테이블 문서",
                "tables": ["| 열1 | 열2 |"],
                "metadata": {},
            }
        ]
        corpus_rows, _, _ = exporter._build_corpus(docs)

        # 테이블 내용이 청크에 포함되어야 함
        all_content = " ".join(r["contents"] for r in corpus_rows)
        assert "열1" in all_content

    def test_prev_next_id_연결(self, make_config):
        """같은 문서의 청크들이 prev_id/next_id로 연결됩니다."""
        # 작은 chunk_size로 여러 청크 강제 생성
        exporter = _make_autorag_exporter(make_config, chunk_size=100, overlap_chars=16)
        docs = [
            {
                "doc_id": "long_doc",
                "content": "가" * 500,
                "title": "긴 문서",
                "metadata": {},
            }
        ]
        corpus_rows, _, _ = exporter._build_corpus(docs)

        assert len(corpus_rows) >= 2

        # 첫 청크: prev_id=None, next_id 존재
        assert corpus_rows[0]["metadata"]["prev_id"] is None
        assert corpus_rows[0]["metadata"]["next_id"] is not None

        # 마지막 청크: next_id=None, prev_id 존재
        assert corpus_rows[-1]["metadata"]["next_id"] is None
        assert corpus_rows[-1]["metadata"]["prev_id"] is not None

    def test_결정적_UUID(self, make_config):
        """같은 문서를 두 번 처리하면 같은 doc_id가 생성됩니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = [
            {
                "doc_id": "stable",
                "content": "재현 가능한 ID 테스트",
                "title": "안정 문서",
                "metadata": {},
            }
        ]

        rows_1, _, _ = exporter._build_corpus(docs)
        rows_2, _, _ = exporter._build_corpus(docs)

        assert rows_1[0]["doc_id"] == rows_2[0]["doc_id"]


# ---------------------------------------------------------------------------
# AutoRAGExporter._build_qa
# ---------------------------------------------------------------------------


class TestBuildQA:
    """AutoRAGExporter._build_qa 메서드의 테스트입니다."""

    def test_정상_QA_변환(self, make_config):
        """정상 QA 쌍이 올바른 형식으로 변환되는지 확인합니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)
        qa_rows = exporter._build_qa(_sample_qa_pairs(), doc_chunk_map, corpus_rows)

        assert len(qa_rows) == 3

        row = qa_rows[0]
        assert "qid" in row
        assert "query" in row
        assert "retrieval_gt" in row
        assert "generation_gt" in row

    def test_retrieval_gt_이중리스트(self, make_config):
        """retrieval_gt가 list[list[str]] 형식인지 확인합니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)
        qa_rows = exporter._build_qa(_sample_qa_pairs(), doc_chunk_map, corpus_rows)

        for row in qa_rows:
            assert isinstance(row["retrieval_gt"], list)
            assert len(row["retrieval_gt"]) > 0
            assert isinstance(row["retrieval_gt"][0], list)
            for chunk_id in row["retrieval_gt"][0]:
                assert isinstance(chunk_id, str)

    def test_generation_gt_리스트(self, make_config):
        """generation_gt가 list[str] 형식인지 확인합니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)
        qa_rows = exporter._build_qa(_sample_qa_pairs(), doc_chunk_map, corpus_rows)

        for row in qa_rows:
            assert isinstance(row["generation_gt"], list)
            assert len(row["generation_gt"]) > 0
            assert isinstance(row["generation_gt"][0], str)

    def test_alpaca_형식_QA_지원(self, make_config):
        """instruction/output 형식의 QA도 변환됩니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)

        alpaca_qa = [
            {
                "instruction": "딥러닝이란?",
                "output": "기계 학습과 딥러닝은 인공지능의 하위 분야입니다.",
                "source_doc": "doc_001",
            }
        ]
        qa_rows = exporter._build_qa(alpaca_qa, doc_chunk_map, corpus_rows)

        assert len(qa_rows) == 1
        assert qa_rows[0]["query"] == "딥러닝이란?"

    def test_빈_질문_답변_건너뜀(self, make_config):
        """질문이나 답변이 비어 있는 QA는 건너뜁니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)

        bad_qa = [
            {"question": "", "answer": "답변", "source_doc": "doc_001"},
            {"question": "질문", "answer": "", "source_doc": "doc_001"},
        ]
        qa_rows = exporter._build_qa(bad_qa, doc_chunk_map, corpus_rows)

        assert len(qa_rows) == 0

    def test_source_doc_미매칭시_건너뜀(self, make_config):
        """source_doc가 doc_chunk_map에 없으면 해당 QA를 건너뜁니다."""
        exporter = _make_autorag_exporter(make_config, chunk_size=2000)
        docs = _sample_parsed_docs()
        corpus_rows, doc_chunk_map, _doc_full_text = exporter._build_corpus(docs)

        orphan_qa = [
            {
                "question": "연결 안 되는 질문",
                "answer": "연결 안 되는 답변",
                "source_doc": "nonexistent_doc",
            }
        ]
        qa_rows = exporter._build_qa(orphan_qa, doc_chunk_map, corpus_rows)

        assert len(qa_rows) == 0


# ---------------------------------------------------------------------------
# AutoRAGExporter.export (통합)
# ---------------------------------------------------------------------------


class TestExport통합:
    """AutoRAGExporter.export 메서드의 통합 테스트입니다."""

    def test_parquet_파일_생성(self, make_config, tmp_path):
        """export()가 corpus.parquet과 qa.parquet 파일을 생성하는지 확인합니다."""
        config = make_config(
            autorag_export={
                "enabled": True,
                "output_dir": "autorag",
                "chunk_size": 2000,
            },
            paths={"output": str(tmp_path), "documents": str(tmp_path / "docs")},
        )
        exporter = AutoRAGExporter(config)

        corpus_path, qa_path = exporter.export(
            _sample_parsed_docs(), _sample_qa_pairs()
        )

        assert corpus_path.exists()
        assert qa_path.exists()
        assert corpus_path.name == "corpus.parquet"
        assert qa_path.name == "qa.parquet"

    def test_parquet_읽기_가능(self, make_config, tmp_path):
        """생성된 parquet 파일이 pandas로 읽을 수 있는지 확인합니다."""
        import pandas as pd

        config = make_config(
            autorag_export={
                "enabled": True,
                "output_dir": "autorag",
                "chunk_size": 2000,
            },
            paths={"output": str(tmp_path), "documents": str(tmp_path / "docs")},
        )
        exporter = AutoRAGExporter(config)

        corpus_path, qa_path = exporter.export(
            _sample_parsed_docs(), _sample_qa_pairs()
        )

        corpus_df = pd.read_parquet(corpus_path)
        qa_df = pd.read_parquet(qa_path)

        # corpus 스키마 확인
        assert "doc_id" in corpus_df.columns
        assert "contents" in corpus_df.columns
        assert len(corpus_df) > 0

        # qa 스키마 확인
        assert "qid" in qa_df.columns
        assert "query" in qa_df.columns
        assert "retrieval_gt" in qa_df.columns
        assert "generation_gt" in qa_df.columns
        assert len(qa_df) == 3

    def test_빈_문서_빈_QA(self, make_config, tmp_path):
        """문서와 QA가 모두 비어 있어도 에러 없이 실행됩니다."""
        config = make_config(
            autorag_export={"enabled": True, "output_dir": "autorag"},
            paths={"output": str(tmp_path), "documents": str(tmp_path / "docs")},
        )
        exporter = AutoRAGExporter(config)

        corpus_path, qa_path = exporter.export([], [])

        assert corpus_path.exists()
        assert qa_path.exists()
