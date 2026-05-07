"""청킹·재생성 기능의 통합 테스트입니다.

개별 모듈 단위가 아닌 모듈 간 데이터 흐름을 검증합니다.
LLM 호출은 mock하지만 데이터 구조는 실제 객체를 사용합니다.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slm_factory.config import SLMConfig
from slm_factory.models import ParsedDocument, QAPair
from slm_factory.pipeline import Pipeline
from slm_factory.teacher.qa_generator import QAGenerator, chunk_document


def _make_pipeline(make_config, tmp_path, **overrides) -> Pipeline:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = {"paths": {"output": str(output_dir), "documents": str(docs_dir)}}
    defaults.update(overrides)
    config = make_config(**defaults)
    return Pipeline(config)


def _long_content(paragraphs: int = 10, chars_per_paragraph: int = 2000) -> str:
    return "\n\n".join(
        f"문단 {i + 1}. " + "한국어 테스트 문장입니다. " * (chars_per_paragraph // 15)
        for i in range(paragraphs)
    )


# ---------------------------------------------------------------------------
# 청킹 → QA 생성 체인
# ---------------------------------------------------------------------------


class TestChunkingToQAChain:
    """청킹이 활성화되면 QA 생성이 청크 단위로 동작하는지 검증합니다."""

    def test_청킹_활성화시_여러_청크에서_QA_생성(self, make_config, mocker):
        """긴 문서에서 청킹이 활성화되면 chunk별로 QA 생성 호출이 발생하는지 확인합니다."""
        config = make_config(
            chunking={"enabled": True, "chunk_size": 3000, "overlap_chars": 200},
            teacher={"max_context_chars": 12000},
            questions={"categories": {"test": ["테스트 질문?"]}},
        )

        long_content = _long_content(paragraphs=8, chars_per_paragraph=1500)
        doc = ParsedDocument(
            doc_id="long.pdf",
            title="긴 문서",
            content=long_content,
            tables=[],
            metadata={},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.questions_config = config.questions
        generator.teacher_config = config.teacher
        generator.chunking_config = config.chunking
        generator.max_context = config.teacher.max_context_chars

        chunks = generator._get_doc_chunks(doc)

        assert len(chunks) > 1
        for chunk_content, chunk_info in chunks:
            assert len(chunk_content) > 0
        assert chunks[-1][1] is not None
        assert "Part" in chunks[0][1]

    def test_청킹_비활성화시_단일_청크_반환(self, make_config):
        """청킹 비활성화 시 원본 문서가 단일 청크로 반환되는지 확인합니다."""
        config = make_config(
            chunking={"enabled": False},
            teacher={"max_context_chars": 12000},
        )

        doc = ParsedDocument(
            doc_id="test.pdf",
            title="문서",
            content="짧은 내용",
            tables=[],
            metadata={},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.chunking_config = config.chunking
        generator.max_context = config.teacher.max_context_chars

        chunks = generator._get_doc_chunks(doc)
        assert len(chunks) == 1
        assert chunks[0] == ("짧은 내용", None)

    def test_청크_정보가_프롬프트에_포함(self, make_config):
        """chunk_info가 프롬프트의 문서 제목에 반영되는지 확인합니다."""
        config = make_config(
            teacher={"max_context_chars": 50000},
            questions={"categories": {"test": ["질문?"]}},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.questions_config = config.questions
        generator.teacher_config = config.teacher
        generator.max_context = config.teacher.max_context_chars

        prompt = generator.build_prompt(
            doc_title="테스트 문서",
            content="내용",
            question="질문?",
            chunk_info="Part 2/5",
        )

        assert "테스트 문서 (Part 2/5)" in prompt

    def test_파이프라인_step_generate에서_청킹_동작(
        self, make_config, tmp_path, mocker
    ):
        """Pipeline.step_generate가 청킹 활성화 시 generate_all_async를 올바르게 호출하는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config,
            tmp_path,
            chunking={"enabled": True, "chunk_size": 5000, "overlap_chars": 300},
            questions={"categories": {"test": ["질문?"]}},
        )

        doc = ParsedDocument(
            doc_id="test.pdf",
            title="문서",
            content=_long_content(5, 2000),
            tables=[],
            metadata={},
        )

        mock_qa = QAPair(
            question="질문?",
            answer="답변",
            instruction="질문?",
            source_doc="test.pdf",
            category="",
        )

        mock_generator_cls = mocker.patch(
            "slm_factory.teacher.qa_generator.QAGenerator"
        )
        mock_generator = mock_generator_cls.return_value
        mock_generator.save_alpaca = MagicMock()

        mock_coro = AsyncMock(return_value=[mock_qa])
        mock_generator.generate_all_async = mock_coro
        mocker.patch(
            "slm_factory.pipeline.run_async",
            side_effect=lambda c: asyncio.run(c) if asyncio.iscoroutine(c) else c,
        )
        mocker.patch("slm_factory.pipeline.run_async", return_value=[mock_qa])

        pairs = pipeline.step_generate([doc])

        assert len(pairs) == 1
        mock_generator.generate_all_async.assert_called_once()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 전체 체인 통합 + 엣지 케이스
# ---------------------------------------------------------------------------


class TestFullChainIntegration:
    """전체 파이프라인 체인(파싱→청킹→QA→검증→스코어→재생성)을 검증합니다."""

    def test_run_메서드에서_전체_체인_실행(self, make_config, tmp_path, mocker):
        """Pipeline.run()이 청킹을 포함해 전체 체인을 실행하는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config,
            tmp_path,
            chunking={"enabled": True, "chunk_size": 5000, "overlap_chars": 300},
            scoring={"enabled": True, "threshold": 3.0},
        )

        mock_docs = [MagicMock()]
        mock_pairs = [MagicMock()]
        mock_training_path = tmp_path / "training.jsonl"
        mock_adapter = tmp_path / "adapter"
        mock_export = tmp_path / "export"

        mocker.patch.object(pipeline, "step_parse", return_value=mock_docs)
        mocker.patch.object(pipeline, "step_generate", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_validate", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_score", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_augment", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_analyze")
        mocker.patch.object(pipeline, "step_convert", return_value=mock_training_path)
        mocker.patch.object(pipeline, "step_train", return_value=mock_adapter)
        mocker.patch.object(pipeline, "step_export", return_value=mock_export)
        mocker.patch.object(pipeline, "step_eval", return_value=[])
        mocker.patch.object(
            pipeline, "step_corpus_export", return_value=(Path(), Path())
        )
        mocker.patch.object(pipeline, "step_rag_index", return_value=Path())

        pipeline.run()

        pipeline.step_generate.assert_called_once_with(mock_docs)
        pipeline.step_score.assert_called_once_with(mock_pairs, docs=mock_docs)


class TestEdgeCases:
    """새 기능의 엣지 케이스를 검증합니다."""

    def test_빈_문서_청킹시_단일_빈_청크(self):
        """빈 문서를 청킹하면 빈 문자열 하나가 반환되는지 확인합니다."""
        result = chunk_document("", chunk_size=5000, overlap=500)
        assert result == [""]

    def test_청크사이즈보다_짧은_문서(self):
        """chunk_size보다 짧은 문서는 분할 없이 반환되는지 확인합니다."""
        short = "짧은 문서입니다."
        result = chunk_document(short, chunk_size=10000, overlap=500)
        assert result == [short]

    def test_문단_경계_없는_긴_문서_강제_분할(self):
        """\\n\\n 없는 긴 문자열도 chunk_size에서 강제 분할되는지 확인합니다."""
        no_paragraphs = "가" * 25000
        result = chunk_document(no_paragraphs, chunk_size=10000, overlap=500)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= 10000

    def test_단일_문단_긴_문서(self):
        """단일 문단이지만 chunk_size를 초과하는 문서가 분할되는지 확인합니다."""
        single_paragraph = "이것은 매우 긴 단일 문단입니다. " * 1000
        result = chunk_document(single_paragraph, chunk_size=5000, overlap=300)
        assert len(result) > 1
