"""의미론적 근거 검증기(validator/similarity.py) 모듈의 통합 테스트입니다.

GroundednessChecker의 청크 분할, 유사도 점수 계산, 배치 검증 기능을 검증합니다.
sentence_transformers는 conftest.py에서 sys.modules에 등록된 mock을 사용합니다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from slm_factory.config import GroundednessConfig
from slm_factory.validator.similarity import GroundednessChecker


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_checker(threshold=0.5) -> GroundednessChecker:
    """테스트용 GroundednessChecker를 생성합니다."""
    config = GroundednessConfig(enabled=True, model="test-model", threshold=threshold)
    return GroundednessChecker(config)


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    """GroundednessChecker._chunk_text 메서드의 테스트입니다."""

    def test_짧은_텍스트_1개_청크(self):
        """chunk_size보다 짧은 텍스트는 1개 청크만 반환하는지 확인합니다."""
        checker = _make_checker()
        text = "짧은 텍스트입니다."

        chunks = checker._chunk_text(text, chunk_size=512, overlap=64)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_긴_텍스트_여러_청크(self):
        """chunk_size를 초과하는 텍스트가 여러 청크로 분할되는지 확인합니다."""
        checker = _make_checker()
        # 충분히 긴 텍스트 생성 (chunk_size=100 기준)
        text = "가" * 300

        chunks = checker._chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        # 각 청크가 chunk_size를 크게 초과하지 않는지 확인
        for chunk in chunks:
            assert len(chunk) <= 120  # overlap 여유 포함

    def test_빈_텍스트_폴백(self):
        """빈 텍스트를 입력하면 빈 리스트를 반환하는지 확인합니다."""
        checker = _make_checker()
        text = ""

        chunks = checker._chunk_text(text, chunk_size=512, overlap=64)

        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


class TestScore:
    """GroundednessChecker.score 메서드의 테스트입니다."""

    def test_유사도_점수_반환(self, mocker):
        """모델 encode와 cos_sim을 mock하여 float 점수를 반환하는지 확인합니다."""
        import sys

        checker = _make_checker()

        # _model.encode mock
        mock_answer_emb = MagicMock()
        mock_chunk_embs = MagicMock()
        checker._model.encode = MagicMock(side_effect=[mock_answer_emb, mock_chunk_embs])

        # cos_sim mock — score() 내부에서 `from sentence_transformers import util as st_util`을
        # 사용하므로 sys.modules의 sentence_transformers mock에 직접 설정합니다.
        mock_sim = MagicMock()
        mock_sim.max.return_value = 0.85
        st_mock = sys.modules["sentence_transformers"]
        st_mock.util.cos_sim.return_value = mock_sim

        result = checker.score("테스트 답변", "소스 텍스트 내용입니다.")

        assert isinstance(result, float)
        assert result == 0.85


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


class TestCheck:
    """GroundednessChecker.check 메서드의 테스트입니다."""

    def test_임계값_이상_True(self, mocker, make_qa_pair):
        """점수가 임계값 이상이면 (True, score)를 반환하는지 확인합니다."""
        checker = _make_checker(threshold=0.5)
        mocker.patch.object(checker, "score", return_value=0.8)

        pair = make_qa_pair(answer="충분히 근거가 있는 답변입니다.")
        is_grounded, score = checker.check(pair, "소스 텍스트")

        assert is_grounded is True
        assert score == 0.8

    def test_임계값_미만_False(self, mocker, make_qa_pair):
        """점수가 임계값 미만이면 (False, score)를 반환하는지 확인합니다."""
        checker = _make_checker(threshold=0.5)
        mocker.patch.object(checker, "score", return_value=0.2)

        pair = make_qa_pair(answer="근거가 없는 답변입니다.")
        is_grounded, score = checker.check(pair, "소스 텍스트")

        assert is_grounded is False
        assert score == 0.2


# ---------------------------------------------------------------------------
# check_batch
# ---------------------------------------------------------------------------


class TestCheckBatch:
    """GroundednessChecker.check_batch 메서드의 테스트입니다."""

    def test_혼합_결과_분리(self, mocker, make_qa_pair):
        """근거있는 쌍과 근거없는 쌍이 올바르게 분리되는지 확인합니다."""
        import sys

        checker = _make_checker(threshold=0.5)

        pair_high = make_qa_pair(question="높은 유사도 질문", source_doc="doc1.pdf")
        pair_low = make_qa_pair(question="낮은 유사도 질문", source_doc="doc2.pdf")

        # _model.encode를 mock — 청크 임베딩 + 답변 임베딩 번갈아 호출됨
        mock_chunk_emb1 = MagicMock()  # doc1 청크
        mock_chunk_emb2 = MagicMock()  # doc2 청크
        mock_answer_emb1 = MagicMock()  # pair_high 답변
        mock_answer_emb2 = MagicMock()  # pair_low 답변
        checker._model.encode = MagicMock(side_effect=[
            mock_chunk_emb1, mock_answer_emb1,
            mock_chunk_emb2, mock_answer_emb2,
        ])

        # cos_sim mock — 첫 호출 0.8(굼거있음), 두 번째 0.2(굼거없음)
        mock_sim_high = MagicMock()
        mock_sim_high.max.return_value = 0.8
        mock_sim_low = MagicMock()
        mock_sim_low.max.return_value = 0.2
        st_mock = sys.modules["sentence_transformers"]
        st_mock.util.cos_sim.side_effect = [mock_sim_high, mock_sim_low]

        source_texts = {
            "doc1.pdf": "높은 유사도 소스 텍스트",
            "doc2.pdf": "낮은 유사도 소스 텍스트",
        }

        grounded, ungrounded = checker.check_batch([pair_high, pair_low], source_texts)

        assert len(grounded) == 1
        assert len(ungrounded) == 1
        assert grounded[0].question == "높은 유사도 질문"

    def test_source_texts에_없는_doc은_grounded에_추가(self, mocker, make_qa_pair):
        """source_texts에 해당 문서가 없으면 grounded에 추가하고 건너뛰는지 확인합니다."""
        import sys

        checker = _make_checker(threshold=0.5)

        pair_known = make_qa_pair(question="알려진 문서 질문", source_doc="known.pdf")
        pair_unknown = make_qa_pair(question="미지 문서 질문", source_doc="unknown.pdf")

        # known.pdf만 위한 encode mock: 청크 임베딩 + 답변 임베딩
        mock_chunk_emb = MagicMock()
        mock_answer_emb = MagicMock()
        checker._model.encode = MagicMock(side_effect=[mock_chunk_emb, mock_answer_emb])

        # cos_sim mock — known.pdf는 굼거있음
        mock_sim = MagicMock()
        mock_sim.max.return_value = 0.7
        st_mock = sys.modules["sentence_transformers"]
        st_mock.util.cos_sim.side_effect = None
        st_mock.util.cos_sim.return_value = mock_sim

        source_texts = {
            "known.pdf": "알려진 문서의 내용입니다.",
            # "unknown.pdf"는 의도적으로 누락
        }

        grounded, ungrounded = checker.check_batch(
            [pair_known, pair_unknown], source_texts
        )

        # unknown.pdf 쌍은 source_texts에 없으므로 grounded에 추가됩니다
        assert len(grounded) == 2
        assert len(ungrounded) == 0
