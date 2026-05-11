"""DataAugmenter 테스트 — mock 기반 (teacher.agenerate 모킹)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from slm_factory.config import AugmentConfig, TeacherConfig
from slm_factory.models import QAPair
from slm_factory.augmenter import DataAugmenter, _token_overlap_ratio


@pytest.fixture
def teacher_config():
    return TeacherConfig(backend="openai")


@pytest.fixture
def augment_config():
    return AugmentConfig(enabled=True, num_variants=2, min_similarity=0.0)


@pytest.fixture
def mock_teacher():
    teacher = MagicMock()
    teacher.agenerate = AsyncMock()
    return teacher


@pytest.fixture
def augmenter(mock_teacher, augment_config, teacher_config):
    return DataAugmenter(mock_teacher, augment_config, teacher_config)


@pytest.fixture
def sample_pair():
    return QAPair(
        question="원본 질문입니다?",
        answer="원본 답변입니다.",
        source_doc="test.pdf",
        category="general",
    )


class TestBuildParaphrasePrompt:
    def test_contains_question(self, augmenter):
        prompt = augmenter._build_paraphrase_prompt("테스트 질문?", 2)
        assert "테스트 질문?" in prompt

    def test_contains_num_variants(self, augmenter):
        prompt = augmenter._build_paraphrase_prompt("질문?", 3)
        assert "3개" in prompt

    def test_contains_json_format(self, augmenter):
        prompt = augmenter._build_paraphrase_prompt("질문?", 2)
        assert "JSON" in prompt
        assert "questions" in prompt


class TestParseParaphrases:
    def test_valid_json_dict(self, augmenter):
        text = '{"questions": ["변형1?", "변형2?"]}'
        result = augmenter._parse_paraphrases(text)
        assert result == ["변형1?", "변형2?"]

    def test_valid_json_with_code_fence(self, augmenter):
        text = '```json\n{"questions": ["변형1?", "변형2?"]}\n```'
        result = augmenter._parse_paraphrases(text)
        assert result == ["변형1?", "변형2?"]

    def test_array_fallback(self, augmenter):
        text = '["변형1?", "변형2?"]'
        result = augmenter._parse_paraphrases(text)
        assert result == ["변형1?", "변형2?"]

    def test_parse_failure_returns_empty(self, augmenter):
        text = "이것은 JSON이 아닙니다"
        result = augmenter._parse_paraphrases(text)
        assert result == []

    def test_filters_empty_strings(self, augmenter):
        text = '{"questions": ["변형1?", "", "  ", "변형2?"]}'
        result = augmenter._parse_paraphrases(text)
        assert result == ["변형1?", "변형2?"]


class TestAugmentAll:
    async def test_original_plus_augmented(self, mock_teacher, augment_config, teacher_config):
        augmenter = DataAugmenter(mock_teacher, augment_config, teacher_config)

        pairs = [
            QAPair(question="원본1?", answer="답변1", source_doc="a.pdf", category="g"),
            QAPair(question="원본2?", answer="답변2", source_doc="b.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(side_effect=[
            json.dumps({"questions": ["변형1a?", "변형1b?"]}),
            json.dumps({"questions": ["변형2a?", "변형2b?"]}),
        ])

        result = await augmenter.augment_all(pairs)
        assert len(result) == 6  # 2 original + 4 augmented

    async def test_is_augmented_flag(self, mock_teacher, augment_config, teacher_config):
        augmenter = DataAugmenter(mock_teacher, augment_config, teacher_config)

        pairs = [
            QAPair(question="원본?", answer="답변", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["변형1?", "변형2?"]})
        )

        result = await augmenter.augment_all(pairs)
        originals = [p for p in result if not p.is_augmented]
        augmented = [p for p in result if p.is_augmented]
        assert len(originals) == 1
        assert len(augmented) == 2

    async def test_augmented_preserves_answer(self, mock_teacher, augment_config, teacher_config):
        augmenter = DataAugmenter(mock_teacher, augment_config, teacher_config)

        pairs = [
            QAPair(question="원본?", answer="원본 답변입니다", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["변형?"]})
        )

        result = await augmenter.augment_all(pairs)
        for p in result:
            assert p.answer == "원본 답변입니다"

    async def test_skips_already_augmented(self, mock_teacher, augment_config, teacher_config):
        augmenter = DataAugmenter(mock_teacher, augment_config, teacher_config)

        pairs = [
            QAPair(question="원본?", answer="답변", source_doc="a.pdf", category="g", is_augmented=False),
            QAPair(question="이미증강?", answer="답변", source_doc="a.pdf", category="g", is_augmented=True),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["변형?"]})
        )

        result = await augmenter.augment_all(pairs)
        assert mock_teacher.agenerate.call_count == 1

    async def test_teacher_error_handled(self, mock_teacher, augment_config, teacher_config):
        augmenter = DataAugmenter(mock_teacher, augment_config, teacher_config)

        pairs = [
            QAPair(question="원본?", answer="답변", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(side_effect=RuntimeError("LLM 오류"))

        result = await augmenter.augment_all(pairs)
        assert len(result) == 1
        assert result[0].question == "원본?"

    async def test_num_variants_limit(self, mock_teacher, teacher_config):
        config = AugmentConfig(enabled=True, num_variants=1, min_similarity=0.0)
        augmenter = DataAugmenter(mock_teacher, config, teacher_config)

        pairs = [
            QAPair(question="원본?", answer="답변", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["변형1?", "변형2?", "변형3?"]})
        )

        result = await augmenter.augment_all(pairs)
        augmented = [p for p in result if p.is_augmented]
        assert len(augmented) == 1


class TestTokenOverlapFiltering:
    async def test_높은_겹침_패러프레이즈_유지(self, mock_teacher, teacher_config):
        config = AugmentConfig(enabled=True, num_variants=2, min_similarity=0.3)
        augmenter = DataAugmenter(mock_teacher, config, teacher_config)

        pairs = [
            QAPair(question="데이터베이스 인덱스란 무엇인가?", answer="답변", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["데이터베이스에서 인덱스란 무엇인가?", "인덱스 데이터베이스 개념은?"]})
        )

        result = await augmenter.augment_all(pairs)
        augmented = [p for p in result if p.is_augmented]
        assert len(augmented) >= 1

    async def test_낮은_겹침_패러프레이즈_제거(self, mock_teacher, teacher_config):
        config = AugmentConfig(enabled=True, num_variants=2, min_similarity=0.3)
        augmenter = DataAugmenter(mock_teacher, config, teacher_config)

        pairs = [
            QAPair(question="데이터베이스 인덱스란 무엇인가?", answer="답변", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["오늘 날씨가 좋습니다", "점심 메뉴 추천해주세요"]})
        )

        result = await augmenter.augment_all(pairs)
        augmented = [p for p in result if p.is_augmented]
        assert len(augmented) == 0

    async def test_min_similarity_설정_적용(self, mock_teacher, teacher_config):
        config = AugmentConfig(enabled=True, num_variants=2, min_similarity=0.9)
        augmenter = DataAugmenter(mock_teacher, config, teacher_config)

        pairs = [
            QAPair(question="데이터베이스 인덱스란?", answer="답변", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["데이터베이스 인덱스 개념은?", "DB 인덱스란?"]})
        )

        result = await augmenter.augment_all(pairs)
        augmented_high = [p for p in result if p.is_augmented]

        config_low = AugmentConfig(enabled=True, num_variants=2, min_similarity=0.1)
        augmenter_low = DataAugmenter(mock_teacher, config_low, teacher_config)

        mock_teacher.agenerate = AsyncMock(
            return_value=json.dumps({"questions": ["데이터베이스 인덱스 개념은?", "DB 인덱스란?"]})
        )

        result_low = await augmenter_low.augment_all(pairs)
        augmented_low = [p for p in result_low if p.is_augmented]

        assert len(augmented_low) >= len(augmented_high)


class TestTokenOverlapRatio:
    def test_완전_일치(self):
        ratio = _token_overlap_ratio("데이터베이스 인덱스 설명", "데이터베이스 인덱스 설명")
        assert ratio >= 0.9

    def test_완전_불일치(self):
        ratio = _token_overlap_ratio("데이터베이스 인덱스", "오늘 날씨 좋다")
        assert ratio <= 0.1

    def test_빈_문자열(self):
        ratio = _token_overlap_ratio("", "테스트")
        assert ratio == 0.0
        ratio2 = _token_overlap_ratio("테스트", "")
        assert ratio2 == 0.0
        ratio3 = _token_overlap_ratio("", "")
        assert ratio3 == 0.0

    def test_한국어_조사_변형_높은_유사도(self):
        """한국어 교착어 조사 변형이 높은 유사도로 평가됩니다."""
        ratio = _token_overlap_ratio(
            "데이터베이스 인덱스란 무엇인가?",
            "데이터베이스에서 인덱스란 무엇인가?",
        )
        assert ratio >= 0.5
