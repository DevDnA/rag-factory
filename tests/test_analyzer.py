"""DataAnalyzer 및 AnalysisReport 테스트 — fixture 기반, mock 불필요."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_factory.analyzer import AnalysisReport, DataAnalyzer
from slm_factory.models import QAPair


@pytest.fixture
def analyzer():
    return DataAnalyzer()


@pytest.fixture
def sample_pairs():
    return [
        QAPair(question="질문1?", answer="답변1입니다.", source_doc="doc1.pdf", category="general"),
        QAPair(question="질문2?", answer="답변2입니다. 좀 더 긴 답변.", source_doc="doc1.pdf", category="general"),
        QAPair(question="질문3?", answer="답변3.", source_doc="doc2.pdf", category="technical"),
    ]


@pytest.fixture
def augmented_pairs():
    return [
        QAPair(question="원본 질문?", answer="답변입니다.", source_doc="doc1.pdf", category="general", is_augmented=False),
        QAPair(question="패러프레이즈 질문?", answer="답변입니다.", source_doc="doc1.pdf", category="general", is_augmented=True),
        QAPair(question="또 다른 패러프레이즈?", answer="답변입니다.", source_doc="doc1.pdf", category="general", is_augmented=True),
    ]


class TestAnalysisReport:
    def test_default_values(self):
        report = AnalysisReport()
        assert report.total_pairs == 0
        assert report.original_pairs == 0
        assert report.augmented_pairs == 0
        assert report.category_distribution == {}
        assert report.source_doc_distribution == {}
        assert report.answer_length_stats == {}
        assert report.question_length_stats == {}
        assert report.quality_score_stats == {}
        assert report.warnings == []


class TestDataAnalyzer:
    def test_analyze_empty_list(self, analyzer):
        report = analyzer.analyze([])
        assert report.total_pairs == 0
        assert "데이터가 비어 있습니다." in report.warnings

    def test_analyze_normal_data(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        assert report.total_pairs == 3
        assert report.original_pairs == 3
        assert report.augmented_pairs == 0

    def test_category_distribution(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        assert "general" in report.category_distribution
        assert "technical" in report.category_distribution
        assert report.category_distribution["general"] == 2
        assert report.category_distribution["technical"] == 1

    def test_source_doc_distribution(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        assert "doc1.pdf" in report.source_doc_distribution
        assert "doc2.pdf" in report.source_doc_distribution
        assert report.source_doc_distribution["doc1.pdf"] == 2
        assert report.source_doc_distribution["doc2.pdf"] == 1

    def test_answer_length_stats(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        stats = report.answer_length_stats
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "stdev" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_question_length_stats(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        stats = report.question_length_stats
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] > 0

    def test_imbalance_warning(self, analyzer):
        pairs = [
            QAPair(question="q", answer="a" * 20, source_doc="doc1.pdf", category="cat1")
        ] * 11 + [
            QAPair(question="q", answer="a" * 20, source_doc="doc2.pdf", category="cat1")
        ]
        report = analyzer.analyze(pairs)
        imbalance_warnings = [w for w in report.warnings if "불균형" in w]
        assert len(imbalance_warnings) > 0

    def test_insufficient_data_warning(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        data_warnings = [w for w in report.warnings if "적습니다" in w]
        assert len(data_warnings) > 0

    def test_single_category_warning(self, analyzer):
        pairs = [
            QAPair(question="q1?", answer="a" * 20, source_doc="doc1.pdf", category="only_one"),
            QAPair(question="q2?", answer="a" * 20, source_doc="doc1.pdf", category="only_one"),
        ]
        report = analyzer.analyze(pairs)
        cat_warnings = [w for w in report.warnings if "카테고리가 1개" in w]
        assert len(cat_warnings) > 0

    def test_save_report(self, analyzer, sample_pairs, tmp_path):
        report = analyzer.analyze(sample_pairs)
        output_path = tmp_path / "report.json"
        analyzer.save_report(report, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["total_pairs"] == 3
        assert "category_distribution" in data

    def test_is_augmented_count(self, analyzer, augmented_pairs):
        report = analyzer.analyze(augmented_pairs)
        assert report.total_pairs == 3
        assert report.original_pairs == 1
        assert report.augmented_pairs == 2

    def test_compute_stats_single_value(self, analyzer):
        stats = analyzer._compute_stats([42])
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["mean"] == 42.0
        assert stats["stdev"] == 0.0

    def test_compute_stats_empty(self, analyzer):
        stats = analyzer._compute_stats([])
        assert stats == {}


class TestDiversityMetrics:
    """다양성 메트릭 분석 테스트."""

    def test_다양성_메트릭_존재(self, analyzer, sample_pairs):
        """analyze()가 report.diversity에 다양성 메트릭을 채우는지 확인합니다."""
        report = analyzer.analyze(sample_pairs)

        assert "unique_bigram_ratio" in report.diversity
        assert "question_type_coverage" in report.diversity
        assert "unique_question_types" in report.diversity

    def test_질문_유형_분류(self, analyzer, sample_pairs):
        """analyze()가 report.question_type_distribution을 채우는지 확인합니다."""
        report = analyzer.analyze(sample_pairs)

        assert isinstance(report.question_type_distribution, dict)
        assert len(report.question_type_distribution) > 0
        total = sum(report.question_type_distribution.values())
        assert total == len(sample_pairs)

    def test_단일_유형_질문_경고(self, analyzer):
        """모든 질문이 동일 유형이면 제한적 유형 경고가 발생합니다."""
        pairs = [
            QAPair(question="무엇이 핵심인가?", answer="답변입니다.", source_doc="d.pdf", category="c"),
            QAPair(question="무엇이 중요한가?", answer="답변입니다.", source_doc="d.pdf", category="c"),
            QAPair(question="무엇이 차이인가?", answer="답변입니다.", source_doc="d.pdf", category="c"),
        ]
        report = analyzer.analyze(pairs)

        type_warnings = [w for w in report.warnings if "질문 유형" in w and "제한적" in w]
        assert len(type_warnings) > 0

    def test_낮은_다양성_경고(self, analyzer):
        """동일한 질문이 반복되면 낮은 bigram 다양성 경고가 발생합니다."""
        pairs = [
            QAPair(question="동일한 질문입니다", answer="답변입니다.", source_doc="d.pdf", category="c")
            for _ in range(20)
        ]
        report = analyzer.analyze(pairs)

        diversity_warnings = [w for w in report.warnings if "다양성" in w and "낮" in w]
        assert len(diversity_warnings) > 0
        assert report.diversity["unique_bigram_ratio"] < 0.3


class TestClassifyQuestionType:
    """_classify_question_type 질문 유형 분류 테스트."""

    def test_what_유형(self, analyzer):
        """'무엇' 계열 질문이 'what'으로 분류되는지 확인합니다."""
        assert analyzer._classify_question_type("무엇이 핵심인가?") == "what"

    def test_how_유형(self, analyzer):
        """'어떻게' 계열 질문이 'how'로 분류되는지 확인합니다."""
        assert analyzer._classify_question_type("어떻게 작동하나?") == "how"

    def test_why_유형(self, analyzer):
        """'왜' 계열 질문이 'why'로 분류되는지 확인합니다."""
        assert analyzer._classify_question_type("왜 이런 방식인가?") == "why"

    def test_other_유형(self, analyzer):
        """분류 규칙에 매칭되지 않는 질문은 'other'로 분류됩니다."""
        assert analyzer._classify_question_type("설명해주세요") == "other"

    def test_한국어_중간_위치_의문사(self, analyzer):
        """한국어 의문사가 문장 중간에 있어도 올바르게 분류됩니다."""
        assert analyzer._classify_question_type("인덱스란 무엇인가?") == "what"
        assert analyzer._classify_question_type("이 기능은 어떻게 동작하나요?") == "how"
        assert analyzer._classify_question_type("그것은 왜 필요한가?") == "why"

    def test_누락_패턴_무슨(self, analyzer):
        """'무슨' 계열 질문이 'what'으로 분류됩니다."""
        assert analyzer._classify_question_type("무슨 차이가 있나요?") == "what"

    def test_누락_패턴_어째서(self, analyzer):
        """'어째서' 계열 질문이 'why'로 분류됩니다."""
        assert analyzer._classify_question_type("어째서 이렇게 되었나요?") == "why"
