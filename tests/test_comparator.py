"""ModelComparator 테스트 — mock 기반 (httpx + evaluate 패키지 모킹)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from slm_factory.config import CompareConfig, EvalConfig, ProjectConfig, SLMConfig, TeacherConfig
from slm_factory.comparator import ModelComparator
from slm_factory.models import CompareResult, QAPair


@pytest.fixture
def slm_config():
    return SLMConfig(
        teacher=TeacherConfig(api_base="http://localhost:11434", timeout=30, max_concurrency=2),
        eval=EvalConfig(enabled=True, metrics=["bleu", "rouge"], max_samples=10),
        compare=CompareConfig(
            enabled=True,
            base_model="base-model",
            finetuned_model="finetuned-model",
            max_samples=10,
        ),
    )


@pytest.fixture
def comparator(slm_config):
    return ModelComparator(slm_config)


@pytest.fixture
def sample_pairs():
    return [
        QAPair(question="한국의 수도는?", answer="서울입니다.", source_doc="geo.pdf", category="지리"),
        QAPair(question="1+1은?", answer="2입니다.", source_doc="math.pdf", category="수학"),
    ]


@pytest.fixture
def mock_bleu():
    m = MagicMock()
    m.compute.return_value = {"bleu": 0.75}
    return m


@pytest.fixture
def mock_rouge():
    m = MagicMock()
    m.compute.return_value = {"rouge1": 0.8, "rouge2": 0.6, "rougeL": 0.7}
    return m


def _mock_httpx_response(text: str):
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    return resp


class TestComputeScores:
    def test_bleu_and_rouge(self, comparator, mock_bleu, mock_rouge):
        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
        ):
            scores = comparator._compute_scores("서울입니다.", "서울", "서울입니다.")

        assert "base_bleu" in scores
        assert "finetuned_bleu" in scores
        assert "base_rouge1" in scores
        assert "finetuned_rougeL" in scores
        assert mock_bleu.compute.call_count == 2
        assert mock_rouge.compute.call_count == 2

    def test_bleu_only(self, slm_config, mock_bleu):
        slm_config.compare.metrics = ["bleu"]
        comp = ModelComparator(slm_config)
        with patch("slm_factory.comparator._load_bleu", return_value=mock_bleu):
            scores = comp._compute_scores("ref", "base", "ft")

        assert "base_bleu" in scores
        assert "finetuned_bleu" in scores
        assert "base_rouge1" not in scores

    def test_rouge_only(self, slm_config, mock_rouge):
        slm_config.compare.metrics = ["rouge"]
        comp = ModelComparator(slm_config)
        with patch("slm_factory.comparator._load_rouge", return_value=mock_rouge):
            scores = comp._compute_scores("ref", "base", "ft")

        assert "base_rouge1" in scores
        assert "finetuned_rouge1" in scores
        assert "base_bleu" not in scores

    def test_no_metrics(self, slm_config):
        slm_config.compare.metrics = []
        comp = ModelComparator(slm_config)
        scores = comp._compute_scores("ref", "base", "ft")
        assert scores == {}


class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_response_text(self, comparator):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_httpx_response("서울입니다."))

        result = await comparator._generate(mock_client, "test-model", "한국의 수도는?")

        assert result == "서울입니다."
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_response(self, comparator):
        resp = MagicMock()
        resp.json.return_value = {}
        resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)

        result = await comparator._generate(mock_client, "test-model", "질문")
        assert result == ""


class TestCompareOne:
    @pytest.mark.asyncio
    async def test_returns_compare_result(self, comparator, mock_bleu, mock_rouge):
        pair = QAPair(question="수도는?", answer="서울", source_doc="d.pdf", category="c")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_httpx_response("서울"))

        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
        ):
            result = await comparator._compare_one(mock_client, pair)

        assert isinstance(result, CompareResult)
        assert result.question == "수도는?"
        assert result.reference_answer == "서울"
        assert result.base_answer == "서울"
        assert result.finetuned_answer == "서울"
        assert "base_bleu" in result.scores
        assert mock_client.post.call_count == 2


class TestCompareAsync:
    @pytest.mark.asyncio
    async def test_processes_all_pairs(self, comparator, sample_pairs, mock_bleu, mock_rouge):
        mock_response = _mock_httpx_response("답변")

        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await comparator._compare_async(sample_pairs)

        assert len(results) == 2
        assert all(isinstance(r, CompareResult) for r in results)

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, comparator, mock_bleu, mock_rouge):
        pairs = [QAPair(question="q?", answer="a", source_doc="d.pdf", category="c")]

        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await comparator._compare_async(pairs)

        assert len(results) == 0


class TestCompare:
    def test_returns_compare_results(self, comparator, sample_pairs, mock_bleu, mock_rouge):
        mock_response = _mock_httpx_response("답변")

        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = comparator.compare(sample_pairs)

        assert len(results) == 2
        assert all(isinstance(r, CompareResult) for r in results)
        assert results[0].base_answer == "답변"
        assert results[0].finetuned_answer == "답변"

    def test_empty_input(self, comparator):
        results = comparator.compare([])
        assert results == []

    def test_max_samples_limit(self, slm_config, mock_bleu, mock_rouge):
        slm_config.compare.max_samples = 2
        comp = ModelComparator(slm_config)
        pairs = [
            QAPair(question=f"q{i}?", answer=f"a{i}", source_doc="d.pdf", category="c")
            for i in range(5)
        ]

        mock_response = _mock_httpx_response("answer")

        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = comp.compare(pairs)

        assert len(results) == 2


class TestSaveResults:
    def test_writes_valid_json(self, comparator, tmp_path):
        results = [
            CompareResult(
                question="q1?",
                reference_answer="a1",
                base_answer="b1",
                finetuned_answer="f1",
                scores={"base_bleu": 0.5, "finetuned_bleu": 0.7},
            ),
            CompareResult(
                question="q2?",
                reference_answer="a2",
                base_answer="b2",
                finetuned_answer="f2",
                scores={"base_bleu": 0.6, "finetuned_bleu": 0.8},
            ),
        ]

        out_path = tmp_path / "results" / "compare.json"
        comparator.save_results(results, out_path)

        assert out_path.is_file()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["question"] == "q1?"
        assert data[0]["base_answer"] == "b1"
        assert data[0]["scores"]["base_bleu"] == 0.5

    def test_empty_results(self, comparator, tmp_path):
        out_path = tmp_path / "empty.json"
        comparator.save_results([], out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data == []


class TestPrintSummary:
    def test_runs_without_error(self, comparator, capsys):
        results = [
            CompareResult(
                question="q?", reference_answer="a", base_answer="b", finetuned_answer="f",
                scores={"base_bleu": 0.5, "finetuned_bleu": 0.7},
            ),
        ]
        comparator.print_summary(results)
        out = capsys.readouterr().out
        assert "비교 결과 요약" in out
        assert "1건 비교 완료" in out

    def test_empty_results(self, comparator, capsys):
        comparator.print_summary([])
        out = capsys.readouterr().out
        assert "비교 결과가 없습니다" in out

    def test_multiple_metrics(self, comparator, capsys):
        results = [
            CompareResult(
                question="q1?", reference_answer="a1", base_answer="b1", finetuned_answer="f1",
                scores={"base_bleu": 0.5, "finetuned_bleu": 0.7, "base_rouge1": 0.6, "finetuned_rouge1": 0.8},
            ),
            CompareResult(
                question="q2?", reference_answer="a2", base_answer="b2", finetuned_answer="f2",
                scores={"base_bleu": 0.6, "finetuned_bleu": 0.9, "base_rouge1": 0.7, "finetuned_rouge1": 0.9},
            ),
        ]
        comparator.print_summary(results)
        out = capsys.readouterr().out
        assert "bleu" in out
        assert "rouge1" in out
        assert "2건 비교 완료" in out

    def test_zero_base_score_shows_na(self, comparator, capsys):
        results = [
            CompareResult(
                question="q?", reference_answer="a", base_answer="b", finetuned_answer="f",
                scores={"base_bleu": 0.0, "finetuned_bleu": 0.5},
            ),
        ]
        comparator.print_summary(results)
        out = capsys.readouterr().out
        assert "N/A" in out


class TestKoreanTokenizer:
    """한국어 형태소 분석 토크나이저 비교 테스트입니다."""

    def test_한국어_프로젝트_토크나이저_사용(self, mock_bleu, mock_rouge):
        """language='ko'일 때 한국어 토크나이저가 적용되는지 확인합니다."""
        config = SLMConfig(
            project=ProjectConfig(language="ko"),
            teacher=TeacherConfig(api_base="http://localhost:11434", timeout=30, max_concurrency=2),
            eval=EvalConfig(enabled=True, metrics=["bleu", "rouge"], max_samples=10),
            compare=CompareConfig(
                enabled=True,
                base_model="base-model",
                finetuned_model="finetuned-model",
                max_samples=10,
            ),
        )
        comp = ModelComparator(config)

        mock_tokenizer = MagicMock(side_effect=lambda text: text.split())
        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
            patch("slm_factory.comparator._load_korean_tokenizer", return_value=mock_tokenizer),
        ):
            comp._compute_scores("서울", "서울", "서울입니다")

        assert mock_tokenizer.call_count >= 3

    def test_영어_프로젝트_토크나이저_미사용(self, mock_bleu, mock_rouge):
        """language='en'일 때 한국어 토크나이저가 사용되지 않는지 확인합니다."""
        config = SLMConfig(
            project=ProjectConfig(language="en"),
            teacher=TeacherConfig(api_base="http://localhost:11434", timeout=30, max_concurrency=2),
            eval=EvalConfig(enabled=True, metrics=["bleu", "rouge"], max_samples=10),
            compare=CompareConfig(
                enabled=True,
                base_model="base-model",
                finetuned_model="finetuned-model",
                max_samples=10,
            ),
        )
        comp = ModelComparator(config)

        with (
            patch("slm_factory.comparator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.comparator._load_rouge", return_value=mock_rouge),
            patch("slm_factory.comparator._load_korean_tokenizer") as mock_load,
        ):
            comp._compute_scores("hello", "hello", "hello world")

        mock_load.assert_not_called()
