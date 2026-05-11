"""CLI(cli.py) 모듈의 통합 테스트입니다.

typer.testing.CliRunner를 사용하여 각 명령어의 동작을 검증합니다.
외부 파이프라인 호출은 mock으로 대체합니다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from slm_factory.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionCommand:
    """version 명령어의 테스트입니다."""

    def test_버전_출력(self):
        """출력에 '0.1.0'이 포함되는지 확인합니다."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class TestInitCommand:
    """init 명령어의 테스트입니다."""

    def test_프로젝트_디렉토리_생성(self, tmp_path):
        """프로젝트 디렉토리, documents, output, project.yaml이 생성되는지 확인합니다."""
        result = runner.invoke(
            app,
            [
                "init",
                "test-proj",
                "--path",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        project_dir = tmp_path / "test-proj"
        assert project_dir.is_dir()
        assert (project_dir / "documents").is_dir()
        assert (project_dir / "output").is_dir()
        assert (project_dir / "project.yaml").is_file()

        # project.yaml에 프로젝트명이 포함되는지 확인
        content = (project_dir / "project.yaml").read_text(encoding="utf-8")
        assert "test-proj" in content


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRunCommand:
    """run 명령어의 테스트입니다."""

    def test_존재하지_않는_config_exit_code_1(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["tune", "--config", "/nonexistent/path.yaml"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


class TestParseCommand:
    """run --until parse 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        """_load_pipeline이 올바르게 호출되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()

        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(
            app, ["tune", "--until", "parse", "--config", "test.yaml"]
        )

        assert result.exit_code == 0
        mock_pipeline.step_parse.assert_called_once()


# ---------------------------------------------------------------------------
# no args
# ---------------------------------------------------------------------------


class TestNoArgs:
    """인자 없이 실행했을 때의 테스트입니다."""

    def test_도움말_출력(self):
        """인자 없이 실행하면 도움말 메시지가 출력되는지 확인합니다."""
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    """run --until generate 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        """_load_pipeline이 올바르게 호출되고 파싱+생성이 수행되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()

        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(
            app, ["tune", "--until", "generate", "--config", "test.yaml"]
        )

        assert result.exit_code == 0
        mock_pipeline.step_parse.assert_called_once()
        mock_pipeline.step_generate.assert_called_once()


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """config.py 필드 검증 테스트입니다."""

    def test_프로젝트명_빈문자열_거부(self):
        """ProjectConfig.name에 빈 문자열을 넣으면 ValidationError가 발생하는지 확인합니다."""
        from pydantic import ValidationError
        from slm_factory.config import ProjectConfig

        with pytest.raises(ValidationError):
            ProjectConfig(name="")

    def test_교사모델명_빈문자열_거부(self):
        """TeacherConfig.model에 빈 문자열을 넣으면 ValidationError가 발생하는지 확인합니다."""
        from pydantic import ValidationError
        from slm_factory.config import TeacherConfig

        with pytest.raises(ValidationError):
            TeacherConfig(model="")

    def test_교사_api_base_빈문자열_거부(self):
        from pydantic import ValidationError
        from slm_factory.config import TeacherConfig

        with pytest.raises(ValidationError):
            TeacherConfig(api_base="")

    def test_학생모델명_빈문자열_거부(self):
        from pydantic import ValidationError
        from slm_factory.config import StudentConfig

        with pytest.raises(ValidationError):
            StudentConfig(model="")

    def test_ollama_모델명_빈문자열_거부(self):
        from pydantic import ValidationError
        from slm_factory.config import OllamaExportConfig

        with pytest.raises(ValidationError):
            OllamaExportConfig(model_name="")

    def test_정상값_통과(self):
        """기본값이 유효한지 확인합니다."""
        from slm_factory.config import (
            ProjectConfig,
            TeacherConfig,
            StudentConfig,
            OllamaExportConfig,
        )

        # 기본값으로 생성 시 에러 없어야 함
        ProjectConfig()
        TeacherConfig()
        StudentConfig()
        OllamaExportConfig()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


class TestCheckCommand:
    """check 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["check", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """status 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["status", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


class TestCleanCommand:
    """clean 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["clean", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


class TestScoreCommand:
    """run --until score 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.step_validate.return_value = [MagicMock()]
        mock_pipeline.step_score.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(
            app, ["tune", "--until", "score", "--config", "test.yaml"]
        )
        assert result.exit_code == 0
        mock_pipeline.step_score.assert_called_once()


# ---------------------------------------------------------------------------
# augment
# ---------------------------------------------------------------------------


class TestAugmentCommand:
    """run --until augment 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.step_validate.return_value = [MagicMock()]
        mock_pipeline.step_score.return_value = [MagicMock()]
        mock_pipeline.step_augment.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(
            app, ["tune", "--until", "augment", "--config", "test.yaml"]
        )
        assert result.exit_code == 0
        mock_pipeline.step_augment.assert_called_once()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    """run --until analyze 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.step_validate.return_value = [MagicMock()]
        mock_pipeline.step_score.return_value = [MagicMock()]
        mock_pipeline.step_augment.return_value = [MagicMock()]
        mock_pipeline.step_analyze.return_value = None
        mock_pipeline.config.paths.ensure_dirs = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(
            app, ["tune", "--until", "analyze", "--config", "test.yaml"]
        )
        assert result.exit_code == 0
        mock_pipeline.step_analyze.assert_called_once()


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


class TestConvertCommand:
    """convert 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        result = runner.invoke(
            app, ["tool", "convert", "--config", "/nonexistent/path.yaml"]
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExportCommand:
    """export 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        result = runner.invoke(app, ["export", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# _load_qa_data helper
# ---------------------------------------------------------------------------


class TestLoadQaData:
    """_load_qa_data 헬퍼 함수의 테스트입니다."""

    def test_명시적_경로_파일_존재(self, tmp_path):
        """--data 옵션으로 지정한 파일이 존재하면 해당 파일을 로드하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        data_file = tmp_path / "test.json"
        data_file.write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline._load_pairs.return_value = [mock_pair]

        result = _load_qa_data(pipeline, str(data_file))

        pipeline._load_pairs.assert_called_once_with(data_file)
        assert result == [mock_pair]

    def test_명시적_경로_파일_미존재(self):
        """--data 옵션으로 지정한 파일이 없으면 Exit 예외가 발생하는지 확인합니다."""
        from click.exceptions import Exit as ClickExit

        from slm_factory.cli import _load_qa_data

        pipeline = MagicMock()

        with pytest.raises(ClickExit):
            _load_qa_data(pipeline, "/nonexistent/file.json")

    def test_자동감지_qa_augmented(self, tmp_path):
        """출력 디렉토리에서 qa_augmented.json을 자동 감지하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        qa_file = tmp_path / "qa_augmented.json"
        qa_file.write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._load_pairs.return_value = [mock_pair]

        result = _load_qa_data(pipeline, None)

        pipeline._load_pairs.assert_called_once_with(qa_file)
        assert result == [mock_pair]

    def test_자동감지_우선순위(self, tmp_path):
        """qa_augmented > qa_scored > qa_alpaca 우선순위로 감지하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        (tmp_path / "qa_augmented.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_scored.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_alpaca.json").write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._load_pairs.return_value = [mock_pair]

        _load_qa_data(pipeline, None)

        pipeline._load_pairs.assert_called_once_with(tmp_path / "qa_augmented.json")

    def test_extra_candidates_우선(self, tmp_path):
        """extra_candidates가 기본 후보보다 우선하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        (tmp_path / "qa_reviewed.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_augmented.json").write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._load_pairs.return_value = [mock_pair]

        _load_qa_data(pipeline, None, extra_candidates=["qa_reviewed.json"])

        pipeline._load_pairs.assert_called_once_with(tmp_path / "qa_reviewed.json")

    def test_파일_미발견_exit(self, tmp_path):
        """출력 디렉토리에 QA 파일이 없으면 Exit 예외가 발생하는지 확인합니다."""
        from click.exceptions import Exit as ClickExit

        from slm_factory.cli import _load_qa_data

        pipeline = MagicMock()
        pipeline.output_dir = tmp_path

        with pytest.raises(ClickExit):
            _load_qa_data(pipeline, None)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


class TestEvalCommand:
    """eval run / eval compare 명령어의 테스트입니다."""

    def test_모델_평가_실행(self, mocker, tmp_path):
        """파이프라인과 평가기를 올바르게 호출하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.config.eval.output_file = "eval_results.json"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = [MagicMock()]
        mocker.patch(
            "slm_factory.evaluator.ModelEvaluator",
            return_value=mock_evaluator,
        )

        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "--config",
                "test.yaml",
                "--model",
                "test-model",
            ],
        )

        assert result.exit_code == 0
        mock_evaluator.evaluate.assert_called_once()
        mock_evaluator.save_results.assert_called_once()
        mock_evaluator.print_summary.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "--config",
                "/nonexistent/path.yaml",
                "--model",
                "test",
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdateCommand:
    """tool update 명령어의 테스트입니다."""

    def test_증분_업데이트_실행(self, mocker, tmp_path):
        """변경 문서 감지 및 증분 업데이트가 수행되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline._load_pairs.return_value = []
        mock_pipeline.config.incremental.merge_strategy = "append"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        mock_tracker = MagicMock()
        mock_tracker.get_changed_files.return_value = [Path("doc1.pdf")]
        mock_tracker.merge_qa_pairs.return_value = [MagicMock()]
        mocker.patch(
            "slm_factory.incremental.IncrementalTracker",
            return_value=mock_tracker,
        )

        result = runner.invoke(app, ["tool", "update", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_tracker.get_changed_files.assert_called_once()
        mock_pipeline.step_parse.assert_called_once()
        mock_pipeline.step_generate.assert_called_once()

    def test_변경_없음_조기종료(self, mocker):
        """변경된 문서가 없으면 조기 종료하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        mock_tracker = MagicMock()
        mock_tracker.get_changed_files.return_value = []
        mocker.patch(
            "slm_factory.incremental.IncrementalTracker",
            return_value=mock_tracker,
        )

        result = runner.invoke(app, ["tool", "update", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_pipeline.step_parse.assert_not_called()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(
            app,
            [
                "tool",
                "update",
                "--config",
                "/nonexistent/path.yaml",
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompareCommand:
    """eval compare 명령어의 테스트입니다."""

    def test_모델_비교_실행(self, mocker, tmp_path):
        """파이프라인과 ModelComparator를 올바르게 호출하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.config.compare.output_file = "compare_results.json"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_comparator = MagicMock()
        mock_comparator.compare.return_value = [MagicMock()]
        mocker.patch(
            "slm_factory.comparator.ModelComparator",
            return_value=mock_comparator,
        )

        result = runner.invoke(
            app,
            [
                "eval",
                "compare",
                "--config",
                "test.yaml",
                "--base-model",
                "base",
                "--ft",
                "finetuned",
            ],
        )

        assert result.exit_code == 0
        mock_comparator.compare.assert_called_once()
        mock_comparator.save_results.assert_called_once()
        mock_comparator.print_summary.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(
            app,
            [
                "eval",
                "compare",
                "--config",
                "/nonexistent/path.yaml",
                "--base-model",
                "base",
                "--ft",
                "finetuned",
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


class TestReviewCommand:
    """review 명령어의 테스트입니다."""

    def test_리뷰_실행(self, mocker, tmp_path):
        """파이프라인과 QAReviewerApp이 올바르게 호출되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.config.review.output_file = "qa_reviewed.json"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_reviewer_cls = MagicMock()
        mock_reviewer_cls.count_statuses.return_value = {
            "approved": 1,
            "rejected": 0,
            "pending": 0,
        }
        mocker.patch(
            "slm_factory.tui.reviewer.QAReviewerApp",
            mock_reviewer_cls,
        )

        result = runner.invoke(app, ["tool", "review", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_reviewer_cls.return_value.run.assert_called_once()
        mock_reviewer_cls.count_statuses.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(
            app,
            [
                "tool",
                "review",
                "--config",
                "/nonexistent/path.yaml",
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# compare-data
# ---------------------------------------------------------------------------


class TestCompareDataCommand:
    """tool compare-data 명령어의 테스트입니다."""

    def _make_qa_file(self, path: Path, pairs: list[dict]) -> Path:
        import json as _json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def test_두_파일_비교_정상_실행(self, tmp_path):
        """두 QA 파일을 비교하여 정상 실행되는지 확인합니다."""
        baseline_pairs = [
            {
                "question": "Q1?",
                "answer": "A" * 50,
                "instruction": "Q1?",
                "source_doc": "a.pdf",
                "category": "test",
            },
            {
                "question": "Q2?",
                "answer": "B" * 30,
                "instruction": "Q2?",
                "source_doc": "a.pdf",
                "category": "test",
            },
        ]
        target_pairs = [
            {
                "question": "Q1?",
                "answer": "A" * 80,
                "instruction": "Q1?",
                "source_doc": "a.pdf",
                "category": "test",
            },
            {
                "question": "Q2?",
                "answer": "B" * 60,
                "instruction": "Q2?",
                "source_doc": "a.pdf",
                "category": "test",
            },
            {
                "question": "Q3?",
                "answer": "C" * 40,
                "instruction": "Q3?",
                "source_doc": "b.pdf",
                "category": "test",
            },
        ]

        baseline = self._make_qa_file(tmp_path / "baseline.json", baseline_pairs)
        target = self._make_qa_file(tmp_path / "target.json", target_pairs)

        result = runner.invoke(
            app,
            [
                "tool",
                "compare-data",
                "--baseline",
                str(baseline),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0
        assert "기본 통계 비교" in result.output
        assert "baseline.json" in result.output
        assert "target.json" in result.output

    def test_기준_파일_미존재시_exit_code_1(self, tmp_path):
        """기준 파일이 존재하지 않으면 exit code 1로 종료하는지 확인합니다."""
        target = self._make_qa_file(tmp_path / "target.json", [])

        result = runner.invoke(
            app,
            [
                "tool",
                "compare-data",
                "--baseline",
                "/nonexistent/file.json",
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 1

    def test_대상_파일_미존재시_exit_code_1(self, tmp_path):
        """대상 파일이 존재하지 않으면 exit code 1로 종료하는지 확인합니다."""
        baseline = self._make_qa_file(tmp_path / "baseline.json", [])

        result = runner.invoke(
            app,
            [
                "tool",
                "compare-data",
                "--baseline",
                str(baseline),
                "--target",
                "/nonexistent/file.json",
            ],
        )

        assert result.exit_code == 1

    def test_빈_파일_비교(self, tmp_path):
        """빈 QA 데이터 비교 시에도 정상 실행되는지 확인합니다."""
        baseline = self._make_qa_file(tmp_path / "baseline.json", [])
        target = self._make_qa_file(tmp_path / "target.json", [])

        result = runner.invoke(
            app,
            [
                "tool",
                "compare-data",
                "--baseline",
                str(baseline),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0

    def test_카테고리_분포_비교_출력(self, tmp_path):
        """서로 다른 카테고리 분포가 비교 테이블에 표시되는지 확인합니다."""
        baseline_pairs = [
            {
                "question": "Q?",
                "answer": "A" * 50,
                "instruction": "Q?",
                "source_doc": "a.pdf",
                "category": "overview",
            },
        ]
        target_pairs = [
            {
                "question": "Q?",
                "answer": "A" * 50,
                "instruction": "Q?",
                "source_doc": "a.pdf",
                "category": "overview",
            },
            {
                "question": "Q2?",
                "answer": "B" * 50,
                "instruction": "Q2?",
                "source_doc": "a.pdf",
                "category": "technical",
            },
        ]

        baseline = self._make_qa_file(tmp_path / "baseline.json", baseline_pairs)
        target = self._make_qa_file(tmp_path / "target.json", target_pairs)

        result = runner.invoke(
            app,
            [
                "tool",
                "compare-data",
                "--baseline",
                str(baseline),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0
        assert "카테고리 분포 비교" in result.output

    def test_help_출력(self):
        """compare-data --help가 정상 출력되는지 확인합니다."""
        result = runner.invoke(app, ["tool", "compare-data", "--help"])
        assert result.exit_code == 0
        assert "비교" in result.output
