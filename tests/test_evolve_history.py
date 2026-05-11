"""진화 히스토리 관리(evolve_history.py) 모듈의 단위 테스트입니다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_factory.config import EvolveConfig
from slm_factory.evolve_history import EvolveHistory
from slm_factory.models import CompareResult


# ---------------------------------------------------------------------------
# EvolveConfig 기본값 및 검증
# ---------------------------------------------------------------------------


class TestEvolveConfigDefaults:

    def test_기본값(self):
        cfg = EvolveConfig()
        assert cfg.quality_gate is True
        assert cfg.gate_metric == "rougeL"
        assert cfg.gate_min_improvement == 0.0
        assert cfg.version_format == "date"
        assert cfg.history_file == "evolve_history.json"
        assert cfg.keep_previous_versions == 3

    def test_invalid_gate_metric_raises_error(self):
        with pytest.raises(ValueError, match="gate_metric"):
            EvolveConfig(gate_metric="invalid_metric")

    def test_valid_gate_metrics(self):
        for metric in ["bleu", "rouge1", "rouge2", "rougeL"]:
            cfg = EvolveConfig(gate_metric=metric)
            assert cfg.gate_metric == metric

    def test_negative_keep_previous_versions_raises_error(self):
        with pytest.raises(ValueError, match="keep_previous_versions"):
            EvolveConfig(keep_previous_versions=-1)

    def test_zero_keep_previous_versions_allowed(self):
        cfg = EvolveConfig(keep_previous_versions=0)
        assert cfg.keep_previous_versions == 0


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestEvolveHistoryLoad:

    def test_파일_없으면_기본값_반환(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        result = history.load()

        assert result == {"versions": [], "current": None}

    def test_빈_파일_로드(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        history_file.write_text("{}", encoding="utf-8")

        history = EvolveHistory(config)
        result = history.load()

        assert result == {}

    def test_기존_파일_로드(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220"}
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)
        result = history.load()

        assert result == data
        assert len(result["versions"]) == 1
        assert result["current"] == "v20250220"


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestEvolveHistorySave:

    def test_파일_생성(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)
        data = {"versions": [], "current": None}

        history.save(data)

        assert (tmp_path / "evolve_history.json").is_file()

    def test_올바른_내용_저장(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220"}
            ],
            "current": "v20250220",
        }

        history.save(data)

        loaded = json.loads(
            (tmp_path / "evolve_history.json").read_text(encoding="utf-8")
        )
        assert loaded == data

    def test_부모_디렉토리_자동_생성(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path / "nested" / "output")}
        )
        history = EvolveHistory(config)
        data = {"versions": [], "current": None}

        history.save(data)

        assert (tmp_path / "nested" / "output" / "evolve_history.json").is_file()


# ---------------------------------------------------------------------------
# is_first_run
# ---------------------------------------------------------------------------


class TestEvolveHistoryIsFirstRun:

    def test_버전_없으면_true(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        assert history.is_first_run() is True

    def test_버전_있으면_false(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220"}
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)
        assert history.is_first_run() is False

    def test_빈_버전_리스트_true(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        history_file.write_text(json.dumps({"versions": []}), encoding="utf-8")

        history = EvolveHistory(config)
        assert history.is_first_run() is True


# ---------------------------------------------------------------------------
# get_current_model_name
# ---------------------------------------------------------------------------


class TestEvolveHistoryGetCurrentModelName:

    def test_current_없으면_none(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        assert history.get_current_model_name() is None

    def test_current_있으면_모델명_반환(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {
                    "version": "v20250220",
                    "model_name": "my-model-v20250220",
                }
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)
        assert history.get_current_model_name() == "my-model-v20250220"

    def test_current_버전_없으면_none(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {
                    "version": "v20250220",
                    "model_name": "my-model-v20250220",
                }
            ],
            "current": "v20250221",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)
        assert history.get_current_model_name() is None


# ---------------------------------------------------------------------------
# generate_version_name
# ---------------------------------------------------------------------------


class TestEvolveHistoryGenerateVersionName:

    def test_첫_실행_버전명(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        with patch("slm_factory.evolve_history.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 2, 20, 10, 30, 0, tzinfo=timezone.utc
            )
            version = history.generate_version_name()

        assert version == "v20250220"

    def test_같은_날_두_번째_버전(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220"}
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)

        with patch("slm_factory.evolve_history.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 2, 20, 15, 45, 0, tzinfo=timezone.utc
            )
            version = history.generate_version_name()

        assert version == "v20250220-2"

    def test_같은_날_세_번째_버전(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220"},
                {"version": "v20250220-2", "model_name": "test-v20250220-2"},
            ],
            "current": "v20250220-2",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)

        with patch("slm_factory.evolve_history.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 2, 20, 20, 0, 0, tzinfo=timezone.utc
            )
            version = history.generate_version_name()

        assert version == "v20250220-3"

    def test_다른_날_새_버전(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220"}
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        history = EvolveHistory(config)

        with patch("slm_factory.evolve_history.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 2, 21, 10, 0, 0, tzinfo=timezone.utc
            )
            version = history.generate_version_name()

        assert version == "v20250221"


# ---------------------------------------------------------------------------
# generate_model_name
# ---------------------------------------------------------------------------


class TestEvolveHistoryGenerateModelName:

    def test_모델명_생성(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            export={"ollama": {"model_name": "my-model"}},
        )
        history = EvolveHistory(config)

        model_name = history.generate_model_name("v20250220")

        assert model_name == "my-model-v20250220"

    def test_버전_포함된_모델명(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            export={"ollama": {"model_name": "domain-expert"}},
        )
        history = EvolveHistory(config)

        model_name = history.generate_model_name("v20250220-2")

        assert model_name == "domain-expert-v20250220-2"


# ---------------------------------------------------------------------------
# check_quality_gate
# ---------------------------------------------------------------------------


class TestEvolveHistoryCheckQualityGate:

    def test_빈_결과_실패(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        passed, scores = history.check_quality_gate([])

        assert passed is False
        assert scores == {}

    def test_개선_있으면_통과(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"gate_metric": "rougeL", "gate_min_improvement": 0.0},
        )
        history = EvolveHistory(config)

        results = [
            CompareResult(
                question="Q1",
                reference_answer="Ref",
                base_answer="Base",
                finetuned_answer="FT",
                scores={
                    "base_rougeL": 0.5,
                    "finetuned_rougeL": 0.6,
                },
            ),
        ]

        passed, scores = history.check_quality_gate(results)

        assert passed is True
        assert scores["base_avg"] == 0.5
        assert scores["finetuned_avg"] == 0.6
        assert scores["improvement_pct"] == 20.0

    def test_개선_없으면_실패(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"gate_metric": "rougeL", "gate_min_improvement": 5.0},
        )
        history = EvolveHistory(config)

        results = [
            CompareResult(
                question="Q1",
                reference_answer="Ref",
                base_answer="Base",
                finetuned_answer="FT",
                scores={
                    "base_rougeL": 0.5,
                    "finetuned_rougeL": 0.51,
                },
            ),
        ]

        passed, scores = history.check_quality_gate(results)

        assert passed is False
        assert scores["improvement_pct"] == 2.0

    def test_메트릭_없으면_실패(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"gate_metric": "rougeL"},
        )
        history = EvolveHistory(config)

        results = [
            CompareResult(
                question="Q1",
                reference_answer="Ref",
                base_answer="Base",
                finetuned_answer="FT",
                scores={"base_bleu": 0.5, "finetuned_bleu": 0.6},
            ),
        ]

        passed, scores = history.check_quality_gate(results)

        assert passed is False
        assert scores == {}

    def test_여러_결과_평균_계산(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"gate_metric": "rougeL", "gate_min_improvement": 0.0},
        )
        history = EvolveHistory(config)

        results = [
            CompareResult(
                question="Q1",
                reference_answer="Ref1",
                base_answer="Base1",
                finetuned_answer="FT1",
                scores={
                    "base_rougeL": 0.4,
                    "finetuned_rougeL": 0.5,
                },
            ),
            CompareResult(
                question="Q2",
                reference_answer="Ref2",
                base_answer="Base2",
                finetuned_answer="FT2",
                scores={
                    "base_rougeL": 0.6,
                    "finetuned_rougeL": 0.7,
                },
            ),
        ]

        passed, scores = history.check_quality_gate(results)

        assert passed is True
        assert scores["base_avg"] == 0.5
        assert scores["finetuned_avg"] == 0.6
        assert scores["improvement_pct"] == 20.0

    def test_base_avg_zero_처리(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"gate_metric": "rougeL", "gate_min_improvement": 0.0},
        )
        history = EvolveHistory(config)

        results = [
            CompareResult(
                question="Q1",
                reference_answer="Ref",
                base_answer="Base",
                finetuned_answer="FT",
                scores={
                    "base_rougeL": 0.0,
                    "finetuned_rougeL": 0.5,
                },
            ),
        ]

        passed, scores = history.check_quality_gate(results)

        assert passed is True
        assert scores["improvement_pct"] == 100.0


# ---------------------------------------------------------------------------
# record_version
# ---------------------------------------------------------------------------


class TestEvolveHistoryRecordVersion:

    def test_버전_기록(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        history.record_version("v20250220", "test-v20250220")

        loaded = history.load()
        assert len(loaded["versions"]) == 1
        assert loaded["versions"][0]["version"] == "v20250220"
        assert loaded["versions"][0]["model_name"] == "test-v20250220"

    def test_promoted_true_current_설정(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        history.record_version(
            "v20250220", "test-v20250220", promoted=True
        )

        loaded = history.load()
        assert loaded["current"] == "v20250220"

    def test_promoted_false_current_미설정(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        history.record_version(
            "v20250220", "test-v20250220", promoted=False
        )

        loaded = history.load()
        assert loaded.get("current") is None

    def test_점수_포함_기록(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)
        scores = {"base_avg": 0.5, "finetuned_avg": 0.6}

        history.record_version(
            "v20250220", "test-v20250220", scores=scores
        )

        loaded = history.load()
        assert loaded["versions"][0]["scores"] == scores

    def test_qa_count_포함_기록(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        history.record_version(
            "v20250220", "test-v20250220", qa_count=42
        )

        loaded = history.load()
        assert loaded["versions"][0]["qa_count"] == 42

    def test_여러_버전_누적(self, make_config, tmp_path):
        config = make_config(paths={"output": str(tmp_path)})
        history = EvolveHistory(config)

        history.record_version("v20250220", "test-v20250220", promoted=True)
        history.record_version("v20250220-2", "test-v20250220-2", promoted=True)

        loaded = history.load()
        assert len(loaded["versions"]) == 2
        assert loaded["current"] == "v20250220-2"


# ---------------------------------------------------------------------------
# cleanup_old_versions
# ---------------------------------------------------------------------------


class TestEvolveHistoryCleanupOldVersions:

    def test_keep_zero_정리_안함(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"keep_previous_versions": 0},
        )
        history = EvolveHistory(config)
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220", "promoted": True},
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        removed = history.cleanup_old_versions()

        assert removed == []

    def test_버전_수_미만_정리_안함(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"keep_previous_versions": 3},
        )
        history = EvolveHistory(config)
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220", "promoted": True},
                {"version": "v20250220-2", "model_name": "test-v20250220-2", "promoted": True},
            ],
            "current": "v20250220-2",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        removed = history.cleanup_old_versions()

        assert removed == []

    def test_promoted_버전만_정리(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"keep_previous_versions": 1},
        )
        history = EvolveHistory(config)
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220", "promoted": True},
                {"version": "v20250220-2", "model_name": "test-v20250220-2", "promoted": False},
                {"version": "v20250220-3", "model_name": "test-v20250220-3", "promoted": True},
            ],
            "current": "v20250220-3",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(history, "_ollama_rm", return_value=True) as mock_rm:
            removed = history.cleanup_old_versions()

        assert "test-v20250220" in removed
        assert "test-v20250220-2" not in removed
        mock_rm.assert_called_once_with("test-v20250220")

    def test_current_버전_정리_안함(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"keep_previous_versions": 1},
        )
        history = EvolveHistory(config)
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220", "promoted": True},
                {"version": "v20250220-2", "model_name": "test-v20250220-2", "promoted": True},
            ],
            "current": "v20250220",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(history, "_ollama_rm", return_value=True) as mock_rm:
            removed = history.cleanup_old_versions()

        assert removed == []
        mock_rm.assert_not_called()

    def test_여러_버전_정리(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"keep_previous_versions": 1},
        )
        history = EvolveHistory(config)
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "model_name": "test-v20250220", "promoted": True},
                {"version": "v20250220-2", "model_name": "test-v20250220-2", "promoted": True},
                {"version": "v20250220-3", "model_name": "test-v20250220-3", "promoted": True},
            ],
            "current": "v20250220-3",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(history, "_ollama_rm", return_value=True) as mock_rm:
            removed = history.cleanup_old_versions()

        assert len(removed) == 2
        assert "test-v20250220" in removed
        assert "test-v20250220-2" in removed
        assert mock_rm.call_count == 2

    def test_모델명_없는_버전_스킵(self, make_config, tmp_path):
        config = make_config(
            paths={"output": str(tmp_path)},
            evolve={"keep_previous_versions": 1},
        )
        history = EvolveHistory(config)
        history_file = tmp_path / "evolve_history.json"
        data = {
            "versions": [
                {"version": "v20250220", "promoted": True},
                {"version": "v20250220-2", "model_name": "test-v20250220-2", "promoted": True},
            ],
            "current": "v20250220-2",
        }
        history_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(history, "_ollama_rm", return_value=True) as mock_rm:
            removed = history.cleanup_old_versions()

        assert removed == []
        mock_rm.assert_not_called()
