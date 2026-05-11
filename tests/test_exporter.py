"""내보내기(exporter) 모듈의 통합 테스트입니다.

HFExporter와 OllamaExporter의 내보내기, 병합, Modelfile 생성 기능을 검증합니다.
ML 라이브러리(peft, transformers, torch)와 subprocess는 mock으로 대체합니다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slm_factory.exporter.hf_export import HFExporter
from slm_factory.exporter.ollama_export import OllamaExporter


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_hf_exporter(make_config, merge_lora=True):
    """테스트용 HFExporter를 생성합니다."""
    config = make_config(
        student={"model": "test-model"},
        export={"merge_lora": merge_lora, "output_format": "safetensors"},
    )
    return HFExporter(config)


def _make_ollama_exporter(make_config):
    """테스트용 OllamaExporter를 생성합니다."""
    config = make_config(
        export={
            "ollama": {
                "enabled": True,
                "model_name": "test-ollama-model",
                "system_prompt": "도움이 되는 어시스턴트입니다.",
                "parameters": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 4096},
            }
        }
    )
    return OllamaExporter(config)


# ---------------------------------------------------------------------------
# HFExporter.__init__
# ---------------------------------------------------------------------------


class TestHFExporterInit:
    """HFExporter 초기화 테스트입니다."""

    def test_필드_설정(self, make_config):
        """student_model, merge_lora, output_format 필드가 올바르게 설정되는지 확인합니다."""
        exporter = _make_hf_exporter(make_config, merge_lora=True)

        assert exporter.student_model == "test-model"
        assert exporter.merge_lora is True
        assert exporter.output_format == "safetensors"


# ---------------------------------------------------------------------------
# HFExporter.export
# ---------------------------------------------------------------------------


class TestHFExporterExport:
    """HFExporter.export 메서드의 테스트입니다."""

    def test_merge_lora_True이면_merge_and_save_호출(self, make_config, mocker, tmp_path):
        """merge_lora=True일 때 merge_and_save가 호출되는지 확인합니다."""
        exporter = _make_hf_exporter(make_config, merge_lora=True)
        expected_path = tmp_path / "merged"

        mocker.patch.object(exporter, "merge_and_save", return_value=expected_path)

        result = exporter.export(tmp_path / "adapter")

        exporter.merge_and_save.assert_called_once()
        assert result == expected_path

    def test_merge_lora_False이면_save_adapter_only_호출(self, make_config, mocker, tmp_path):
        """merge_lora=False일 때 save_adapter_only가 호출되는지 확인합니다."""
        exporter = _make_hf_exporter(make_config, merge_lora=False)
        expected_path = tmp_path / "adapter_copy"

        mocker.patch.object(exporter, "save_adapter_only", return_value=expected_path)

        result = exporter.export(tmp_path / "adapter")

        exporter.save_adapter_only.assert_called_once()
        assert result == expected_path


# ---------------------------------------------------------------------------
# HFExporter.save_adapter_only
# ---------------------------------------------------------------------------


class TestHFExporterSaveAdapterOnly:
    """HFExporter.save_adapter_only 메서드의 테스트입니다."""

    def test_shutil_copytree_호출(self, make_config, tmp_path, mocker):
        """shutil.copytree가 올바르게 호출되는지 확인합니다."""
        exporter = _make_hf_exporter(make_config, merge_lora=False)

        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_model.bin").touch()

        output_dir = tmp_path / "output"
        mock_copytree = mocker.patch("shutil.copytree", return_value=str(output_dir))

        result = exporter.save_adapter_only(adapter_path, output_dir=output_dir)

        mock_copytree.assert_called_once()


# ---------------------------------------------------------------------------
# OllamaExporter.__init__
# ---------------------------------------------------------------------------


class TestOllamaExporterInit:
    """OllamaExporter 초기화 테스트입니다."""

    def test_필드_설정(self, make_config):
        """model_name, system_prompt, parameters 필드가 올바르게 설정되는지 확인합니다."""
        exporter = _make_ollama_exporter(make_config)

        assert exporter.model_name == "test-ollama-model"
        assert exporter.system_prompt == "도움이 되는 어시스턴트입니다."
        assert exporter.parameters["temperature"] == 0.7
        assert exporter.parameters["top_p"] == 0.9
        assert exporter.parameters["num_ctx"] == 4096


# ---------------------------------------------------------------------------
# OllamaExporter.generate_modelfile
# ---------------------------------------------------------------------------


class TestOllamaExporterGenerateModelfile:
    """OllamaExporter.generate_modelfile 메서드의 테스트입니다."""

    def test_Modelfile_내용_확인(self, make_config, tmp_path):
        """Modelfile에 FROM, SYSTEM, PARAMETER가 포함되는지 확인합니다."""
        exporter = _make_ollama_exporter(make_config)

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_path = tmp_path / "Modelfile"

        result = exporter.generate_modelfile(model_dir, output_path=output_path)

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "FROM" in content
        assert "SYSTEM" in content
        assert "PARAMETER" in content
        assert "도움이 되는 어시스턴트입니다." in content


# ---------------------------------------------------------------------------
# OllamaExporter.create_model
# ---------------------------------------------------------------------------


class TestOllamaExporterCreateModel:
    """OllamaExporter.create_model 메서드의 테스트입니다."""

    def test_정상_실행_True(self, make_config, tmp_path, mocker):
        """subprocess.run이 정상 실행되면 True를 반환하는지 확인합니다."""
        exporter = _make_ollama_exporter(make_config)

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.touch()

        result = exporter.create_model(modelfile_path)

        assert result is True
        mock_run.assert_called_once()

    def test_FileNotFoundError시_False(self, make_config, tmp_path, mocker):
        """ollama 명령어가 없어서 FileNotFoundError 발생 시 False를 반환하는지 확인합니다."""
        exporter = _make_ollama_exporter(make_config)

        mocker.patch("subprocess.run", side_effect=FileNotFoundError("ollama not found"))

        modelfile_path = tmp_path / "Modelfile"
        modelfile_path.touch()

        result = exporter.create_model(modelfile_path)

        assert result is False


# ---------------------------------------------------------------------------
# OllamaExporter.export
# ---------------------------------------------------------------------------


class TestOllamaExporterExport:
    """OllamaExporter.export 메서드의 테스트입니다."""

    def test_ollama_사용_불가시_Modelfile만_생성(self, make_config, tmp_path, mocker):
        """ollama가 설치되어 있지 않으면 Modelfile만 생성하고 create_model을 건너뛰는지 확인합니다."""
        exporter = _make_ollama_exporter(make_config)

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # generate_modelfile은 정상 동작
        modelfile_path = tmp_path / "Modelfile"
        mocker.patch.object(exporter, "generate_modelfile", return_value=modelfile_path)

        # create_model은 False 반환 (ollama 없음)
        mocker.patch.object(exporter, "create_model", return_value=False)

        result = exporter.export(model_dir, output_dir=tmp_path)

        exporter.generate_modelfile.assert_called_once()
        assert isinstance(result, Path)
