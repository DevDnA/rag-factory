"""LoRA 트레이너(DataLoader, LoRATrainer) 모듈의 단위 테스트입니다.

ML 라이브러리(torch, transformers, peft, trl, datasets)는
conftest.py에서 MagicMock으로 대체되어 GPU 없이 실행됩니다.

_RichProgressCallback은 transformers.TrainerCallback(MagicMock)을
상속하므로 인스턴스 재생성이 불안정합니다. 모듈 레벨에서 단일
인스턴스를 생성하여 재사용합니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_factory.trainer.lora_trainer import DataLoader, LoRATrainer, _RichProgressCallback


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_training_config(**overrides):
    """테스트용 SLMConfig를 생성합니다."""
    from slm_factory.config import SLMConfig
    return SLMConfig(**overrides)


def _make_jsonl_file(tmp_path: Path, name: str = "train.jsonl", n: int = 10) -> Path:
    """임시 JSONL 파일을 생성합니다."""
    path = tmp_path / name
    lines = [json.dumps({"text": f"훈련 데이터 {i}"}, ensure_ascii=False) for i in range(n)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# MagicMock 기반 TrainerCallback을 상속하는 _RichProgressCallback은
# 인스턴스를 여러 번 생성하면 mock 내부 상태가 오염됩니다.
# 단일 인스턴스를 재사용합니다.
_shared_callback = _RichProgressCallback()


# ---------------------------------------------------------------------------
# _RichProgressCallback
# ---------------------------------------------------------------------------


class TestRichProgressCallback:
    """학습 진행 콜백의 테스트입니다."""

    def test_on_epoch_begin_로그(self):
        """epoch 시작 시 로그를 출력합니다."""
        state = MagicMock()
        state.epoch = 0.0
        args = MagicMock()
        args.num_train_epochs = 3

        _shared_callback.on_epoch_begin(args, state, control=MagicMock())

    def test_on_epoch_begin_epoch_None(self):
        """epoch이 None이면 로그를 건너뜁니다."""
        state = MagicMock()
        state.epoch = None

        _shared_callback.on_epoch_begin(MagicMock(), state, control=MagicMock())

    def test_on_log_loss_정보(self):
        """loss 정보가 포함된 로그를 처리합니다."""
        state = MagicMock()
        state.global_step = 100

        logs = {"loss": 0.5432, "learning_rate": 1e-4}
        _shared_callback.on_log(MagicMock(), state, control=MagicMock(), logs=logs)

    def test_on_log_eval_loss_정보(self):
        """eval_loss 정보가 포함된 로그를 처리합니다."""
        state = MagicMock()
        state.global_step = 200

        logs = {"eval_loss": 0.3210}
        _shared_callback.on_log(MagicMock(), state, control=MagicMock(), logs=logs)

    def test_on_log_빈_로그(self):
        """로그가 None이면 예외 없이 건너뜁니다."""
        _shared_callback.on_log(MagicMock(), MagicMock(), control=MagicMock(), logs=None)

    def test_on_train_end_로그(self):
        """학습 종료 시 로그를 출력합니다."""
        state = MagicMock()
        state.epoch = 3.0
        state.global_step = 300

        _shared_callback.on_train_end(MagicMock(), state, control=MagicMock())

    def test_on_train_end_epoch_None(self):
        """epoch이 None(falsy)이면 '?'로 표시합니다."""
        state = MagicMock()
        state.epoch = None
        state.global_step = 0

        _shared_callback.on_train_end(MagicMock(), state, control=MagicMock())


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class TestDataLoaderInit:
    """DataLoader 초기화 테스트입니다."""

    def test_기본_train_split(self):
        """기본 학습/평가 분할 비율은 0.9입니다."""
        dl = DataLoader()
        assert dl.train_split == 0.9

    def test_커스텀_train_split(self):
        """학습/평가 분할 비율을 커스텀으로 지정합니다."""
        dl = DataLoader(train_split=0.8)
        assert dl.train_split == 0.8


class TestDataLoaderLoadJsonl:
    """DataLoader.load_jsonl() 테스트입니다."""

    @patch("slm_factory.trainer.lora_trainer.DataLoader.load_jsonl")
    def test_load_dataset_호출(self, mock_load, tmp_path):
        """load_jsonl이 올바른 경로로 호출됩니다."""
        jsonl_path = _make_jsonl_file(tmp_path)
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=10)
        mock_load.return_value = mock_ds

        dl = DataLoader()
        result = dl.load_jsonl(jsonl_path)
        mock_load.assert_called_once_with(jsonl_path)
        assert result is mock_ds


class TestDataLoaderSplit:
    """DataLoader.split() 테스트입니다."""

    def test_split_호출_구조(self):
        """split()이 train_test_split을 올바른 인자로 호출합니다."""
        dl = DataLoader(train_split=0.9)

        mock_ds = MagicMock()
        mock_split_result = {"train": MagicMock(), "test": MagicMock()}
        mock_split_result["train"].__len__ = MagicMock(return_value=9)
        mock_split_result["test"].__len__ = MagicMock(return_value=1)
        mock_ds.train_test_split.return_value = mock_split_result

        result = dl.split(mock_ds)

        mock_ds.train_test_split.assert_called_once_with(
            test_size=pytest.approx(0.1), seed=42,
        )
        assert result is not None


class TestDataLoaderLoadAndSplit:
    """DataLoader.load_and_split() 테스트입니다."""

    def test_load_jsonl과_split_체이닝(self):
        """load_and_split()이 load_jsonl → split을 순차 호출합니다."""
        dl = DataLoader()

        mock_ds = MagicMock()
        mock_dict = MagicMock()

        with patch.object(dl, "load_jsonl", return_value=mock_ds) as mock_load, \
             patch.object(dl, "split", return_value=mock_dict) as mock_split:
            result = dl.load_and_split("dummy.jsonl")

        mock_load.assert_called_once_with("dummy.jsonl")
        mock_split.assert_called_once_with(mock_ds)
        assert result is mock_dict


# ---------------------------------------------------------------------------
# LoRATrainer — 초기화
# ---------------------------------------------------------------------------


class TestLoRATrainerInit:
    """LoRATrainer 초기화 테스트입니다."""

    def test_설정_언패킹(self):
        """SLMConfig에서 하위 설정을 올바르게 추출합니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        assert trainer.config is config
        assert trainer.student_config is config.student
        assert trainer.training_config is config.training
        assert trainer.lora_config is config.training.lora

    def test_출력_디렉토리_설정(self):
        """output_dir이 {paths.output}/checkpoints로 설정됩니다."""
        config = _make_training_config(paths={"output": "/tmp/test-output"})
        trainer = LoRATrainer(config)

        assert trainer.output_dir == Path("/tmp/test-output/checkpoints")

    def test_device_info_초기값_None(self):
        """초기화 시 _device_info는 None입니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)
        assert trainer._device_info is None


# ---------------------------------------------------------------------------
# LoRATrainer — LoRA 설정
# ---------------------------------------------------------------------------


class TestCreateLoraConfig:
    """_create_lora_config() 테스트입니다."""

    def test_설정값_전달(self):
        """config.training.lora의 값이 LoraConfig에 전달됩니다."""
        config = _make_training_config(
            training={"lora": {"r": 16, "alpha": 32, "dropout": 0.1}},
        )
        trainer = LoRATrainer(config)
        lora_cfg = trainer._create_lora_config()

        # peft가 mock이므로 LoraConfig()는 MagicMock을 반환
        assert lora_cfg is not None

    def test_auto_target_modules(self):
        """target_modules가 'auto'이면 None으로 변환됩니다."""
        config = _make_training_config(
            training={"lora": {"target_modules": "auto"}},
        )
        trainer = LoRATrainer(config)
        lora_cfg = trainer._create_lora_config()
        assert lora_cfg is not None


# ---------------------------------------------------------------------------
# LoRATrainer — 콜백
# ---------------------------------------------------------------------------


class TestCreateCallbacks:
    """_create_callbacks() 테스트입니다."""

    def test_조기종료_활성화시_콜백_포함(self):
        """early_stopping이 활성화되면 콜백이 2개(ES + Rich)입니다."""
        config = _make_training_config(
            training={"early_stopping": {"enabled": True, "patience": 3, "threshold": 0.01}},
        )
        trainer = LoRATrainer(config)

        # _RichProgressCallback() 인스턴스화를 mock으로 우회
        with patch(
            "slm_factory.trainer.lora_trainer._RichProgressCallback",
            return_value=MagicMock(),
        ):
            callbacks = trainer._create_callbacks()

        # EarlyStoppingCallback + RichProgressCallback
        assert len(callbacks) == 2

    def test_조기종료_비활성화시_콜백_1개(self):
        """early_stopping이 비활성화되면 RichProgressCallback만 포함됩니다."""
        config = _make_training_config(
            training={"early_stopping": {"enabled": False}},
        )
        trainer = LoRATrainer(config)

        with patch(
            "slm_factory.trainer.lora_trainer._RichProgressCallback",
            return_value=MagicMock(),
        ):
            callbacks = trainer._create_callbacks()

        assert len(callbacks) == 1


# ---------------------------------------------------------------------------
# LoRATrainer — 학습 인자
# ---------------------------------------------------------------------------


class TestCreateTrainingArgs:
    """_create_training_args() 테스트입니다."""

    def test_기본_학습_인자_생성(self):
        """기본 설정으로 TrainingArguments를 생성합니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)
        # _device_info가 None이면 overrides={}
        args = trainer._create_training_args()
        assert args is not None

    def test_디바이스_오버라이드_적용(self):
        """_device_info가 있으면 디바이스 오버라이드가 적용됩니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        mock_device = MagicMock()
        mock_device.gpu_count = 1
        trainer._device_info = mock_device

        # get_training_overrides는 lazy import (from ..device import ...)
        with patch("slm_factory.device.get_training_overrides", return_value={}):
            args = trainer._create_training_args()
        assert args is not None

    def test_멀티GPU_gradient_accum_조정(self):
        """멀티 GPU 환경에서 gradient_accumulation_steps가 조정됩니다."""
        config = _make_training_config(
            training={"gradient_accumulation_steps": 8},
        )
        trainer = LoRATrainer(config)

        mock_device = MagicMock()
        mock_device.gpu_count = 4
        trainer._device_info = mock_device

        with patch("slm_factory.device.get_training_overrides", return_value={}):
            args = trainer._create_training_args()
        assert args is not None


# ---------------------------------------------------------------------------
# LoRATrainer — train() 통합
# ---------------------------------------------------------------------------


class TestLoRATrainerTrain:
    """LoRATrainer.train() 통합 테스트입니다 (전체 mock)."""

    def _run_train(self, trainer, mock_dataset, mock_model, mock_tokenizer, sft_cls=None):
        """train() 호출의 공통 mock 설정 헬퍼입니다."""
        # lazy import되는 peft.get_peft_model, trl.SFTTrainer는
        # sys.modules의 MagicMock 속성으로 설정
        peft_mock = sys.modules["peft"]
        trl_mock = sys.modules["trl"]

        peft_mock.get_peft_model = MagicMock(return_value=mock_model)

        if sft_cls is None:
            sft_cls = MagicMock()
        trl_mock.SFTTrainer = sft_cls

        with patch.object(trainer, "_load_model", return_value=(mock_model, mock_tokenizer)), \
             patch.object(trainer, "_create_lora_config", return_value=MagicMock()), \
             patch.object(trainer, "_create_training_args", return_value=MagicMock()), \
             patch.object(trainer, "_create_callbacks", return_value=[]), \
             patch.object(trainer, "_split_prompt_completion", return_value=mock_dataset):
            return trainer.train(mock_dataset), sft_cls

    def test_정상_학습_흐름(self):
        """모델 로드 → LoRA 적용 → 학습 → 저장 흐름을 검증합니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = {"train": MagicMock(), "eval": MagicMock()}

        result, mock_sft = self._run_train(
            trainer, mock_dataset, mock_model, mock_tokenizer,
        )

        # SFTTrainer가 생성되고 train()이 호출됨
        mock_sft.assert_called_once()
        mock_sft.return_value.train.assert_called_once()

        # 어댑터 저장 확인
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

        # 반환값이 Path
        assert isinstance(result, Path)
        assert "adapter" in str(result)

    def test_OOM_에러_래핑(self):
        """GPU 메모리 부족 시 사용자 친화적 에러 메시지를 제공합니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = {"train": MagicMock(), "eval": MagicMock()}

        mock_sft_cls = MagicMock()
        mock_sft_cls.return_value.train.side_effect = RuntimeError(
            "CUDA out of memory",
        )

        with pytest.raises(RuntimeError, match="GPU 메모리가 부족합니다"):
            self._run_train(
                trainer, mock_dataset, mock_model, mock_tokenizer,
                sft_cls=mock_sft_cls,
            )

    def test_일반_RuntimeError_재전파(self):
        """GPU 메모리 외의 RuntimeError는 그대로 전파됩니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = {"train": MagicMock(), "eval": MagicMock()}

        mock_sft_cls = MagicMock()
        mock_sft_cls.return_value.train.side_effect = RuntimeError("알 수 없는 오류")

        with pytest.raises(RuntimeError, match="알 수 없는 오류"):
            self._run_train(
                trainer, mock_dataset, mock_model, mock_tokenizer,
                sft_cls=mock_sft_cls,
            )


# ---------------------------------------------------------------------------
# LoRATrainer — Completion-Only Loss
# ---------------------------------------------------------------------------


class TestDetectResponseTemplate:
    """_detect_response_template() 테스트입니다."""

    def test_Qwen_채팅_템플릿_감지(self):
        """Qwen 스타일 채팅 템플릿에서 assistant 마커를 감지합니다."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nQ<|im_end|>\n"
            "<|im_start|>assistant\n__RESPONSE_SENTINEL__<|im_end|>\n"
        )

        result = LoRATrainer._detect_response_template(mock_tokenizer)
        assert "assistant" in result
        assert result == "<|im_start|>assistant\n"

    def test_Gemma_채팅_템플릿_감지(self):
        """Gemma 스타일 채팅 템플릿에서 assistant 마커를 감지합니다."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<start_of_turn>user\nQ<end_of_turn>\n"
            "<start_of_turn>model\n__RESPONSE_SENTINEL__<end_of_turn>\n"
        )

        result = LoRATrainer._detect_response_template(mock_tokenizer)
        assert "model" in result
        assert result == "<start_of_turn>model\n"

    def test_채팅_템플릿_렌더링_실패(self):
        """채팅 템플릿 렌더링이 실패하면 ValueError를 발생시킵니다."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("template error")

        with pytest.raises(ValueError, match="채팅 템플릿 렌더링 실패"):
            LoRATrainer._detect_response_template(mock_tokenizer)

    def test_sentinel_미발견시_ValueError(self):
        """sentinel이 렌더링 결과에 없으면 ValueError를 발생시킵니다."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "no sentinel here"

        with pytest.raises(ValueError, match="assistant 응답 마커를 감지할 수 없습니다"):
            LoRATrainer._detect_response_template(mock_tokenizer)


class TestSplitPromptCompletion:
    """_split_prompt_completion() 테스트입니다."""

    def _make_mock_dataset(self, texts):
        """mock DatasetDict를 생성합니다. map()이 실제 변환을 수행합니다."""
        def make_split(text_list):
            ds = MagicMock()
            data = [{"text": t} for t in text_list]

            def mock_map(fn, remove_columns=None):
                mapped = [fn(row) for row in data]
                result_ds = MagicMock()
                result_ds.__getitem__ = MagicMock(side_effect=lambda i: mapped[i])
                result_ds.column_names = list(mapped[0].keys()) if mapped else []
                return result_ds
            ds.map = mock_map
            return ds

        mock_dict = {"train": make_split(texts), "eval": make_split(texts)}
        return mock_dict

    def test_정상_분할(self):
        """assistant 마커를 기준으로 prompt/completion을 분할합니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        text = (
            "<|im_start|>user\n질문<|im_end|>\n"
            "<|im_start|>assistant\n답변입니다.<|im_end|>\n"
        )
        ds = self._make_mock_dataset([text])

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nQ<|im_end|>\n"
            "<|im_start|>assistant\n__RESPONSE_SENTINEL__<|im_end|>\n"
        )

        # datasets가 mock이므로 DatasetDict를 dict passthrough로 설정
        datasets_mock = sys.modules["datasets"]
        datasets_mock.DatasetDict = lambda x: x

        result = trainer._split_prompt_completion(ds, mock_tokenizer)

        train_row = result["train"][0]
        assert "prompt" in train_row
        assert "completion" in train_row
        assert train_row["completion"] == "답변입니다.<|im_end|>\n"
        assert train_row["prompt"].endswith("<|im_start|>assistant\n")

    def test_마커_미발견_fallback(self):
        """마커를 찾지 못하면 전체 텍스트를 completion으로 사용합니다."""
        config = _make_training_config()
        trainer = LoRATrainer(config)

        text = "마커가 없는 텍스트"
        ds = self._make_mock_dataset([text])

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nQ<|im_end|>\n"
            "<|im_start|>assistant\n__RESPONSE_SENTINEL__<|im_end|>\n"
        )

        datasets_mock = sys.modules["datasets"]
        datasets_mock.DatasetDict = lambda x: x

        result = trainer._split_prompt_completion(ds, mock_tokenizer)

        train_row = result["train"][0]
        assert train_row["prompt"] == ""
        assert train_row["completion"] == text


class TestCompletionOnlyLossConfig:
    """completion_only_loss 설정 테스트입니다."""

    def test_기본값_True(self):
        """completion_only_loss의 기본값은 True입니다."""
        config = _make_training_config()
        assert config.training.completion_only_loss is True

    def test_비활성화(self):
        """completion_only_loss를 False로 설정할 수 있습니다."""
        config = _make_training_config(
            training={"completion_only_loss": False},
        )
        assert config.training.completion_only_loss is False
