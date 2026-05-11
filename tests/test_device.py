"""디바이스 감지 모듈 테스트 — DeviceInfo, detect_device, get_training_overrides를 검증합니다."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from slm_factory.device import DeviceInfo, get_training_overrides


# ---------------------------------------------------------------------------
# DeviceInfo 데이터클래스 테스트
# ---------------------------------------------------------------------------


class TestDeviceInfo:
    """DeviceInfo 필드 및 프로퍼티를 검증합니다."""

    def test_기본_gpu_count(self):
        """gpu_count 기본값이 1인지 확인합니다."""
        info = DeviceInfo(
            type="cuda",
            name="test",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=True,
            device_map="auto",
        )
        assert info.gpu_count == 1

    def test_멀티_GPU_gpu_count(self):
        """gpu_count를 명시적으로 설정할 수 있는지 확인합니다."""
        info = DeviceInfo(
            type="cuda",
            name="test",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=True,
            device_map="auto",
            gpu_count=4,
        )
        assert info.gpu_count == 4

    def test_CPU_gpu_count_0(self):
        """CPU 디바이스의 gpu_count가 0일 수 있는지 확인합니다."""
        info = DeviceInfo(
            type="cpu",
            name="CPU",
            dtype_name="float32",
            bf16=False,
            fp16=False,
            quantization_available=False,
            device_map=None,
            gpu_count=0,
        )
        assert info.gpu_count == 0

    def test_is_gpu_cuda(self):
        """CUDA 디바이스에서 is_gpu가 True입니다."""
        info = DeviceInfo(
            type="cuda",
            name="test",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=True,
            device_map="auto",
            gpu_count=2,
        )
        assert info.is_gpu is True

    def test_is_gpu_cpu(self):
        """CPU 디바이스에서 is_gpu가 False입니다."""
        info = DeviceInfo(
            type="cpu",
            name="CPU",
            dtype_name="float32",
            bf16=False,
            fp16=False,
            quantization_available=False,
            device_map=None,
            gpu_count=0,
        )
        assert info.is_gpu is False

    def test_recommended_optimizer_cuda(self):
        """CUDA에서 adamw_torch_fused를 권장합니다."""
        info = DeviceInfo(
            type="cuda",
            name="test",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=True,
            device_map="auto",
            gpu_count=1,
        )
        assert info.recommended_optimizer == "adamw_torch_fused"


# ---------------------------------------------------------------------------
# detect_device 테스트
# ---------------------------------------------------------------------------


class TestDetectDevice:
    """detect_device 함수의 디바이스 감지를 검증합니다."""

    def test_CUDA_멀티GPU_감지(self):
        """CUDA 환경에서 gpu_count가 올바르게 감지되는지 확인합니다."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        props = MagicMock()
        props.total_memory = 80 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.version.cuda = "12.1"

        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("slm_factory.device._check_bitsandbytes", return_value=True),
        ):
            from importlib import reload
            import slm_factory.device as dev_mod

            reload(dev_mod)
            info = dev_mod.detect_device()

        assert info.gpu_count == 4
        assert info.type == "cuda"
        assert "x4" in info.name

    def test_CUDA_단일GPU_감지(self):
        """단일 CUDA GPU에서 gpu_count=1이고 이름에 'x' 없는지 확인합니다."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        props = MagicMock()
        props.total_memory = 24 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.version.cuda = "12.4"

        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("slm_factory.device._check_bitsandbytes", return_value=False),
        ):
            from importlib import reload
            import slm_factory.device as dev_mod

            reload(dev_mod)
            info = dev_mod.detect_device()

        assert info.gpu_count == 1
        assert "x1" not in info.name

    def test_CPU_gpu_count_0(self):
        """CPU 폴백 시 gpu_count=0인지 확인합니다."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from importlib import reload
            import slm_factory.device as dev_mod

            reload(dev_mod)
            info = dev_mod.detect_device()

        assert info.gpu_count == 0
        assert info.type == "cpu"


# ---------------------------------------------------------------------------
# get_training_overrides 테스트
# ---------------------------------------------------------------------------


class TestGetTrainingOverrides:
    """get_training_overrides의 멀티 GPU 관련 오버라이드를 검증합니다."""

    def test_멀티GPU_ddp_설정(self):
        """멀티 GPU에서 ddp_find_unused_parameters=False가 설정되는지 확인합니다."""
        info = DeviceInfo(
            type="cuda",
            name="test",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=True,
            device_map="auto",
            gpu_count=4,
        )
        overrides = get_training_overrides(info)
        assert overrides["ddp_find_unused_parameters"] is False

    def test_단일GPU_ddp_미설정(self):
        """단일 GPU에서 ddp_find_unused_parameters가 설정되지 않는지 확인합니다."""
        info = DeviceInfo(
            type="cuda",
            name="test",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=True,
            device_map="auto",
            gpu_count=1,
        )
        overrides = get_training_overrides(info)
        assert "ddp_find_unused_parameters" not in overrides

    def test_MPS_gradient_checkpointing_비활성(self):
        """MPS에서 gradient_checkpointing이 False인지 확인합니다."""
        info = DeviceInfo(
            type="mps",
            name="M1",
            dtype_name="float16",
            bf16=False,
            fp16=True,
            quantization_available=False,
            device_map=None,
            gpu_count=1,
        )
        overrides = get_training_overrides(info)
        assert overrides["gradient_checkpointing"] is False

    def test_CPU_기본_오버라이드(self):
        """CPU에서 기본 오버라이드(bf16=False, fp16=False)를 확인합니다."""
        info = DeviceInfo(
            type="cpu",
            name="CPU",
            dtype_name="float32",
            bf16=False,
            fp16=False,
            quantization_available=False,
            device_map=None,
            gpu_count=0,
        )
        overrides = get_training_overrides(info)
        assert overrides["bf16"] is False
        assert overrides["fp16"] is False
