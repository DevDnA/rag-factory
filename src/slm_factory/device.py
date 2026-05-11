"""컴퓨팅 디바이스 자동 감지 — CUDA / MPS(Apple Silicon) / CPU.

시스템 환경을 자동으로 감지하여 최적의 디바이스, 데이터 타입,
학습 파라미터를 결정합니다. 학습(Train)과 내보내기(Export) 단계에서
하드웨어에 맞는 설정을 자동으로 적용합니다.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .utils import get_logger

logger = get_logger("device")


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    """감지된 컴퓨팅 디바이스 정보.

    Attributes
    ----------
    type : str
        디바이스 종류 (``"cuda"``, ``"mps"``, ``"cpu"``).
    name : str
        사람이 읽을 수 있는 디바이스 이름.
    dtype_name : str
        권장 torch dtype 이름 (``"bfloat16"``, ``"float16"``, ``"float32"``).
    bf16 : bool
        bfloat16 혼합 정밀도 지원 여부.
    fp16 : bool
        float16 혼합 정밀도 지원 여부.
    quantization_available : bool
        BitsAndBytes 양자화 사용 가능 여부.
    device_map : str | None
        HuggingFace ``device_map`` 권장값.
    gpu_count : int
        감지된 GPU 수 (CUDA만 해당, MPS/CPU는 1 또는 0).
    """

    type: str
    name: str
    dtype_name: str
    bf16: bool
    fp16: bool
    quantization_available: bool
    device_map: str | None
    gpu_count: int = 1

    @property
    def torch_dtype(self) -> Any:
        """권장 ``torch.dtype`` 객체를 반환합니다."""
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]

    @property
    def is_gpu(self) -> bool:
        """GPU(CUDA 또는 MPS)를 사용하는지 여부."""
        return self.type in ("cuda", "mps")

    @property
    def recommended_optimizer(self) -> str:
        """디바이스에 적합한 옵티마이저를 반환합니다.

        ``adamw_torch_fused``는 CUDA에서만 지원되므로, MPS와 CPU에서는
        ``adamw_torch``로 대체합니다.
        """
        if self.type == "cuda":
            return "adamw_torch_fused"
        return "adamw_torch"


def _check_bitsandbytes() -> bool:
    """bitsandbytes 라이브러리 사용 가능 여부를 확인합니다."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def detect_device() -> DeviceInfo:
    """현재 시스템의 최적 컴퓨팅 디바이스를 자동 감지합니다.

    감지 우선순위: CUDA → MPS (Apple Silicon) → CPU.

    반환값
    -------
    DeviceInfo
        감지된 디바이스 정보와 권장 설정.
    """
    import torch

    # ── CUDA (NVIDIA GPU) ─────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda or "unknown"
        bnb_available = _check_bitsandbytes()

        gpu_suffix = f" x{gpu_count}" if gpu_count > 1 else ""
        info = DeviceInfo(
            type="cuda",
            name=f"{device_name} ({vram_gb:.1f}GB, CUDA {cuda_version}){gpu_suffix}",
            dtype_name="bfloat16",
            bf16=True,
            fp16=True,
            quantization_available=bnb_available,
            device_map="auto",
            gpu_count=gpu_count,
        )
        logger.info("디바이스 감지: %s", info.name)
        if gpu_count > 1:
            logger.info("멀티 GPU 감지: %d개의 GPU를 사용합니다", gpu_count)
        return info

    # ── MPS (Apple Silicon) ───────────────────────────────────────
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import platform

        chip = platform.processor() or "Apple Silicon"

        info = DeviceInfo(
            type="mps",
            name=f"{chip} (MPS, Unified Memory)",
            dtype_name="float16",
            bf16=False,
            fp16=True,
            quantization_available=False,
            device_map=None,
            gpu_count=1,
        )
        logger.info("디바이스 감지: %s", info.name)
        return info

    # ── CPU 폴백 ──────────────────────────────────────────────────
    import platform

    cpu_name = platform.processor() or platform.machine()

    info = DeviceInfo(
        type="cpu",
        name=f"CPU ({cpu_name})",
        dtype_name="float32",
        bf16=False,
        fp16=False,
        quantization_available=False,
        device_map=None,
        gpu_count=0,
    )
    logger.info("디바이스 감지: %s (GPU 미감지)", info.name)
    return info


def get_training_overrides(device: DeviceInfo) -> dict[str, Any]:
    """디바이스에 맞게 TrainingArguments를 재정의할 값을 반환합니다.

    사용자 설정(``project.yaml``)의 ``training`` 섹션과 병합하여,
    디바이스 비호환 파라미터를 안전한 값으로 대체합니다.

    반환값
    -------
    dict
        ``TrainingArguments``에 전달할 재정의 값.
    """
    overrides: dict[str, Any] = {
        "bf16": device.bf16,
        "fp16": device.fp16 and not device.bf16,
        "optim": device.recommended_optimizer,
    }

    if device.type == "mps":
        overrides["gradient_checkpointing"] = False

    if device.gpu_count > 1:
        overrides["ddp_find_unused_parameters"] = False

    return overrides


def print_device_summary(device: DeviceInfo) -> None:
    """감지된 디바이스 요약을 Rich 패널로 출력합니다.

    CLI의 ``check`` 명령에서 호출됩니다.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(
        title="컴퓨팅 디바이스",
        show_header=True,
        title_style="bold",
    )
    table.add_column("항목", style="cyan")
    table.add_column("값", style="bold")

    # 디바이스 타입 + 이름
    type_display = {
        "cuda": "[green]NVIDIA GPU (CUDA)[/green]",
        "mps": "[green]Apple Silicon GPU (MPS)[/green]",
        "cpu": "[yellow]CPU (GPU 미감지)[/yellow]",
    }
    table.add_row("디바이스", type_display.get(device.type, device.type))
    table.add_row("이름", device.name)
    if device.gpu_count > 1:
        table.add_row("GPU 수", f"[green]{device.gpu_count}개[/green] (멀티 GPU)")
    elif device.gpu_count == 1 and device.type != "cpu":
        table.add_row("GPU 수", "1개")

    # 데이터 타입
    dtype_display = {
        "bfloat16": "[green]bfloat16[/green]",
        "float16": "[green]float16[/green]",
        "float32": "[yellow]float32[/yellow]",
    }
    table.add_row(
        "학습 정밀도", dtype_display.get(device.dtype_name, device.dtype_name)
    )

    # 양자화
    if device.quantization_available:
        table.add_row("4bit 양자화", "[green]사용 가능[/green] (bitsandbytes)")
    elif device.type == "mps":
        table.add_row("4bit 양자화", "[dim]미지원[/dim] (Unified Memory로 대체)")
    else:
        table.add_row("4bit 양자화", "[yellow]미설치[/yellow] (uv sync --extra cuda)")

    # 옵티마이저
    table.add_row("옵티마이저", device.recommended_optimizer)

    console.print(table)

    # MPS 특화 안내
    if device.type == "mps":
        console.print(
            "  [dim]ℹ Apple Silicon의 Unified Memory를 사용합니다. "
            "시스템 RAM 전체를 GPU가 공유하므로 양자화 없이도 비교적 큰 모델을 "
            "로드할 수 있습니다.[/dim]"
        )

    # CPU 경고
    if device.type == "cpu":
        console.print(
            "  [yellow]⚠ GPU가 감지되지 않았습니다. CPU로 학습하면 "
            "매우 느릴 수 있습니다.[/yellow]"
        )
