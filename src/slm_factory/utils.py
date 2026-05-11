"""Rich를 사용한 구조화된 로깅 및 비동기 유틸리티."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable

    import httpx
    from rich.progress import Progress

T = TypeVar("T")

console = Console()

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """루트 slm-factory 로거를 구성하고 반환합니다."""
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(console=console, rich_tracebacks=True, show_path=False)
            ],
        )
        _configured = True
    return logging.getLogger("slm_factory")


def get_logger(name: str) -> logging.Logger:
    """slm_factory 네임스페이스 아래의 자식 로거를 가져옵니다."""
    return logging.getLogger(f"slm_factory.{name}")


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """파일의 해시값을 계산합니다."""
    import hashlib

    h = hashlib.new(algorithm)
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def run_bounded(
    semaphore: asyncio.Semaphore,
    coro: Awaitable[T],
    progress: Progress,
    task_id: int,
) -> T:
    """세마포어 제한 하에 코루틴을 실행하고 진행률을 갱신합니다."""
    async with semaphore:
        result = await coro
        progress.advance(task_id)
        return result


async def ollama_generate(
    client: httpx.AsyncClient,
    api_base: str,
    model_name: str,
    question: str,
    timeout: float,
    *,
    max_tokens: int = 512,
    max_retries: int = 3,
    format: str | None = None,
    think: bool | None = None,
) -> str:
    """Ollama /api/generate 엔드포인트로 답변을 생성합니다.

    일시적 오류(HTTP 5xx, 타임아웃, 연결 오류)에 대해
    지수 백오프로 재시도합니다.

    매개변수
    ----------
    format:
        응답 형식입니다. ``"json"``이면 JSON 형식으로 응답을 제한합니다.
    think:
        사고 과정(thinking/reasoning) 활성화 여부입니다.
        ``False``이면 사고 과정 없이 즉시 응답합니다.
    """
    import asyncio as _asyncio

    body: dict = {
        "model": model_name,
        "prompt": question,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if format is not None:
        body["format"] = format
    if think is not None:
        body["think"] = think

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{api_base}/api/generate",
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2**attempt
                get_logger("utils").debug(
                    "ollama_generate 재시도 %d/%d (%s): %.1f초 대기",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                await _asyncio.sleep(wait)
    if last_exc is None:
        raise RuntimeError(
            "ollama_generate: max_retries에 도달했으나 예외가 기록되지 않았습니다"
        )
    raise last_exc


def run_async(coro: Awaitable[T]) -> T:
    """이벤트 루프 유무에 관계없이 코루틴을 실행합니다.

    일반 환경에서는 ``asyncio.run()``을 사용하고,
    Jupyter 등 이미 이벤트 루프가 실행 중인 환경에서는
    ``nest_asyncio``를 적용하여 중첩 실행을 허용합니다.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 실행 중인 루프 없음 — 일반적인 경우
        return asyncio.run(coro)

    # 이미 이벤트 루프가 실행 중 (Jupyter 등)
    try:
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(coro)
    except ImportError:
        raise RuntimeError(
            "이미 실행 중인 이벤트 루프에서 호출되었습니다. "
            "Jupyter 등에서 사용하려면 nest_asyncio를 설치하세요: "
            "uv pip install nest_asyncio"
        )
