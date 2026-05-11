"""Ollama 교사 백엔드 — 로컬 Ollama REST API를 호출합니다."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from ..config import TeacherConfig
from ..utils import get_logger
from .base import BaseTeacher

logger = get_logger("teacher.ollama")

# Retry 설정
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # 초

import re

# 사고 과정 추론 태그 — 응답 후처리에서 제거합니다.
# Ollama API의 stop 파라미터로 전달하면 Qwen3 등 thinking 모델이
# <think> 태그를 생성하자마자 출력이 중단되는 문제가 발생합니다.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_thinking_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


class OllamaTeacher(BaseTeacher):
    """Ollama 인스턴스와 통신하는 교사 구현입니다.

    매개변수
    ----------
    config:
        ``backend="ollama"``인 :class:`TeacherConfig`.
    """

    def __init__(self, config: TeacherConfig) -> None:
        self.model: str = config.model
        self.api_base: str = config.api_base.rstrip("/")
        self.temperature: float = config.temperature
        self.timeout: int = config.timeout
        logger.info(
            "OllamaTeacher initialised  model=%s  api_base=%s",
            self.model,
            self.api_base,
        )

    # ------------------------------------------------------------------
    # BaseTeacher 인터페이스
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """*prompt*를 Ollama ``/api/generate`` 엔드포인트로 전송합니다.

        키워드 인자는 ``model``, ``temperature``를 오버라이드하고
        ``format``을 추가할 수 있습니다(예: ``"json"``).

        ``stream=True``를 전달하면 NDJSON 스트리밍 모드를 사용하여
        토큰이 생성될 때마다 점진적으로 수신합니다. 긴 응답에서
        타임아웃을 방지하는 데 유용합니다.
        """
        url = f"{self.api_base}/api/generate"

        model: str = kwargs.pop("model", self.model)
        temperature: float = kwargs.pop("temperature", self.temperature)
        fmt: str | None = kwargs.pop("format", None)
        stream: bool = kwargs.pop("stream", False)
        think: bool | None = kwargs.pop("think", None)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think

        logger.debug(
            "POST %s  model=%s  temp=%.2f  stream=%s",
            url, model, temperature, stream,
        )

        if stream:
            return self._generate_stream(url, payload, model)

        resp: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = httpx.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 요청 타임아웃 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama 요청이 {_MAX_RETRIES}회 시도 후에도 타임아웃되었습니다 "
                    f"(model={model}, url={url}, timeout={self.timeout}s). "
                    f"서버 상태를 확인하거나 teacher.timeout 값을 늘려보세요."
                ) from e
            except httpx.ConnectError as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 연결 실패 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama({self.api_base})에 {_MAX_RETRIES}회 시도 후에도 연결할 수 없습니다. "
                    "서버가 실행 중인지 확인하세요: ollama serve"
                ) from e
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ) from exc
            except httpx.HTTPError as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 요청 실패 (시도 %d/%d, %s), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, type(e).__name__, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama 요청이 {_MAX_RETRIES}회 시도 후에도 실패했습니다: {e}"
                ) from e

        if resp is None:
            raise RuntimeError("Ollama 요청에 대한 응답을 받지 못했습니다")
        data: dict[str, Any] = resp.json()
        text: str = _clean_thinking_tags(data.get("response", ""))

        if not text:
            thinking = data.get("thinking", "")
            if thinking:
                text = _clean_thinking_tags(thinking)

        if not text:
            logger.warning("Ollama returned an empty response for model=%s", model)

        return text

    def _generate_stream(
        self,
        url: str,
        payload: dict[str, Any],
        model: str,
    ) -> str:
        """NDJSON 스트리밍으로 응답을 수신합니다.

        Ollama ``stream: true``는 한 줄당 하나의 JSON 객체를 반환합니다:
        ``{"response": "token", "done": false}``.
        마지막 줄은 ``{"done": true, ...stats}``입니다.
        """
        chunks: list[str] = []
        thinking_chunks: list[str] = []

        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.stream(
                    "POST", url, json=payload, timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("스트리밍 라인 JSON 파싱 실패: %s", line[:100])
                            continue
                        token = obj.get("response", "")
                        if token:
                            chunks.append(token)
                        thinking_token = obj.get("thinking", "")
                        if thinking_token:
                            thinking_chunks.append(thinking_token)
                        if obj.get("done", False):
                            break
                break
            except httpx.TimeoutException as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 스트리밍 타임아웃 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    chunks.clear()
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama 스트리밍이 {_MAX_RETRIES}회 시도 후에도 타임아웃되었습니다 "
                    f"(model={model}, url={url}, timeout={self.timeout}s). "
                    f"서버 상태를 확인하거나 teacher.timeout 값을 늘려보세요."
                ) from e
            except httpx.ConnectError as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 연결 실패 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    chunks.clear()
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama({url})에 {_MAX_RETRIES}회 시도 후에도 연결할 수 없습니다. "
                    "서버가 실행 중인지 확인하세요: ollama serve"
                ) from e
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ) from exc
            except httpx.HTTPError as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 스트리밍 실패 (시도 %d/%d, %s), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, type(e).__name__, delay,
                    )
                    chunks.clear()
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama 스트리밍이 {_MAX_RETRIES}회 시도 후에도 실패했습니다: {e}"
                ) from e

        text = _clean_thinking_tags("".join(chunks))
        if not text and thinking_chunks:
            text = _clean_thinking_tags("".join(thinking_chunks))
            logger.debug(
                "Using thinking tokens as response (thinking-model streaming, model=%s)",
                model,
            )
        if not text:
            logger.warning("Ollama returned an empty streaming response for model=%s", model)
        return text

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """비동기 변형 — 동시 요청을 위해 ``httpx.AsyncClient``를 사용합니다."""
        url = f"{self.api_base}/api/generate"

        model: str = kwargs.pop("model", self.model)
        temperature: float = kwargs.pop("temperature", self.temperature)
        fmt: str | None = kwargs.pop("format", None)
        stream: bool = kwargs.pop("stream", False)
        think: bool | None = kwargs.pop("think", None)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think

        if stream:
            return await self._agenerate_stream(url, payload, model)

        resp: httpx.Response | None = None
        async with httpx.AsyncClient() as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.post(url, json=payload, timeout=self.timeout)
                    resp.raise_for_status()
                    break
                except httpx.TimeoutException as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Ollama 요청 타임아웃 (시도 %d/%d), %.0f초 후 재시도...",
                            attempt + 1, _MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Ollama 요청이 {_MAX_RETRIES}회 시도 후에도 타임아웃되었습니다 "
                        f"(model={model}, url={url}, timeout={self.timeout}s). "
                        f"서버 상태를 확인하거나 teacher.timeout 값을 늘려보세요."
                    ) from e
                except httpx.ConnectError as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Ollama 연결 실패 (시도 %d/%d), %.0f초 후 재시도...",
                            attempt + 1, _MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Ollama({self.api_base})에 {_MAX_RETRIES}회 시도 후에도 연결할 수 없습니다. "
                        "서버가 실행 중인지 확인하세요: ollama serve"
                    ) from e
                except httpx.HTTPStatusError as exc:
                    raise RuntimeError(
                        f"Ollama HTTP {exc.response.status_code}: "
                        f"{exc.response.text[:200]}"
                    ) from exc
                except httpx.HTTPError as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Ollama 요청 실패 (시도 %d/%d, %s), %.0f초 후 재시도...",
                            attempt + 1, _MAX_RETRIES, type(e).__name__, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Ollama 요청이 {_MAX_RETRIES}회 시도 후에도 실패했습니다: {e}"
                    ) from e

        if resp is None:
            raise RuntimeError("Ollama 요청에 대한 응답을 받지 못했습니다")
        data: dict[str, Any] = resp.json()
        text: str = _clean_thinking_tags(data.get("response", ""))

        if not text:
            thinking = data.get("thinking", "")
            if thinking:
                text = _clean_thinking_tags(thinking)

        if not text:
            logger.warning("Ollama returned an empty response for model=%s", model)

        return text

    async def _agenerate_stream(
        self,
        url: str,
        payload: dict[str, Any],
        model: str,
    ) -> str:
        """비동기 NDJSON 스트리밍으로 응답을 수신합니다."""
        chunks: list[str] = []
        thinking_chunks: list[str] = []

        async with httpx.AsyncClient() as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    async with client.stream(
                        "POST", url, json=payload, timeout=self.timeout,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                logger.debug("스트리밍 라인 JSON 파싱 실패: %s", line[:100])
                                continue
                            token = obj.get("response", "")
                            if token:
                                chunks.append(token)
                            thinking_token = obj.get("thinking", "")
                            if thinking_token:
                                thinking_chunks.append(thinking_token)
                            if obj.get("done", False):
                                break
                    break
                except httpx.TimeoutException as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Ollama 스트리밍 타임아웃 (시도 %d/%d), %.0f초 후 재시도...",
                            attempt + 1, _MAX_RETRIES, delay,
                        )
                        chunks.clear()
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Ollama 스트리밍이 {_MAX_RETRIES}회 시도 후에도 타임아웃되었습니다 "
                        f"(model={model}, url={url}, timeout={self.timeout}s). "
                        f"서버 상태를 확인하거나 teacher.timeout 값을 늘려보세요."
                    ) from e
                except httpx.ConnectError as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Ollama 연결 실패 (시도 %d/%d), %.0f초 후 재시도...",
                            attempt + 1, _MAX_RETRIES, delay,
                        )
                        chunks.clear()
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Ollama({url})에 {_MAX_RETRIES}회 시도 후에도 연결할 수 없습니다. "
                        "서버가 실행 중인지 확인하세요: ollama serve"
                    ) from e
                except httpx.HTTPStatusError as exc:
                    raise RuntimeError(
                        f"Ollama HTTP {exc.response.status_code}: "
                        f"{exc.response.text[:200]}"
                    ) from exc
                except httpx.HTTPError as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Ollama 스트리밍 실패 (시도 %d/%d, %s), %.0f초 후 재시도...",
                            attempt + 1, _MAX_RETRIES, type(e).__name__, delay,
                        )
                        chunks.clear()
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Ollama 스트리밍이 {_MAX_RETRIES}회 시도 후에도 실패했습니다: {e}"
                    ) from e

        text = _clean_thinking_tags("".join(chunks))
        if not text and thinking_chunks:
            text = _clean_thinking_tags("".join(thinking_chunks))
            logger.debug(
                "Using thinking tokens as response (thinking-model streaming, model=%s)",
                model,
            )
        if not text:
            logger.warning("Ollama returned an empty streaming response for model=%s", model)
        return text

    def health_check(self) -> bool:
        """``/api/tags``를 핑하여 Ollama에 도달할 수 있는지 확인합니다."""
        url = f"{self.api_base}/api/tags"
        try:
            resp = httpx.get(url, timeout=10)
            resp.raise_for_status()
            logger.debug("Ollama health-check OK")
            return True
        except (httpx.HTTPError, httpx.ConnectError):
            logger.warning("Ollama health-check FAILED at %s", url)
            return False
