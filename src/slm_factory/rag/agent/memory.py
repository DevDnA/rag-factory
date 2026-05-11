"""ConversationCompressor — 긴 대화의 오래된 턴을 LLM 요약으로 압축.

긴 대화는 ``format_history``의 토큰 예산을 초과하거나 synthesis 프롬프트를
과도하게 부풀립니다. 오래된 턴들을 한 줄 요약으로 압축하면 최근 턴의 맥락은
그대로 보존하면서 token 소비를 줄일 수 있습니다.

설계 원칙
---------
- **Never-raise**: LLM 실패 시 원본 메시지를 그대로 두고 ``None`` 반환 → 호출 측이
  압축 스킵 결정.
- **Idempotent**: 이미 압축된 세션을 재압축해도 동일 결과 (오래된 턴 개수 기준).
"""

from __future__ import annotations

import re
from typing import Any

from ...utils import get_logger
from .prompts import MEMORY_COMPRESSION_PROMPT
from .session import Message

logger = get_logger("rag.agent.memory")


class ConversationCompressor:
    """대화 요약 생성기.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate`` 호출용.
    ollama_model:
        요약 모델. 작은 모델이 적합.
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        요청 타임아웃(초).
    max_tokens:
        생성 최대 토큰.
    target_chars:
        요약 결과 목표 길이(문자).
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 30.0,
        max_tokens: int = 400,
        target_chars: int = 500,
        *,
        keep_alive: str = "5m",
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._target_chars = target_chars
        self._keep_alive = keep_alive

    async def summarize(self, messages: list[Message]) -> str | None:
        """메시지 목록을 한 단락 요약으로 반환 — 실패 시 ``None``."""
        if not messages:
            return None

        transcript = "\n".join(
            f"{self._format_role(m.role)}: {m.content}" for m in messages if m.content
        )
        if not transcript.strip():
            return None

        try:
            raw = await self._generate(transcript)
        except Exception as exc:
            logger.warning("Memory compressor LLM 실패: %s", exc)
            return None

        cleaned = self._clean(raw)
        if not cleaned:
            return None
        return cleaned[: self._target_chars * 3]  # 여유 한도

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    @staticmethod
    def _format_role(role: str) -> str:
        if role == "user":
            return "사용자"
        if role == "assistant":
            return "어시스턴트"
        return role

    async def _generate(self, transcript: str) -> str:
        prompt = MEMORY_COMPRESSION_PROMPT.format(
            transcript=transcript,
            target_chars=self._target_chars,
        )
        response = await self._http_client.post(
            f"{self._api_base}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": self._keep_alive,
                "options": {"num_predict": self._max_tokens},
            },
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "") or data.get("thinking", "")

    @staticmethod
    def _clean(raw: str) -> str:
        """<think> 태그 제거 + 공백 정규화."""
        if not raw:
            return ""
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        return re.sub(r"\s+", " ", text).strip()


__all__ = ["ConversationCompressor"]
