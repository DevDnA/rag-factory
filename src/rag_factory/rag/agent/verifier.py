"""수집된 컨텍스트의 충분성(sufficiency) 판정기.

``Verifier``는 지금까지 도구 호출로 수집한 정보가 질문에 답하기에 충분한지를
LLM으로 평가합니다. 부족하면 추가 검색을 제안합니다.

설계 원칙
---------
- **절대 raise하지 않음**: LLM 실패·파싱 실패 시 ``sufficient=True``로
  반환해서 orchestrator가 무한 루프에 빠지지 않도록 합니다. 확실하지 않으면
  "충분하다고 가정하고 답변을 시도"가 안전한 기본값입니다.
- **Repair 횟수 제한**: Verifier 자체는 단일 판정만 수행합니다. 반복 제한은
  호출 측(orchestrator)에서 ``verifier_max_repairs`` 설정으로 통제합니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ...utils import get_logger
from .prompts import SUFFICIENCY_PROMPT

logger = get_logger("rag.agent.verifier")

# 컨텍스트 길이 제한 — 프롬프트 폭발 방지.
_CONTEXT_CHAR_LIMIT = 2000


@dataclass
class VerifierDecision:
    """충분성 판정 결과."""

    sufficient: bool
    reason: str = ""
    suggested_query: str | None = None

    @property
    def needs_repair(self) -> bool:
        """추가 검색이 필요한지 — orchestrator가 참조하는 편의 속성."""
        return not self.sufficient and bool(self.suggested_query)


class Verifier:
    """수집된 컨텍스트의 충분성을 LLM으로 평가합니다.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate``를 호출할 ``httpx.AsyncClient``.
    ollama_model:
        평가용 Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        단일 요청 타임아웃(초). 빠른 실패를 위해 메인 타임아웃보다 짧게 설정.
    max_tokens:
        Ollama ``num_predict``. 판정 JSON은 짧으므로 낮게 설정.
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 30.0,
        max_tokens: int = 200,
        *,
        keep_alive: str = "5m",
        native_thinking: bool = False,
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive
        self._native_thinking = native_thinking

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(self, query: str, collected_context: str) -> VerifierDecision:
        """컨텍스트 충분성을 평가합니다 — never raises."""
        try:
            raw = await self._generate(query, collected_context)
        except Exception as exc:
            logger.warning("Verifier LLM 호출 실패: %s — 충분 처리", exc)
            return VerifierDecision(
                sufficient=True,
                reason="verifier unavailable — assumed sufficient",
            )

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("Verifier JSON 파싱 실패 — 충분 처리")
            return VerifierDecision(
                sufficient=True,
                reason="verifier parse failure — assumed sufficient",
            )

        return self._to_decision(parsed)

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(self, query: str, context: str) -> str:
        prompt = SUFFICIENCY_PROMPT.format(
            query=query,
            context=context[:_CONTEXT_CHAR_LIMIT],
        )
        response = await self._http_client.post(
            f"{self._api_base}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": self._native_thinking,
                "format": "json",
                "keep_alive": self._keep_alive,
                "options": {"num_predict": self._max_tokens},
            },
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "") or data.get("thinking", "")

    # ------------------------------------------------------------------
    # JSON 파싱
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> dict | None:
        if not raw:
            return None
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start == -1 or brace_end <= brace_start:
            return None
        try:
            parsed = json.loads(raw[brace_start : brace_end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _to_decision(data: dict) -> VerifierDecision:
        """파싱된 dict를 ``VerifierDecision``으로 변환합니다."""
        raw_sufficient = data.get("sufficient")
        # LLM이 문자열 "true"/"false"로 반환할 수 있음 — 방어적 처리.
        if isinstance(raw_sufficient, str):
            sufficient = raw_sufficient.strip().lower() in ("true", "yes", "예", "충분")
        else:
            sufficient = bool(raw_sufficient) if raw_sufficient is not None else True

        reason = str(data.get("reason", ""))[:300]
        suggestion_raw = data.get("suggestion") or data.get("suggested_query")
        # LLM이 list로 응답할 수 있음 — 첫 원소만 채택. (str(list)이 그대로
        # 검색 query로 들어가 재검색을 망가뜨리는 사고 방지.)
        if isinstance(suggestion_raw, list):
            suggestion_raw = suggestion_raw[0] if suggestion_raw else None
        suggestion = str(suggestion_raw).strip() if suggestion_raw else None
        if suggestion == "":
            suggestion = None

        return VerifierDecision(
            sufficient=sufficient,
            reason=reason,
            suggested_query=suggestion,
        )


__all__ = ["Verifier", "VerifierDecision"]
