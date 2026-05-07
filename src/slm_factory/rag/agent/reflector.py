"""답변 자기 검증기(Reflector) — 생성된 답변의 품질을 LLM으로 평가합니다.

Planner → Verifier(사전 충분성 판정)에 이어 Reflector는 **사후** 답변 품질을
평가합니다. "답변이 질문에 충분히 답하는가? 근거가 명확한가? 추가 정보가
필요한가?" 를 판단하여 필요 시 재시도를 제안합니다.

설계 원칙
---------
- **절대 raise하지 않음**: LLM 실패·JSON 파싱 실패 시 ``answer_ok=True``로
  반환해서 orchestrator가 무한 retry에 빠지지 않도록 합니다.
- **Verifier와 다른 역할**: Verifier는 "검색 결과가 답변하기에 충분한가"를
  판단하는 사전 게이트입니다. Reflector는 "실제 생성된 답변이 질문에
  부합하는가"를 판단하는 사후 검수자입니다. 함께 사용하면 양방향 품질 검증.
- **Retry 횟수 제한**: Reflector는 단일 판정만 수행합니다. 반복은 orchestrator의
  ``reflector_max_retries`` 설정으로 통제합니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ...utils import get_logger
from .prompts import REFLECTOR_PROMPT

logger = get_logger("rag.agent.reflector")

# 답변·컨텍스트 길이 제한 — 프롬프트 폭발 방지.
_ANSWER_CHAR_LIMIT = 2000
_SOURCES_PREVIEW_LIMIT = 1500


@dataclass
class ReflectorDecision:
    """답변 품질 판정 결과."""

    answer_ok: bool
    reason: str = ""
    missing_info_query: str | None = None

    @property
    def needs_retry(self) -> bool:
        """재시도가 필요한지 — orchestrator의 판단용 편의 속성."""
        return not self.answer_ok and bool(self.missing_info_query)


class Reflector:
    """생성된 답변의 품질을 LLM으로 평가합니다.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate``를 호출할 ``httpx.AsyncClient``.
    ollama_model:
        평가용 Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        단일 요청 타임아웃(초).
    max_tokens:
        Ollama ``num_predict``. 판정 JSON은 짧으므로 낮게 설정.
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 30.0,
        max_tokens: int = 250,
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

    async def reflect(
        self,
        query: str,
        answer: str,
        sources: list[dict] | None = None,
    ) -> ReflectorDecision:
        """답변 품질을 평가합니다 — never raises."""
        if not answer.strip():
            return ReflectorDecision(
                answer_ok=False,
                reason="empty-answer",
                missing_info_query=query,
            )

        try:
            raw = await self._generate(query, answer, sources or [])
        except Exception as exc:
            logger.warning("Reflector LLM 호출 실패: %s — 답변 통과 처리", exc)
            return ReflectorDecision(
                answer_ok=True,
                reason="reflector unavailable — assumed ok",
            )

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("Reflector JSON 파싱 실패 — 답변 통과 처리")
            return ReflectorDecision(
                answer_ok=True,
                reason="reflector parse failure — assumed ok",
            )

        return self._to_decision(parsed)

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(
        self, query: str, answer: str, sources: list[dict]
    ) -> str:
        sources_preview = self._format_sources(sources)
        prompt = REFLECTOR_PROMPT.format(
            query=query,
            answer=answer[:_ANSWER_CHAR_LIMIT],
            sources_preview=sources_preview,
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

    @staticmethod
    def _format_sources(sources: list[dict]) -> str:
        """Reflector 프롬프트용 soruces 요약을 생성합니다."""
        if not sources:
            return "(수집된 참고 문서 없음)"
        lines: list[str] = []
        total = 0
        for i, src in enumerate(sources, start=1):
            if not isinstance(src, dict):
                continue
            doc_id = src.get("doc_id", "?")
            content = str(src.get("content", ""))[:200]
            line = f"[문서 {i}] (ID: {doc_id}) {content}"
            if total + len(line) > _SOURCES_PREVIEW_LIMIT:
                lines.append("...(일부 생략)")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

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
    def _to_decision(data: dict) -> ReflectorDecision:
        raw_ok = data.get("answer_ok")
        if raw_ok is None:
            raw_ok = data.get("ok")
        if isinstance(raw_ok, str):
            answer_ok = raw_ok.strip().lower() in ("true", "yes", "예", "ok", "좋음")
        else:
            answer_ok = bool(raw_ok) if raw_ok is not None else True

        reason = str(data.get("reason", ""))[:300]
        raw_query = (
            data.get("missing_info_query")
            or data.get("missing_info")
            or data.get("next_query")
        )
        missing = str(raw_query).strip() if raw_query else None
        if missing == "":
            missing = None

        return ReflectorDecision(
            answer_ok=answer_ok,
            reason=reason,
            missing_info_query=missing,
        )


__all__ = ["Reflector", "ReflectorDecision"]
