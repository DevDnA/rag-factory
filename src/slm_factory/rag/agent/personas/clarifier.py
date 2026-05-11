"""Clarifier persona — 모호한 질의에 명확화 역질문을 생성합니다.

``IntentClassifier``가 ``ambiguous``로 분류한 질의는 이 persona로 라우팅되어
사용자에게 1~2개의 구체적인 후속 질문을 반환합니다. 답변 생성 없이 대화를
반환 형태로 종료.

설계 원칙
---------
- **절대 raise하지 않음**: LLM·파싱 실패 시 일반적 fallback 질문 제공.
- **대화 연속성**: 세션의 이전 대화 내용을 프롬프트에 포함하여 맥락 반영.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ....utils import get_logger
from ..prompts import CLARIFIER_PROMPT
from .base import Persona, PersonaResult

logger = get_logger("rag.agent.personas.clarifier")


class Clarifier(Persona):
    """모호한 질의 → 명확화 질문 생성기.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate`` 호출용.
    ollama_model:
        생성 모델명.
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        요청 타임아웃(초).
    max_tokens:
        생성 최대 토큰 수.
    max_questions:
        반환할 질문의 최대 개수. 1~3 범위가 현실적.
    """

    name = "clarifier"
    description = "모호한 질의에 명확화 역질문을 생성하는 persona"
    allowed_tools = frozenset()  # 도구 사용 금지 — 질문만 생성

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 15.0,
        max_tokens: int = 250,
        max_questions: int = 2,
        *,
        keep_alive: str = "5m",
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._max_questions = max_questions
        self._keep_alive = keep_alive

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_questions(
        self, query: str, history: str = ""
    ) -> PersonaResult:
        """명확화 질문 목록을 반환합니다 — never raises."""
        try:
            raw = await self._generate(query, history)
        except Exception as exc:
            logger.warning(
                "Clarifier LLM 호출 실패 type=%s msg=%r — fallback 질문",
                type(exc).__name__,
                str(exc),
            )
            return self._fallback(query, reason="llm-error")

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("Clarifier JSON 파싱 실패 — fallback")
            return self._fallback(query, reason="parse-error")

        questions = self._extract_questions(parsed)
        if not questions:
            return self._fallback(query, reason="empty-questions")

        return PersonaResult(
            kind="clarification",
            questions=questions[: self._max_questions],
            metadata={"persona": self.name, "is_fallback": False},
        )

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    async def _generate(self, query: str, history: str) -> str:
        prompt = CLARIFIER_PROMPT.format(
            query=query,
            history=f"{history}\n" if history else "",
            max_questions=self._max_questions,
        )
        response = await self._http_client.post(
            f"{self._api_base}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": False,
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
    def _extract_questions(data: dict) -> list[str]:
        raw = data.get("questions")
        if not isinstance(raw, list):
            return []
        cleaned: list[str] = []
        for q in raw:
            if isinstance(q, str) and q.strip():
                cleaned.append(q.strip()[:300])
        return cleaned

    def _fallback(self, query: str, *, reason: str) -> PersonaResult:
        """LLM 실패 시 generic 후속 질문."""
        default_questions = [
            "조금 더 구체적으로 어떤 부분이 궁금하신가요?",
            "질문의 맥락 (예: 어느 문서·어느 주제)을 알려주시겠어요?",
        ]
        return PersonaResult(
            kind="clarification",
            questions=default_questions[: self._max_questions],
            metadata={
                "persona": self.name,
                "is_fallback": True,
                "fallback_reason": reason,
            },
        )


__all__ = ["Clarifier"]
