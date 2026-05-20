"""합성 답변의 적대적 검증기 (Phase 14).

``AnswerVerifier``는 합성된 답변이 인용된 chunk에서 *직접 지지되는지*를 LLM으로
판정합니다. autoclaude ``Argus.md`` 패턴을 RFP RAG 도메인에 맞춰 축소 이식:
컨텍스트 충분성을 보는 :class:`Verifier`와 달리 **답변 자체**의 hallucination·
모순·unsupported claim을 검출합니다.

설계 원칙
---------
- **절대 raise하지 않음**: LLM 실패·파싱 실패 시 ``verdict="PASS"``로 반환.
  사용자가 받은 답변을 임의로 막지 않는다는 정책 (CLAUDE.md "never-raise" 일관성).
- **Repair는 호출 측에서**: ``AnswerVerifier`` 자체는 단일 판정만. 재합성 반복은
  ``orchestrator``가 ``answer_verifier_max_repairs`` 설정으로 통제.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ...utils import get_logger
from .prompts import ANSWER_VERIFIER_PROMPT

logger = get_logger("rag.agent.answer_verifier")

# 답변·컨텍스트 길이 제한 — 프롬프트 폭발 방지.
_CONTEXT_CHAR_LIMIT = 2000
_ANSWER_CHAR_LIMIT = 3000

Verdict = Literal["PASS", "FAIL", "PARTIAL"]


@dataclass
class AnswerVerdict:
    """답변 검증 결과.

    Attributes
    ----------
    verdict:
        ``"PASS"`` (지지됨) / ``"FAIL"`` (unsupported claim 있음) / ``"PARTIAL"``
        (일부 지지되나 핵심 주장 불명).
    issues:
        검출된 문제 목록 (FAIL/PARTIAL 시). 사용자 노출 가능.
    repair_hint:
        재합성 시 강조할 지침 1줄. FAIL 시에만 의미 있음.
    """

    verdict: Verdict
    issues: list[str] = field(default_factory=list)
    repair_hint: str = ""

    @property
    def needs_repair(self) -> bool:
        """재합성이 필요한지 — orchestrator가 참조하는 편의 속성."""
        return self.verdict == "FAIL" and bool(self.repair_hint)


class AnswerVerifier:
    """합성된 답변의 적대적 검증을 LLM으로 수행합니다.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate``를 호출할 ``httpx.AsyncClient``.
    ollama_model:
        검증용 Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        단일 요청 타임아웃(초).
    max_tokens:
        Ollama ``num_predict``. 검증 JSON은 짧으므로 낮게 설정.
    keep_alive:
        Ollama keep_alive 문자열.
    native_thinking:
        Qwen3·DeepSeek-R1 등 ``think=True`` 지원 모델에서 reasoning 활성화.
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 30.0,
        max_tokens: int = 300,
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

    async def evaluate(
        self, query: str, answer: str, context: str
    ) -> AnswerVerdict:
        """답변이 컨텍스트에서 지지되는지 판정 — never raises."""
        if not answer.strip():
            # 빈 답변은 검증 의미 없음 — PASS (사용자 노출 금지가 아니라 단순 noop).
            return AnswerVerdict(verdict="PASS")

        try:
            raw = await self._generate(query, answer, context)
        except Exception as exc:
            logger.warning("AnswerVerifier LLM 호출 실패: %s — PASS 처리", exc)
            return AnswerVerdict(verdict="PASS")

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("AnswerVerifier JSON 파싱 실패 — PASS 처리")
            return AnswerVerdict(verdict="PASS")

        return self._to_verdict(parsed)

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(self, query: str, answer: str, context: str) -> str:
        prompt = ANSWER_VERIFIER_PROMPT.format(
            query=query,
            answer=answer[:_ANSWER_CHAR_LIMIT],
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
    def _to_verdict(data: dict) -> AnswerVerdict:
        """파싱된 dict를 :class:`AnswerVerdict`로 변환합니다."""
        raw_verdict = data.get("verdict")
        # LLM이 소문자·다른 표기로 반환할 수 있음 — 정규화.
        if isinstance(raw_verdict, str):
            normalized = raw_verdict.strip().upper()
            if normalized in ("PASS", "FAIL", "PARTIAL"):
                verdict: Verdict = normalized  # type: ignore[assignment]
            else:
                # 인식 못 하는 값은 안전 default PASS.
                verdict = "PASS"
        else:
            verdict = "PASS"

        # issues가 단일 문자열로 올 수도 있음 — list로 정규화.
        raw_issues = data.get("issues", [])
        if isinstance(raw_issues, str):
            issues = [raw_issues] if raw_issues.strip() else []
        elif isinstance(raw_issues, list):
            issues = [str(x)[:300] for x in raw_issues if str(x).strip()]
        else:
            issues = []

        repair_hint = str(data.get("repair_hint", ""))[:500].strip()

        return AnswerVerdict(
            verdict=verdict,
            issues=issues,
            repair_hint=repair_hint,
        )


__all__ = ["AnswerVerifier", "AnswerVerdict", "Verdict"]
