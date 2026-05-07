"""질의 → 실행 계획(``ExecutionPlan``) 생성기.

``Planner``는 사용자 질의를 받아 LLM에 구조화된 검색 계획을 요청합니다.
계획은 도구 호출 시퀀스(``PlanStep``)로 구성되며 orchestrator가 step-by-step
으로 실행합니다.

설계 원칙
---------
- **절대 raise하지 않음**: LLM 실패·JSON 파싱 실패·타임아웃 등 어떤 경우에도
  ``plan()``은 항상 유효한 ``ExecutionPlan``을 반환합니다. 실패 시
  ``_default_plan()``이 "단일 검색" fallback을 생성합니다.
- **도구 이름 화이트리스트 검증**: LLM이 존재하지 않는 도구를 제안하면 해당
  step을 drop하고 fallback으로 보정합니다.
- **JSON mode 사용**: Ollama ``format=json``으로 출력 포맷을 강제합니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ...utils import get_logger
from .prompts import PLANNER_PROMPT

logger = get_logger("rag.agent.planner")


PlanStrategy = Literal["fact", "compare", "decompose"]
"""계획 전략 — 단일 사실 | 비교 | 다단계 분해."""

# Planner가 허용하는 도구 이름 — orchestrator가 등록한 ToolRegistry와 정렬.
_ALLOWED_TOOLS: frozenset[str] = frozenset({"search", "lookup", "compare"})

# 최대 step 수 — 프롬프트 폭발·비용 폭주 방지.
_DEFAULT_MAX_STEPS = 3


@dataclass
class PlanStep:
    """단일 도구 호출 계획."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ExecutionPlan:
    """orchestrator가 실행할 전체 계획."""

    strategy: PlanStrategy
    steps: list[PlanStep]
    rationale: str = ""

    @property
    def is_fallback(self) -> bool:
        """기본 fallback 계획 여부 — 디버깅·메트릭용."""
        return self.rationale.startswith("fallback:")


class Planner:
    """질의 복잡도에 맞는 실행 계획을 LLM으로 생성합니다.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate``를 호출할 ``httpx.AsyncClient``.
    ollama_model:
        Planner용 Ollama 모델명 (보통 Teacher 모델과 동일).
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        단일 요청 타임아웃(초). 빠른 실패를 위해 메인 타임아웃보다 짧게 설정.
    max_steps:
        계획에 포함 가능한 최대 step 수.
    max_tokens:
        Ollama ``num_predict``. 계획 JSON은 일반적으로 작으므로 낮게 설정.
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 30.0,
        max_steps: int = _DEFAULT_MAX_STEPS,
        max_tokens: int = 400,
        *,
        keep_alive: str = "5m",
        native_thinking: bool = False,
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive
        self._native_thinking = native_thinking

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(self, query: str) -> ExecutionPlan:
        """질의에 대한 ``ExecutionPlan``을 반환합니다 — never raises."""
        try:
            raw = await self._generate(query)
        except Exception as exc:
            logger.warning("Planner LLM 호출 실패: %s — fallback 사용", exc)
            return self._default_plan(query, reason="llm-error")

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("Planner JSON 파싱 실패 — fallback 사용")
            return self._default_plan(query, reason="parse-error")

        plan = self._validate(parsed, query)
        if not plan.steps:
            return self._default_plan(query, reason="empty-steps")
        return plan

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(self, query: str) -> str:
        prompt = PLANNER_PROMPT.format(
            query=query,
            allowed_tools=", ".join(sorted(_ALLOWED_TOOLS)),
            max_steps=self._max_steps,
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
    # JSON 파싱 + 검증
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> dict | None:
        """LLM 응답에서 JSON 객체를 추출합니다."""
        if not raw:
            return None
        # ``format: json``을 썼어도 일부 모델은 prefix/suffix를 붙이므로 방어적으로 추출.
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

    def _validate(self, data: dict, query: str) -> ExecutionPlan:
        """파싱된 dict를 ``ExecutionPlan``으로 검증·변환합니다."""
        strategy = data.get("strategy", "fact")
        if strategy not in ("fact", "compare", "decompose"):
            strategy = "fact"

        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raw_steps = []

        steps: list[PlanStep] = []
        for item in raw_steps[: self._max_steps]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip().lower()
            if tool not in _ALLOWED_TOOLS:
                logger.debug("planner가 알 수 없는 도구 '%s' 제안 — 스킵", tool)
                continue
            args = item.get("args") or {}
            if not isinstance(args, dict):
                continue
            reason = str(item.get("reason", ""))[:200]
            steps.append(PlanStep(tool=tool, args=args, reason=reason))

        rationale = str(data.get("rationale", ""))[:500]
        return ExecutionPlan(strategy=strategy, steps=steps, rationale=rationale)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _default_plan(query: str, *, reason: str) -> ExecutionPlan:
        """LLM 실패 시 단일 search로 구성된 안전한 fallback 계획."""
        return ExecutionPlan(
            strategy="fact",
            steps=[
                PlanStep(
                    tool="search",
                    args={"query": query},
                    reason="fallback plan — single search",
                )
            ],
            rationale=f"fallback: {reason}",
        )


__all__ = ["Planner", "ExecutionPlan", "PlanStep", "PlanStrategy"]
