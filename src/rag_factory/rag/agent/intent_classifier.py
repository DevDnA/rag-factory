"""IntentClassifier — LLM 기반 질의 의도 분류기.

키워드 휴리스틱만으로 구분하기 어려운 질의의 "진짜 의도"를 LLM이 분류합니다.
oh-my-openagent의 ``IntentGate`` 패턴을 RAG Q&A 컨텍스트에 맞게 적응.

의도 카테고리
------------
- ``factual``: 단일 사실·정의·조항 확인 ("제15조는?", "금리는?")
- ``comparative``: 비교·차이·대조 ("A와 B 차이?")
- ``analytical``: 분석·인과·종합 ("왜 변경됐나?", "영향은?")
- ``procedural``: 절차·방법·단계 ("어떻게 신청하나?")
- ``exploratory``: 탐색·개요·목록 ("약관에 어떤 조항이?")
- ``chitchat``: 인사·잡담 — 짧은 사회적 발화 ("안녕하세요", "너 누구야?")
- ``general``: 코퍼스 외 일반 지식·코드·잡학 — LLM 직답이 적절 ("파이썬이 뭐야?")
- ``ambiguous``: 모호·맥락 부족 ("그거 어떻게 돼요?")

설계 원칙
---------
- **절대 raise하지 않음**: LLM·파싱 실패 시 ``ambiguous`` 또는 ``factual``로
  안전하게 fallback. 호출 측에서 결정을 내릴 수 있도록.
- **TTL 캐싱**: 동일 질의에 대한 중복 LLM 호출 방지.
- **JSON 모드**: Ollama ``format=json``으로 출력 형식 강제.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from ...utils import get_logger
from .prompts import INTENT_CLASSIFIER_PROMPT

logger = get_logger("rag.agent.intent_classifier")


IntentCategory = Literal[
    "factual",
    "comparative",
    "analytical",
    "procedural",
    "exploratory",
    "chitchat",
    "general",
    "ambiguous",
]

_VALID_INTENTS: frozenset[str] = frozenset(
    {
        "factual",
        "comparative",
        "analytical",
        "procedural",
        "exploratory",
        "chitchat",
        "general",
        "ambiguous",
    }
)


@dataclass(frozen=True)
class IntentDecision:
    """의도 분류 결과."""

    intent: IntentCategory
    confidence: float
    reason: str = ""

    @property
    def is_agent_intent(self) -> bool:
        """복합 질의(agent 경로) 성격인지 여부."""
        return self.intent != "factual"


class IntentClassifier:
    """LLM 기반 의도 분류기 with TTL 캐시.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate`` 호출용 ``httpx.AsyncClient``.
    ollama_model:
        분류용 모델명 (가벼운 모델 권장).
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        요청 타임아웃(초). 분류는 빠르게 실패하고 fallback하도록 짧게.
    max_tokens:
        생성 최대 토큰. 분류 JSON은 짧으므로 낮게.
    cache_ttl:
        같은 질의에 대한 캐싱 시간(초). 0이면 비활성.
    cache_max_size:
        캐시 최대 엔트리 수. 초과 시 최고령 항목 퇴거.
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 10.0,
        max_tokens: int = 150,
        cache_ttl: int = 300,
        cache_max_size: int = 512,
        *,
        keep_alive: str = "5m",
        corpus_profile: Any = None,
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._cache_ttl = cache_ttl
        self._cache_max_size = cache_max_size
        self._keep_alive = keep_alive
        self._cache: dict[str, tuple[IntentDecision, float]] = {}
        # CorpusProfile 인스턴스(또는 None) — to_prompt_header()로 LLM 컨텍스트 주입.
        self._corpus_profile = corpus_profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(self, query: str) -> IntentDecision:
        """질의의 의도를 분류합니다 — never raises."""
        normalized = self._normalize(query)

        cached = self._get_cached(normalized)
        if cached is not None:
            return cached

        try:
            raw = await self._generate(normalized)
        except Exception as exc:
            logger.warning("IntentClassifier LLM 호출 실패: %s — ambiguous 반환", exc)
            return self._fallback(reason="llm-error")

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("IntentClassifier JSON 파싱 실패 — ambiguous 반환")
            return self._fallback(reason="parse-error")

        decision = self._to_decision(parsed)
        self._set_cached(normalized, decision)
        return decision

    # ------------------------------------------------------------------
    # 캐시 관리
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """캐시 키용 정규화 — 공백 압축·양쪽 trim·lowercase."""
        return re.sub(r"\s+", " ", query).strip().lower()

    def _get_cached(self, key: str) -> IntentDecision | None:
        if self._cache_ttl <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        decision, expires_at = entry
        if time.time() >= expires_at:
            self._cache.pop(key, None)
            return None
        return decision

    def _set_cached(self, key: str, decision: IntentDecision) -> None:
        if self._cache_ttl <= 0:
            return
        if len(self._cache) >= self._cache_max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            self._cache.pop(oldest, None)
        self._cache[key] = (decision, time.time() + self._cache_ttl)

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(self, query: str) -> str:
        # Corpus profile이 주입되어 있으면 도메인 컨텍스트를 헤더로 prepend.
        # 빈 profile이면 to_prompt_header()가 빈 문자열을 반환해 동작 변화 없음.
        corpus_header = ""
        if self._corpus_profile is not None:
            header = self._corpus_profile.to_prompt_header()
            if header:
                corpus_header = f"{header}\n\n"
        prompt = INTENT_CLASSIFIER_PROMPT.format(
            query=query, corpus_header=corpus_header
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
                "options": {
                    "num_predict": self._max_tokens,
                    # Phase 14 — 결정성 강제. 분류 task는 greedy decoding이 정답.
                    # 베이스라인(default sampling)에서 같은 query를 5회 반복 시
                    # q1/q3/q4가 intent를 갈아치워 route mode가 simple/agent/general
                    # 사이에서 흔들리는 비결정성을 관측 → 사용자 경험 일관성 손상.
                    # temperature=0 + 고정 seed로 동일 query → 동일 결과 보장.
                    "temperature": 0,
                    "seed": 42,
                    "top_p": 1.0,
                    "top_k": 1,
                },
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
    def _to_decision(data: dict) -> IntentDecision:
        raw_intent = str(data.get("intent", "")).strip().lower()
        if raw_intent not in _VALID_INTENTS:
            raw_intent = "ambiguous"

        raw_conf = data.get("confidence", 0.5)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reason = str(data.get("reason", ""))[:300]

        return IntentDecision(
            intent=raw_intent,  # type: ignore[arg-type]
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _fallback(*, reason: str) -> IntentDecision:
        """LLM 실패 시 안전한 기본값 — ambiguous로 두면 Clarifier 또는 agent 경로 유도."""
        return IntentDecision(
            intent="ambiguous",
            confidence=0.0,
            reason=f"fallback: {reason}",
        )


__all__ = ["IntentClassifier", "IntentDecision", "IntentCategory"]
