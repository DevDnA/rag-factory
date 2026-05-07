"""질의 복잡도 기반 라우팅 결정기.

``QueryRouter``는 사용자 질의를 검사하여 단순 RAG(``simple``) 또는
Agent RAG(``agent``) 경로 중 하나를 선택합니다. 현재 휴리스틱은
server.py에 인라인되어 있던 ``_is_complex_query`` 규칙을 그대로
옮겨와 동작이 동일함을 보장합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .intent_classifier import IntentCategory, IntentClassifier

RouteMode = Literal["simple", "agent", "chitchat", "general"]


# 복합 질문 판단용 키워드 — 비교, 변경 이력, 인과, 조건 등.
_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "비교", "차이", "다른점", "대비", "vs", "versus",
    "변경", "바뀐", "달라진", "개정",
    "왜", "이유", "원인", "근거",
    "어떤 경우", "조건", "해당하는", "자격",
    "장단점", "장점과 단점", "pros", "cons",
    "관계", "연관", "영향", "따르면",
    "종합", "분석", "검토",
)

# "A와 B의 차이", "A와 B를 비교" 형태의 비교 절 패턴.
# "와/과" 양옆에 공백을 요구하여 단어 내부 매치(예: "교과정"의 "과") false positive 차단.
_COMPARISON_RE = re.compile(
    r"(?:^|\s)[\w가-힣]{2,}\s+[와과]\s+[\w가-힣]{2,}(?:의|을|를|에|는|가|와|과)"
)

# 잡담 / 인사 / 감사 패턴 — RAG 검색이 불필요한 일반 대화를 빠르게 분기.
# oh-my-openagent의 `keyword-detector`와 동일한 정규식 사전 필터 패턴(다국어).
# 매치 시 LLM 호출 없이 즉시 ``chitchat`` 모드로 라우팅합니다.
_CHITCHAT_RE = re.compile(
    r"^\s*("
    # 한국어 인사·감사·작별 — 어간별 활용형 명시 enumerate
    r"안녕(하세요|히|히\s*가세요|히\s*계세요)?"
    r"|반갑(다|네|네요|군요|습니다|습니까)?"
    r"|반가(워|워요|웠어|웠어요|웠습니다)"
    r"|좋은\s*(아침|점심|저녁|밤|하루)(이에요|이네요|입니다)?"
    r"|잘\s*(있어|있어요|있으세요|가|가요|가세요|자|자요|주무세요|지내|지내요|지내세요)"
    r"|어서\s*(와|와요|오세요)"
    r"|고맙(다|네|네요|군요|습니다|습니까)?"
    r"|고마(워|워요|웠어요)"
    r"|감사(합니다|해요|하다|드립니다|드려요)?"
    r"|땡큐|쌩큐"
    r"|미안(하다|합니다|해요|해)?"
    r"|죄송(하다|합니다|해요|해)?"
    r"|괜찮(다|아|아요|네|네요|습니다)?"
    r"|수고(하셨습니다|하셨어요|하셨네요|많으셨습니다|많으셨어요|많으셨네요|했어요|했어)?"
    r"|네\s*$|아니(요|오)?\s*$|응\s*$|예\s*$"
    # 자기소개·정체성 질의
    r"|너\s*(는\s*)?누구(야|니|냐|세요|이세요|예요)?"
    r"|당신은\s*누구(인가요|예요|이세요)?"
    r"|니가\s*누구(야|니|냐)?"
    r"|(네|니)\s*이름(이|은)?\s*(뭐|무엇)(야|니|이야|예요|입니까)?"
    r"|이름이\s*(뭐|무엇)(야|니|이야|예요|입니까)?"
    r"|뭐\s*할\s*수\s*있(어|어요|니|냐|습니까|나요)?"
    r"|무엇을\s*할\s*수\s*있(어|어요|니|냐|습니까|나요)?"
    # 영어 greetings/thanks
    r"|(hi|hello|hey|yo|howdy)(\s+(there|guys|all|everyone|y'?all|folks))?"
    r"|good\s*(morning|afternoon|evening|night|day)"
    r"|how\s*are\s*you|how'?s\s*it\s*going|what'?s\s*up|sup"
    r"|thanks?|thank\s*you|thx|ty|cheers"
    r"|sorry|my\s*bad"
    r"|bye|goodbye|see\s*(you|ya)|cya|farewell"
    r"|ok(ay)?|alright|sure|nope?|yep|yeah"
    r"|who\s*are\s*you|what\s*can\s*you\s*do"
    # 일본어
    r"|こんにちは|こんばんは|おはよう(ございます)?|お疲れ(さま|様)?(です)?"
    r"|ありがとう(ございます)?|すみません|さようなら|またね"
    # 중국어
    r"|你好|您好|早上好|晚上好|谢谢|对不起|再见"
    # 베트남어
    r"|xin\s*chào|cảm\s*ơn|tạm\s*biệt"
    r")[\s.!?。、,，~～]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    """라우팅 결정 결과.

    Attributes
    ----------
    mode:
        ``"simple"`` 또는 ``"agent"``.
    reason:
        라우팅 근거 — 로깅 및 디버깅용 사람이 읽을 수 있는 설명.
    complexity:
        감지된 복잡도 — 0.0 (단순) ~ 1.0 (복합).
    matched_keyword:
        복합으로 분류된 경우 첫 매칭 키워드. 단순일 땐 ``None``.
    intent:
        ``IntentClassifier``가 활성화된 경우 감지된 의도 카테고리.
        키워드 기반 라우팅만 사용한 경우 ``None``.
    """

    mode: RouteMode
    reason: str
    complexity: float
    matched_keyword: str | None = None
    intent: "IntentCategory | None" = None


class QueryRouter:
    """질의 복잡도 휴리스틱 + 선택적 LLM 의도 분류 기반 라우터.

    Parameters
    ----------
    agent_enabled:
        Agent 모드가 전반적으로 활성화되어 있는지. ``False``이면
        어떤 질의든 ``simple``로 라우팅됩니다.
    intent_classifier:
        선택적 LLM 기반 분류기. 제공되면 ``route_async()``가 먼저
        사용하고, 실패 시 키워드 휴리스틱으로 fallback.
    """

    def __init__(
        self,
        agent_enabled: bool = True,
        intent_classifier: "IntentClassifier | None" = None,
    ) -> None:
        self._agent_enabled = agent_enabled
        self._classifier = intent_classifier

    def route(self, query: str) -> RouteDecision:
        """주어진 질의의 라우팅을 결정합니다 (동기·키워드 전용).

        라우팅 우선순위:
        1. ``agent_enabled=False`` → simple 즉시 반환
        2. 복합 키워드(비교/이유/관계 등) — 도메인 의도가 강하므로 chitchat보다 우선
        3. 잡담 정규식 — 인사/감사/자기소개 등 RAG 무관 발화
        4. 다중 물음표 / 비교 절 패턴 → agent
        5. 그 외 → simple (단순 RAG)
        """
        if not self._agent_enabled:
            return RouteDecision(
                mode="simple",
                reason="agent mode disabled",
                complexity=0.0,
            )

        lowered = query.lower()

        # 도메인 신호(복합 키워드)가 잡담 신호보다 강함 — chitchat보다 먼저 검사.
        for keyword in _COMPLEX_KEYWORDS:
            if keyword in lowered:
                return RouteDecision(
                    mode="agent",
                    reason=f"복합 키워드 '{keyword}' 감지",
                    complexity=0.8,
                    matched_keyword=keyword,
                )

        # 잡담 fast-path — 짧은 인사·감사·정체성 질의는 LLM 분류 없이 즉시 chitchat.
        if _CHITCHAT_RE.match(query):
            return RouteDecision(
                mode="chitchat",
                reason="잡담 패턴 감지 — RAG 검색 우회",
                complexity=0.0,
            )

        if query.count("?") >= 2 or query.count("？") >= 2:
            return RouteDecision(
                mode="agent",
                reason="다중 물음표 — 복합 질문으로 판단",
                complexity=0.7,
            )

        if _COMPARISON_RE.search(query):
            return RouteDecision(
                mode="agent",
                reason="비교 절 패턴(~와/과 ~) 감지",
                complexity=0.75,
            )

        return RouteDecision(
            mode="simple",
            reason="복합 신호 없음 — 단순 RAG",
            complexity=0.1,
        )

    async def route_async(self, query: str) -> RouteDecision:
        """비동기 라우팅 — ``IntentClassifier``가 주입되어 있으면 사용.

        LLM 분류가 실패하거나 ``ambiguous`` 저신뢰로 반환되면 키워드 휴리스틱으로
        fallback합니다. 키워드가 ``agent``로 분류했고 LLM은 ``factual``이라면
        키워드 신호를 우선시합니다 (명시적 비교·분석 키워드가 LLM 모호성보다 강함).
        """
        if not self._agent_enabled or self._classifier is None:
            return self.route(query)

        keyword_decision = self.route(query)

        # 키워드가 이미 chitchat으로 분류했으면 LLM 호출 없이 그대로 반환 — 비용 절감.
        if keyword_decision.mode == "chitchat":
            return keyword_decision

        try:
            intent = await self._classifier.classify(query)
        except Exception:  # pragma: no cover — classifier는 자체 never-raise
            return keyword_decision

        # LLM 분류 → 라우팅 모드 매핑.
        llm_mode: RouteMode
        if intent.intent == "chitchat" and intent.confidence >= 0.7:
            llm_mode = "chitchat"
        elif intent.intent == "general" and intent.confidence >= 0.7:
            # corpus 외 일반 지식 — chitchat과 같이 RAG 우회하지만 별도 합성 프롬프트 사용.
            llm_mode = "general"
        elif intent.intent == "factual" and intent.confidence >= 0.7:
            llm_mode = "simple"
        else:
            llm_mode = "agent"

        # 키워드가 agent라고 판단한 경우, LLM이 명확히 chitchat/general(고신뢰)
        # 라고 분류했다면 LLM 우선 — corpus 컨텍스트를 본 LLM이 도메인 외라고
        # 확신하면 키워드 휴리스틱(비교/이유 등)은 false positive로 보고 무시.
        if keyword_decision.mode == "agent":
            if (
                intent.intent in ("chitchat", "general")
                and intent.confidence >= 0.85
            ):
                return RouteDecision(
                    mode=llm_mode,
                    reason=(
                        f"intent={intent.intent} conf={intent.confidence:.2f} "
                        f"(키워드 '{keyword_decision.matched_keyword}' 무시 — "
                        f"LLM이 도메인 외 확신)"
                    ),
                    complexity=intent.confidence,
                    matched_keyword=None,
                    intent=intent.intent,
                )
            return RouteDecision(
                mode="agent",
                reason=(
                    f"{keyword_decision.reason} + intent={intent.intent} "
                    f"conf={intent.confidence:.2f}"
                ),
                complexity=max(keyword_decision.complexity, intent.confidence),
                matched_keyword=keyword_decision.matched_keyword,
                intent=intent.intent,
            )

        return RouteDecision(
            mode=llm_mode,
            reason=f"intent={intent.intent} conf={intent.confidence:.2f} ({intent.reason})",
            complexity=intent.confidence if llm_mode == "agent" else 1.0 - intent.confidence,
            matched_keyword=None,
            intent=intent.intent,
        )


__all__ = ["QueryRouter", "RouteDecision", "RouteMode"]
