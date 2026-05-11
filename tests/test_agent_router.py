"""QueryRouter 테스트 — 복잡도 휴리스틱과 agent_enabled 게이팅을 검증합니다."""

from __future__ import annotations

import pytest

from slm_factory.rag.agent.router import QueryRouter, RouteDecision


class TestAgentDisabled:
    """agent_enabled=False일 때는 어떤 질의든 simple로 라우팅되어야 합니다."""

    def test_복합_질의도_simple로_라우팅(self):
        router = QueryRouter(agent_enabled=False)
        decision = router.route("A와 B의 차이를 비교해줘")
        assert decision.mode == "simple"
        assert "disabled" in decision.reason

    def test_단순_질의도_simple(self):
        router = QueryRouter(agent_enabled=False)
        decision = router.route("오늘 날씨")
        assert decision.mode == "simple"


class TestAgentEnabled:
    """agent_enabled=True일 때의 복잡도 감지."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        return QueryRouter(agent_enabled=True)

    def test_단순_키워드_질의는_simple(self, router):
        decision = router.route("서울 인구")
        assert decision.mode == "simple"
        assert decision.complexity < 0.5

    @pytest.mark.parametrize(
        "query, expected_kw",
        [
            ("A와 B 비교", "비교"),
            ("무엇이 다른점인가", "다른점"),
            ("정책 변경 사항", "변경"),
            ("그 이유가 무엇인가", "이유"),
            ("어떤 경우에 해당하나요", "어떤 경우"),
            ("장단점을 알려줘", "장단점"),
            ("서로 관계가 있나", "관계"),
            ("종합적으로 분석해줘", "종합"),
        ],
    )
    def test_복합_키워드_매칭은_agent(self, router, query, expected_kw):
        decision = router.route(query)
        assert decision.mode == "agent"
        assert decision.matched_keyword == expected_kw
        assert decision.complexity >= 0.5

    def test_대소문자_무시_vs_키워드(self, router):
        decision = router.route("React VS Vue")
        assert decision.mode == "agent"
        assert decision.matched_keyword == "vs"

    def test_다중_물음표는_agent(self, router):
        decision = router.route("A는 뭔가요? B도 같나요?")
        assert decision.mode == "agent"
        assert "물음표" in decision.reason

    def test_전각_다중_물음표도_agent(self, router):
        decision = router.route("이건 뭐？ 저건 뭐？")
        assert decision.mode == "agent"

    def test_비교_절_패턴은_agent(self, router):
        # "빨간사과와 파란사과의" — 2글자 이상 + 와/과 + 2글자 이상 + 조사
        decision = router.route("빨간사과와 파란사과의 차이점")
        assert decision.mode == "agent"
        # "차이점"이 키워드로 먼저 매칭되므로 그쪽 경로로 분류됨
        assert decision.matched_keyword == "차이"


class TestRouteDecision:
    """RouteDecision 자체의 속성 검증."""

    def test_frozen_dataclass(self):
        d = RouteDecision(mode="simple", reason="x", complexity=0.0)
        with pytest.raises(Exception):
            d.mode = "agent"  # type: ignore[misc]

    def test_matched_keyword_기본값은_None(self):
        d = RouteDecision(mode="simple", reason="x", complexity=0.0)
        assert d.matched_keyword is None

    def test_intent_기본값은_None(self):
        d = RouteDecision(mode="simple", reason="x", complexity=0.0)
        assert d.intent is None


class TestChitchatFastPath:
    """잡담 정규식 fast-path — LLM 호출 없이 chitchat 모드로 라우팅."""

    @pytest.fixture
    def router(self) -> QueryRouter:
        return QueryRouter(agent_enabled=True)

    @pytest.mark.parametrize(
        "query",
        [
            "안녕",
            "안녕하세요",
            "안녕히 가세요",
            "반갑습니다",
            "좋은 아침",
            "좋은 저녁이에요",
            "감사합니다",
            "고맙습니다",
            "고마워",
            "땡큐",
            "수고하셨습니다",
            "너 누구야?",
            "당신은 누구인가요",
            "네 이름이 뭐야",
            "뭐 할 수 있어?",
            "Hello",
            "hi there",
            "thanks",
            "thank you",
            "Good morning",
            "How are you?",
            "bye",
            "goodbye",
            "OK",
            "okay",
            "sure",
            "こんにちは",
            "ありがとう",
            "你好",
            "谢谢",
        ],
    )
    def test_인사_감사_정체성_질의는_chitchat(self, router, query):
        decision = router.route(query)
        assert decision.mode == "chitchat", f"{query!r} should be chitchat"
        assert decision.complexity == 0.0

    def test_도메인_복합_키워드는_chitchat보다_우선(self, router):
        # "안녕"이 포함되어도 "비교"가 더 강한 신호 → agent.
        decision = router.route("안녕 A와 B 비교해줘")
        assert decision.mode == "agent"
        assert decision.matched_keyword == "비교"

    def test_긴_도메인_질의는_chitchat_아님(self, router):
        decision = router.route("RFP에서 NMS 개발자가 무엇을 만들어야 하는지 알려줘")
        assert decision.mode == "simple"

    def test_감사_뒤에_도메인_질의가_붙으면_chitchat_아님(self, router):
        # 정규식은 ^ ... $ 앵커이므로 단순 인사로 끝나지 않으면 매치 X.
        decision = router.route("감사합니다 그런데 NMS는 무엇입니까")
        assert decision.mode == "simple"


class TestChitchatLLMPath:
    """LLM이 chitchat 의도를 반환하면 chitchat 모드로 라우팅."""

    class _FakeClassifier:
        def __init__(self, decisions):
            self._decisions = list(decisions)
            self.calls = 0

        async def classify(self, query):
            self.calls += 1
            if not self._decisions:
                from slm_factory.rag.agent.intent_classifier import IntentDecision
                return IntentDecision(intent="ambiguous", confidence=0.0)
            return self._decisions.pop(0)

    @pytest.mark.asyncio
    async def test_LLM_chitchat_고신뢰는_chitchat(self):
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        # 정규식엔 매치 안 되지만 LLM이 chitchat으로 분류 가능한 잡담.
        classifier = self._FakeClassifier([
            IntentDecision(intent="chitchat", confidence=0.92, reason="일상 잡담"),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("오늘 점심으로 김밥 먹었는데 맛있더라")
        assert d.mode == "chitchat"
        assert d.intent == "chitchat"
        assert classifier.calls == 1

    @pytest.mark.asyncio
    async def test_LLM_chitchat_저신뢰는_agent_fallback(self):
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        classifier = self._FakeClassifier([
            IntentDecision(intent="chitchat", confidence=0.4, reason="확신 없음"),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("음 그냥 뭐랄까")
        # 저신뢰 chitchat은 안전을 위해 agent로 fallback (검색 누락보다 검색 시도가 안전).
        assert d.mode == "agent"

    @pytest.mark.asyncio
    async def test_정규식_매치는_LLM_호출_없이_즉시_chitchat(self):
        classifier = self._FakeClassifier([])  # LLM 호출되면 ambiguous 반환되어 라우팅 달라짐.
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("안녕하세요")
        assert d.mode == "chitchat"
        assert classifier.calls == 0  # 정규식이 fast-path로 끊었으므로 LLM 호출 X.


class TestRouteAsync:
    """Phase 5 — IntentClassifier 주입 기반 route_async."""

    class _FakeClassifier:
        def __init__(self, decisions):
            self._decisions = list(decisions)
            self.calls = 0

        async def classify(self, query):
            self.calls += 1
            if not self._decisions:
                from slm_factory.rag.agent.intent_classifier import IntentDecision
                return IntentDecision(intent="ambiguous", confidence=0.0)
            return self._decisions.pop(0)

    @pytest.mark.asyncio
    async def test_classifier_없으면_키워드_route로_위임(self):
        router = QueryRouter(agent_enabled=True)
        d = await router.route_async("단순 질의")
        assert d.mode == "simple"
        assert d.intent is None

    @pytest.mark.asyncio
    async def test_factual_고신뢰는_simple(self):
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        classifier = self._FakeClassifier([
            IntentDecision(intent="factual", confidence=0.92, reason="단일 사실"),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("어떤 내용인가요")
        assert d.mode == "simple"
        assert d.intent == "factual"
        assert classifier.calls == 1

    @pytest.mark.asyncio
    async def test_factual_저신뢰는_agent(self):
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        classifier = self._FakeClassifier([
            IntentDecision(intent="factual", confidence=0.3),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("음")
        assert d.mode == "agent"

    @pytest.mark.asyncio
    async def test_comparative는_agent(self):
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        classifier = self._FakeClassifier([
            IntentDecision(intent="comparative", confidence=0.9),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("대조")
        assert d.mode == "agent"
        assert d.intent == "comparative"

    @pytest.mark.asyncio
    async def test_ambiguous는_agent_route(self):
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        classifier = self._FakeClassifier([
            IntentDecision(intent="ambiguous", confidence=0.6),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("그거")
        assert d.mode == "agent"
        assert d.intent == "ambiguous"

    @pytest.mark.asyncio
    async def test_키워드가_agent이면_LLM이_factual여도_agent_유지(self):
        """명시적 비교 키워드가 있으면 LLM 결과와 상관없이 agent 유지."""
        from slm_factory.rag.agent.intent_classifier import IntentDecision

        classifier = self._FakeClassifier([
            IntentDecision(intent="factual", confidence=0.95),
        ])
        router = QueryRouter(agent_enabled=True, intent_classifier=classifier)
        d = await router.route_async("A와 B 비교")
        assert d.mode == "agent"
        assert d.intent == "factual"  # LLM 결과도 보존
        assert d.matched_keyword == "비교"

    @pytest.mark.asyncio
    async def test_agent_비활성이면_LLM_호출_안함(self):
        classifier = self._FakeClassifier([])
        router = QueryRouter(agent_enabled=False, intent_classifier=classifier)
        d = await router.route_async("A와 B 비교")
        assert d.mode == "simple"
        assert classifier.calls == 0
