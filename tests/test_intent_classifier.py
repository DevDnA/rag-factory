"""IntentClassifier 테스트 — corpus_profile 주입·프롬프트 헤더 동작."""

from __future__ import annotations

import json

import pytest

from rag_factory.rag.agent.intent_classifier import IntentClassifier
from rag_factory.rag.corpus_profile import CorpusProfile


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttp:
    """post(...)가 호출될 때마다 마지막 prompt를 self.last_prompt에 저장."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.last_prompt: str = ""
        self.calls = 0

    async def post(self, url, json=None, timeout=None):
        self.calls += 1
        self.last_prompt = (json or {}).get("prompt", "")
        return _FakeResponse(self._payload)


class TestCorpusProfileInjection:
    """헤더 주입 검증 — '명칭:' / '핵심 키워드:' 행이 프롬프트 상단에 prepend되는지 확인.

    프롬프트 본문이 ``[본 corpus 도메인 정보]`` 문자열을 안내 문구로 포함하므로,
    그 문자열만으로는 주입 여부를 구별할 수 없습니다. 실제 헤더 본문(``명칭:`` /
    ``핵심 키워드:``)이 질의 앞에 등장하는지로 판단합니다.
    """

    @pytest.mark.asyncio
    async def test_profile_없으면_헤더_본문_주입_없음(self):
        payload = {"response": json.dumps({"intent": "factual", "confidence": 0.9})}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=None,
        )
        await classifier.classify("질의")
        # corpus profile이 없으면 '명칭:' / '핵심 키워드:' 본문이 프롬프트에 없음.
        assert "명칭:" not in http.last_prompt
        assert "핵심 키워드:" not in http.last_prompt

    @pytest.mark.asyncio
    async def test_빈_profile은_헤더_본문_주입_없음(self):
        payload = {"response": json.dumps({"intent": "factual", "confidence": 0.9})}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=CorpusProfile(),
        )
        await classifier.classify("질의")
        assert "명칭:" not in http.last_prompt
        assert "핵심 키워드:" not in http.last_prompt

    @pytest.mark.asyncio
    async def test_profile_채워지면_헤더_본문_주입(self):
        payload = {"response": json.dumps({"intent": "factual", "confidence": 0.9})}
        http = _FakeHttp(payload)
        profile = CorpusProfile(
            name="한국 통신사 RFP",
            summary="통신 인프라 사양 요구사항.",
            keywords=["NMS", "BIS", "MIMO"],
        )
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=profile,
        )
        await classifier.classify("NMS는 무엇입니까")

        prompt = http.last_prompt
        assert "한국 통신사 RFP" in prompt
        assert "NMS" in prompt and "BIS" in prompt
        # 헤더 본문(명칭:)이 사용자 질의보다 먼저 등장해야 LLM 컨텍스트로 작동.
        assert prompt.index("명칭:") < prompt.index("질의:")

    @pytest.mark.asyncio
    async def test_헤더_있어도_분류_결과는_그대로_파싱(self):
        # 헤더가 추가돼도 LLM 응답 파싱 로직은 영향 없음 — 분류 정확도만 변함.
        payload = {"response": json.dumps({"intent": "comparative", "confidence": 0.88, "reason": "비교"})}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http,
            ollama_model="m",
            api_base="http://x",
            cache_ttl=0,
            corpus_profile=CorpusProfile(name="N", keywords=["k"]),
        )
        decision = await classifier.classify("질의")
        assert decision.intent == "comparative"
        assert decision.confidence == pytest.approx(0.88)


class TestInDomainGate:
    """LLM이 general/chitchat으로 분류했어도 query에 corpus 키워드가 매칭되면
    confidence를 낮춰 router가 agent 경로로 흐르도록 유도하는 후처리 게이트.

    `intent` 자체는 변경하지 않음(원본 LLM reasoning 보존) — confidence만 0.5로 강제.
    router.py가 general/chitchat 분기에서 ≥0.7 임계를 쓰므로 자동으로 else(agent)로
    흘러간다.
    """

    @pytest.mark.asyncio
    async def test_general이지만_corpus_keyword_매칭이면_confidence_낮춤(self):
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.95, "reason": "out-of-domain"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(
            name="버스 공공와이파이 RFP",
            summary="버스 와이파이",
            keywords=["WiFi 7", "5G", "정부", "지자체", "AP"],
        )
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify("정부와 지자체의 차이점은 무엇인가요?")
        # intent는 보존 — LLM의 reasoning trail 살아남음
        assert decision.intent == "general"
        # confidence는 0.5 이하로 강제 — router의 general(>=0.7) 분기 미달
        assert decision.confidence <= 0.5
        # reason에 gate 발동 흔적 + 원본 reason 포함
        assert "in-domain gate" in decision.reason
        assert "정부" in decision.reason or "지자체" in decision.reason

    @pytest.mark.asyncio
    async def test_general이고_매칭_없으면_원본_유지(self):
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.92, "reason": "프로그래밍 일반 지식"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(
            name="버스 공공와이파이 RFP",
            keywords=["WiFi 7", "5G", "정부", "지자체"],
        )
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify("파이썬에서 list와 tuple의 차이는?")
        # 매칭 없음 — confidence 그대로
        assert decision.intent == "general"
        assert decision.confidence == pytest.approx(0.92)
        assert "in-domain gate" not in decision.reason

    @pytest.mark.asyncio
    async def test_chitchat도_corpus_매칭이면_낮춤(self):
        # 잡담 fast-path(_CHITCHAT_RE)는 router.py에서 LLM 호출 전에 가로채므로
        # LLM이 chitchat을 반환하는 경우는 LLM 자체가 분류한 결과. 이 경우에도
        # 게이트가 corpus 매칭을 검사한다.
        payload = {"response": json.dumps(
            {"intent": "chitchat", "confidence": 0.9, "reason": "사회적 발화로 보임"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(keywords=["WiFi 7", "NMS"])
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify("WiFi 7 설명 좀")
        assert decision.intent == "chitchat"
        assert decision.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_factual은_게이트_미적용(self):
        payload = {"response": json.dumps(
            {"intent": "factual", "confidence": 0.95, "reason": "fact"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(keywords=["WiFi 7", "정부"])
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify("정부 정책은 무엇입니까?")
        # factual은 게이트 미적용 — 원본 confidence 유지
        assert decision.intent == "factual"
        assert decision.confidence == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_빈_profile은_no_op(self):
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.9, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=CorpusProfile(),  # 빈 profile
        )
        decision = await classifier.classify("정부 차이")
        # profile이 비면 게이트가 동작 안 함 — original confidence 보존
        assert decision.confidence == pytest.approx(0.9)
        assert "in-domain gate" not in decision.reason

    @pytest.mark.asyncio
    async def test_None_profile은_no_op(self):
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.9, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=None,
        )
        decision = await classifier.classify("WiFi 7 무엇")
        assert decision.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_단일_문자_keyword는_매칭_제외(self):
        # 한 글자 키워드는 false positive 위험이 커서 게이트 매칭에서 제외 (최소 길이 2).
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.9, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(keywords=["A", "X"])  # 1자 keywords만
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify("AX 정책")  # "A", "X" 둘 다 substring
        assert decision.confidence == pytest.approx(0.9)  # gate 미발동

    @pytest.mark.asyncio
    async def test_case_insensitive_매칭(self):
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.9, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(keywords=["WiFi 7"])
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        # query는 _normalize에서 lowercase되므로 "wifi 7"로 정규화됨
        decision = await classifier.classify("wifi 7 도입은?")
        assert decision.confidence <= 0.5


class TestContextFreeCommandFormGate:
    """Context-free 명령형 query 감지 보조 게이트 — keyword 매칭 실패 시 보완.

    발동 조건 (세 조건 AND):
      1) 명령형 동사 ("설명하시오/논하시오/서술/제시/기술/요약/정리/...")
      2) ASCII 영문/숫자 토큰 (길이 ≥ 2) 없음 — 고유 entity name 부재
      3) generic referent ("주제/내용/항목/사항/이것/그것/세 가지/...")

    이 조건들이 모두 만족하면 elliptical in-domain query로 판정, conf를 0.5로 강제.
    """

    @pytest.mark.asyncio
    async def test_F1_failing_query_케이스_세_가지_주제_설명하시오(self):
        # 실제 baseline 평가에서 wrongful_refusal 발생한 F1 land mine 쿼리.
        # corpus_profile의 keywords와는 0개 overlap이라 primary 게이트 fire 불가.
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.95, "reason": "out-of-domain"}
        )}
        http = _FakeHttp(payload)
        # 실제 allganize 코퍼스와 같이 일반 동사·referent는 포함하지 않음.
        profile = CorpusProfile(
            name="앙골라 내전 보고서",
            keywords=["UNITA", "MPLA", "다이아몬드", "내전", "평화협상"],
        )
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify(
            "세 가지 주제에 대해 중점과제들을 구체적으로 설명하시오."
        )
        assert decision.intent == "general"
        # context-free 명령형 → conf 0.5 이하로 강제
        assert decision.confidence <= 0.5
        assert "context-free 명령형" in decision.reason

    @pytest.mark.asyncio
    async def test_명령형이지만_ASCII_token_있으면_미발동(self):
        # "S&P 500" 같은 고유 entity가 있으면 OOD 가능성이 더 높음 → 보수적으로 보류.
        # corpus keyword 매칭이 별도로 fire하지 않는 한 원본 confidence 유지.
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.9, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(keywords=["다이아몬드"])
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify(
            "S&P 500 지수의 변화를 설명하시오."
        )
        # ASCII token "500", "S&P"(s&p 정규화 후 letters) → command-form 게이트 미발동.
        # corpus keyword "다이아몬드"도 없음 → 원본 conf 유지.
        assert decision.confidence == pytest.approx(0.9)
        assert "in-domain gate" not in decision.reason

    @pytest.mark.asyncio
    async def test_generic_referent_없으면_미발동(self):
        # 명령형 동사 + ASCII 부재 + generic referent 부재 → 미발동.
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.92, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        profile = CorpusProfile(keywords=["다이아몬드"])
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify("좋은 점심 메뉴를 추천해.")
        # "추천해"는 명령형이지만 _COMMAND_FORM_VERBS에 없음, generic referent도 없음.
        assert decision.confidence == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_corpus_profile_없어도_command_form_단독_발동(self):
        # context-free 명령형 게이트는 corpus_profile 부재여도 독립적으로 fire.
        # 안전망: profile 없이도 명백한 OOD가 아닌 elliptical query는 agent로 흘려야 함.
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.95, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=None,
        )
        decision = await classifier.classify(
            "세 가지 주제에 대해 중점과제들을 구체적으로 설명하시오."
        )
        # profile 없어도 command-form 보조 게이트가 fire → conf 0.5 이하.
        assert decision.confidence <= 0.5
        assert "context-free 명령형" in decision.reason

    @pytest.mark.asyncio
    async def test_factual은_command_form_게이트_미적용(self):
        # 게이트는 general/chitchat만 대상. factual은 그대로 둠 (router에서 simple로 흐름).
        payload = {"response": json.dumps(
            {"intent": "factual", "confidence": 0.95, "reason": "fact"}
        )}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=None,
        )
        decision = await classifier.classify(
            "세 가지 주제에 대해 중점과제들을 구체적으로 설명하시오."
        )
        assert decision.intent == "factual"
        assert decision.confidence == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_keyword_매칭이_있으면_command_form_보다_keyword_reason_우선(self):
        # 둘 다 fire 가능한 query — corpus keyword 매칭이 reason에 명시되어야 함
        # (디버깅 일관성: primary 신호를 더 강하게 표시).
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.95, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        # "주제"가 keyword에 명시 → corpus 매칭 fire. command-form 신호도 충족.
        profile = CorpusProfile(keywords=["주제"])
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=profile,
        )
        decision = await classifier.classify(
            "세 가지 주제에 대해 중점과제들을 구체적으로 설명하시오."
        )
        # 둘 다 fire 가능하지만 primary(keyword)가 reason에 명시.
        assert decision.confidence <= 0.5
        assert "corpus keyword 매칭" in decision.reason
        assert "context-free 명령형" not in decision.reason

    @pytest.mark.asyncio
    async def test_여러_가지_사항_요약해줘(self):
        # 명령형 + generic referent + ASCII 부재 → 발동.
        payload = {"response": json.dumps(
            {"intent": "general", "confidence": 0.9, "reason": "ood"}
        )}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=CorpusProfile(),
        )
        decision = await classifier.classify("여러 가지 사항을 요약해줘.")
        assert decision.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_general이_아니면_미발동(self):
        # ambiguous/analytical 등은 router가 이미 agent 경로로 흘리므로 게이트 미적용.
        payload = {"response": json.dumps(
            {"intent": "analytical", "confidence": 0.92, "reason": "분석"}
        )}
        http = _FakeHttp(payload)
        classifier = IntentClassifier(
            http_client=http, ollama_model="m", api_base="http://x",
            cache_ttl=0, corpus_profile=None,
        )
        decision = await classifier.classify(
            "세 가지 주제에 대해 중점과제들을 구체적으로 설명하시오."
        )
        # analytical은 그대로 — router는 어차피 agent로 흘림.
        assert decision.intent == "analytical"
        assert decision.confidence == pytest.approx(0.92)
