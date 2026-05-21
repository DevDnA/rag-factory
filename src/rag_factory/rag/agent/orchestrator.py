"""`/auto` 엔드포인트를 위한 라우팅·스트리밍 오케스트레이터.

``AgentOrchestrator``는 질의를 받아 다음을 수행합니다.

1. ``QueryRouter``로 ``simple`` | ``agent`` 경로 결정
2. ``{"type": "route"}`` 이벤트 발행
3. 선택된 경로의 이벤트 스트림을 그대로 전달

이벤트는 dict 형태로 yield되며, ``server.py``는 SSE로 framing만 담당합니다.
이렇게 분리하면 라우팅·세션 관리·agent 이벤트 매핑 로직을 HTTP 레이어
없이 단독으로 테스트할 수 있습니다.

동작 보존 원칙
--------------
기존 ``server.py``의 ``/auto`` 핸들러가 발행하던 이벤트 순서와 필드는
**바이트 수준으로 동일하게** 유지됩니다. 본 모듈은 순수 추출 리팩터링이며
새로운 기능(planner, verifier 등)은 포함하지 않습니다.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Callable, Iterable

from ...utils import get_logger
from .persona_router import PersonaRouter
from .personas.base import Persona
from .router import QueryRouter

logger = get_logger("rag.agent.orchestrator")


SimpleStreamFn = Callable[[str], AsyncGenerator[dict, None]]
"""단순 RAG 스트림 함수 — query를 받아 이벤트 dict를 yield합니다."""

# 컨텍스트 합성 시 prompt에 삽입되는 참고 문서의 최대 길이.
_SYNTHESIS_CONTEXT_CHAR_LIMIT = 6000

# 합성 안전망 — num_predict가 -1(무제한)일 때 적용할 하드 캡.
# Ollama 기본 EOS 도달 실패 + 빈약한 컨텍스트로 인한 무한 paraphrase 루프 방어.
_SYNTHESIS_NUM_PREDICT_CAP = 1500

# 거부 게이트가 발동될 때의 응답 메시지.
# (임계값은 ``rag.agent.refusal_min_score`` 설정으로 조절합니다 — 0.0이면 비활성.)
_REFUSAL_MESSAGE = (
    "관련 정보를 문서에서 찾지 못했습니다. "
    "질문을 다른 방식으로 표현하거나 관련 문서를 추가해 주세요."
)


class AgentOrchestrator:
    """``/auto`` 경로의 라우팅·스트리밍 오케스트레이터.

    Parameters
    ----------
    router:
        복잡도 기반 라우팅 결정기.
    app_state:
        FastAPI ``app.state`` — 런타임에 ``agent_session_manager``,
        ``agent_tool_registry``, ``http_client``를 조회합니다.
    config:
        ``SLMConfig`` — ``rag.agent``, ``rag.request_timeout``,
        ``rag.max_tokens`` 등을 참조합니다.
    ollama_model:
        Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    rag_max_tokens:
        LLM 생성 최대 토큰.
    simple_stream_fn:
        단순 RAG 스트림을 생성하는 async generator factory — ``app.state``가
        보유한 Qdrant·임베딩·reranker 등 의존성을 클로저로 캡처한 함수를
        주입받습니다.
    """

    def __init__(
        self,
        *,
        router: QueryRouter,
        app_state: Any,
        config: Any,
        ollama_model: str,
        api_base: str,
        rag_max_tokens: int,
        simple_stream_fn: SimpleStreamFn,
    ) -> None:
        self._router = router
        self._app_state = app_state
        self._config = config
        self._ollama_model = ollama_model
        self._api_base = api_base
        self._rag_max_tokens = rag_max_tokens
        self._simple_stream_fn = simple_stream_fn
        self._persona_router = PersonaRouter(
            enabled=getattr(config.rag.agent, "personas_enabled", False),
            custom_registry=self._build_custom_personas(config),
        )
        self._skill_registry = self._build_skill_registry(config)
        self._hook_registry = self._build_hook_registry(config)
        # observation 이벤트로 클라이언트에 보낼 때의 길이 제한 — config에서 캐시.
        self._obs_preview_limit = getattr(
            config.rag.agent, "observation_preview_limit", 300
        )
        # 모든 LLM 호출에 사용할 Ollama keep_alive 값 — config에서 캐시.
        self._keep_alive = getattr(
            config.rag.agent, "ollama_keep_alive", "5m"
        )

    @staticmethod
    def _build_hook_registry(config: Any):
        """config.builtin_hooks로 지정된 hook들을 등록한 registry 반환."""
        from .hooks import build_default_registry

        enabled = getattr(config.rag.agent, "hooks_enabled", False)
        names = list(getattr(config.rag.agent, "builtin_hooks", []) or [])
        return build_default_registry(enabled=enabled, builtin_names=names)

    def register_hook(self, point: str, fn):
        """외부 코드가 orchestrator에 사용자 정의 hook을 등록할 수 있도록 제공."""
        self._hook_registry.register(point, fn)

    def _model_for(self, slot: str) -> str:
        """Phase 9 — 컴포넌트별 모델 슬롯 조회. 빈 값이면 기본 모델로 fallback."""
        models_cfg = getattr(self._config.rag.agent, "models", None)
        if models_cfg is None:
            return self._ollama_model
        value = getattr(models_cfg, f"{slot}_model", "") or ""
        return value.strip() or self._ollama_model

    def _native_thinking(self) -> bool:
        """품질 경로(Planner/Verifier/Reflector/synthesis)에 Ollama native thinking 적용 여부."""
        return bool(getattr(self._config.rag.agent, "native_thinking", False))

    @staticmethod
    def _build_custom_personas(config: Any):
        """Phase 14 — custom_personas_dir가 설정되면 YAML에서 로드."""
        from .persona_loader import CustomPersonaRegistry, load_custom_personas

        path = (getattr(config.rag.agent, "custom_personas_dir", "") or "").strip()
        if not path:
            return None
        try:
            personas = load_custom_personas(path)
        except Exception as exc:  # pragma: no cover — loader는 자체 never-raise
            logger.warning("Custom personas 로드 실패: %s", exc)
            personas = []
        if personas:
            logger.info(
                "Custom personas 로드: %d개 (%s)",
                len(personas),
                ", ".join(p.name for p in personas),
            )
        return CustomPersonaRegistry(personas)

    @staticmethod
    def _build_skill_registry(config: Any):
        """Skills 디렉터리에서 Skill 목록을 로드 — 실패 시 빈 registry."""
        from .skills import SkillRegistry, load_skills_from_dir

        if not getattr(config.rag.agent, "skills_enabled", False):
            return SkillRegistry()
        skills_dir = getattr(config.rag.agent, "skills_dir", "skills")
        try:
            skills = load_skills_from_dir(skills_dir)
        except Exception as exc:  # pragma: no cover — loader는 never-raise
            logger.warning("Skills 로드 실패: %s — 빈 registry 사용", exc)
            skills = []
        if skills:
            logger.info(
                "Skills 로드: %d개 (%s)",
                len(skills),
                ", ".join(s.name for s in skills),
            )
        return SkillRegistry(skills)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_auto(
        self, query: str, session_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """``/auto``의 전체 이벤트 스트림을 생성합니다.

        raw_query는 사용자 입력 그대로 보존되어 세션 history에 저장되고,
        normalized_query는 pre_query hook을 거쳐 router/planner/synthesis 등
        downstream 단계에 전달됩니다 (의미: 대화 history와 LLM 컨텍스트가
        동일한 사용자 발화를 보장).
        """
        raw_query = query
        normalized_query = await self._hook_registry.run("pre_query", query)
        # IntentClassifier가 주입된 router는 ``route_async()``를 통해 LLM 분류를 수행.
        decision = await self._router.route_async(normalized_query)
        logger.info(
            "라우팅 결정: mode=%s complexity=%.2f reason=%s intent=%s",
            decision.mode,
            decision.complexity,
            decision.reason,
            decision.intent,
        )

        route_event: dict[str, Any] = {"type": "route", "mode": decision.mode}
        if decision.intent is not None:
            route_event["intent"] = decision.intent
        yield route_event

        # Intent Verbalization (oh-my-openagent의 'Verbalize Intent' 패턴) —
        # 라우팅 결정의 근거를 짧은 thought 이벤트로 표면화하여 follow-up
        # 처리의 일관성과 디버깅 가시성을 높입니다.
        if (
            getattr(self._config.rag.agent, "intent_verbalization_enabled", False)
            and self._config.rag.agent.stream_reasoning
        ):
            verbalization = self._verbalize_intent(decision)
            if verbalization:
                yield {
                    "type": "thought",
                    "content": verbalization,
                    "iteration": 0,
                }

        # Chitchat: 인사·짧은 사회적 발화 — Qdrant 우회 + chitchat 합성 프롬프트.
        if decision.mode == "chitchat":
            async for event in self._stream_chitchat(
                normalized_query, session_id, raw_query=raw_query
            ):
                yield event
            return

        # General: corpus 외 일반 지식·코드·잡학 — Qdrant 우회 + general 합성 프롬프트.
        # 안전망(2단 임계): IntentClassifier가 general(OOD)로 분류했더라도 corpus
        # 의미적 유사도로 in-domain 정정. ``_check_corpus_override`` 참조.
        if decision.mode == "general":
            override = await self._check_corpus_override(normalized_query, decision)
            if override is not None:
                async for event in self._emit_corpus_override(
                    normalized_query, override, source_label="general"
                ):
                    yield event
                return
            async for event in self._stream_general(
                normalized_query, session_id, raw_query=raw_query
            ):
                yield event
            return

        # Agent override 안전망 — IntentClassifier가 exploratory/analytical 등으로
        # agent에 보낸 query라도 corpus가 사용자 원문에 직접 잘 매칭되면 simple로
        # 정정. planner가 검색어를 추상화해 매칭이 폭락하는 비대칭(예: "표형식"은
        # general→corpus override→simple로 성공하지만 "그래프형식"은 exploratory→
        # agent→planner abstraction→0.03 매칭 실패)을 해소합니다.
        # 제외 조건:
        #   · matched_keyword (비교/이유/차이 등) — 사용자가 명시적으로 분해/비교를
        #     요청한 신호이므로 corpus가 잘 맞아도 agent 경로 유지
        #   · intent="ambiguous" — clarifier에 위임해 사용자에게 명확화 질문 반환
        if (
            decision.mode == "agent"
            and decision.matched_keyword is None
            and decision.intent != "ambiguous"
        ):
            override = await self._check_corpus_override(normalized_query, decision)
            if override is not None:
                async for event in self._emit_corpus_override(
                    normalized_query, override, source_label="agent"
                ):
                    yield event
                return

        # Clarifier: ambiguous 의도 + clarifier 활성화. corpus profile이 비어
        # 있지 않으면 query에 corpus 키워드가 하나도 없을 때 out-of-domain
        # ambiguous로 보고 clarifier 대신 general 경로로 라우팅 — 사용자에게
        # 같은 질문 반복 요구하지 않음. profile이 비면 종전 clarifier 행동 유지.
        if (
            decision.intent == "ambiguous"
            and self._config.rag.agent.clarifier_enabled
        ):
            # ambiguous는 항상 clarifier로 — corpus keyword 누락(추출 실패)이
            # OOD가 아니므로, 사용자에게 직접 명확화 질문을 던져 in-domain 여부를
            # 자연스럽게 가리도록 합니다. 키워드 매칭 기반 OOD 추측은 짧은
            # 도메인 query("임차운영 vs 직접구축")를 잘못 거절하는 land mine.
            async for event in self._stream_clarifier(
                normalized_query, session_id, raw_query=raw_query
            ):
                yield event
            return

        persona = self._persona_router.select(decision.intent)
        tabular_override = False
        if self._has_tabular_intent(normalized_query):
            from .personas.comparator import Comparator
            persona = Comparator()
            tabular_override = True
            logger.info("Persona override: 표·비교 키워드 감지 → Comparator")
        persona, persona_trace = self._compose_persona_with_trace(
            decision, persona, normalized_query=normalized_query
        )
        persona_trace["tabular_override"] = tabular_override
        if persona is not None:
            logger.info("Persona 선택: %s", persona.name)
        # Phase 14 — persona 선택 trace를 thought 이벤트로 발행해 사용자가
        # composition 발동 여부와 trigger를 reasoning panel에서 즉시 확인 가능.
        # simple 경로에서는 persona가 사용되지 않으므로 발행 생략. stream_reasoning이
        # 꺼져 있으면 다른 thought 이벤트와 동일하게 발행 생략 (관측성 토글 존중).
        if (
            decision.mode != "simple"
            and persona is not None
            and getattr(self._config.rag.agent, "stream_reasoning", True)
        ):
            line = self._format_persona_trace(persona_trace)
            if tabular_override:
                line = f"{line} [tabular override]"
            yield {"type": "thought", "content": line}

        if decision.mode == "simple":
            async for event in self._wrap_simple_with_verification(
                normalized_query
            ):
                yield event
        else:
            async for event in self._stream_agent(
                normalized_query, session_id, persona=persona, raw_query=raw_query
            ):
                yield event

    async def handle_agent(
        self, query: str, session_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """``/agent`` stream 모드 — 라우팅 없이 항상 agent 경로.

        ``handle_auto``와 달리 ``{type: route}`` 이벤트를 발행하지 않으며,
        ``planner_enabled`` 설정에 따라 planner 또는 legacy 경로로 분기합니다.
        """
        raw_query = query
        normalized_query = await self._hook_registry.run("pre_query", query)
        async for event in self._stream_agent(
            normalized_query, session_id, raw_query=raw_query
        ):
            yield event

    # ------------------------------------------------------------------
    # 내부: Clarifier 경로 — ambiguous 의도에 대한 역질문
    # ------------------------------------------------------------------

    async def _stream_clarifier(
        self,
        query: str,
        session_id: str | None,
        *,
        raw_query: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Clarifier persona로 명확화 질문을 생성·반환합니다.

        ``raw_query``는 사용자가 입력한 원문이며 세션 history에 그대로 저장.
        ``query``는 정규화된 텍스트로 LLM 프롬프트(history)에 전달됩니다.
        """
        from .personas.clarifier import Clarifier
        from .session import Message

        session_store = self._app_state.agent_session_manager
        http_client = self._app_state.http_client
        aux_timeout = min(self._config.rag.request_timeout, 30.0)

        sid, _ = session_store.get_or_create(session_id)
        # Clarifier 경로에서도 긴 대화는 압축이 필요함 — 기록 전에 시도해 history를 줄임.
        await self._maybe_compress_memory(
            session_store, sid, http_client, aux_timeout
        )
        history = session_store.format_history(sid)
        # 세션에는 사용자가 실제로 입력한 raw_query를 저장 (history와 입력 일치).
        session_store.add_message(
            sid,
            Message(role="user", content=raw_query if raw_query is not None else query),
        )

        clarifier = Clarifier(
            http_client=http_client,
            ollama_model=self._model_for("clarifier"),
            api_base=self._api_base,
            request_timeout=min(self._config.rag.request_timeout, 15.0),
            max_questions=self._config.rag.agent.clarifier_max_questions,
            keep_alive=self._keep_alive,
        )
        result = await clarifier.generate_questions(query, history=history)

        # 세션에 assistant 턴으로 기록 — 다음 턴에 이전 역질문 맥락을 이어감.
        summary = "명확화 질문: " + " / ".join(result.questions)
        session_store.add_message(sid, Message(role="assistant", content=summary))

        yield {
            "type": "clarification",
            "questions": result.questions,
            "is_fallback": result.metadata.get("is_fallback", False),
        }
        yield {"type": "done", "session_id": sid}

    # ------------------------------------------------------------------
    # 내부: Chitchat 경로 — RAG 검색 우회 LLM 직답
    # ------------------------------------------------------------------

    async def _stream_chitchat(
        self,
        query: str,
        session_id: str | None,
        *,
        raw_query: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """잡담·인사·자기소개 등 RAG 무관 발화에 LLM으로 직답합니다.

        Qdrant·planner·verifier 등 모든 게이트를 우회하고 synthesis 모델로
        짧게 응답합니다. 세션 history는 유지되어 다중 턴 잡담이 자연스럽게 이어집니다.
        """
        from .prompts import CHITCHAT_SYNTHESIS_PROMPT
        from .session import Message

        session_store = self._app_state.agent_session_manager
        http_client = self._app_state.http_client

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)
        session_store.add_message(
            sid,
            Message(
                role="user",
                content=raw_query if raw_query is not None else query,
            ),
        )

        prompt = CHITCHAT_SYNTHESIS_PROMPT.format(
            history=f"{history}\n" if history else "",
            query=query,
        )

        payload = {
            "model": self._model_for("synthesis"),
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": self._rag_max_tokens if self._rag_max_tokens > 0 else _SYNTHESIS_NUM_PREDICT_CAP,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.25,
            },
        }

        answer_parts: list[str] = []
        try:
            async with http_client.stream(
                "POST",
                f"{self._api_base}/api/generate",
                json=payload,
                timeout=self._config.rag.request_timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        answer_parts.append(token)
                        yield {"type": "token", "content": token}
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.error("Chitchat 합성 실패: %s", exc)
            fallback = "안녕하세요. 무엇을 도와드릴까요?"
            answer_parts = [fallback]
            yield {"type": "token", "content": fallback}

        answer = "".join(answer_parts).strip()
        if answer:
            session_store.add_message(sid, Message(role="assistant", content=answer))

        yield {"type": "done", "session_id": sid}

    # ------------------------------------------------------------------
    # 내부: General 경로 — 코퍼스 외 일반 지식 LLM 직답
    # ------------------------------------------------------------------

    async def _stream_general(
        self,
        query: str,
        session_id: str | None,
        *,
        raw_query: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """본 corpus 도메인 외 질의를 정중히 거절하고 도메인 안내로 유도합니다.

        본 시스템은 RAG 기반 도메인 추론 전용이며, 도메인 외 질의에 LLM 학습 지식으로
        답변하지 않습니다 (학습 시점 이후 사실이 바뀌었거나 부정확할 위험). corpus
        profile 헤더의 도메인 정보를 근거로 사용자에게 어떤 질문에 도움을 드릴 수
        있는지 안내합니다.
        """
        from .prompts import GENERAL_SYNTHESIS_PROMPT
        from .session import Message

        session_store = self._app_state.agent_session_manager
        http_client = self._app_state.http_client

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)
        session_store.add_message(
            sid,
            Message(
                role="user",
                content=raw_query if raw_query is not None else query,
            ),
        )

        # corpus profile 헤더 — 거절문에 "어떤 도메인 질문에 답할 수 있는지" 안내를
        # 생성하기 위한 근거.
        corpus_header = ""
        profile = getattr(self._app_state, "corpus_profile", None)
        if profile is not None:
            header_text = profile.to_prompt_header() if hasattr(profile, "to_prompt_header") else ""
            if header_text:
                corpus_header = f"{header_text}\n\n"

        prompt = GENERAL_SYNTHESIS_PROMPT.format(
            history=f"{history}\n" if history else "",
            corpus_header=corpus_header,
            query=query,
        )

        payload = {
            "model": self._model_for("synthesis"),
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": self._rag_max_tokens if self._rag_max_tokens > 0 else _SYNTHESIS_NUM_PREDICT_CAP,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.25,
            },
        }

        answer_parts: list[str] = []
        try:
            async with http_client.stream(
                "POST",
                f"{self._api_base}/api/generate",
                json=payload,
                timeout=self._config.rag.request_timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        answer_parts.append(token)
                        yield {"type": "token", "content": token}
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.error("General 합성 실패: %s", exc)
            fallback = "죄송합니다. 답변 생성 중 문제가 발생했습니다."
            answer_parts = [fallback]
            yield {"type": "token", "content": fallback}

        answer = "".join(answer_parts).strip()
        if answer:
            session_store.add_message(sid, Message(role="assistant", content=answer))

        yield {"type": "done", "session_id": sid}

    # ------------------------------------------------------------------
    # 내부: agent 경로 디스패치
    # ------------------------------------------------------------------

    async def _stream_agent(
        self,
        query: str,
        session_id: str | None,
        *,
        persona: Persona | None = None,
        raw_query: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """``planner_enabled`` 설정에 따라 planner 또는 legacy 경로로 디스패치합니다."""
        if self._config.rag.agent.planner_enabled:
            async for ev in self._stream_agent_planner(
                query, session_id, persona=persona, raw_query=raw_query
            ):
                yield ev
        else:
            async for ev in self._stream_agent_legacy(
                query, session_id, raw_query=raw_query
            ):
                yield ev

    # ------------------------------------------------------------------
    # 내부: legacy agent 경로 — 기존 ReAct AgentLoop
    # ------------------------------------------------------------------

    async def _stream_agent_legacy(
        self,
        query: str,
        session_id: str | None,
        *,
        raw_query: str | None = None,
        skip_user_message: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """Agent RAG legacy 경로 — 세션 관리 + AgentLoop run_stream 이벤트 매핑.

        ``skip_user_message=True``는 planner 경로에서 fallback으로 진입할 때
        이미 user 메시지가 기록되어 있는 경우에 사용합니다 (이중 기록 방지).
        """
        from .loop import AgentLoop
        from .session import Message

        session_store = self._app_state.agent_session_manager
        tool_registry = self._app_state.agent_tool_registry
        http_client = self._app_state.http_client

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)

        agent = AgentLoop(
            http_client=http_client,
            tool_registry=tool_registry,
            ollama_model=self._model_for("synthesis"),
            api_base=self._api_base,
            max_iterations=self._config.rag.agent.max_iterations,
            max_tokens=self._rag_max_tokens,
            request_timeout=self._config.rag.request_timeout,
            keep_alive=self._keep_alive,
        )

        if not skip_user_message:
            session_store.add_message(
                sid,
                Message(
                    role="user",
                    content=raw_query if raw_query is not None else query,
                ),
            )

        stream_reasoning = self._config.rag.agent.stream_reasoning
        answer_parts: list[str] = []
        final_sources: list[dict] = []
        preview_limit = self._obs_preview_limit

        try:
            async for event in agent.run_stream(query, history):
                if event.type == "thought" and stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": event.content,
                        "iteration": event.iteration,
                    }
                elif event.type == "action" and stream_reasoning:
                    yield {
                        "type": "action",
                        "content": event.content,
                        "input": event.metadata.get("input", {}),
                        "iteration": event.iteration,
                    }
                elif event.type == "observation" and stream_reasoning:
                    obs_preview = event.content[:preview_limit]
                    if len(event.content) > preview_limit:
                        obs_preview += "..."
                    yield {
                        "type": "observation",
                        "content": obs_preview,
                        "iteration": event.iteration,
                    }
                elif event.type == "token":
                    answer_parts.append(event.content)
                    yield {"type": "token", "content": event.content}
                elif event.type == "done":
                    final_sources = event.metadata.get("sources", [])
                elif event.type == "error":
                    yield {
                        "type": "token",
                        "content": "[오류] 처리 중 문제가 발생했습니다.",
                    }
        except Exception as exc:
            logger.error("Agent 스트리밍 오류: %s", exc)
            yield {
                "type": "token",
                "content": "[오류] 처리 중 문제가 발생했습니다.",
            }

        answer = "".join(answer_parts)
        if answer.strip():
            session_store.add_message(sid, Message(role="assistant", content=answer))
            sources_payload = [
                {
                    "content": s.get("content", ""),
                    "doc_id": s.get("doc_id", ""),
                    "source_doc_id": s.get("source_doc_id", ""),
                    "score": s.get("score", 0.0),
                }
                for s in final_sources
            ]
            yield {"type": "sources", "sources": sources_payload}
        yield {"type": "done", "session_id": sid}

    # ------------------------------------------------------------------
    # 내부: planner 경로 — plan → execute → verify → synthesize
    # ------------------------------------------------------------------

    async def _stream_agent_planner(
        self,
        query: str,
        session_id: str | None,
        *,
        persona: Persona | None = None,
        raw_query: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Planner 기반 오케스트레이션 경로.

        설계 원칙
        ---------
        - **드래프트 vs 발행 분리**: 합성은 ``_collect_synthesis``로 한 번에
          수집해 yield하지 않고, 합성 완료 후 **최종 답변만** chunk 단위
          ``token`` 이벤트로 발행합니다 — 답변 중복 yield 방지(HIGH-1/HIGH-2).
        - **세션 user 메시지 우선 기록**: planner.plan() 호출 전에 user 메시지를
          기록해 follow-up 질의의 plan이 history를 반영하도록 합니다(HIGH-3).
        - **raw vs normalized**: 세션 history에는 ``raw_query``를, planner/synthesis
          downstream에는 ``query``(normalized)를 사용해 사용자 입력과 컨텍스트를
          분리합니다(MED-1).
        """
        from .planner import Planner
        from .session import Message
        from .verifier import Verifier

        http_client = self._app_state.http_client

        stream_reasoning = self._config.rag.agent.stream_reasoning
        # Planner/Verifier는 메인 timeout보다 짧게 — 빠른 실패로 fallback 경로 확보.
        aux_timeout = min(self._config.rag.request_timeout, 30.0)
        preview_limit = self._obs_preview_limit

        # --- HIGH-3: user 메시지 기록을 planner.plan() **이전**에 수행 -------
        session_store = self._app_state.agent_session_manager
        tool_registry = self._app_state.agent_tool_registry

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)
        session_store.add_message(
            sid,
            Message(
                role="user",
                content=raw_query if raw_query is not None else query,
            ),
        )

        # Plan 생성.
        planner = Planner(
            http_client=http_client,
            ollama_model=self._model_for("planner"),
            api_base=self._api_base,
            request_timeout=aux_timeout,
            keep_alive=self._keep_alive,
            native_thinking=self._native_thinking(),
        )
        plan = await planner.plan(query)

        # Planner가 사용자 query를 추상화한 검색어로 첫 step을 만들면 임베딩 매칭이
        # 폭락해 refusal gate에 걸리는 케이스가 있음(예: "그래프형식으로 정리"가
        # planner에서 "NMS 개발 업체 제안서 준비사항"으로 변환되어 0.03 매칭).
        # 첫 search/lookup step의 ``query`` 인자를 사용자 원문으로 강제해 항상
        # 사용자 원문 grounding을 보장. 이후 step들은 planner의 분해를 그대로
        # 보존해 multi-step decomposition이 유지됩니다.
        if (
            getattr(self._config.rag.agent, "planner_preserve_first_query", False)
            and not plan.is_fallback
        ):
            for step in plan.steps:
                if step.tool not in ("search", "lookup"):
                    continue
                if not isinstance(step.args, dict):
                    break
                planner_query = str(step.args.get("query", "")).strip()
                if planner_query == query.strip():
                    break
                step.args = {**step.args, "query": query}
                if not step.reason:
                    step.reason = "사용자 원문 보존 검색"
                break

        # Persona가 도구 권한을 제한하면 plan step을 필터링.
        if persona is not None and persona.allowed_tools is not None:
            allowed = persona.allowed_tools
            if allowed:
                original_count = len(plan.steps)
                plan.steps = [s for s in plan.steps if s.tool in allowed]
                if len(plan.steps) < original_count:
                    logger.debug(
                        "Persona '%s' 도구 화이트리스트로 step %d → %d",
                        persona.name,
                        original_count,
                        len(plan.steps),
                    )
            # 빈 allowed_tools는 "도구 없음"이므로 plan을 비움.
            else:
                plan.steps = []

        # Fallback 게이트 — planner가 구조적으로 실패했으면 legacy 경로로 위임.
        # user 메시지는 이미 위에서 기록했으므로 legacy에는 ``skip_user_message=True``.
        # 또한 planner가 생성한 sid를 그대로 사용해 동일 세션에 assistant 답변이
        # 기록되도록 합니다 (이중 기록 + 세션 분기 방지).
        if plan.is_fallback and self._config.rag.agent.legacy_fallback_enabled:
            logger.warning(
                "Planner fallback (%s) — legacy AgentLoop 경로로 전환",
                plan.rationale,
            )
            async for event in self._stream_agent_legacy(
                query,
                sid,
                raw_query=raw_query,
                skip_user_message=True,
            ):
                yield event
            return

        # plan.rationale이 있을 때만 초기 요약 thought를 발행합니다.
        # rationale이 비어 있으면 "계획: fact 전략, 1개 step" 같은 저정보 텍스트만
        # 나가 UI 노이즈가 되므로, 후속 action 이벤트가 상태 표시를 대신하도록 둡니다.
        if stream_reasoning and plan.rationale:
            yield {
                "type": "thought",
                "content": f"계획({plan.strategy}): {plan.rationale}",
                "iteration": 0,
            }

        all_sources: list[dict] = []
        seen_doc_ids: set[str] = set()
        context_parts: list[str] = []

        # --- Plan step 실행 -------------------------------------------
        # 병렬 조건: parallel_steps=True + 모든 step이 parallel_safe + 2개 이상.
        # ToolSpec.parallel_safe 메타를 신뢰해 read-only 도구만 병렬화합니다.
        def _tool_is_parallel_safe(tool_name: str) -> bool:
            # ToolRegistry는 ``get`` 또는 ``_tools`` 사전을 노출 — get으로 조회.
            getter = getattr(tool_registry, "get", None)
            if not callable(getter):
                # MagicMock 등 테스트 fixture 호환: search-only fallback 정책 유지.
                return tool_name == "search"
            spec = getter(tool_name)
            if spec is None:
                return False
            return bool(getattr(spec, "parallel_safe", False))

        can_parallelize = (
            self._config.rag.agent.parallel_steps
            and len(plan.steps) >= 2
            and all(_tool_is_parallel_safe(step.tool) for step in plan.steps)
        )

        if can_parallelize:
            # 동시 실행 후 결과를 plan 순서대로 이벤트 발행.
            try:
                results = await asyncio.gather(
                    *[
                        tool_registry.execute(step.tool, step.args)
                        for step in plan.steps
                    ],
                    return_exceptions=True,
                )
            except Exception as exc:
                logger.warning("병렬 step 실행 실패: %s", exc)
                results = [exc] * len(plan.steps)

            for i, (step, result) in enumerate(zip(plan.steps, results), start=1):
                if stream_reasoning and step.reason:
                    yield {
                        "type": "thought",
                        "content": step.reason,
                        "iteration": i,
                    }
                if stream_reasoning:
                    yield {
                        "type": "action",
                        "content": step.tool,
                        "input": step.args,
                        "iteration": i,
                    }
                if isinstance(result, Exception):
                    logger.warning(
                        "병렬 step '%s' 실패: %s — 건너뜁니다", step.tool, result
                    )
                    continue

                self._dedup_extend(all_sources, seen_doc_ids, result.sources)
                context_parts.append(result.text)

                if stream_reasoning:
                    yield {
                        "type": "observation",
                        "content": self._format_observation_summary(result),
                        "iteration": i,
                    }
        else:
            # 직렬 실행 — 기본 경로. 도구 간 의존성이 있을 수 있어 안전.
            for i, step in enumerate(plan.steps, start=1):
                if stream_reasoning and step.reason:
                    yield {
                        "type": "thought",
                        "content": step.reason,
                        "iteration": i,
                    }
                if stream_reasoning:
                    yield {
                        "type": "action",
                        "content": step.tool,
                        "input": step.args,
                        "iteration": i,
                    }

                try:
                    result = await tool_registry.execute(step.tool, step.args)
                except Exception as exc:
                    logger.warning("도구 '%s' 실행 실패: %s", step.tool, exc)
                    continue

                self._dedup_extend(all_sources, seen_doc_ids, result.sources)
                context_parts.append(result.text)

                if stream_reasoning:
                    yield {
                        "type": "observation",
                        "content": self._format_observation_summary(result),
                        "iteration": i,
                    }

        # --- Verifier 기반 repair 루프 ---------------------------------
        repair_iteration = len(plan.steps)
        if self._config.rag.agent.verifier_enabled:
            verifier = Verifier(
                http_client=http_client,
                ollama_model=self._model_for("verifier"),
                api_base=self._api_base,
                request_timeout=aux_timeout,
                keep_alive=self._keep_alive,
                native_thinking=self._native_thinking(),
            )
            max_repairs = self._config.rag.agent.verifier_max_repairs
            for _ in range(max_repairs):
                context_str = "\n\n".join(context_parts)
                decision = await verifier.evaluate(query, context_str)
                if not decision.needs_repair:
                    break

                repair_iteration += 1
                suggested = decision.suggested_query or ""

                if stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": (
                            f"추가 검색 필요: {decision.reason} → '{suggested}'"
                        ),
                        "iteration": repair_iteration,
                    }
                    yield {
                        "type": "action",
                        "content": "search",
                        "input": {"query": suggested},
                        "iteration": repair_iteration,
                    }

                try:
                    repair_result = await tool_registry.execute(
                        "search", {"query": suggested}
                    )
                except Exception as exc:
                    logger.warning("Repair search 실패: %s", exc)
                    break

                self._dedup_extend(
                    all_sources, seen_doc_ids, repair_result.sources
                )
                context_parts.append(repair_result.text)

                if stream_reasoning:
                    yield {
                        "type": "observation",
                        "content": self._format_observation_summary(repair_result),
                        "iteration": repair_iteration,
                    }

        # --- 답변 합성(드래프트) ---------------------------------------
        # 이전 턴의 참조 문서를 synthesis 컨텍스트에 주입(follow-up 연속성).
        prior_context = self._format_prior_context(session_store, sid)

        def _build_context() -> str:
            """현재 시점 context_parts + prior_context + skill_addon로 컨텍스트를 합성합니다."""
            ctx = "\n\n".join(context_parts)
            if prior_context:
                ctx = f"{prior_context}\n\n{ctx}" if ctx else prior_context
            addon = self._format_active_skills(query)
            if addon:
                ctx = f"{addon}\n\n{ctx}" if ctx else addon
            return ctx

        context_str = _build_context()

        # post_search hook — 수집된 source 목록을 후처리(dedup, boosting 등).
        all_sources = await self._hook_registry.run("post_search", all_sources)

        # --- 거부 게이트 — 무관한 chunk 위에서 합성하면 hallucinate/loop 위험 ---
        # 두 가지 게이트가 OR로 결합되며 둘 중 하나라도 발동하면 거부:
        #  1) 절대 임계 ``refusal_min_score`` — best_score가 이 값 미만이면 거부.
        #     0.0이면 비활성. corpus·reranker에 따라 score 절대값이 크게 다르므로
        #     default(0.01)는 "garbage(≈0)만 차단"하는 안전 하한 역할만 한다.
        #     이 게이트가 활성일 때(>0) 빈 sources도 거부에 포함된다.
        #  2) 동적 게이트 ``refusal_relative_margin`` — best_score가 나머지 sources
        #     평균 대비 ``(1 + margin)``배 미만이면 거부 (signal/noise 분리 안 됨).
        #     corpus·reranker score 절대값에 무관한 corpus-relative gate.
        #     sources가 2개 미만이면 비교 대상이 없어 무시된다.
        #
        # 두 게이트 모두 0.0이면 게이트 완전 비활성 — 빈 sources여도 합성을 시도한다
        # (백워드 호환 / 테스트 용도).
        refusal_threshold = self._config.rag.agent.refusal_min_score
        relative_margin = getattr(
            self._config.rag.agent, "refusal_relative_margin", 0.0
        )
        scores_desc = sorted(
            (s.get("score", 0.0) for s in all_sources), reverse=True
        )
        best_score = scores_desc[0] if scores_desc else 0.0

        absolute_block = refusal_threshold > 0 and (
            not all_sources or best_score < refusal_threshold
        )
        # 동적 게이트 — 나머지(rest)가 1개 이상일 때만 적용.
        relative_block = False
        if relative_margin > 0 and len(scores_desc) >= 2:
            rest = scores_desc[1:]
            mean_rest = sum(rest) / len(rest)
            if mean_rest > 0:
                # best가 mean_rest의 (1 + margin)배 미만이면 signal이 약하다고 판단.
                relative_block = best_score < mean_rest * (1.0 + relative_margin)

        if absolute_block or relative_block:
            yield {"type": "token", "content": _REFUSAL_MESSAGE}
            session_store.add_message(
                sid, Message(role="assistant", content=_REFUSAL_MESSAGE)
            )
            sources_payload = [
                {
                    "content": s.get("content", ""),
                    "doc_id": s.get("doc_id", ""),
                    "source_doc_id": s.get("source_doc_id", ""),
                    "score": s.get("score", 0.0),
                }
                for s in all_sources
            ]
            yield {"type": "sources", "sources": sources_payload}
            yield {"type": "done", "session_id": sid}
            return

        # 답변 합성 — Phase 14에서 answer_verifier/citation audit가 활성화된 경우
        # collect-then-emit로 전환 (TTFT 트레이드오프이나 답변 중복 yield 방지).
        # 기본 경로는 토큰 단위 진짜 스트리밍 유지 (기존 SSE 시퀀스 바이트 호환).
        agent_cfg = self._config.rag.agent
        verifier_enabled = bool(getattr(agent_cfg, "answer_verifier_enabled", False))
        citations_required = bool(
            getattr(agent_cfg, "synthesis_require_citations", False)
        )
        needs_post_check = verifier_enabled or citations_required

        answer_parts: list[str] = []
        if not needs_post_check:
            async for token in self._stream_synthesis(
                query, context_str, history, persona=persona
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        else:
            # collect → verify → emit. 빈 답변/오류 token은 그대로 사용자에게 전달.
            collected = await self._collect_synthesis(
                query, context_str, history, persona=persona
            )

            verification_event: dict | None = None
            if verifier_enabled and collected.strip():
                from .answer_verifier import AnswerVerifier

                verifier = AnswerVerifier(
                    http_client=http_client,
                    ollama_model=self._model_for("answer_verifier"),
                    api_base=self._api_base,
                    request_timeout=min(self._config.rag.request_timeout, 30.0),
                    keep_alive=self._keep_alive,
                    native_thinking=self._native_thinking(),
                )
                verdict = await verifier.evaluate(query, collected, context_str)

                # FAIL + repair_hint이면 1회 재합성 시도.
                max_repairs = int(
                    getattr(agent_cfg, "answer_verifier_max_repairs", 1) or 0
                )
                if verdict.needs_repair and max_repairs > 0:
                    repair_context = (
                        context_str
                        + "\n\n[검증 피드백]\n"
                        + verdict.repair_hint
                        + "\n위 피드백을 반영해 인용 문서로만 직접 지지되는 답변을"
                        + " 다시 작성하세요."
                    )
                    retried = await self._collect_synthesis(
                        query, repair_context, history, persona=persona
                    )
                    if retried.strip():
                        collected = retried
                        # 재합성이 성공했으면 verdict 표시는 첫 결과의 issues로 유지.
                        # (사용자에게는 retry가 일어났다는 사실만 issues에 기록.)
                        verification_event = {
                            "type": "verification",
                            "verdict": verdict.verdict,
                            "issues": verdict.issues + ["1회 재합성 수행"],
                        }
                    else:
                        verification_event = {
                            "type": "verification",
                            "verdict": verdict.verdict,
                            "issues": verdict.issues,
                        }
                elif verdict.verdict != "PASS":
                    verification_event = {
                        "type": "verification",
                        "verdict": verdict.verdict,
                        "issues": verdict.issues,
                    }

            # verification 이벤트는 답변 본문 emit 직전에 발행.
            if verification_event is not None:
                yield verification_event

            # citation audit — synthesis_require_citations일 때만.
            if citations_required and collected.strip():
                from .citation_audit import audit_citations

                missing = audit_citations(collected, all_sources)
                if missing:
                    yield {
                        "type": "warning",
                        "content": f"근거 없는 인용 {len(missing)}개 감지",
                        "items": missing,
                    }

            # 답변 본문 1회 token 이벤트로 발행 — chat.html이 누적 렌더.
            answer_parts.append(collected)
            yield {"type": "token", "content": collected}

        # post_synthesis hook은 누적된 답변에 적용되므로 세션 저장본만 갱신합니다.
        # (현재 등록된 subscriber 없음. 표시본 수정이 필요해지면 별도 patch 이벤트 도입 필요.)
        answer = "".join(answer_parts)
        answer = await self._hook_registry.run("post_synthesis", answer)

        # --- 답변 후처리 (세션 저장 + 메모리 압축) ----------------------
        if answer.strip():
            session_store.add_message(sid, Message(role="assistant", content=answer))
            await self._maybe_compress_memory(session_store, sid, http_client, aux_timeout)

            sources_payload = [
                {
                    "content": s.get("content", ""),
                    "doc_id": s.get("doc_id", ""),
                    "source_doc_id": s.get("source_doc_id", ""),
                    "score": s.get("score", 0.0),
                }
                for s in all_sources
            ]

            # 다음 턴을 위해 현재 참조 문서를 세션에 저장.
            if self._config.rag.agent.session_source_reuse and sources_payload:
                limit = self._config.rag.agent.session_source_reuse_limit
                set_last = getattr(session_store, "set_last_sources", None)
                if callable(set_last):
                    set_last(sid, sources_payload[:limit])

            yield {"type": "sources", "sources": sources_payload}

        yield {"type": "done", "session_id": sid}

    _TABULAR_KEYWORDS: tuple[str, ...] = (
        "표로", "표 형식", "표를", "표 보여", "표 만들",
        "비교해", "비교 ", "대비", "대조", "차이",
        " vs ", " vs.", "v.s.",
        "tabular", "table",
    )

    _ANALYTICAL_KEYWORDS: tuple[str, ...] = (
        "왜", "이유", "원인", "배경", "시사점", "영향", "분석",
        "why", "reason", "cause", "implication",
    )

    @classmethod
    def _has_tabular_intent(cls, query: str) -> bool:
        """질의에 표·비교 형식을 명시 요청하는 키워드가 있는지 판정."""
        lowered = query.lower()
        return any(k.lower() in lowered for k in cls._TABULAR_KEYWORDS)

    @classmethod
    def _has_analytical_intent(cls, query: str) -> bool:
        """질의에 원인·이유·분석 키워드가 있는지 판정."""
        lowered = query.lower()
        return any(k.lower() in lowered for k in cls._ANALYTICAL_KEYWORDS)

    def _maybe_compose_persona(
        self, decision: Any, primary: Persona | None,
        normalized_query: str = "",
    ) -> Persona | None:
        """Phase 14 — primary persona에 보조 persona를 합성합니다.

        ``_compose_persona_with_trace`` 의 가벼운 wrapper — 호환성용. 신규
        호출자는 trace를 함께 얻을 수 있는 ``_compose_persona_with_trace``를
        사용하세요.
        """
        persona, _ = self._compose_persona_with_trace(decision, primary, normalized_query)
        return persona

    def _compose_persona_with_trace(
        self, decision: Any, primary: Persona | None,
        normalized_query: str = "",
    ) -> tuple[Persona | None, dict[str, Any]]:
        """Phase 14 — primary persona 합성 + trace 정보 반환.

        조건:
          1. ``persona_composition_enabled=True``
          2. primary가 ``None``이 아님
          3. confidence < ``persona_composition_confidence_threshold`` **또는**
             query에 secondary 키워드 신호가 있음

        쌍 정책 — primary 카운터파트:
          - ``comparator`` ↔ ``analyst`` (비교 + 분석 — hybrid intent의 90%)
          - ``analyst`` ↔ ``comparator``
          - 그 외(researcher / procedural)는 합성 안 함 (단일 의도 명확)

        Returns
        -------
        ``(persona, trace)`` — persona는 합성 결과(또는 primary 그대로 / None).
        trace는 ``handle_auto`` 가 ``thought`` SSE 이벤트로 발행할 진단 정보.
        """
        # decision.confidence가 명시적으로 None인 경우만 1.0 fallback. 0.0은 그대로
        # 사용(낮은 신뢰도 신호이므로 1.0으로 치환하면 의도와 정반대). 과거 `or 1.0`
        # 패턴은 0.0을 falsy로 잘못 흡수하던 버그.
        raw_conf = getattr(decision, "confidence", None)
        confidence = float(raw_conf) if raw_conf is not None else 1.0
        enabled = bool(
            getattr(self._config.rag.agent, "persona_composition_enabled", False)
        )
        # threshold·primary 의존 항목은 enabled/None 게이트 이후에 채워 넣는다.
        # 테스트 픽스처가 SimpleNamespace로 일부 config 필드만 노출해도 안전하도록.
        trace: dict[str, Any] = {
            "primary": primary.name if primary is not None else None,
            "secondary": None,
            "composed": False,
            "enabled": enabled,
            "confidence": confidence,
            "threshold": None,
            "triggers": [],
            "eligible_pair": False,
            "skip_reason": None,
        }

        if not enabled:
            trace["skip_reason"] = "composition_disabled"
            return primary, trace
        if primary is None:
            trace["skip_reason"] = "no_primary"
            return primary, trace

        threshold = getattr(
            self._config.rag.agent, "persona_composition_confidence_threshold", 0.7
        )
        trace["threshold"] = threshold

        # 카운터파트 쌍 결정.
        from .personas.analyst import Analyst
        from .personas.comparator import Comparator

        primary_name = primary.name
        if primary_name == "comparator":
            secondary_cls: type[Persona] | None = Analyst
        elif primary_name == "analyst":
            secondary_cls = Comparator
        else:
            trace["skip_reason"] = "primary_not_composable"
            return primary, trace

        trace["eligible_pair"] = True

        # 신호 1: 낮은 confidence
        low_conf = confidence < threshold  # noqa: F821 — threshold는 위에서 정의됨

        # 신호 2: secondary 의도 키워드가 query에 등장.
        # **사용자 원문(normalized_query)을 우선 검사** — 과거 구현은 decision.reason
        # (LLM 분류기 reasoning 텍스트)에서 키워드를 찾았는데, LLM이 reasoning에
        # 키워드를 그대로 echo하지 않으면 hybrid_signal이 영영 false인 silent dead
        # path였음. query를 직접 검사하면 "...왜 그렇게 설계됐는지" 같은 정답
        # 케이스가 안정적으로 trigger됨. reason은 보조 fallback.
        query_lower = (normalized_query or "").lower()
        reason_text = (getattr(decision, "reason", "") or "").lower()
        haystack = f"{query_lower}\n{reason_text}"
        hybrid_signal = False
        matched_hybrid: list[str] = []
        if primary_name == "comparator":
            # secondary=Analyst — analytical 키워드 신호
            for k in self._ANALYTICAL_KEYWORDS:
                if k in haystack:
                    matched_hybrid.append(k)
            hybrid_signal = bool(matched_hybrid)
        elif primary_name == "analyst":
            # secondary=Comparator — tabular 키워드 신호
            for k in self._TABULAR_KEYWORDS:
                if k in haystack:
                    matched_hybrid.append(k)
            hybrid_signal = bool(matched_hybrid)

        if low_conf:
            trace["triggers"].append("low_conf")
        if hybrid_signal:
            trace["triggers"].append(f"hybrid_signal({','.join(matched_hybrid[:3])})")

        if not (low_conf or hybrid_signal):
            trace["skip_reason"] = "no_trigger"
            return primary, trace

        from .personas.composite import CompositePersona

        secondary = secondary_cls()
        composite = CompositePersona(primary, secondary)
        trace["secondary"] = secondary.name
        trace["composed"] = True
        logger.info(
            "Persona composition: %s (conf=%.2f, low_conf=%s, hybrid_signal=%s)",
            composite.name,
            confidence,
            low_conf,
            hybrid_signal,
        )
        return composite, trace

    @staticmethod
    def _format_persona_trace(trace: dict[str, Any]) -> str:
        """trace dict을 사용자 가시 SSE thought 한 줄로 직렬화.

        디버깅성을 위해 단일 persona 케이스도 짧게 보고 — composition이 *왜*
        발동되지 않았는지를 한눈에 확인할 수 있어야 한다.
        """
        primary = trace.get("primary") or "default"
        conf = trace.get("confidence", 0.0)
        if trace.get("composed"):
            secondary = trace.get("secondary") or "?"
            triggers = ", ".join(trace.get("triggers") or [])
            return (
                f"Persona: {primary}+{secondary} (composition, "
                f"trigger={triggers or 'n/a'}, conf={conf:.2f})"
            )
        skip = trace.get("skip_reason") or "n/a"
        return f"Persona: {primary} (composition skip: {skip}, conf={conf:.2f})"

    async def _check_corpus_override(
        self, query: str, decision: Any
    ) -> dict | None:
        """corpus 유사도 + corpus_profile 키워드 매칭으로 override 발동 판정.

        Returns
        -------
        ``{"score", "threshold", "trigger", "matched"}`` dict — override 발동.
        ``None`` — 미발동.

        ``trigger``는 ``"vector"`` 또는 ``"keyword"`` — emit 시 사유 분기.

        규칙:
          · conf < 0.95: 벡터 score ≥ threshold **또는** corpus 키워드 매칭이면 override.
          · conf ≥ 0.95: 벡터 score ≥ 0.65(strong) **또는**
            (키워드 매칭 AND 벡터 score ≥ threshold) — 매우 확신 분류에선
            키워드 우연 일치만으로 정정하지 않도록 보조 벡터 확인.

        짧은 도메인 쿼리(예: "RFP 요약해줘")가 벡터 임계 아래로 떨어지는 비대칭을
        해소합니다 — corpus_profile.name/keywords의 토큰이 명시적으로 등장하면
        이는 LLM의 OOD 판정을 뒤집을 만한 직접 신호입니다.
        """
        threshold = self._config.rag.agent.in_domain_score_threshold
        if threshold <= 0.0:
            return None
        score = await self._corpus_relevance_score(query)
        keyword_matched = self._query_has_corpus_token(query)
        very_confident = getattr(decision, "confidence", 0.0) >= 0.95
        strong_signal_threshold = max(threshold, 0.65)

        if very_confident:
            vector_ok = score >= strong_signal_threshold
            keyword_ok = keyword_matched and score >= threshold
        else:
            vector_ok = score >= threshold
            keyword_ok = keyword_matched

        if not (vector_ok or keyword_ok):
            return None
        trigger = "vector" if vector_ok else "keyword"
        return {
            "score": score,
            "threshold": threshold,
            "trigger": trigger,
            "matched": keyword_matched,
        }

    def _query_has_corpus_token(self, query: str) -> bool:
        """query에 corpus_profile의 name/keywords 토큰이 포함됐는지 판정.

        영문/숫자 토큰은 ASCII 한정 단어 경계로 매칭 — 한국어 조사가 결합된
        ``"SLA가"`` 같은 케이스도 매칭됨 (한글은 ASCII letter/digit이 아니므로
        경계 역할). 한국어 토큰은 substring 매칭. corpus_profile 부재 시 ``False``.
        """
        profile = getattr(self._app_state, "corpus_profile", None)
        if profile is None or not hasattr(profile, "merged_keywords"):
            return False
        tokens = profile.merged_keywords()
        if not tokens:
            return False
        q_lower = query.lower()
        for tok in tokens:
            if not tok or len(tok) < 2:
                continue
            t_lower = tok.lower()
            if all(c.isascii() for c in t_lower):
                pattern = rf"(?:^|[^a-z0-9]){re.escape(t_lower)}(?:[^a-z0-9]|$)"
                if re.search(pattern, q_lower):
                    return True
            elif t_lower in q_lower:
                return True
        return False

    async def _emit_corpus_override(
        self,
        query: str,
        override: dict,
        *,
        source_label: str,
    ) -> AsyncGenerator[dict, None]:
        """Override 결정에 따른 thought + route + simple 스트림을 emit합니다."""
        score = override["score"]
        threshold = override["threshold"]
        trigger = override.get("trigger", "vector")
        if self._config.rag.agent.stream_reasoning:
            if trigger == "keyword":
                reason = (
                    f"corpus 키워드 매칭(유사도 {score:.2f})"
                )
            else:
                reason = (
                    f"corpus 유사도 {score:.2f} ≥ 임계 {threshold:.2f}"
                )
            yield {
                "type": "thought",
                "content": (
                    f"의도 보정: {reason} — {source_label} 분류 무시하고 단순 RAG로 정정"
                ),
                "iteration": 0,
            }
        yield {"type": "route", "mode": "simple", "intent": "factual"}
        async for event in self._wrap_simple_with_verification(query):
            yield event

    async def _corpus_relevance_score(self, query: str) -> float:
        """query와 corpus의 vector similarity 최대값을 반환합니다.

        general 라우팅 안전망용 — IntentClassifier가 도메인 query를 OOD로 잘못
        분류했을 때 corpus 자체를 신호로 in-domain 정정에 사용합니다. 도메인
        무관(corpus가 무엇이든 의미 가까우면 hit). 실패 시 ``0.0`` (안전하게
        general로 처리).

        Top-3 검색의 max score를 사용해 단일 outlier가 아닌 안정적 신호로
        활용합니다. embedding/search는 동기 호출이라 executor로 우회.
        """
        try:
            embedding_model = getattr(self._app_state, "embedding_model", None)
            qdrant_client = getattr(self._app_state, "qdrant_client", None)
            if embedding_model is None or qdrant_client is None:
                return 0.0
            collection = self._config.rag.collection_name
            loop = asyncio.get_event_loop()
            query_emb = await loop.run_in_executor(
                None,
                lambda: embedding_model.encode(
                    query, prompt_name="query", show_progress_bar=False
                ).tolist(),
            )
            results = await loop.run_in_executor(
                None,
                lambda: qdrant_client.query_points(
                    collection_name=collection,
                    query=query_emb,
                    limit=3,
                    with_payload=False,
                ),
            )
            if not results.points:
                return 0.0
            return max(float(p.score) for p in results.points)
        except Exception as exc:
            logger.warning("Corpus relevance probe 실패: %s", exc)
            return 0.0

    @staticmethod
    def _format_observation_summary(result: Any) -> str:
        """검색 도구 결과를 raw 청크 텍스트가 아닌 메타 요약으로 포맷.

        UI 추론 패널에 ``[문서 N] (ID: ..., 유사도: ...) ...`` 형태의 raw 청크
        텍스트가 노출되어 가독성을 해치는 문제를 해결합니다. raw 텍스트는
        최종 ``sources`` 이벤트로 별도 노출되므로 도구 실행 흔적은 메타만 충분.

        LLM 호출 없음 — ``result.sources`` dict 리스트의 단순 집계.
        """
        sources = getattr(result, "sources", None) or []
        if not isinstance(sources, list) or not sources:
            return "검색 결과 없음"
        n = len(sources)
        try:
            max_score = max(
                (float(s.get("score", 0.0) or 0.0) for s in sources if isinstance(s, dict)),
                default=0.0,
            )
        except (TypeError, ValueError):
            max_score = 0.0
        ids: list[str] = []
        for s in sources[:3]:
            if not isinstance(s, dict):
                continue
            did = str(s.get("doc_id", ""))
            ids.append(did.split("-")[0] if "-" in did else did[:8])
        suffix = "..." if n > 3 else ""
        joined = ", ".join(i for i in ids if i)
        return f"검색 완료 {n}건, 최고 유사도 {max_score:.2f}, doc IDs: {joined}{suffix}"

    @staticmethod
    def _verbalize_intent(decision) -> str:
        """라우팅 결정을 한국어 자연 발화로 정리합니다.

        oh-my-openagent의 'Verbalize Intent' 패턴 — 의도 분류 결과를 명시
        텍스트로 노출하여 LLM 라우팅의 투명성·follow-up 일관성을 확보합니다.
        분류기가 비활성/실패해 ``intent``가 ``None``이면 mode만 발화합니다.
        """
        intent = getattr(decision, "intent", None)
        mode = getattr(decision, "mode", "agent")
        reason = (getattr(decision, "reason", "") or "").strip()
        intent_label_map = {
            "fact": "사실 확인",
            "compare": "비교/대조",
            "explain": "설명·해설",
            "howto": "절차/방법",
            "ambiguous": "모호 — 명확화 필요",
        }
        if intent is None:
            head = "라우팅"
            kind = ""
        else:
            head = "의도"
            kind = intent_label_map.get(intent, str(intent))
        path = (
            "agent 경로" if mode == "agent"
            else "잡담 직답 경로" if mode == "chitchat"
            else "일반 지식 직답 경로" if mode == "general"
            else "단순 RAG 경로"
        )
        parts = [f"[{head}] {kind}".strip(), f"→ {path}"]
        if reason:
            parts.append(f"({reason})")
        return " ".join(p for p in parts if p)

    async def _maybe_compress_memory(
        self,
        session_store: Any,
        sid: str,
        http_client: Any,
        aux_timeout: float,
    ) -> None:
        """Phase 12 — 세션 이력이 임계값을 넘으면 오래된 턴을 요약으로 압축 — never raises."""
        cfg = self._config.rag.agent
        if not getattr(cfg, "memory_compression_enabled", False):
            return
        compress_after = getattr(cfg, "compress_after_turns", 10)
        target_chars = getattr(cfg, "compress_target_chars", 500)

        # 현재 메시지 수 조회.
        try:
            _, msgs = session_store.get_or_create(sid)
        except Exception:
            return
        # compress_after_turns는 user+assistant 쌍 기준 → 메시지 수 2 * turns.
        threshold = compress_after * 2
        if len(msgs) <= threshold:
            return

        # 최신 (compress_after) 턴만 남기고 나머지를 요약.
        keep_recent = compress_after  # 메시지 개수
        to_summarize = msgs[:-keep_recent] if keep_recent > 0 else list(msgs)
        if not to_summarize:
            return

        from .memory import ConversationCompressor

        compressor = ConversationCompressor(
            http_client=http_client,
            ollama_model=self._model_for("verifier"),  # 가벼운 모델 활용
            api_base=self._api_base,
            request_timeout=aux_timeout,
            target_chars=target_chars,
        )
        try:
            summary = await compressor.summarize(to_summarize)
        except Exception as exc:
            logger.warning("Memory compression 실패: %s — 건너뜀", exc)
            return
        if not summary:
            return

        compress_fn = getattr(session_store, "compress_old_turns", None)
        if not callable(compress_fn):
            return
        removed = compress_fn(sid, keep_recent, summary)
        if removed:
            logger.info("세션 %s 압축: 메시지 %d개 → 요약 1개", sid, removed)

    def _format_active_skills(self, query: str) -> str:
        """질의에 매칭되는 skill들의 prompt_addon을 조립합니다. 없으면 빈 문자열."""
        if len(self._skill_registry) == 0:
            return ""
        matches = self._skill_registry.select_for_query(query, limit=3)
        if not matches:
            return ""
        return self._skill_registry.format_addons(matches)

    def _format_prior_context(self, session_store: Any, sid: str) -> str:
        """이전 턴의 참조 문서를 synthesis 프롬프트 컨텍스트로 포맷합니다."""
        if not self._config.rag.agent.session_source_reuse:
            return ""
        get_last = getattr(session_store, "get_last_sources", None)
        if not callable(get_last):
            return ""
        prior = get_last(sid)
        if not prior:
            return ""
        limit = self._config.rag.agent.session_source_reuse_limit
        lines = ["[이전 대화 참조 문서]"]
        for i, src in enumerate(prior[:limit], start=1):
            doc_id = src.get("doc_id", "?")
            content = src.get("content", "")[:300]
            lines.append(f"[이전 문서 {i}] (ID: {doc_id})\n{content}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 내부: 답변 합성 — Ollama 스트리밍
    # ------------------------------------------------------------------

    async def _stream_synthesis(
        self,
        query: str,
        context: str,
        history: str,
        *,
        persona: Persona | None = None,
    ) -> AsyncGenerator[str, None]:
        """수집된 컨텍스트로 최종 답변을 LLM에 스트리밍 요청합니다."""
        from .prompts import ANSWER_SYNTHESIS_PROMPT

        http_client = self._app_state.http_client
        if not context.strip():
            context = "(수집된 문서 없음)"

        template = (
            persona.synthesis_prompt_template
            if persona is not None and persona.synthesis_prompt_template
            else ANSWER_SYNTHESIS_PROMPT
        )
        # Phase 14 — synthesis_require_citations이면 인용 강제 preamble prepend.
        citations_required = getattr(
            self._config.rag.agent, "synthesis_require_citations", False
        )
        if citations_required:
            from .prompts import CITATION_EVIDENCE_PREAMBLE

            template = CITATION_EVIDENCE_PREAMBLE + template
        prompt = template.format(
            history=f"{history}\n" if history else "",
            context=context[:_SYNTHESIS_CONTEXT_CHAR_LIMIT],
            query=query,
        )
        # Recency-bias 대응 — 긴 시스템 프롬프트의 중간에 묻힌 인용 규칙을 끝에서
        # 다시 한 번 강조해 LLM이 ``[doc:파일명]`` 토큰을 빠뜨리지 않게 한다.
        if citations_required and prompt.endswith("답변:"):
            prompt = prompt[:-3] + (
                "\n[마지막 알림] 답변의 각 사실 주장 끝에 ``[doc:파일명]`` "
                "토큰을 반드시 붙이세요. 헤더의 doc: 뒤 텍스트를 그대로 복사. "
                "이 규칙을 빠뜨리면 답변이 거부됩니다.\n답변:"
            )

        payload = {
            "model": self._model_for("synthesis"),
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": self._rag_max_tokens if self._rag_max_tokens > 0 else _SYNTHESIS_NUM_PREDICT_CAP,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.25,
            },
        }
        try:
            async with http_client.stream(
                "POST",
                f"{self._api_base}/api/generate",
                json=payload,
                timeout=self._config.rag.request_timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.error("답변 합성 실패: %s", exc)
            yield "[오류] 답변 생성 중 문제가 발생했습니다."

    async def _collect_synthesis(
        self,
        query: str,
        context: str,
        history: str,
        *,
        persona: Persona | None = None,
    ) -> str:
        """``_stream_synthesis``를 소비해 최종 텍스트만 반환합니다.

        합성 토큰을 yield하지 않고 누적해 반환하므로, 호출 측은 완성된
        답변을 chunk 단위로 발행할 수 있습니다 — 답변 중복 yield 방지.
        """
        parts: list[str] = []
        async for token in self._stream_synthesis(
            query, context, history, persona=persona
        ):
            parts.append(token)
        return "".join(parts)

    async def _wrap_simple_with_verification(
        self, query: str
    ) -> AsyncGenerator[dict, None]:
        """Phase 14 — simple 경로의 합성 결과에 answer_verifier + citation_audit 적용.

        ``answer_verifier_enabled``·``synthesis_require_citations`` 둘 다 False이면
        ``_simple_stream_fn``의 이벤트를 그대로 통과시켜 원본 streaming 동작 보존
        (TTFT 회귀 차단). 둘 중 하나라도 True이면 token 이벤트를 collect-then-emit
        으로 전환해 합성 끝난 후 verifier/audit 실행, ``verification``/``warning``
        이벤트 발행. citation preamble 자체는 ``_simple_stream_fn`` 안(server.py)
        에서 prompt에 prepend됩니다 — 본 함수는 합성 *후* 검증만 담당.

        simple 경로는 planner처럼 단일 합성 호출이 아니라 검색→Ollama stream 한 번
        이라 재합성(repair)은 비용 대비 효과가 낮아 생략. FAIL/PARTIAL 결과는
        verification 이벤트로 사용자에게 노출되며 답변 본문은 그대로 전달.
        """
        agent_cfg = self._config.rag.agent
        verifier_enabled = bool(getattr(agent_cfg, "answer_verifier_enabled", False))
        citations_required = bool(
            getattr(agent_cfg, "synthesis_require_citations", False)
        )
        needs_post_check = verifier_enabled or citations_required

        if not needs_post_check:
            # fast path — 원본 그대로 (회귀 차단)
            async for event in self._simple_stream_fn(query):
                yield event
            return

        # collect-then-emit path — 합성 끝나기까지 token을 누적, 그 사이 다른
        # 이벤트(thought 등)는 즉시 통과.
        answer_parts: list[str] = []
        sources_payload: list[dict] = []
        done_event: dict | None = None

        async for event in self._simple_stream_fn(query):
            et = event.get("type")
            if et == "token":
                # 토큰은 누적만 (collect-then-emit)
                answer_parts.append(event.get("content", ""))
            elif et == "sources":
                sources_payload = event.get("sources", []) or []
                # sources는 마지막 답변 emit 뒤에 다시 발행 (원본 순서 보존)
            elif et == "done":
                done_event = event
            else:
                # route 등 다른 이벤트는 즉시 통과
                yield event

        final_answer = "".join(answer_parts)
        # context는 sources의 content들을 concat — verifier 입력으로 사용
        context_str = "\n\n".join(
            s.get("content", "") for s in sources_payload if s.get("content")
        )

        # answer_verifier — FAIL/PARTIAL일 때만 verification 이벤트 발행
        if verifier_enabled and final_answer.strip():
            from .answer_verifier import AnswerVerifier

            http_client = self._app_state.http_client
            verifier = AnswerVerifier(
                http_client=http_client,
                ollama_model=self._model_for("answer_verifier"),
                api_base=self._api_base,
                request_timeout=min(self._config.rag.request_timeout, 30.0),
                keep_alive=self._keep_alive,
                native_thinking=self._native_thinking(),
            )
            verdict = await verifier.evaluate(query, final_answer, context_str)
            if verdict.verdict != "PASS":
                yield {
                    "type": "verification",
                    "verdict": verdict.verdict,
                    "issues": verdict.issues,
                }

        # citation_audit — 미매칭 인용 토큰이 있으면 warning 이벤트
        if citations_required and final_answer.strip():
            from .citation_audit import audit_citations

            missing = audit_citations(final_answer, sources_payload)
            if missing:
                yield {
                    "type": "warning",
                    "content": f"근거 없는 인용 {len(missing)}개 감지",
                    "items": missing,
                }

        # 답변 본문 단일 token 이벤트로 발행 — chat.html이 누적 렌더
        if final_answer:
            yield {"type": "token", "content": final_answer}

        # sources / done 원본 순서 보존
        yield {"type": "sources", "sources": sources_payload}
        if done_event is not None:
            yield done_event
        else:
            yield {"type": "done"}

    @staticmethod
    def _dedup_extend(
        all_sources: list[dict],
        seen: set[str],
        new_sources: Iterable[dict],
    ) -> None:
        """``new_sources``를 doc_id 기준으로 dedup하여 ``all_sources``에 추가.

        이미 존재하는 doc_id는 score가 높을 때만 in-place 갱신합니다.
        ``seen``은 호출 측이 보유한 doc_id set으로, in-place 갱신됩니다.
        """
        for src in new_sources:
            if not isinstance(src, dict):
                continue
            doc_id = str(src.get("doc_id", ""))
            if doc_id and doc_id in seen:
                # 기존 항목과 score 비교 후 갱신
                try:
                    new_score = float(src.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    new_score = 0.0
                for i, existing in enumerate(all_sources):
                    if str(existing.get("doc_id", "")) != doc_id:
                        continue
                    try:
                        old_score = float(existing.get("score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        old_score = 0.0
                    if new_score > old_score:
                        all_sources[i] = src
                    break
                continue
            all_sources.append(src)
            if doc_id:
                seen.add(doc_id)


__all__ = ["AgentOrchestrator", "SimpleStreamFn"]
