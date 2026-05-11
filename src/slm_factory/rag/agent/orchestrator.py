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

# 최종 답변의 pseudo-streaming chunk 파라미터.
# quality loop가 끝난 뒤 확정된 answer를 여러 token 이벤트로 쪼개 발행하여
# UI 타자기 효과를 복원합니다. 단일 LLM 호출 결과를 재생만 하는 것이므로
# HIGH-1/HIGH-2 중복 yield 제약은 그대로 유지됩니다.
_FINAL_ANSWER_CHUNK_CHARS = 12
_FINAL_ANSWER_CHUNK_DELAY_SEC = 0.012


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
        # 안전망: LLM이 general로 분류했더라도 corpus와의 의미적 유사도가 충분히
        # 높으면(`in_domain_score_threshold` 이상) in-domain query를 잘못 분류한
        # 경우로 보고 simple로 정정합니다. corpus 자체를 신호로 사용하므로 도메인
        # 무관 — corpus_profile keyword 추출 품질이나 도메인 특성에 의존 안 함.
        if decision.mode == "general":
            threshold = self._config.rag.agent.in_domain_score_threshold
            if threshold > 0.0:
                score = await self._corpus_relevance_score(normalized_query)
                if score >= threshold:
                    if self._config.rag.agent.stream_reasoning:
                        yield {
                            "type": "thought",
                            "content": (
                                f"의도 보정: corpus 유사도 {score:.2f} "
                                f"≥ 임계 {threshold:.2f} — general 분류 무시하고 RAG로 정정"
                            ),
                            "iteration": 0,
                        }
                    yield {
                        "type": "route",
                        "mode": "simple",
                        "intent": "factual",
                    }
                    async for event in self._simple_stream_fn(normalized_query):
                        yield event
                    return
            async for event in self._stream_general(
                normalized_query, session_id, raw_query=raw_query
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
        if self._has_tabular_intent(normalized_query):
            from .personas.comparator import Comparator
            persona = Comparator()
            logger.info("Persona override: 표·비교 키워드 감지 → Comparator")
        if persona is not None:
            logger.info("Persona 선택: %s", persona.name)

        if decision.mode == "simple":
            async for event in self._simple_stream_fn(normalized_query):
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
                "num_predict": self._rag_max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
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
                "num_predict": self._rag_max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
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

        # 답변 합성 — token yield 없이 한 번에 수집해 chunk 단위로 발행.
        answer = await self._collect_synthesis(
            query, context_str, history, persona=persona
        )
        answer = await self._hook_registry.run("post_synthesis", answer)

        # --- 최종 답변 발행 (chunk 단위 pseudo-streaming) -----------------
        if answer.strip():
            session_store.add_message(sid, Message(role="assistant", content=answer))
            chunk_size = _FINAL_ANSWER_CHUNK_CHARS
            chunk_delay = _FINAL_ANSWER_CHUNK_DELAY_SEC
            for i in range(0, len(answer), chunk_size):
                yield {"type": "token", "content": answer[i : i + chunk_size]}
                if chunk_delay > 0:
                    await asyncio.sleep(chunk_delay)
            await self._maybe_compress_memory(session_store, sid, http_client, aux_timeout)

            sources_payload = [
                {
                    "content": s.get("content", ""),
                    "doc_id": s.get("doc_id", ""),
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

    @classmethod
    def _has_tabular_intent(cls, query: str) -> bool:
        """질의에 표·비교 형식을 명시 요청하는 키워드가 있는지 판정."""
        lowered = query.lower()
        return any(k.lower() in lowered for k in cls._TABULAR_KEYWORDS)

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
        prompt = template.format(
            history=f"{history}\n" if history else "",
            context=context[:_SYNTHESIS_CONTEXT_CHAR_LIMIT],
            query=query,
        )

        payload = {
            "model": self._model_for("synthesis"),
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": self._rag_max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
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
