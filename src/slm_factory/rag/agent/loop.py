"""ReAct 에이전트 루프 — 질문 분석, 도구 사용, 답변 생성을 오케스트레이트합니다.

``AgentLoop``은 LLM과 도구를 반복적으로 호출하여 복합 질문에
다단계 검색·추론으로 답변합니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from ...utils import get_logger
from .parser import parse_react_output, _SPECIAL_TAG_RE
from .prompts import AGENT_SYSTEM_PROMPT, DECOMPOSE_PROMPT, FORCE_ANSWER_PROMPT
from .tools import ToolRegistry

logger = get_logger("rag.agent.loop")

# 스크래치패드 최대 길이 (프롬프트 폭발 방지)
_MAX_SCRATCHPAD_LEN = 6000
# Observation 최대 길이 (스크래치패드에 추가 시)
_MAX_OBSERVATION_LEN = 1500


# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """ReAct 루프 이벤트 — 스트리밍 시 클라이언트로 전송됩니다."""

    type: str  # "thought", "action", "observation", "token", "done", "error"
    content: str = ""
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """ReAct 루프 최종 결과."""

    answer: str
    sources: list[dict[str, Any]]
    iterations: int
    events: list[AgentEvent]


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """ReAct 에이전트 루프.

    Parameters
    ----------
    http_client:
        Ollama API 호출용 httpx.AsyncClient.
    tool_registry:
        에이전트가 사용할 도구 레지스트리.
    ollama_model:
        Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    max_iterations:
        ReAct 루프 최대 반복 횟수.
    max_tokens:
        LLM 생성 최대 토큰 수. -1이면 무제한.
    request_timeout:
        LLM 요청 타임아웃(초).
    """

    def __init__(
        self,
        http_client: Any,
        tool_registry: ToolRegistry,
        ollama_model: str,
        api_base: str,
        max_iterations: int = 5,
        max_tokens: int = -1,
        request_timeout: float = 120.0,
        *,
        keep_alive: str = "5m",
    ) -> None:
        self._http_client = http_client
        self._tools = tool_registry
        self._model = ollama_model
        self._api_base = api_base
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._keep_alive = keep_alive
        # 동일 query 분해 결과 캐시 — LLM 호출 제거.
        self._decompose_cache: dict[str, list[str]] = {}
        self._decompose_cache_max = 50

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(self, prompt: str) -> str:
        """Ollama /api/generate를 비스트리밍으로 호출하여 텍스트를 생성합니다."""

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": self._keep_alive,
        }
        if self._max_tokens > 0:
            payload["options"] = {"num_predict": self._max_tokens}

        response = await self._http_client.post(
            f"{self._api_base}/api/generate",
            json=payload,
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    async def _generate_stream_tokens(
        self, prompt: str
    ) -> AsyncGenerator[str, None]:
        """Ollama /api/generate를 스트리밍으로 호출하여 토큰을 yield합니다."""
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": self._keep_alive,
        }
        if self._max_tokens > 0:
            payload["options"] = {"num_predict": self._max_tokens}

        async with self._http_client.stream(
            "POST",
            f"{self._api_base}/api/generate",
            json=payload,
            timeout=self._request_timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token

    # ------------------------------------------------------------------
    # 프롬프트 조립
    # ------------------------------------------------------------------

    def _build_prompt(
        self, query: str, history: str, scratchpad: str
    ) -> str:
        """ReAct 시스템 프롬프트를 조립합니다."""
        conversation_history = f"{history}\n" if history else ""
        prompt = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self._tools.get_tool_descriptions(),
            max_iterations=self._max_iterations,
            conversation_history=conversation_history,
            query=query,
        )
        if scratchpad:
            # 스크래치패드 길이 제한 — 프롬프트 폭발 방지
            if len(scratchpad) > _MAX_SCRATCHPAD_LEN:
                scratchpad = "...(이전 단계 생략)\n" + scratchpad[-_MAX_SCRATCHPAD_LEN:]
            prompt += f"\n{scratchpad}"
        return prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _decompose_query(self, query: str) -> list[str]:
        """복합 질문을 하위 질문으로 분해합니다. 단순 질문이면 그대로 반환.

        동일 query에 대한 LLM 호출을 막기 위해 in-memory dict로 LRU-ish
        캐시를 운용합니다 (size 제한만 적용 — 단순한 정책으로 충분).
        """
        cached = self._decompose_cache.get(query)
        if cached is not None:
            return list(cached)
        try:
            prompt = DECOMPOSE_PROMPT.format(query=query)
            response = await self._generate(prompt)
            # JSON 추출
            brace_start = response.find("{")
            brace_end = response.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                data = json.loads(response[brace_start:brace_end + 1])
                subs = data.get("sub_questions", [])
                if isinstance(subs, list) and len(subs) >= 1:
                    result = [str(s) for s in subs[:3]]  # 최대 3개
                    self._cache_decompose(query, result)
                    return result
        except Exception:
            pass
        fallback = [query]
        self._cache_decompose(query, fallback)
        return fallback

    def _cache_decompose(self, query: str, result: list[str]) -> None:
        """decompose 결과를 size 제한 캐시에 저장합니다."""
        if len(self._decompose_cache) >= self._decompose_cache_max:
            # 가장 오래된(삽입 순서 첫) 엔트리 1개 퇴거.
            try:
                oldest = next(iter(self._decompose_cache))
                self._decompose_cache.pop(oldest, None)
            except StopIteration:
                pass
        self._decompose_cache[query] = list(result)

    async def run(self, query: str, history: str = "") -> AgentResult:
        """동기적 실행 — 최종 결과를 반환합니다."""
        events: list[AgentEvent] = []
        all_sources: list[dict[str, Any]] = []
        scratchpad = ""

        # 질문 분해: 복합 질문이면 서브 질문별로 사전 검색
        sub_questions = await self._decompose_query(query)
        if len(sub_questions) > 1:
            logger.info("질문 분해: %d개 하위 질문", len(sub_questions))
            for sq in sub_questions:
                tool_result = await self._tools.execute("search", {"query": sq})
                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)
                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: 하위 질문 '{sq}'에 대해 검색합니다.\n"
                scratchpad += "Action: search\n"
                scratchpad += f"Action Input: {json.dumps({'query': sq}, ensure_ascii=False)}\n"
                scratchpad += f"Observation: {obs_text}\n"
                events.append(AgentEvent(type="thought", content=f"하위 질문: {sq}", iteration=0))
                events.append(AgentEvent(type="observation", content=obs_text, iteration=0))

        for iteration in range(1, self._max_iterations + 1):
            prompt = self._build_prompt(query, history, scratchpad)

            try:
                llm_output = await self._generate(prompt)
            except Exception as e:
                logger.error("LLM 호출 실패 (iteration %d): %s", iteration, e)
                events.append(AgentEvent(
                    type="error", content="LLM 호출 중 문제가 발생했습니다.", iteration=iteration
                ))
                break

            step = parse_react_output(llm_output)

            # Thought 이벤트
            if step.thought:
                events.append(AgentEvent(
                    type="thought", content=step.thought, iteration=iteration
                ))

            # Final Answer — 루프 종료
            if step.final_answer is not None:
                events.append(AgentEvent(
                    type="done", content=step.final_answer, iteration=iteration,
                    metadata={"sources": all_sources},
                ))
                return AgentResult(
                    answer=step.final_answer,
                    sources=all_sources,
                    iterations=iteration,
                    events=events,
                )

            # Action — 도구 실행
            if step.action:
                events.append(AgentEvent(
                    type="action", content=step.action, iteration=iteration,
                    metadata={"input": step.action_input or {}},
                ))

                tool_result = await self._tools.execute(
                    step.action, step.action_input or {}
                )

                # 구조화된 소스 수집 (regex 파싱 불필요)
                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)

                events.append(AgentEvent(
                    type="observation", content=tool_result.text, iteration=iteration
                ))

                # 스크래치패드에 추가 (observation 길이 제한)
                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: {step.thought}\n"
                scratchpad += f"Action: {step.action}\n"
                scratchpad += f"Action Input: {json.dumps(step.action_input or {}, ensure_ascii=False)}\n"
                scratchpad += f"Observation: {obs_text}\n"
            else:
                # Action도 Final Answer도 없는 경우 — 답변으로 처리
                answer = step.thought or llm_output.strip()
                events.append(AgentEvent(
                    type="done", content=answer, iteration=iteration,
                    metadata={"sources": all_sources},
                ))
                return AgentResult(
                    answer=answer,
                    sources=all_sources,
                    iterations=iteration,
                    events=events,
                )

        # max_iterations 도달 — 강제 답변 생성
        return await self._force_answer(query, scratchpad, all_sources, events)

    async def run_stream(
        self, query: str, history: str = ""
    ) -> AsyncGenerator[AgentEvent, None]:
        """스트리밍 실행 — Final Answer 감지 후 실시간 토큰 전달."""

        all_sources: list[dict[str, Any]] = []
        scratchpad = ""

        _FINAL_ANSWER_MARKER = re.compile(
            r"(?:Final Answer|최종 답변)\s*:\s*", re.IGNORECASE
        )
        _REASONING_STOP = re.compile(
            r"\n\s*(?:Thought|생각|Action|행동|Observation|관찰)\s*:", re.IGNORECASE
        )

        # 질문 분해: 복합 질문이면 서브 질문별로 사전 검색
        sub_questions = await self._decompose_query(query)
        if len(sub_questions) > 1:
            logger.info("질문 분해: %d개 하위 질문", len(sub_questions))
            for sq in sub_questions:
                yield AgentEvent(type="thought", content=f"하위 질문: {sq}", iteration=0)
                tool_result = await self._tools.execute("search", {"query": sq})
                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)
                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: 하위 질문 '{sq}'에 대해 검색합니다.\n"
                scratchpad += "Action: search\n"
                scratchpad += f"Action Input: {json.dumps({'query': sq}, ensure_ascii=False)}\n"
                scratchpad += f"Observation: {obs_text}\n"
                yield AgentEvent(type="observation", content=obs_text[:300], iteration=0)

        for iteration in range(1, self._max_iterations + 1):
            prompt = self._build_prompt(query, history, scratchpad)

            # 스트리밍으로 토큰을 받으면서 Final Answer를 감지
            buffer = ""
            found_final = False
            try:
                async for token in self._generate_stream_tokens(prompt):
                    buffer += token

                    if not found_final:
                        # Final Answer: 패턴 감지
                        m = _FINAL_ANSWER_MARKER.search(buffer)
                        if m:
                            found_final = True
                            # 패턴 앞부분에서 Thought 추출
                            pre_text = buffer[:m.start()]
                            step = parse_react_output(pre_text)
                            if step.thought:
                                yield AgentEvent(
                                    type="thought", content=step.thought,
                                    iteration=iteration,
                                )
                            # 패턴 뒤의 텍스트를 즉시 전달
                            after = _SPECIAL_TAG_RE.sub("", buffer[m.end():])
                            if after:
                                yield AgentEvent(
                                    type="token", content=after,
                                    iteration=iteration,
                                )
                    else:
                        # Final Answer 이후 — 추론 키워드 나오면 중단
                        # 영문/한국어 마커 모두 탐색 (loop:336 _FINAL_ANSWER_MARKER와 일치)
                        marker_pos = -1
                        for marker in ("Final Answer", "최종 답변"):
                            pos = buffer.rfind(marker)
                            if pos > marker_pos:
                                marker_pos = pos
                        tail = buffer[marker_pos:] if marker_pos != -1 else buffer[-200:]
                        if _REASONING_STOP.search(tail):
                            break
                        # 특수 태그 필터링
                        clean_token = _SPECIAL_TAG_RE.sub("", token)
                        if clean_token:
                            yield AgentEvent(
                                type="token", content=clean_token,
                                iteration=iteration,
                            )

            except Exception as e:
                logger.error("LLM 호출 실패 (iteration %d): %s", iteration, e)
                yield AgentEvent(
                    type="error", content="LLM 호출 중 문제가 발생했습니다.",
                    iteration=iteration,
                )
                return

            # thinking 태그 제거
            buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL).strip()

            if found_final:
                # 스트리밍 완료
                yield AgentEvent(
                    type="done", content="", iteration=iteration,
                    metadata={"sources": all_sources},
                )
                return

            # Final Answer가 없었으면 — Thought/Action 파싱
            step = parse_react_output(buffer)

            if step.thought:
                yield AgentEvent(
                    type="thought", content=step.thought, iteration=iteration,
                )

            if step.action:
                yield AgentEvent(
                    type="action", content=step.action, iteration=iteration,
                    metadata={"input": step.action_input or {}},
                )

                tool_result = await self._tools.execute(
                    step.action, step.action_input or {}
                )

                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)

                yield AgentEvent(
                    type="observation", content=tool_result.text,
                    iteration=iteration,
                )

                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: {step.thought}\n"
                scratchpad += f"Action: {step.action}\n"
                scratchpad += f"Action Input: {json.dumps(step.action_input or {}, ensure_ascii=False)}\n"
                scratchpad += f"Observation: {obs_text}\n"
            else:
                # Action도 Final Answer도 없음 — fallback 답변.
                # thought/buffer가 모두 비어있으면 빈 답변이 되므로 error로 escalate.
                answer = step.thought or buffer
                if answer:
                    yield AgentEvent(
                        type="token", content=answer, iteration=iteration,
                    )
                else:
                    yield AgentEvent(
                        type="error",
                        content="LLM 응답이 비어 있습니다 — 답변을 생성하지 못했습니다.",
                        iteration=iteration,
                    )
                yield AgentEvent(
                    type="done", content="", iteration=iteration,
                    metadata={"sources": all_sources},
                )
                return

        # max_iterations 도달 — 강제 답변도 실시간 스트리밍.
        # 클라이언트가 "강제 답변" 단계임을 인지하도록 thought 이벤트를 먼저 발행.
        logger.warning("max_iterations(%d) 도달 — 강제 답변 생성", self._max_iterations)
        yield AgentEvent(
            type="thought",
            content=(
                f"max_iterations({self._max_iterations}) 도달 — "
                "수집된 정보로 강제 답변 생성"
            ),
            iteration=self._max_iterations,
        )
        force_prompt = FORCE_ANSWER_PROMPT.format(
            gathered_context=scratchpad[-3000:] if scratchpad else "(수집된 정보 없음)",
            query=query,
        )
        try:
            async for token in self._generate_stream_tokens(force_prompt):
                yield AgentEvent(
                    type="token", content=token,
                    iteration=self._max_iterations,
                )
        except Exception as e:
            logger.error("강제 답변 생성 실패: %s", e)
            yield AgentEvent(
                type="token",
                content="답변 생성 중 문제가 발생했습니다. 다시 시도해 주세요.",
                iteration=self._max_iterations,
            )
        yield AgentEvent(
            type="done", content="", iteration=self._max_iterations,
            metadata={"sources": all_sources},
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    async def _force_answer(
        self,
        query: str,
        scratchpad: str,
        all_sources: list[dict],
        events: list[AgentEvent],
    ) -> AgentResult:
        """max_iterations 도달 시 수집된 컨텍스트로 강제 답변을 생성합니다."""
        logger.warning("max_iterations(%d) 도달 — 강제 답변 생성", self._max_iterations)

        # 강제 답변 단계임을 사용자/디버깅용 thought 이벤트로 기록.
        events.append(AgentEvent(
            type="thought",
            content=(
                f"max_iterations({self._max_iterations}) 도달 — "
                "수집된 정보로 강제 답변 생성"
            ),
            iteration=self._max_iterations,
        ))

        prompt = FORCE_ANSWER_PROMPT.format(
            gathered_context=scratchpad[-3000:] if scratchpad else "(수집된 정보 없음)",
            query=query,
        )

        try:
            answer = await self._generate(prompt)
        except Exception as e:
            logger.error("강제 답변 생성 실패: %s", e)
            answer = "답변 생성 중 문제가 발생했습니다. 다시 시도해 주세요."

        events.append(AgentEvent(
            type="done", content=answer, iteration=self._max_iterations,
            metadata={"sources": all_sources, "forced": True},
        ))
        return AgentResult(
            answer=answer,
            sources=all_sources,
            iterations=self._max_iterations,
            events=events,
        )
