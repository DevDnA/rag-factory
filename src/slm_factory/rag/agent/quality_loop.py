"""Ralph 스타일 답변 품질 통합 루프 — oh-my-openagent의 self-referential loop을
RAG에 맞게 적응한 컨트롤러.

설계 동기
----------
oh-my-openagent의 ``ralph-loop``은 LLM이 ``<promise>DONE</promise>`` 토큰을
방출할 때까지(또는 max_iterations에 도달할 때까지) 작업을 자기 참조 반복합니다.
``ultrawork-loop``은 여기에 Oracle 검증을 더합니다.

slm-factory 기존 quality 체인은 reflector → review-work → self-improvement를
**직렬**로 한 번씩만 실행했습니다. 이를 단일 **반복** 루프로 통합하여:

1. 매 반복마다 reflector + reviewers + scorer를 **병렬** 평가
2. 모든 게이트 통과 + scorer 점수 ≥ threshold → ``promise`` 이벤트 발행 후 종료
3. 실패 시 ``strategy``(reset/continue)에 따라 피드백 누적, 보완 검색,
   재합성 후 다음 반복
4. 최대 반복에 도달하면 마지막 답변을 그대로 사용

영속화
------
``LoopStateStore``는 ``state_dir/<session_id>.json``으로 매 반복 상태를 저장
합니다. 디렉터리가 빈 문자열이면 비영속. 다음 턴에서 같은 session으로 재진입
시 ``last_score``·``iteration``을 참고해 추적 메트릭을 누적할 수 있습니다.

기존 코드 호환성
----------------
``ralph_loop_enabled=False``(기본값)이면 이 모듈은 호출되지 않으며 기존 직렬
체인이 그대로 동작합니다. 활성화 시에만 ``orchestrator``가 본 루프를 진입점
으로 사용합니다.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from ...utils import get_logger

logger = get_logger("rag.agent.quality_loop")


# ---------------------------------------------------------------------------
# 상수 — oh-my-openagent의 컨벤션을 가져옵니다.
# ---------------------------------------------------------------------------

COMPLETION_TAG_PATTERN = re.compile(
    r"<promise>(.*?)</promise>", re.IGNORECASE | re.DOTALL
)
"""``<promise>DONE</promise>`` 형태의 완료 약속 토큰 매칭 패턴."""

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_COMPLETION_PROMISE = "DONE"

# 피드백 누적 텍스트의 최대 길이 — 프롬프트 폭발 방지.
_FEEDBACK_BLOCK_CHAR_LIMIT = 1500


# ---------------------------------------------------------------------------
# 상태 모델 + 영속화
# ---------------------------------------------------------------------------


@dataclass
class RalphLoopState:
    """Ralph 루프 1회 실행의 상태.

    필드 의미
    ---------
    iteration:
        현재 반복 번호(0부터 시작 — 0은 최초 합성).
    completion_promise:
        통과 시 발행할 토큰. 사용자 정의 가능.
    started_at:
        루프 시작 시간(epoch 초).
    session_id:
        세션 식별자(영속화 시 파일명).
    last_score:
        가장 최근 scorer 점수(없으면 ``None``).
    last_promise_emitted:
        DONE 발행 여부.
    strategy:
        피드백 누적 전략.
    status:
        ``running`` | ``completed`` | ``max_reached``.
    """

    iteration: int = 0
    completion_promise: str = DEFAULT_COMPLETION_PROMISE
    started_at: float = field(default_factory=time.time)
    session_id: str = ""
    last_score: float | None = None
    last_promise_emitted: bool = False
    strategy: Literal["reset", "continue"] = "continue"
    status: Literal["running", "completed", "max_reached"] = "running"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoopStateStore:
    """Ralph 루프 상태를 JSON 파일로 영속화합니다.

    ``directory``가 빈 문자열이면 모든 메서드는 no-op (read는 ``None`` 반환).
    파일 I/O 실패는 절대 raise하지 않고 warning 로그만 남깁니다 — 루프 자체의
    안정성을 깨지 않기 위함.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory) if directory else None

    def _path_for(self, session_id: str) -> Path | None:
        if self._dir is None or not session_id:
            return None
        return self._dir / f"{session_id}.json"

    def write(self, state: RalphLoopState) -> bool:
        path = self._path_for(state.session_id)
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except OSError as exc:
            logger.warning("Ralph 루프 상태 저장 실패(%s): %s", path, exc)
            return False

    def read(self, session_id: str) -> RalphLoopState | None:
        path = self._path_for(session_id)
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            return RalphLoopState(**data)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Ralph 루프 상태 로드 실패(%s): %s", path, exc)
            return None

    def clear(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        if path is None or not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            logger.warning("Ralph 루프 상태 삭제 실패(%s): %s", path, exc)
            return False


# ---------------------------------------------------------------------------
# 약속 토큰 검출 + 피드백 포맷
# ---------------------------------------------------------------------------


def extract_promise_tag(text: str) -> str | None:
    """``<promise>...</promise>`` 안의 토큰을 추출합니다. 없으면 ``None``."""
    if not text:
        return None
    match = COMPLETION_TAG_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1).strip() or None


def has_promise(text: str, expected: str) -> bool:
    """텍스트에 ``<promise>{expected}</promise>``가 포함되어 있는지."""
    found = extract_promise_tag(text)
    if found is None:
        return False
    return found.casefold() == expected.casefold()


def append_promise(text: str, promise: str) -> str:
    """답변 끝에 promise 태그를 1회 부착합니다 — 이미 있으면 중복 추가하지 않음."""
    if has_promise(text, promise):
        return text
    suffix = f"\n\n<promise>{promise}</promise>"
    return text.rstrip() + suffix


def format_feedback_block(
    *,
    scorer_feedback: str = "",
    scorer_improvements: list[str] | None = None,
    reflector_reason: str = "",
    failed_reviewers: list[str] | None = None,
    reviewer_reasons: list[str] | None = None,
    iteration: int,
) -> str:
    """게이트 실패 신호들을 재합성 프롬프트에 주입할 한국어 텍스트로 정리합니다.

    중복·빈 항목은 자동으로 제거합니다. 결과 문자열이 비면 빈 문자열을 반환.
    """
    lines: list[str] = [f"[이전 반복 #{iteration} 개선 지침]"]
    if scorer_feedback:
        lines.append(f"- scorer 총평: {scorer_feedback}")
    for i, imp in enumerate(scorer_improvements or [], start=1):
        if imp:
            lines.append(f"- scorer 개선 {i}: {imp}")
    if reflector_reason:
        lines.append(f"- reflector 지적: {reflector_reason}")
    if failed_reviewers:
        lines.append(f"- review-work 실패 항목: {', '.join(failed_reviewers)}")
    for i, r in enumerate(reviewer_reasons or [], start=1):
        if r:
            lines.append(f"- reviewer 사유 {i}: {r}")
    if len(lines) == 1:
        return ""
    text = "\n".join(lines)
    if len(text) > _FEEDBACK_BLOCK_CHAR_LIMIT:
        text = text[:_FEEDBACK_BLOCK_CHAR_LIMIT] + "\n...(생략)"
    return text


# ---------------------------------------------------------------------------
# 통합 루프 컨트롤러
# ---------------------------------------------------------------------------


SynthesizeFn = Callable[[str, str, str], Awaitable[str]]
"""(query, context, history) → 답변. 프로토콜 단순화를 위한 alias."""

GateFn = Callable[..., Awaitable[Any]]


@dataclass
class GateOutcome:
    """1회 반복의 모든 게이트 결과를 모은 구조."""

    scorer_score: float | None
    scorer_ok: bool
    scorer_feedback: str
    scorer_improvements: list[str]
    reflector_ok: bool
    reflector_reason: str
    reflector_missing: str | None
    reviewer_passed: bool
    reviewer_failed: list[str]
    reviewer_reasons: list[str]
    reviewer_missing: str | None

    def all_passed(self, threshold: float) -> bool:
        """모든 게이트가 통과했고 점수가 임계값 이상인지."""
        if not self.reflector_ok:
            return False
        if not self.reviewer_passed:
            return False
        # scorer가 호출 불가였으면(ok=False) 점수 게이트는 무시 — 무한 retry 방지.
        if self.scorer_ok and self.scorer_score is not None:
            if self.scorer_score < threshold:
                return False
        return True

    def composite_quality(self) -> float:
        """반복 간 답변 품질 비교용 종합 점수.

        모든 게이트 통과(reflector + reviewer + scorer ≥ threshold)는
        ``all_passed``로 별도 판정합니다. 본 메서드는 *부분 통과* 답변들 중
        어느 것이 더 나은지 ranking하기 위한 보조 지표 — 게이트 통과 가중치 +
        scorer 점수.
        """
        score = self.scorer_score if (
            self.scorer_ok and self.scorer_score is not None
        ) else 7.0
        bonus = 0.0
        if self.reflector_ok:
            bonus += 5.0
        if self.reviewer_passed:
            bonus += 10.0
        return score + bonus

    def best_missing_query(self) -> str | None:
        """보완 검색에 사용할 추천 질의 — reflector 우선, 없으면 reviewer."""
        return self.reflector_missing or self.reviewer_missing


@dataclass
class RalphLoopResult:
    """루프 종료 결과."""

    answer: str
    state: RalphLoopState
    promise_emitted: bool
    iterations_run: int
    final_outcome: GateOutcome | None


__all__ = [
    "COMPLETION_TAG_PATTERN",
    "DEFAULT_COMPLETION_PROMISE",
    "DEFAULT_MAX_ITERATIONS",
    "GateOutcome",
    "LoopStateStore",
    "RalphLoopResult",
    "RalphLoopState",
    "append_promise",
    "extract_promise_tag",
    "format_feedback_block",
    "has_promise",
]


# ---------------------------------------------------------------------------
# 컨트롤러 — orchestrator가 호출하는 진입점
# ---------------------------------------------------------------------------


class RAGQualityLoop:
    """orchestrator의 quality 체인을 대체하는 통합 반복 루프.

    호출 측은 ``run()`` async generator를 소비합니다. 매 반복마다 thought/
    action/observation 이벤트를 yield하며, 종료 시 ``RalphLoopResult``를 들고
    있는 sentinel 이벤트(``{"type": "ralph_done", ...}``)를 yield 합니다.

    이 클래스 자체는 reflector/reviewers/scorer/tool 의존성을 콜러블 hook으로
    주입받아 단위 테스트·다른 backend로의 교체가 쉽도록 설계됩니다.
    """

    def __init__(
        self,
        *,
        max_iterations: int,
        quality_threshold: float,
        strategy: Literal["reset", "continue"],
        completion_promise: str,
        synthesize: SynthesizeFn,
        run_reflector: GateFn,
        run_reviewers: GateFn,
        run_scorer: GateFn,
        execute_search: Callable[[str], Awaitable[Any]] | None = None,
        state_store: LoopStateStore | None = None,
        preview_limit: int = 300,
        stream_reasoning: bool = True,
    ) -> None:
        self._max_iters = max(1, max_iterations)
        self._threshold = quality_threshold
        self._strategy = strategy
        self._promise = completion_promise or DEFAULT_COMPLETION_PROMISE
        self._synthesize = synthesize
        self._run_reflector = run_reflector
        self._run_reviewers = run_reviewers
        self._run_scorer = run_scorer
        self._execute_search = execute_search
        self._state_store = state_store
        self._preview_limit = preview_limit
        self._stream_reasoning = stream_reasoning

    async def run(
        self,
        *,
        query: str,
        initial_answer: str,
        history: str,
        build_context: Callable[[], str],
        all_sources: list[dict],
        seen_doc_ids: set[str],
        context_parts: list[str],
        dedup_extend: Callable[[list[dict], set[str], Any], None],
        session_id: str,
        starting_iteration: int = 0,
    ):
        """초기 답변에서 시작해 통과/최대반복까지 반복합니다.

        이 함수는 async generator로, 각 step에서 SSE 이벤트 dict를 yield 합니다.
        마지막 이벤트는 항상 ``{"type": "ralph_done", "result": RalphLoopResult}``
        입니다. orchestrator는 이 sentinel을 가로채 답변·약속 발행을 처리합니다.
        """
        answer = initial_answer
        feedback_history: list[str] = []
        last_outcome: GateOutcome | None = None
        promise_emitted = False
        iteration = starting_iteration
        # 최고 품질 답변 트래킹 — 마지막 답변이 이전보다 나쁠 때 열화 방지.
        # 9b judge처럼 일관성 부족한 모델에서 retry가 점수를 떨어뜨릴 수 있어,
        # 매 반복의 composite_quality를 비교해 최선의 답변을 선택합니다.
        best_answer = answer
        best_quality: float | None = None
        best_outcome: GateOutcome | None = None

        state = RalphLoopState(
            iteration=iteration,
            completion_promise=self._promise,
            session_id=session_id,
            strategy=self._strategy,
        )
        if self._state_store is not None:
            self._state_store.write(state)

        for step in range(self._max_iters):
            iteration += 1
            state.iteration = iteration

            outcome = await self._evaluate(query, answer, all_sources)
            last_outcome = outcome
            state.last_score = outcome.scorer_score

            # 최고 품질 답변 갱신 — 이번 반복이 이전보다 좋으면 best로 저장.
            this_quality = outcome.composite_quality()
            if best_quality is None or this_quality > best_quality:
                best_quality = this_quality
                best_answer = answer
                best_outcome = outcome

            if self._stream_reasoning:
                yield {
                    "type": "ralph_iteration",
                    "iteration": iteration,
                    "score": outcome.scorer_score,
                    "reflector_ok": outcome.reflector_ok,
                    "reviewer_passed": outcome.reviewer_passed,
                    "failed_reviewers": outcome.reviewer_failed,
                }

            if outcome.all_passed(self._threshold):
                promise_emitted = True
                state.last_promise_emitted = True
                state.status = "completed"
                if self._state_store is not None:
                    self._state_store.write(state)
                if self._stream_reasoning:
                    yield {
                        "type": "promise",
                        "content": self._promise,
                        "iteration": iteration,
                    }
                break

            # 게이트 실패 — 보완 검색 + 재합성 시도.
            missing_q = outcome.best_missing_query()
            if missing_q and self._execute_search is not None:
                if self._stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": (
                            f"Ralph #{iteration}: 게이트 실패 → 보완 검색 '{missing_q}'"
                        ),
                        "iteration": iteration,
                    }
                    yield {
                        "type": "action",
                        "content": "search",
                        "input": {"query": missing_q},
                        "iteration": iteration,
                    }
                try:
                    extra = await self._execute_search(missing_q)
                except Exception as exc:
                    logger.warning("Ralph 보완 검색 실패: %s", exc)
                    extra = None
                if extra is not None:
                    dedup_extend(all_sources, seen_doc_ids, getattr(extra, "sources", []))
                    extra_text = getattr(extra, "text", "")
                    if extra_text:
                        context_parts.append(extra_text)
                    if self._stream_reasoning and extra_text:
                        preview = extra_text[: self._preview_limit]
                        if len(extra_text) > self._preview_limit:
                            preview += "..."
                        yield {
                            "type": "observation",
                            "content": preview,
                            "iteration": iteration,
                        }

            # 피드백 누적 — strategy에 따라 reset/continue.
            block = format_feedback_block(
                scorer_feedback=outcome.scorer_feedback,
                scorer_improvements=outcome.scorer_improvements,
                reflector_reason=outcome.reflector_reason,
                failed_reviewers=outcome.reviewer_failed,
                reviewer_reasons=outcome.reviewer_reasons,
                iteration=iteration,
            )
            if self._strategy == "reset":
                feedback_history = [block] if block else []
            else:
                if block:
                    feedback_history.append(block)

            # 마지막 반복이면 더 합성하지 않고 종료(현재 답변 유지).
            if step == self._max_iters - 1:
                break

            current_context = build_context()
            if feedback_history:
                joined = "\n\n".join(feedback_history)
                current_context = (
                    f"{joined}\n\n{current_context}" if current_context else joined
                )
            new_answer = await self._synthesize(query, current_context, history)
            if new_answer:
                answer = new_answer

            if self._state_store is not None:
                self._state_store.write(state)

        if state.status == "running":
            state.status = "max_reached"
            if self._state_store is not None:
                self._state_store.write(state)

        # 최종 답변 선정 — promise 발행 시는 통과한 답변, 아니면 best_quality 답변.
        # 마지막 반복이 열화된 경우(예: 9b judge 노이즈로 score 9 → 4 하락)
        # best_answer가 더 나은 결과를 보존합니다.
        final_answer = answer if promise_emitted else best_answer
        if promise_emitted:
            final_answer = append_promise(final_answer, self._promise)

        # 결과의 final_outcome도 최선 답변과 짝을 맞춥니다.
        final_outcome = last_outcome if promise_emitted else (best_outcome or last_outcome)

        result = RalphLoopResult(
            answer=final_answer,
            state=state,
            promise_emitted=promise_emitted,
            iterations_run=iteration - starting_iteration,
            final_outcome=final_outcome,
        )
        yield {"type": "ralph_done", "result": result}

    async def _evaluate(
        self, query: str, answer: str, all_sources: list[dict]
    ) -> GateOutcome:
        """reflector + reviewers + scorer를 병렬 평가하고 GateOutcome으로 종합."""
        sources_payload = [
            {
                "content": s.get("content", ""),
                "doc_id": s.get("doc_id", ""),
                "score": s.get("score", 0.0),
            }
            for s in all_sources
            if isinstance(s, dict)
        ]
        results = await asyncio.gather(
            self._run_reflector(query, answer, sources_payload),
            self._run_reviewers(query, answer, sources_payload),
            self._run_scorer(query, answer, sources_payload),
            return_exceptions=True,
        )
        reflector_res, reviewers_res, scorer_res = results

        # Reflector 정규화.
        if isinstance(reflector_res, Exception):
            logger.warning("Ralph reflector 예외: %s — pass 처리", reflector_res)
            ref_ok, ref_reason, ref_missing = True, "", None
        else:
            ref_ok = bool(getattr(reflector_res, "answer_ok", True))
            ref_reason = str(getattr(reflector_res, "reason", "") or "")
            ref_missing = getattr(reflector_res, "missing_info_query", None) or None

        # Reviewer 정규화.
        if isinstance(reviewers_res, Exception):
            logger.warning("Ralph reviewers 예외: %s — pass 처리", reviewers_res)
            rv_passed = True
            rv_failed: list[str] = []
            rv_reasons: list[str] = []
            rv_missing = None
        else:
            rv_passed = bool(getattr(reviewers_res, "overall_passed", True))
            rv_failed = list(getattr(reviewers_res, "failed_reviewers", []) or [])
            verdicts = list(getattr(reviewers_res, "verdicts", []) or [])
            rv_reasons = [
                str(getattr(v, "reason", "") or "")
                for v in verdicts
                if not getattr(v, "passed", True)
            ]
            rv_missing = getattr(reviewers_res, "missing_info_query", None) or None

        # Scorer 정규화.
        if isinstance(scorer_res, Exception):
            logger.warning("Ralph scorer 예외: %s — 중립 처리", scorer_res)
            sc_score: float | None = None
            sc_ok = False
            sc_feedback = ""
            sc_imps: list[str] = []
        else:
            raw_score = getattr(scorer_res, "score", None)
            sc_score = (
                float(raw_score) if isinstance(raw_score, (int, float)) else None
            )
            sc_ok = bool(getattr(scorer_res, "ok", True))
            sc_feedback = str(getattr(scorer_res, "feedback", "") or "")
            sc_imps = list(getattr(scorer_res, "improvements", []) or [])

        return GateOutcome(
            scorer_score=sc_score,
            scorer_ok=sc_ok,
            scorer_feedback=sc_feedback,
            scorer_improvements=sc_imps,
            reflector_ok=ref_ok,
            reflector_reason=ref_reason,
            reflector_missing=ref_missing,
            reviewer_passed=rv_passed,
            reviewer_failed=rv_failed,
            reviewer_reasons=rv_reasons,
            reviewer_missing=rv_missing,
        )


__all__ += ["RAGQualityLoop"]
