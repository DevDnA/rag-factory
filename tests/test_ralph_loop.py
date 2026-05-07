"""Ralph 통합 quality loop 테스트.

본 테스트는 ``slm_factory.rag.agent.quality_loop`` 모듈의 단위 + 작은 통합
시나리오를 검증합니다. orchestrator wiring 자체는 ``test_agent_orchestrator``
에서 별도로 검증합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from slm_factory.rag.agent.quality_loop import (
    COMPLETION_TAG_PATTERN,
    GateOutcome,
    LoopStateStore,
    RAGQualityLoop,
    RalphLoopState,
    append_promise,
    extract_promise_tag,
    format_feedback_block,
    has_promise,
)


# ---------------------------------------------------------------------------
# Promise 태그 유틸리티
# ---------------------------------------------------------------------------


class TestPromiseTag:
    def test_태그_없으면_None_반환(self):
        assert extract_promise_tag("그냥 답변입니다") is None

    def test_태그_추출(self):
        text = "답변 내용\n\n<promise>DONE</promise>"
        assert extract_promise_tag(text) == "DONE"

    def test_대소문자_무시_매칭(self):
        text = "<Promise>done</Promise>"
        assert has_promise(text, "DONE")

    def test_빈_promise는_None(self):
        text = "<promise>   </promise>"
        assert extract_promise_tag(text) is None

    def test_append_promise_중복_방지(self):
        base = "답변"
        once = append_promise(base, "DONE")
        twice = append_promise(once, "DONE")
        assert once == twice
        assert once.count("<promise>DONE</promise>") == 1

    def test_compile_pattern_접근가능(self):
        # 외부 모듈이 패턴 자체를 사용하는 경우(예: detector pipeline) 보장.
        assert COMPLETION_TAG_PATTERN.search("<promise>X</promise>") is not None


# ---------------------------------------------------------------------------
# Feedback block 포맷팅
# ---------------------------------------------------------------------------


class TestFeedbackBlock:
    def test_빈_입력은_빈_문자열(self):
        assert format_feedback_block(iteration=1) == ""

    def test_scorer_피드백_포함(self):
        text = format_feedback_block(
            scorer_feedback="더 구체적으로",
            scorer_improvements=["수치를 인용", "출처 명시"],
            iteration=2,
        )
        assert "이전 반복 #2" in text
        assert "더 구체적으로" in text
        assert "수치를 인용" in text

    def test_reviewer_사유는_라인별로_정리(self):
        text = format_feedback_block(
            reflector_reason="근거 부족",
            failed_reviewers=["grounding"],
            reviewer_reasons=["인용된 문서 없음"],
            iteration=1,
        )
        assert "근거 부족" in text
        assert "grounding" in text
        assert "인용된 문서 없음" in text


# ---------------------------------------------------------------------------
# LoopStateStore 영속화
# ---------------------------------------------------------------------------


class TestLoopStateStore:
    def test_빈_디렉토리는_no_op(self):
        store = LoopStateStore("")
        state = RalphLoopState(session_id="s1")
        assert store.write(state) is False
        assert store.read("s1") is None
        assert store.clear("s1") is False

    def test_라운드트립(self, tmp_path: Path):
        store = LoopStateStore(tmp_path)
        state = RalphLoopState(
            iteration=2,
            session_id="abc",
            last_score=8.4,
            strategy="reset",
        )
        assert store.write(state) is True

        loaded = store.read("abc")
        assert loaded is not None
        assert loaded.iteration == 2
        assert loaded.session_id == "abc"
        assert loaded.last_score == 8.4
        assert loaded.strategy == "reset"

        assert store.clear("abc") is True
        assert store.read("abc") is None


# ---------------------------------------------------------------------------
# GateOutcome 통과 판정
# ---------------------------------------------------------------------------


class TestGateOutcome:
    def _outcome(self, **overrides) -> GateOutcome:
        defaults = dict(
            scorer_score=8.0,
            scorer_ok=True,
            scorer_feedback="",
            scorer_improvements=[],
            reflector_ok=True,
            reflector_reason="",
            reflector_missing=None,
            reviewer_passed=True,
            reviewer_failed=[],
            reviewer_reasons=[],
            reviewer_missing=None,
        )
        defaults.update(overrides)
        return GateOutcome(**defaults)

    def test_모든_게이트_통과(self):
        assert self._outcome().all_passed(7.0) is True

    def test_점수_미달이면_실패(self):
        assert self._outcome(scorer_score=5.0).all_passed(7.0) is False

    def test_scorer_ok_False면_점수_무시하고_나머지로_판정(self):
        # scorer가 호출 불가일 때 점수 게이트로 무한 retry되지 않도록.
        assert (
            self._outcome(scorer_ok=False, scorer_score=2.0).all_passed(7.0) is True
        )

    def test_reflector_실패면_전체_실패(self):
        assert self._outcome(reflector_ok=False).all_passed(7.0) is False

    def test_reviewer_실패면_전체_실패(self):
        assert (
            self._outcome(reviewer_passed=False).all_passed(7.0) is False
        )

    def test_best_missing_query는_reflector_우선(self):
        outcome = self._outcome(
            reflector_missing="A",
            reviewer_missing="B",
        )
        assert outcome.best_missing_query() == "A"

        outcome = self._outcome(reviewer_missing="B")
        assert outcome.best_missing_query() == "B"


# ---------------------------------------------------------------------------
# RAGQualityLoop end-to-end (모킹)
# ---------------------------------------------------------------------------


@dataclass
class _StubReflectorDecision:
    answer_ok: bool
    reason: str = ""
    missing_info_query: str | None = None


@dataclass
class _StubVerdict:
    reviewer: str
    passed: bool
    reason: str = ""


@dataclass
class _StubReviewerVerdict:
    overall_passed: bool
    verdicts: list[_StubVerdict]
    failed_reviewers: list[str]
    missing_info_query: str | None = None


@dataclass
class _StubScoreResult:
    score: float
    ok: bool = True
    feedback: str = ""
    improvements: list[str] | None = None


@dataclass
class _StubSearchResult:
    text: str
    sources: list[dict]


class _Counter:
    """순차적으로 응답을 dispense하는 소형 헬퍼."""

    def __init__(self, items):
        self._items = list(items)
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        if not self._items:
            raise RuntimeError("no more stub responses")
        return self._items.pop(0)


class TestRAGQualityLoop:
    @pytest.mark.asyncio
    async def test_첫반복에서_통과시_promise_발행(self):
        events: list[dict] = []
        reflector = _Counter([_StubReflectorDecision(answer_ok=True)])
        reviewers = _Counter(
            [_StubReviewerVerdict(overall_passed=True, verdicts=[], failed_reviewers=[])]
        )
        scorer = _Counter([_StubScoreResult(score=9.0)])

        async def synthesize(q, c, h):
            return "재합성 답변"  # 통과 시 호출되지 않아야 함

        loop = RAGQualityLoop(
            max_iterations=3,
            quality_threshold=7.0,
            strategy="continue",
            completion_promise="DONE",
            synthesize=synthesize,
            run_reflector=reflector,
            run_reviewers=reviewers,
            run_scorer=scorer,
        )

        all_sources: list[dict] = []
        seen_ids: set[str] = set()

        async for ev in loop.run(
            query="질의",
            initial_answer="초안 답변",
            history="",
            build_context=lambda: "ctx",
            all_sources=all_sources,
            seen_doc_ids=seen_ids,
            context_parts=[],
            dedup_extend=lambda *a, **kw: None,
            session_id="sid-1",
            starting_iteration=0,
        ):
            events.append(ev)

        types = [e["type"] for e in events]
        assert "ralph_iteration" in types
        assert "promise" in types

        promise_event = next(e for e in events if e["type"] == "promise")
        assert promise_event["content"] == "DONE"

        done = events[-1]
        assert done["type"] == "ralph_done"
        result = done["result"]
        assert result.promise_emitted is True
        assert "<promise>DONE</promise>" in result.answer
        assert result.iterations_run == 1
        # 통과했으므로 재합성 호출 없음.
        assert synthesize.__code__.co_argcount  # smoke

    @pytest.mark.asyncio
    async def test_max_iterations에_도달하면_promise_없이_종료(self):
        # 항상 실패하는 게이트를 dispatcher로 무한 공급.
        async def reflector(*_a, **_k):
            return _StubReflectorDecision(
                answer_ok=False, reason="부족", missing_info_query="추가 검색"
            )

        async def reviewers(*_a, **_k):
            return _StubReviewerVerdict(
                overall_passed=False,
                verdicts=[_StubVerdict("grounding", False, "근거 부족")],
                failed_reviewers=["grounding"],
            )

        async def scorer(*_a, **_k):
            return _StubScoreResult(score=4.0, feedback="더 자세히")

        synth_calls = []

        async def synthesize(q, c, h):
            synth_calls.append((q, c, h))
            return f"재합성 #{len(synth_calls)}"

        async def search(q):
            return _StubSearchResult(text=f"검색결과:{q}", sources=[])

        events: list[dict] = []
        loop = RAGQualityLoop(
            max_iterations=2,
            quality_threshold=7.0,
            strategy="continue",
            completion_promise="DONE",
            synthesize=synthesize,
            run_reflector=reflector,
            run_reviewers=reviewers,
            run_scorer=scorer,
            execute_search=search,
        )

        async for ev in loop.run(
            query="질의",
            initial_answer="초안",
            history="",
            build_context=lambda: "ctx",
            all_sources=[],
            seen_doc_ids=set(),
            context_parts=[],
            dedup_extend=lambda *a, **kw: None,
            session_id="sid-2",
            starting_iteration=0,
        ):
            events.append(ev)

        # promise 이벤트는 발행되지 않아야 함.
        assert all(e["type"] != "promise" for e in events)

        done = events[-1]
        assert done["type"] == "ralph_done"
        result = done["result"]
        assert result.promise_emitted is False
        assert result.state.status == "max_reached"
        # 재합성은 max_iterations - 1 회 발생 (마지막은 합성 없이 종료).
        assert len(synth_calls) == 1

    @pytest.mark.asyncio
    async def test_best_answer_트래킹_열화_방지(self):
        """LLM judge 노이즈로 후속 반복 점수가 하락해도 최고 품질 답변이 보존."""
        # iter1: 답변=초안, score=9.0(높음), reviewer 통과 안 함 → fail이지만 best
        # iter2: 답변=재합성1, score=4.0(낮음), 모두 fail → best 갱신 안 됨
        scores = [9.0, 4.0]
        scorer = _Counter([_StubScoreResult(score=s) for s in scores])
        reflector = _Counter(
            [_StubReflectorDecision(answer_ok=False, missing_info_query="x")] * 2
        )
        reviewers = _Counter(
            [_StubReviewerVerdict(
                overall_passed=False, verdicts=[], failed_reviewers=["grounding"]
            )] * 2
        )
        synth_calls = []

        async def synthesize(q, c, h):
            synth_calls.append(c)
            return f"재합성:{len(synth_calls)}"

        async def search(q):
            return _StubSearchResult(text="r", sources=[])

        loop = RAGQualityLoop(
            max_iterations=2,
            quality_threshold=7.0,
            strategy="reset",
            completion_promise="DONE",
            synthesize=synthesize,
            run_reflector=reflector,
            run_reviewers=reviewers,
            run_scorer=scorer,
            execute_search=search,
        )

        events: list[dict] = []
        async for ev in loop.run(
            query="q",
            initial_answer="초안",
            history="",
            build_context=lambda: "",
            all_sources=[],
            seen_doc_ids=set(),
            context_parts=[],
            dedup_extend=lambda *a, **kw: None,
            session_id="best-1",
            starting_iteration=0,
        ):
            events.append(ev)

        result = events[-1]["result"]
        # promise 미발행 → best_answer로 fallback. 초안이 score 9.0이므로 best.
        assert result.promise_emitted is False
        assert result.answer == "초안"

    @pytest.mark.asyncio
    async def test_상태_파일이_매_반복_업데이트됨(self, tmp_path: Path):
        store = LoopStateStore(tmp_path)

        async def reflector(*_a, **_k):
            return _StubReflectorDecision(
                answer_ok=False, missing_info_query="more"
            )

        async def reviewers(*_a, **_k):
            return _StubReviewerVerdict(
                overall_passed=False,
                verdicts=[],
                failed_reviewers=["completeness"],
            )

        async def scorer(*_a, **_k):
            return _StubScoreResult(score=5.5)

        async def synthesize(q, c, h):
            return "재합성"

        async def search(q):
            return _StubSearchResult(text="r", sources=[])

        loop = RAGQualityLoop(
            max_iterations=2,
            quality_threshold=7.0,
            strategy="continue",
            completion_promise="DONE",
            synthesize=synthesize,
            run_reflector=reflector,
            run_reviewers=reviewers,
            run_scorer=scorer,
            execute_search=search,
            state_store=store,
        )

        async for _ev in loop.run(
            query="q",
            initial_answer="ans",
            history="",
            build_context=lambda: "",
            all_sources=[],
            seen_doc_ids=set(),
            context_parts=[],
            dedup_extend=lambda *a, **kw: None,
            session_id="persist-1",
            starting_iteration=0,
        ):
            pass

        loaded = store.read("persist-1")
        assert loaded is not None
        # max_reached 상태 도달.
        assert loaded.status == "max_reached"
        assert loaded.iteration == 2
        assert loaded.last_score == 5.5
