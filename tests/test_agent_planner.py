"""Planner 테스트 — JSON 파싱, 검증, fallback 동작을 검증합니다."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.planner import ExecutionPlan, PlanStep, Planner


def _ollama_response(payload: str) -> MagicMock:
    """Ollama /api/generate 응답을 모방합니다."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_planner(http_client) -> Planner:
    return Planner(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
        max_steps=3,
    )


# ---------------------------------------------------------------------------
# Happy path — 구조화된 JSON 응답
# ---------------------------------------------------------------------------


class TestValidPlan:
    """LLM이 올바른 JSON을 반환할 때."""

    @pytest.mark.asyncio
    async def test_fact_strategy_단일_search(self):
        raw = (
            '{"strategy": "fact", "rationale": "단순 사실", '
            '"steps": [{"tool": "search", "args": {"query": "동의 요건"}, "reason": "검색"}]}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("개인정보 동의 요건")
        assert plan.strategy == "fact"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "search"
        assert plan.steps[0].args == {"query": "동의 요건"}
        assert not plan.is_fallback

    @pytest.mark.asyncio
    async def test_compare_strategy_compare_도구(self):
        raw = (
            '{"strategy": "compare", "rationale": "비교", '
            '"steps": [{"tool": "compare", "args": {"query_a": "A", "query_b": "B"}, '
            '"reason": "두 속성 비교"}]}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("A와 B 비교")
        assert plan.strategy == "compare"
        assert plan.steps[0].tool == "compare"
        assert plan.steps[0].args == {"query_a": "A", "query_b": "B"}

    @pytest.mark.asyncio
    async def test_decompose_여러_step(self):
        raw = (
            '{"strategy": "decompose", "rationale": "복합", '
            '"steps": ['
            '{"tool": "search", "args": {"query": "x"}, "reason": "1"},'
            '{"tool": "search", "args": {"query": "y"}, "reason": "2"},'
            '{"tool": "search", "args": {"query": "z"}, "reason": "3"}'
            ']}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("복합 질의")
        assert plan.strategy == "decompose"
        assert len(plan.steps) == 3

    @pytest.mark.asyncio
    async def test_앞뒤_텍스트가_있어도_JSON_추출(self):
        raw = (
            'Here is the plan:\n'
            '{"strategy": "fact", "steps": [{"tool": "search", "args": {"query": "x"}}]}\n'
            'End of plan.'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert not plan.is_fallback
        assert plan.steps[0].tool == "search"


# ---------------------------------------------------------------------------
# Validation — 잘못된 도구·잘못된 strategy
# ---------------------------------------------------------------------------


class TestValidation:
    """Planner의 화이트리스트 검증."""

    @pytest.mark.asyncio
    async def test_알_수_없는_도구는_drop(self):
        raw = (
            '{"strategy": "fact", "steps": ['
            '{"tool": "evil_tool", "args": {}, "reason": "bad"},'
            '{"tool": "search", "args": {"query": "x"}, "reason": "ok"}'
            ']}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "search"

    @pytest.mark.asyncio
    async def test_잘못된_strategy는_fact로_정규화(self):
        raw = '{"strategy": "weird", "steps": [{"tool": "search", "args": {"query": "x"}}]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert plan.strategy == "fact"

    @pytest.mark.asyncio
    async def test_max_steps_초과는_자름(self):
        steps_json = ",".join(
            f'{{"tool": "search", "args": {{"query": "q{i}"}}}}'
            for i in range(10)
        )
        raw = f'{{"strategy": "decompose", "steps": [{steps_json}]}}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert len(plan.steps) == 3

    @pytest.mark.asyncio
    async def test_steps가_비면_fallback(self):
        raw = '{"strategy": "fact", "steps": []}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "search"

    @pytest.mark.asyncio
    async def test_모든_도구가_drop되면_fallback(self):
        raw = '{"strategy": "fact", "steps": [{"tool": "evil", "args": {}}]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback


# ---------------------------------------------------------------------------
# Fallback — LLM 실패, 파싱 실패
# ---------------------------------------------------------------------------


class TestFallback:
    """plan()은 어떤 입력에서도 raise하지 않습니다."""

    @pytest.mark.asyncio
    async def test_HTTP_타임아웃시_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback
        assert "llm-error" in plan.rationale
        assert plan.steps[0].tool == "search"
        assert plan.steps[0].args == {"query": "질의"}

    @pytest.mark.asyncio
    async def test_HTTP_500시_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
        )

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback

    @pytest.mark.asyncio
    async def test_빈_응답시_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(""))

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback
        assert "parse-error" in plan.rationale

    @pytest.mark.asyncio
    async def test_잘못된_JSON_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("{not json at all"))

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback

    @pytest.mark.asyncio
    async def test_JSON이지만_array는_dict가_아님_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response('[1, 2, 3]'))

        plan = await _make_planner(http).plan("질의")
        assert plan.is_fallback

    @pytest.mark.asyncio
    async def test_think_태그_제거(self):
        raw = (
            '<think>이건 제거되어야 함</think>'
            '{"strategy": "fact", "steps": [{"tool": "search", "args": {"query": "x"}}]}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        plan = await _make_planner(http).plan("질의")
        assert not plan.is_fallback
        assert plan.steps[0].args == {"query": "x"}


class TestExecutionPlan:
    """ExecutionPlan dataclass 헬퍼."""

    def test_is_fallback_속성(self):
        fallback = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search")],
            rationale="fallback: parse-error",
        )
        normal = ExecutionPlan(
            strategy="fact",
            steps=[PlanStep(tool="search")],
            rationale="일반 계획",
        )
        assert fallback.is_fallback
        assert not normal.is_fallback
