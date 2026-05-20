"""AnswerVerifier 테스트 — 적대적 답변 검증, JSON 파싱, never-raise 동작."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from rag_factory.rag.agent.answer_verifier import AnswerVerdict, AnswerVerifier


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_verifier(http_client) -> AnswerVerifier:
    return AnswerVerifier(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
    )


# ---------------------------------------------------------------------------
# Happy path — 정상 verdict
# ---------------------------------------------------------------------------


class TestValidVerdict:
    """구조화된 JSON 응답."""

    @pytest.mark.asyncio
    async def test_pass_verdict(self):
        raw = '{"verdict": "PASS", "issues": [], "repair_hint": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("질문", "답변 본문", "컨텍스트")
        assert verdict.verdict == "PASS"
        assert verdict.issues == []
        assert verdict.repair_hint == ""
        assert not verdict.needs_repair

    @pytest.mark.asyncio
    async def test_fail_verdict_with_issues_and_hint(self):
        raw = (
            '{"verdict": "FAIL", "issues": ["X 주장 출처 없음", "Y와 모순"], '
            '"repair_hint": "주장 X는 인용 문서에 없음을 명시"}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "FAIL"
        assert len(verdict.issues) == 2
        assert "X" in verdict.issues[0]
        assert verdict.repair_hint.startswith("주장 X")
        assert verdict.needs_repair

    @pytest.mark.asyncio
    async def test_partial_verdict(self):
        raw = '{"verdict": "PARTIAL", "issues": ["일부 hedge로 덮인 추측"], "repair_hint": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PARTIAL"
        assert len(verdict.issues) == 1
        # PARTIAL은 repair_hint 없으면 needs_repair=False (FAIL만 재합성 대상)
        assert not verdict.needs_repair


# ---------------------------------------------------------------------------
# Verdict 문자열 변형 방어
# ---------------------------------------------------------------------------


class TestVerdictCoercion:
    """LLM이 verdict를 소문자·다른 표기로 반환할 때."""

    @pytest.mark.parametrize(
        "raw_value, expected",
        [
            ('"PASS"', "PASS"),
            ('"pass"', "PASS"),
            ('"Pass"', "PASS"),
            ('"FAIL"', "FAIL"),
            ('"fail"', "FAIL"),
            ('"PARTIAL"', "PARTIAL"),
            ('"partial"', "PARTIAL"),
        ],
    )
    @pytest.mark.asyncio
    async def test_case_insensitive_verdict(self, raw_value, expected):
        raw = f'{{"verdict": {raw_value}, "issues": [], "repair_hint": ""}}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == expected

    @pytest.mark.asyncio
    async def test_unknown_verdict는_PASS로_정규화(self):
        raw = '{"verdict": "MAYBE", "issues": [], "repair_hint": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_verdict_누락은_PASS(self):
        raw = '{"issues": ["x"], "repair_hint": "y"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"


class TestIssuesCoercion:
    """issues 필드의 다양한 형태 정규화."""

    @pytest.mark.asyncio
    async def test_issues_단일_문자열은_리스트로(self):
        raw = '{"verdict": "FAIL", "issues": "주장 X 미지지", "repair_hint": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.issues == ["주장 X 미지지"]

    @pytest.mark.asyncio
    async def test_빈_issues는_빈_리스트(self):
        raw = '{"verdict": "PASS", "issues": [], "repair_hint": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.issues == []

    @pytest.mark.asyncio
    async def test_issues_누락은_빈_리스트(self):
        raw = '{"verdict": "PASS"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.issues == []


# ---------------------------------------------------------------------------
# Never-raise — LLM 실패 시 PASS (사용자 답변 막지 않음)
# ---------------------------------------------------------------------------


class TestNeverRaise:
    """evaluate()는 어떤 실패에서도 raise하지 않고 PASS로 반환."""

    @pytest.mark.asyncio
    async def test_HTTP_타임아웃시_PASS(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"
        assert verdict.issues == []
        assert not verdict.needs_repair

    @pytest.mark.asyncio
    async def test_연결오류시_PASS(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_빈_응답시_PASS(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(""))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패시_PASS(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("garbage {not json"))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_JSON_배열은_dict_아님_PASS(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("[true]"))

        verdict = await _make_verifier(http).evaluate("q", "answer", "ctx")
        assert verdict.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_빈_답변시_LLM_호출_없이_PASS(self):
        """답변 자체가 비어있으면 LLM 호출 자체를 스킵 — 토큰 절약."""
        http = MagicMock()
        http.post = AsyncMock()

        verdict = await _make_verifier(http).evaluate("q", "   ", "ctx")
        assert verdict.verdict == "PASS"
        # LLM이 호출되지 않아야 함
        http.post.assert_not_called()


# ---------------------------------------------------------------------------
# 컨텍스트·답변 길이 제한
# ---------------------------------------------------------------------------


class TestTruncation:
    """긴 컨텍스트·답변은 길이 제한으로 잘려 LLM에 전달."""

    @pytest.mark.asyncio
    async def test_긴_컨텍스트_2000자_제한(self):
        raw = '{"verdict": "PASS", "issues": [], "repair_hint": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        long_ctx = "c" * 5000
        await _make_verifier(http).evaluate("질문", "답변", long_ctx)

        prompt = http.post.call_args.kwargs["json"]["prompt"]
        assert "c" * 2000 in prompt
        assert "c" * 2001 not in prompt

    @pytest.mark.asyncio
    async def test_긴_답변_3000자_제한(self):
        raw = '{"verdict": "PASS", "issues": [], "repair_hint": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        long_answer = "a" * 5000
        await _make_verifier(http).evaluate("질문", long_answer, "ctx")

        prompt = http.post.call_args.kwargs["json"]["prompt"]
        assert "a" * 3000 in prompt
        assert "a" * 3001 not in prompt


# ---------------------------------------------------------------------------
# AnswerVerdict 편의 속성
# ---------------------------------------------------------------------------


class TestAnswerVerdict:
    def test_FAIL_with_hint은_needs_repair(self):
        v = AnswerVerdict(verdict="FAIL", issues=["x"], repair_hint="재합성 지침")
        assert v.needs_repair

    def test_FAIL_without_hint은_needs_repair_False(self):
        v = AnswerVerdict(verdict="FAIL", issues=["x"], repair_hint="")
        assert not v.needs_repair

    def test_PARTIAL은_needs_repair_False(self):
        v = AnswerVerdict(verdict="PARTIAL", issues=["x"], repair_hint="ignored")
        assert not v.needs_repair

    def test_PASS는_needs_repair_False(self):
        v = AnswerVerdict(verdict="PASS")
        assert not v.needs_repair
