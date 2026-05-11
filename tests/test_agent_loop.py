"""AgentLoop 테스트."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slm_factory.rag.agent.loop import AgentLoop, AgentEvent, AgentResult
from slm_factory.rag.agent.tools import ToolResult


class _FakeToolRegistry:
    """테스트용 도구 레지스트리."""

    def get_tool_descriptions(self):
        return "- **search**: 문서 검색\n  입력: {\"query\": \"검색어\"}"

    async def execute(self, name, args):
        if name == "search":
            return ToolResult(
                text="[문서 1] (ID: doc_001, 유사도: 0.95)\n테스트 문서 내용입니다.",
                sources=[{"doc_id": "doc_001", "score": 0.95, "content": "테스트 문서 내용입니다."}],
            )
        return ToolResult(text=f"[오류] 알 수 없는 도구 '{name}'")


def _make_ollama_response(text):
    """Ollama /api/generate 응답을 모방합니다."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": text, "done": True}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@pytest.fixture
def agent():
    """기본 AgentLoop 인스턴스를 생성합니다."""
    http_client = AsyncMock()
    tool_registry = _FakeToolRegistry()
    loop = AgentLoop(
        http_client=http_client,
        tool_registry=tool_registry,
        ollama_model="test-model",
        api_base="http://localhost:11434",
        max_iterations=3,
        max_tokens=512,
    )
    # 질문 분해를 바이패스 — 단순 질문으로 처리
    loop._decompose_query = AsyncMock(return_value=["질문"])
    return loop


class TestAgentLoop:
    """AgentLoop.run() 테스트."""

    @pytest.mark.asyncio
    async def test_단일_검색_후_답변(self, agent):
        """검색 1회 → Final Answer 시나리오."""
        responses = [
            # 1st iteration: search action
            _make_ollama_response(
                "Thought: 문서를 검색해야 합니다.\n"
                "Action: search\n"
                'Action Input: {"query": "테스트"}'
            ),
            # 2nd iteration: final answer
            _make_ollama_response(
                "Thought: 충분한 정보를 얻었습니다.\n"
                "Final Answer: 테스트 문서에 따르면 결과는 다음과 같습니다."
            ),
        ]
        agent._http_client.post = AsyncMock(side_effect=responses)

        result = await agent.run("테스트 질문")
        assert isinstance(result, AgentResult)
        assert "테스트 문서" in result.answer
        assert result.iterations == 2
        # thought, action, observation, thought, done
        types = [e.type for e in result.events]
        assert "thought" in types
        assert "action" in types
        assert "observation" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_즉시_답변(self, agent):
        """첫 반복에서 바로 Final Answer."""
        agent._http_client.post = AsyncMock(return_value=_make_ollama_response(
            "Thought: 이미 알고 있습니다.\nFinal Answer: 바로 답변합니다."
        ))
        result = await agent.run("간단한 질문")
        assert result.answer == "바로 답변합니다."
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_fallback_형식_안따를때(self, agent):
        """LLM이 ReAct 형식을 안 따르면 전체 텍스트를 답변으로 처리."""
        agent._http_client.post = AsyncMock(return_value=_make_ollama_response(
            "그냥 일반적인 텍스트 답변입니다."
        ))
        result = await agent.run("질문")
        assert result.answer == "그냥 일반적인 텍스트 답변입니다."
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_max_iterations_강제_답변(self, agent):
        """max_iterations 도달 시 강제 답변 생성."""
        search_response = _make_ollama_response(
            "Thought: 검색합니다.\n"
            "Action: search\n"
            'Action Input: {"query": "검색어"}'
        )
        forced_response = _make_ollama_response("강제로 생성된 답변입니다.")

        # 3회 반복 모두 검색, 4번째는 강제 답변
        agent._http_client.post = AsyncMock(
            side_effect=[search_response, search_response, search_response, forced_response]
        )
        result = await agent.run("반복 질문")
        assert result.iterations == 3
        assert result.answer  # 강제 답변이 있어야 함

    @pytest.mark.asyncio
    async def test_llm_호출_실패(self, agent):
        """LLM 호출 실패 시 에러 이벤트."""
        agent._http_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        result = await agent.run("질문")
        error_events = [e for e in result.events if e.type == "error"]
        assert len(error_events) == 1
        assert "LLM 호출 중 문제가 발생했습니다" in error_events[0].content

    @pytest.mark.asyncio
    async def test_소스_수집(self, agent):
        """검색 결과에서 소스 정보 추출."""
        responses = [
            _make_ollama_response(
                "Thought: 검색합니다.\n"
                "Action: search\n"
                'Action Input: {"query": "테스트"}'
            ),
            _make_ollama_response(
                "Thought: 답변합니다.\nFinal Answer: 결과입니다."
            ),
        ]
        agent._http_client.post = AsyncMock(side_effect=responses)
        result = await agent.run("질문")
        assert len(result.sources) > 0
        assert result.sources[0]["doc_id"] == "doc_001"


class TestAgentLoopStream:
    """AgentLoop.run_stream() 테스트."""

    @pytest.mark.asyncio
    async def test_스트리밍_이벤트_순서(self, agent):
        """스트리밍 시 thought → action → observation → token → done 순서."""
        call_count = 0

        async def fake_stream(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 1st iteration: search action (토큰 단위로 yield)
                text = (
                    "Thought: 검색합니다.\n"
                    "Action: search\n"
                    'Action Input: {"query": "테스트"}'
                )
            else:
                # 2nd iteration: final answer
                text = "Thought: 답변합니다.\nFinal Answer: 최종 답변."
            for chunk in [text[i:i+10] for i in range(0, len(text), 10)]:
                yield chunk

        agent._generate_stream_tokens = fake_stream
        # run_stream의 Thought/Action 파싱 fallback용
        agent._generate = AsyncMock(side_effect=[
            "Thought: 검색합니다.\nAction: search\nAction Input: {\"query\": \"테스트\"}",
        ])

        events = []
        async for event in agent.run_stream("질문"):
            events.append(event)

        types = [e.type for e in events]
        assert "thought" in types
        assert "action" in types
        assert "observation" in types
        assert "token" in types
        assert types[-1] == "done"
