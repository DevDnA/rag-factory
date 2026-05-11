"""ReAct 파서 테스트."""

from __future__ import annotations

from slm_factory.rag.agent.parser import parse_react_output


class TestParseReactOutput:
    """parse_react_output 함수 테스트."""

    def test_정상_action_파싱(self):
        text = (
            "Thought: 사용자가 개인정보 보호법에 대해 물어봤습니다.\n"
            "Action: search\n"
            'Action Input: {"query": "개인정보 보호법"}'
        )
        step = parse_react_output(text)
        assert step.thought == "사용자가 개인정보 보호법에 대해 물어봤습니다."
        assert step.action == "search"
        assert step.action_input == {"query": "개인정보 보호법"}
        assert step.final_answer is None

    def test_final_answer_파싱(self):
        text = (
            "Thought: 충분한 정보를 수집했습니다.\n"
            "Final Answer: 개인정보 보호법 제15조에 따르면 동의 없이 수집할 수 없습니다."
        )
        step = parse_react_output(text)
        assert step.thought
        assert step.final_answer is not None
        assert "제15조" in step.final_answer
        assert step.action is None

    def test_한국어_레이블(self):
        text = (
            "생각: 검색이 필요합니다.\n"
            "행동: search\n"
            '행동 입력: {"query": "테스트"}'
        )
        step = parse_react_output(text)
        assert step.thought == "검색이 필요합니다."
        assert step.action == "search"
        assert step.action_input == {"query": "테스트"}

    def test_fallback_형식_안_따를때(self):
        text = "그냥 일반적인 답변입니다. 문서를 찾을 수 없습니다."
        step = parse_react_output(text)
        assert step.final_answer == text
        assert step.action is None
        assert step.thought == ""

    def test_빈_문자열(self):
        step = parse_react_output("")
        assert step.final_answer is None
        assert step.action is None
        assert step.raw == ""

    def test_공백만(self):
        step = parse_react_output("   \n\n  ")
        assert step.final_answer is None

    def test_action_input_json_파싱_실패시_폴백(self):
        text = (
            "Thought: 검색합니다.\n"
            "Action: search\n"
            "Action Input: 개인정보 보호법"
        )
        step = parse_react_output(text)
        assert step.action == "search"
        assert step.action_input == {"query": "개인정보 보호법"}

    def test_action_input_없을때(self):
        text = "Thought: 검색합니다.\nAction: list_documents"
        step = parse_react_output(text)
        assert step.action == "list_documents"
        assert step.action_input == {}

    def test_final_answer가_action보다_우선(self):
        text = (
            "Thought: 분석 완료.\n"
            "Action: search\n"
            "Final Answer: 최종 답변입니다."
        )
        step = parse_react_output(text)
        assert step.final_answer == "최종 답변입니다."
        assert step.action is None  # Final Answer가 있으면 action 무시

    def test_raw_필드_보존(self):
        text = "Thought: 테스트\nFinal Answer: 답변"
        step = parse_react_output(text)
        assert step.raw == text

    def test_복잡한_json_action_input(self):
        text = (
            "Thought: 비교 검색합니다.\n"
            "Action: compare\n"
            'Action Input: {"query_a": "개인정보법", "query_b": "정보통신망법"}'
        )
        step = parse_react_output(text)
        assert step.action == "compare"
        assert step.action_input["query_a"] == "개인정보법"
        assert step.action_input["query_b"] == "정보통신망법"
